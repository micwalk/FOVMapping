using System;
using System.Linq;
using UnityEditor;
using UnityEngine;
using FOVMapping;
using Unity.Jobs;
using Unity.Collections;

namespace FOVMapping
{

/// <summary>
/// FOVMapGenerator creates Field of View maps for fog of war systems using a sophisticated three-stage algorithm.
/// 
/// ALGORITHM OVERVIEW:
/// The FOV map generation process consists of three distinct stages that work together to create
/// a comprehensive visibility map for each grid cell across multiple directions.
/// 
/// STAGE 1: GROUND HEIGHT DETECTION
/// - Raycast downward from maximum height (5000 units) at each grid cell center
/// - Find the terrain/ground level to establish the base height for visibility sampling
/// - Calculate the "eye position" by adding eyeHeight to the ground level
/// - One raycast per grid cell (FOVMapWidth × FOVMapHeight total rays)
/// - Cells without ground are filled with white (maximum visibility)
/// 
/// STAGE 2: OBSTACLE DETECTION PER DIRECTION
/// - For each valid grid cell, sample multiple directions (4 channels × layerCount directions)
/// - For each direction, cast multiple rays at different vertical angles (samplesPerDirection rays)
/// - Uses "level-adaptive multisampling" to find maximum visible distance before hitting obstacles
/// - Initial sampling uses fixed vertical angles from -samplingAngle/2 to +samplingAngle/2
/// - Tracks the maximum sight distance achieved before hitting terrain obstacles
/// 
/// STAGE 3: BINARY SEARCH EDGE REFINEMENT
/// - When a ray transitions from "hit obstacle" to "no hit", perform binary search
/// - Find the precise vertical angle where terrain visibility ends (e.g., top of a wall)
/// - Uses binarySearchCount iterations to narrow down the exact angle
/// - Early termination if surface is nearly vertical (blockingSurfaceAngleThreshold)
/// - This creates smooth transitions between visible and blocked areas
/// 
/// PERFORMANCE OPTIMIZATION:
/// This implementation uses Unity's RaycastCommand API for batched raycasting, allowing
/// massive parallelization across all CPU cores. All cells executing the same iteration
/// step are batched together, eliminating the synchronous Physics.Raycast bottleneck.
/// 
/// OUTPUT:
/// Creates a Texture2DArray where each layer represents a direction range, and each
/// texel stores the visibility ratio (0-1) for that direction from that grid position.
/// </summary>
public static class FOVMapGenerator
{
	private const int CHANNELS_PER_TEXEL = 4;

	/// <summary>
	/// Tracks the state of a raycast batch operation
	/// </summary>
	private struct RaycastBatchState
	{
		public NativeArray<RaycastCommand> commands;
		public NativeArray<RaycastHit> results;
		public JobHandle handle;
		public int commandCount;
	}

	/// <summary>
	/// Tracks binary search progress for a specific cell-direction combination
	/// </summary>
	private struct DirectionSamplingState
	{
		public int cellX;
		public int cellZ;
		public int directionIdx;
		public float currentAngle;
		public float angleInterval;
		public int iteration;
		public bool needsBinarySearch;
		public float maxSight;
		public bool obstacleHit;
	}

	/// <summary>
	/// Ground height data for a grid cell
	/// </summary>
	private struct GroundHeightData
	{
		public Vector3 centerPosition;
		public float height;
		public bool hasGround;
	}

	#region Helper Methods for Batched Raycasting

	/// <summary>
	/// Calculates the optimal batch size based on settings and available memory
	/// </summary>
	private static int CalculateOptimalBatchSize(FOVMapGenerationInfo generationInfo, int totalRays)
	{
		return Mathf.Min(generationInfo.maxBatchSize, totalRays);
	}

	/// <summary>
	/// Creates a batch of ground detection raycasts for Stage 1
	/// </summary>
	private static GroundHeightData[] ProcessGroundRaycastsInBatches(FOVMapGenerationInfo generationInfo, Func<int, int, bool> progressAction)
	{
		const float MAX_HEIGHT = 5000.0f;
		int totalCells = generationInfo.FOVMapWidth * generationInfo.FOVMapHeight;
		int batchSize = CalculateOptimalBatchSize(generationInfo, totalCells);
		
		GroundHeightData[] groundData = new GroundHeightData[totalCells];
		
		float planeSizeX = generationInfo.plane.localScale.x;
		float planeSizeZ = generationInfo.plane.localScale.z;
		
		int totalBatches = (totalCells + batchSize - 1) / batchSize; // Ceiling division
		
		for (int startIndex = 0; startIndex < totalCells; startIndex += batchSize)
		{
			int currentBatchSize = Mathf.Min(batchSize, totalCells - startIndex);
			int currentBatch = startIndex / batchSize;
			
			// Update progress (0% to 20% for ground detection)
			int progressPercent = 0 + (currentBatch * 20) / totalBatches;
			if (progressAction.Invoke(progressPercent, 100)) return null;
			
			NativeArray<RaycastCommand> commands = new NativeArray<RaycastCommand>(currentBatchSize, Allocator.TempJob);
			NativeArray<RaycastHit> results = new NativeArray<RaycastHit>(currentBatchSize, Allocator.TempJob);
			
			// Build commands for this batch
			for (int i = 0; i < currentBatchSize; ++i)
			{
				int globalIndex = startIndex + i;
				int squareZ = globalIndex / generationInfo.FOVMapWidth;
				int squareX = globalIndex % generationInfo.FOVMapWidth;
				
				// Position above the sampling point
				Vector3 rayOriginPosition =
					generationInfo.plane.position +
					((squareZ + 0.5f) / generationInfo.FOVMapHeight) * planeSizeZ * generationInfo.plane.forward +
					((squareX + 0.5f) / generationInfo.FOVMapWidth) * planeSizeX * generationInfo.plane.right;
				rayOriginPosition.y = MAX_HEIGHT;

				commands[i] = new RaycastCommand
				{
					from = rayOriginPosition,
					direction = Vector3.down,
					distance = 2 * MAX_HEIGHT,
					queryParameters = new QueryParameters
					{
						layerMask = generationInfo.levelLayer,
						hitTriggers = QueryTriggerInteraction.Ignore
					}
				};
			}
			
			// Execute batch
			JobHandle handle = RaycastCommand.ScheduleBatch(commands, results, 1, 1);
			handle.Complete();
			
			// Process results
			for (int i = 0; i < currentBatchSize; ++i)
			{
				int globalIndex = startIndex + i;
				RaycastHit hit = results[i];
				
				if (hit.collider != null)
				{
					Vector3 centerPosition = hit.point + generationInfo.eyeHeight * Vector3.up;
					float height = hit.point.y - generationInfo.plane.position.y;
					
					groundData[globalIndex] = new GroundHeightData
					{
						centerPosition = centerPosition,
						height = height,
						hasGround = true
					};
				}
				else
				{
					groundData[globalIndex] = new GroundHeightData
					{
						centerPosition = Vector3.zero,
						height = 0,
						hasGround = false
					};
				}
			}
			
			// Cleanup
			commands.Dispose();
			results.Dispose();
		}
		
		return groundData;
	}

    /// <summary>
    /// Processes direction sampling raycasts in configurable batches
    /// Ensures that each state's full set of samples is processed within the same batch
    /// </summary>
    private static DirectionSamplingState[] ProcessDirectionSamplingInBatches(FOVMapGenerationInfo generationInfo, GroundHeightData[] groundData, Func<int, int, bool> progressAction)
    {
        int directionsPerSquare = CHANNELS_PER_TEXEL * generationInfo.layerCount;
        int totalStates = generationInfo.FOVMapWidth * generationInfo.FOVMapHeight * directionsPerSquare;
        DirectionSamplingState[] samplingStates = new DirectionSamplingState[totalStates];

        // Initialize all states first
        int stateIndexInit = 0;
        for (int squareZ = 0; squareZ < generationInfo.FOVMapHeight; ++squareZ)
        {
            for (int squareX = 0; squareX < generationInfo.FOVMapWidth; ++squareX)
            {
                int cellIndex = squareZ * generationInfo.FOVMapWidth + squareX;
                GroundHeightData ground = groundData[cellIndex];

                for (int directionIdx = 0; directionIdx < directionsPerSquare; ++directionIdx)
                {
                    if (!ground.hasGround)
                    {
                        samplingStates[stateIndexInit] = new DirectionSamplingState
                        {
                            cellX = squareX,
                            cellZ = squareZ,
                            directionIdx = directionIdx,
                            needsBinarySearch = false,
                            maxSight = generationInfo.samplingRange,
                            obstacleHit = false
                        };
                    }
                    else
                    {
                        samplingStates[stateIndexInit] = new DirectionSamplingState
                        {
                            cellX = squareX,
                            cellZ = squareZ,
                            directionIdx = directionIdx,
                            // Will be set to the crossing angle if binary search is needed
                            currentAngle = 0.0f,
                            angleInterval = 0.0f,
                            iteration = 0,
                            needsBinarySearch = false,
                            maxSight = 0.0f,
                            obstacleHit = false
                        };
                    }
                    stateIndexInit++;
                }
            }
        }

        // Build a list of eligible states (with ground)
        System.Collections.Generic.List<int> eligibleStateIndices = new System.Collections.Generic.List<int>(totalStates);
        for (int i = 0; i < samplingStates.Length; ++i)
        {
            DirectionSamplingState s = samplingStates[i];
            if (groundData[s.cellZ * generationInfo.FOVMapWidth + s.cellX].hasGround)
            {
                eligibleStateIndices.Add(i);
            }
        }

        int totalEligibleStates = eligibleStateIndices.Count;
        int totalRays = totalEligibleStates * generationInfo.samplesPerDirection;

        // Determine batch sizing and force batches to contain full states (no partial state across batches)
        int rawBatchSize = CalculateOptimalBatchSize(generationInfo, totalRays);
        if (rawBatchSize < generationInfo.samplesPerDirection) rawBatchSize = generationInfo.samplesPerDirection;
        // Round down to a multiple of samplesPerDirection
        int batchSizeRays = rawBatchSize - (rawBatchSize % generationInfo.samplesPerDirection);
        if (batchSizeRays == 0) batchSizeRays = generationInfo.samplesPerDirection;

        int statesPerBatch = Mathf.Max(1, batchSizeRays / generationInfo.samplesPerDirection);
        int totalBatches = (totalEligibleStates + statesPerBatch - 1) / statesPerBatch;

        const float RAY_DISTANCE = 1000.0f;
        float anglePerDirection = 360.0f / directionsPerSquare;
        float anglePerSample = generationInfo.samplingAngle / (generationInfo.samplesPerDirection - 1);

        for (int batchIndex = 0; batchIndex < totalBatches; ++batchIndex)
        {
            // Update progress (25% to 70% for direction sampling)
            int progressPercent = 25 + (batchIndex * 45) / Mathf.Max(1, totalBatches - 1);
            if (progressAction.Invoke(progressPercent, 100)) return null;

            int start = batchIndex * statesPerBatch;
            int endExclusive = Mathf.Min(start + statesPerBatch, totalEligibleStates);
            int batchStatesCount = endExclusive - start;
            if (batchStatesCount <= 0) break;

            int currentBatchSize = batchStatesCount * generationInfo.samplesPerDirection;

            NativeArray<RaycastCommand> commands = new NativeArray<RaycastCommand>(currentBatchSize, Allocator.TempJob);
            NativeArray<RaycastHit> results = new NativeArray<RaycastHit>(currentBatchSize, Allocator.TempJob);

            int commandIndex = 0;
            // Build commands: full set of samples for each state in this batch
            for (int idx = start; idx < endExclusive; ++idx)
            {
                int stateIndex = eligibleStateIndices[idx];
                DirectionSamplingState state = samplingStates[stateIndex];

                GroundHeightData ground = groundData[state.cellZ * generationInfo.FOVMapWidth + state.cellX];
                float angleToward = Vector3.SignedAngle(generationInfo.plane.right, Vector3.right, Vector3.up) + state.directionIdx * anglePerDirection;
                Vector3 samplingDirection = new Vector3(Mathf.Cos(angleToward * Mathf.Deg2Rad), 0.0f, Mathf.Sin(angleToward * Mathf.Deg2Rad));

                for (int samplingIdx = 0; samplingIdx < generationInfo.samplesPerDirection; ++samplingIdx)
                {
                    float samplingAngle = -generationInfo.samplingAngle / 2.0f + samplingIdx * anglePerSample;
                    Vector3 samplingLine = samplingDirection;
                    samplingLine.y = samplingLine.magnitude * Mathf.Tan(samplingAngle * Mathf.Deg2Rad);
                    samplingLine.Normalize();

                    commands[commandIndex++] = new RaycastCommand
                    {
                        from = ground.centerPosition,
                        direction = samplingLine,
                        distance = RAY_DISTANCE,
                        queryParameters = new QueryParameters
                        {
                            layerMask = generationInfo.levelLayer,
                            hitTriggers = QueryTriggerInteraction.Ignore
                        }
                    };
                }
            }

            // Execute batch
            JobHandle handle = RaycastCommand.ScheduleBatch(commands, results, 1, 1);
            handle.Complete();

            // Process results: consume exactly samplesPerDirection results per state, in the same order
            int resultIdx = 0;
            for (int idx = start; idx < endExclusive; ++idx)
            {
                int sIndex = eligibleStateIndices[idx];
                DirectionSamplingState state = samplingStates[sIndex];
                GroundHeightData ground = groundData[state.cellZ * generationInfo.FOVMapWidth + state.cellX];

                float maxSight = 0.0f;
                bool obstacleHit = false;
                bool needsBinarySearch = false;

                // Track the first no-hit above the eye after a hit to seed binary search
                int crossingSamplingIdx = -1;

                for (int samplingIdx = 0; samplingIdx < generationInfo.samplesPerDirection; ++samplingIdx)
                {
                    RaycastHit hit = results[resultIdx++];

                    if (hit.collider != null)
                    {
                        obstacleHit = true;
                        Vector3 centerXZ = new Vector3(ground.centerPosition.x, 0, ground.centerPosition.z);
                        Vector3 hitXZ = new Vector3(hit.point.x, 0, hit.point.z);
                        float blockedDistance = Vector3.Distance(centerXZ, hitXZ);
                        if (blockedDistance > maxSight)
                        {
                            maxSight = Mathf.Clamp(blockedDistance, 0.0f, generationInfo.samplingRange);
                        }
                    }
                    else if (samplingIdx <= (generationInfo.samplesPerDirection + 2 - 1) / 2)
                    {
                        // No hit below the eye line yields a maximum sight
                        maxSight = generationInfo.samplingRange;
                    }
                    else if (obstacleHit)
                    {
                        // Transition detected: previous hit then miss -> seed binary search
                        needsBinarySearch = true;
                        crossingSamplingIdx = samplingIdx;
                        break;
                    }
                }

                // Seed binary search parameters exactly like the original algorithm
                if (needsBinarySearch)
                {
                    float samplingAngleAtCross = -generationInfo.samplingAngle / 2.0f + crossingSamplingIdx * anglePerSample;
                    state.angleInterval = anglePerSample / 2.0f;
                    state.currentAngle = samplingAngleAtCross - state.angleInterval;
                    state.iteration = 0;
                }

                state.maxSight = maxSight;
                state.obstacleHit = obstacleHit;
                state.needsBinarySearch = needsBinarySearch;
                samplingStates[sIndex] = state;
            }

            // Cleanup
            commands.Dispose();
            results.Dispose();
        }

        return samplingStates;
    }



	/// <summary>
	/// Creates a batch of binary search raycasts for one iteration
	/// </summary>
	private static RaycastBatchState CreateBinarySearchBatch(FOVMapGenerationInfo generationInfo, DirectionSamplingState[] samplingStates, GroundHeightData[] groundData, int iteration)
	{
		// Count how many states need binary search at this iteration
		int batchCount = 0;
		foreach (var state in samplingStates)
		{
			if (state.needsBinarySearch && state.iteration == iteration)
			{
				batchCount++;
			}
		}
		
		if (batchCount == 0)
		{
			return new RaycastBatchState { commandCount = 0 };
		}
		
		NativeArray<RaycastCommand> commands = new NativeArray<RaycastCommand>(batchCount, Allocator.TempJob);
		NativeArray<RaycastHit> results = new NativeArray<RaycastHit>(batchCount, Allocator.TempJob);
		const float RAY_DISTANCE = 1000.0f;
		
		int commandIndex = 0;
		
		for (int i = 0; i < samplingStates.Length; ++i)
		{
			DirectionSamplingState state = samplingStates[i];
			
			if (!state.needsBinarySearch || state.iteration != iteration) continue;
			
			int cellIndex = state.cellZ * generationInfo.FOVMapWidth + state.cellX;
			GroundHeightData ground = groundData[cellIndex];
			
			float angleToward = Vector3.SignedAngle(generationInfo.plane.right, Vector3.right, Vector3.up) + state.directionIdx * (360.0f / (CHANNELS_PER_TEXEL * generationInfo.layerCount));
			Vector3 samplingDirection = new Vector3(Mathf.Cos(angleToward * Mathf.Deg2Rad), 0.0f, Mathf.Sin(angleToward * Mathf.Deg2Rad));
			
			Vector3 samplingLine = samplingDirection;
			samplingLine.y = samplingLine.magnitude * Mathf.Tan(state.currentAngle * Mathf.Deg2Rad);
			samplingLine.Normalize();
			
			commands[commandIndex] = new RaycastCommand
			{
				from = ground.centerPosition,
				direction = samplingLine,
				distance = RAY_DISTANCE,
				queryParameters = new QueryParameters
				{
					layerMask = generationInfo.levelLayer,
					hitTriggers = QueryTriggerInteraction.Ignore
				}
			};
			
			commandIndex++;
		}
		
		JobHandle handle = RaycastCommand.ScheduleBatch(commands, results, 1, 1);
		
		return new RaycastBatchState
		{
			commands = commands,
			results = results,
			handle = handle,
			commandCount = batchCount
		};
	}

	/// <summary>
	/// Processes binary search results and updates sampling states
	/// </summary>
	private static void ProcessBinarySearchResults(RaycastBatchState batchState, DirectionSamplingState[] samplingStates, FOVMapGenerationInfo generationInfo, GroundHeightData[] groundData, int iteration)
	{
		batchState.handle.Complete();
		
		int resultIndex = 0;
		int processedSearches = 0;
		int completedSearches = 0;
		
		for (int i = 0; i < samplingStates.Length; ++i)
		{
			DirectionSamplingState state = samplingStates[i];
			
			if (!state.needsBinarySearch || state.iteration != iteration) continue;
			
			processedSearches++;
			RaycastHit hit = batchState.results[resultIndex];
			resultIndex++;
			
			int cellIndex = state.cellZ * generationInfo.FOVMapWidth + state.cellX;
			GroundHeightData ground = groundData[cellIndex];
			
			if (hit.collider != null)
			{
				// Next range is the upper half
				state.currentAngle += state.angleInterval;
				
				// Update maxSight using XZ distance only (horizontal distance)
				Vector3 centerXZ = new Vector3(ground.centerPosition.x, 0, ground.centerPosition.z);
				Vector3 hitXZ = new Vector3(hit.point.x, 0, hit.point.z);
				float searchedDistance = Vector3.Distance(centerXZ, hitXZ);
				if (searchedDistance >= state.maxSight)
				{
					state.maxSight = Mathf.Clamp(searchedDistance, 0.0f, generationInfo.samplingRange);
				}
			}
			else
			{
				// Next range is the lower half
				state.currentAngle -= state.angleInterval;
			}
			
			// Update state
			state.angleInterval /= 2.0f;
			state.iteration++;
			
			// Check if we should continue binary search
			if (state.iteration >= generationInfo.binarySearchCount)
			{
				state.needsBinarySearch = false;
				completedSearches++;
			}
			
			samplingStates[i] = state;
		}
		
		// Debug output for first few iterations
		if (iteration < 3)
		{
			Debug.Log($"FOVMapGenerator: Binary search iteration {iteration + 1} - Processed: {processedSearches}, Completed: {completedSearches}, binarySearchCount: {generationInfo.binarySearchCount}");
		}
	}

	/// <summary>
	/// Processes binary search synchronously for a single direction sampling state
	/// Based on the original FindNearestObstacle algorithm
	/// </summary>
	private static void ProcessBinarySearchSynchronous(DirectionSamplingState state, FOVMapGenerationInfo generationInfo, GroundHeightData[] groundData)
	{
		// Get ground data for this cell
		int cellIndex = state.cellZ * generationInfo.FOVMapWidth + state.cellX;
		GroundHeightData ground = groundData[cellIndex];
		Vector3 centerPosition = ground.centerPosition;
		
		// Calculate direction
		float anglePerDirection = 360.0f / (CHANNELS_PER_TEXEL * generationInfo.layerCount);
		float angleToward = Vector3.SignedAngle(generationInfo.plane.right, Vector3.right, Vector3.up) + state.directionIdx * anglePerDirection;
		Vector3 samplingDirection = new Vector3(Mathf.Cos(angleToward * Mathf.Deg2Rad), 0.0f, Mathf.Sin(angleToward * Mathf.Deg2Rad));
		
		// Calculate angle per sample
		float anglePerSample = generationInfo.samplingAngle / (generationInfo.samplesPerDirection - 1);
		
		// Perform binary search using the original algorithm logic
		float angularInterval = anglePerSample / 2.0f;
		float searchingAngle = state.currentAngle - angularInterval;
		
		const float RAY_DISTANCE = 1000.0f;
		
		for (int i = 0; i < generationInfo.binarySearchCount; ++i)
		{
			angularInterval /= 2.0f;
			
			Vector3 searchingLine = samplingDirection;
			searchingLine.y = searchingLine.magnitude * Mathf.Tan(searchingAngle * Mathf.Deg2Rad);
			
			RaycastHit hitSearched;
			if (Physics.Raycast(centerPosition, searchingLine, out hitSearched, RAY_DISTANCE, generationInfo.levelLayer))
			{
				searchingAngle = searchingAngle + angularInterval; // Next range is the upper half
				
				// Update maxSight using XZ distance only (horizontal distance)
				Vector3 centerXZ = new Vector3(centerPosition.x, 0, centerPosition.z);
				Vector3 hitXZ = new Vector3(hitSearched.point.x, 0, hitSearched.point.z);
				float searchedDistance = Vector3.Distance(centerXZ, hitXZ);
				if (searchedDistance >= state.maxSight)
				{
					state.maxSight = Mathf.Clamp(searchedDistance, 0.0f, generationInfo.samplingRange);
				}
			}
			else
			{
				searchingAngle = searchingAngle - angularInterval; // Next range is the lower half
			}
		}
		
		// Mark binary search as complete
		state.needsBinarySearch = false;
	}

	/// <summary>
	/// Disposes of raycast batch resources
	/// </summary>
	private static void DisposeRaycastBatch(RaycastBatchState batchState)
	{
		if (batchState.commands.IsCreated)
			batchState.commands.Dispose();
		if (batchState.results.IsCreated)
			batchState.results.Dispose();
	}

	#endregion

	public static bool CreateFOVMap(FOVMapGenerationInfo generationInfo, Func<int, int, bool> progressAction)
	{
		string FOVMapPath = $"Assets/{generationInfo.path}/{generationInfo.fileName}.asset";

		Texture2DArray FOVMapArray = GenerateFOVMap(generationInfo, progressAction);
		if (FOVMapArray == null) return false;

		// Save the maps
		FOVMapArray.mipMapBias = generationInfo.samplingRange; // Store sampling range in mipMapBias field to use in FOVManager
		try
		{
			AssetDatabase.CreateAsset(FOVMapArray, FOVMapPath);
			// Don't force refresh - let Unity handle it naturally to avoid domain reload
		}
		catch (Exception e)
		{
			Debug.LogError(e.ToString());
			return false;
		}

		return true;
	}

	private static Texture2DArray GenerateFOVMap(FOVMapGenerationInfo generationInfo, Func<int, int, bool> progressAction)
	{
		// Basic checks
		bool checkPassed = true;

		if (generationInfo.plane == null)
		{
			Debug.LogError("No FOW plane has been assigned.");
			checkPassed = false;
		}

		if (string.IsNullOrEmpty(generationInfo.path) || string.IsNullOrEmpty(generationInfo.fileName))
		{
			Debug.LogError("Either path or file name have not been assigned.");
			checkPassed = false;
		}

		if (generationInfo.FOVMapWidth == 0 || generationInfo.FOVMapHeight == 0)
		{
			Debug.LogError("Incorrect texture size.");
			checkPassed = false;
		}

		if (generationInfo.samplingRange <= 0.0f)
		{
			Debug.LogError("Sampling range must be greater than zero.");
			checkPassed = false;
		}

		if (generationInfo.levelLayer == 0)
		{
			Debug.LogError("Level layer must be non-zero.");
			checkPassed = false;
		}

		if (progressAction == null)
		{
			Debug.LogError("progressAction must be pased.");
			checkPassed = false;
		}

		if (!checkPassed) return null;

		// STAGE 1: Ground Height Detection using batched raycasts
		Debug.Log("FOVMapGenerator: Starting Stage 1 - Ground Height Detection");
		if (progressAction.Invoke(0, 100)) return null; // 0% - Starting ground detection
		
		GroundHeightData[] groundData = ProcessGroundRaycastsInBatches(generationInfo, progressAction);
		
		if (progressAction.Invoke(20, 100)) return null; // 20% - Ground detection complete

		// STAGE 2: Direction Sampling using batched raycasts
		Debug.Log("FOVMapGenerator: Starting Stage 2 - Direction Sampling");
		if (progressAction.Invoke(25, 100)) return null; // 25% - Starting direction sampling
		
		DirectionSamplingState[] samplingStates = ProcessDirectionSamplingInBatches(generationInfo, groundData, progressAction);
		
		// Debug: Check if any obstacles were detected
		int obstaclesDetected = 0;
		int binarySearchesNeeded = 0;
		for (int i = 0; i < samplingStates.Length; ++i)
		{
			if (samplingStates[i].obstacleHit) obstaclesDetected++;
			if (samplingStates[i].needsBinarySearch) binarySearchesNeeded++;
		}
		Debug.Log($"FOVMapGenerator: Stage 2 complete - Obstacles detected: {obstaclesDetected}, Binary searches needed: {binarySearchesNeeded}");
		
		if (progressAction.Invoke(70, 100)) return null; // 70% - Direction sampling complete

		// STAGE 3: Simple Binary Search Edge Refinement (synchronous)
		Debug.Log("FOVMapGenerator: Starting Stage 3 - Binary Search Edge Refinement");
		
		// Process binary search synchronously for each state that needs it
		int totalBinarySearches = 0;
		for (int i = 0; i < samplingStates.Length; ++i)
		{
			if (samplingStates[i].needsBinarySearch)
			{
				totalBinarySearches++;
			}
		}
		
		Debug.Log($"FOVMapGenerator: Processing {totalBinarySearches} binary searches synchronously");
		
		int processedSearches = 0;
		for (int i = 0; i < samplingStates.Length; ++i)
		{
			DirectionSamplingState state = samplingStates[i];
			if (!state.needsBinarySearch) continue;
			
			// Perform binary search synchronously
			ProcessBinarySearchSynchronous(state, generationInfo, groundData);
			samplingStates[i] = state;
			
			processedSearches++;
			
			// Update progress every 100 searches
			if (processedSearches % 100 == 0)
			{
				int progress = 70 + (int)((processedSearches * 25.0f) / totalBinarySearches);
				if (progressAction.Invoke(progress, 100)) return null;
			}
		}
		
		if (progressAction.Invoke(95, 100)) return null; // 95% - Binary search complete

		// Convert sampling states to FOV map texture data
		Debug.Log("FOVMapGenerator: Converting results to texture data");
		
		// Debug: Check final state of sampling states
		int finalObstacles = 0;
		int finalBinarySearches = 0;
		float maxSightValue = 0f;
		int zeroSightCount = 0;
		for (int i = 0; i < samplingStates.Length; ++i)
		{
			if (samplingStates[i].obstacleHit) finalObstacles++;
			if (samplingStates[i].needsBinarySearch) finalBinarySearches++;
			if (samplingStates[i].maxSight > maxSightValue) maxSightValue = samplingStates[i].maxSight;
			if (samplingStates[i].maxSight == 0f) zeroSightCount++;
		}
		Debug.Log($"FOVMapGenerator: Final state - Obstacles: {finalObstacles}, Binary searches: {finalBinarySearches}, Max sight: {maxSightValue}, Zero sight count: {zeroSightCount}");
		
		Color[][] FOVMapTexels = Enumerable.Range(0, generationInfo.layerCount).Select(_ => new Color[generationInfo.FOVMapWidth * generationInfo.FOVMapHeight]).ToArray();
		
		int directionsPerSquare = CHANNELS_PER_TEXEL * generationInfo.layerCount;
		int stateIndex = 0;
		int totalCells = generationInfo.FOVMapWidth * generationInfo.FOVMapHeight;
		int processedCells = 0;
		
		// Debug: Track distance ratio statistics
		int whitePixels = 0;
		int darkPixels = 0;
		float minRatio = float.MaxValue;
		float maxRatio = float.MinValue;
		
		for (int squareZ = 0; squareZ < generationInfo.FOVMapHeight; ++squareZ)
		{
			for (int squareX = 0; squareX < generationInfo.FOVMapWidth; ++squareX)
			{
				for (int directionIdx = 0; directionIdx < directionsPerSquare; ++directionIdx)
				{
					DirectionSamplingState state = samplingStates[stateIndex];
					stateIndex++;
					
					float distanceRatio = state.maxSight == 0.0f ? 1.0f : state.maxSight / generationInfo.samplingRange;
					
					// Debug: Track ratio statistics
					if (distanceRatio >= 0.9f) whitePixels++;
					else darkPixels++;
					if (distanceRatio < minRatio) minRatio = distanceRatio;
					if (distanceRatio > maxRatio) maxRatio = distanceRatio;
					
					// Find the location to store
					int layerIdx = directionIdx / CHANNELS_PER_TEXEL;
					int channelIdx = directionIdx % CHANNELS_PER_TEXEL;
					
					// Store
					FOVMapTexels[layerIdx][squareZ * generationInfo.FOVMapWidth + squareX][channelIdx] = distanceRatio;
				}
				
				processedCells++;
				// Update progress every 5% of cells processed (95% to 99%)
				if (processedCells % (totalCells / 20) == 0)
				{
					int progressPercent = 95 + (processedCells * 4) / totalCells;
					if (progressAction.Invoke(progressPercent, 100)) return null;
				}
			}
		}

		// Debug: Output texture statistics
		Debug.Log($"FOVMapGenerator: Texture conversion complete - White pixels: {whitePixels}, Dark pixels: {darkPixels}, Min ratio: {minRatio:F3}, Max ratio: {maxRatio:F3}");
		
		// Store the FOV info in a texture array
		if (progressAction.Invoke(99, 100)) return null; // 99% - Creating texture array
		
		bool isLinear = (PlayerSettings.colorSpace == ColorSpace.Linear);
		Texture2DArray textureArray = new Texture2DArray(generationInfo.FOVMapWidth, generationInfo.FOVMapHeight, generationInfo.layerCount, TextureFormat.RGBA32, false, isLinear);
		textureArray.filterMode = FilterMode.Bilinear;
		textureArray.wrapMode = TextureWrapMode.Clamp;

		for (int layerIdx = 0; layerIdx < generationInfo.layerCount; ++layerIdx)
		{
			textureArray.SetPixels(FOVMapTexels[layerIdx], layerIdx, 0);
		}

		return textureArray;
	}

	// Original Single threaded version of the nearest obstacle finder
	private static float FindNearestObstacle(FOVMapGenerationInfo generationInfo, float angleToward, float anglePerSample, Vector3 centerPosition) {
		Vector3 DirectionFromAngle(float angle) {
			return new Vector3(Mathf.Cos(angle * Mathf.Deg2Rad), 0.0f, Mathf.Sin(angle * Mathf.Deg2Rad));
		}

		float XZDistance(Vector3 v1, Vector3 v2) {
			v1.y = 0.0f;
			v2.y = 0.0f;
			return Vector3.Distance(v1, v2);
		}
		
		const float RAY_DISTANCE = 1000.0f;
		
		// Level-adaptive multisampling
		float maxSight = 0.0f; // Maximum sight viewed from the center

		Vector3 samplingDirection = DirectionFromAngle(angleToward);
		float samplingInterval = generationInfo.samplingRange / generationInfo.samplesPerDirection;
		bool obstacleHit = false;
		for (int samplingIdx = 0; samplingIdx < generationInfo.samplesPerDirection; ++samplingIdx)
		{
			// For each vertical angle
			float samplingAngle = -generationInfo.samplingAngle / 2.0f + samplingIdx * anglePerSample;

			// Apply the sampling angle
			Vector3 samplingLine = samplingDirection;
			samplingLine.y = samplingLine.magnitude * Mathf.Tan(samplingAngle * Mathf.Deg2Rad);

			// Update max sight
			RaycastHit hitBlocked;
			if (Physics.Raycast(centerPosition, samplingLine, out hitBlocked, RAY_DISTANCE, generationInfo.levelLayer)) // Blocking level exists
			{
				obstacleHit = true;
				float blockedDistance = XZDistance(centerPosition, hitBlocked.point);
				if (blockedDistance > maxSight)
				{
					maxSight = Mathf.Clamp(blockedDistance, 0.0f, generationInfo.samplingRange);
				}

				// If the surface is almost vertical and high enough, stop sampling here
				if (Vector3.Angle(hitBlocked.normal, Vector3.up) >= generationInfo.blockingSurfaceAngleThreshold && samplingAngle >= generationInfo.blockedRayAngleThreshold)
				{
					break;
				}
			}
			else if (samplingIdx <= (generationInfo.samplesPerDirection + 2 - 1) / 2) // No hit below the eye line yields a maximum sight
			{
				maxSight = generationInfo.samplingRange;
			}
			else if (obstacleHit) // Previous ray hit an obstacle, but this one hasn't
			{
				// Binary search to find an edge
				float angularInterval = anglePerSample / 2.0f;
				float searchingAngle = samplingAngle - angularInterval;
				for (int i = 0; i < generationInfo.binarySearchCount; ++i)
				{
					angularInterval /= 2.0f;

					Vector3 searchingLine = samplingDirection;
					searchingLine.y = searchingLine.magnitude * Mathf.Tan(searchingAngle * Mathf.Deg2Rad);

					RaycastHit hitSearched;
					if (Physics.Raycast(centerPosition, searchingLine, out hitSearched, RAY_DISTANCE, generationInfo.levelLayer))
					{
						searchingAngle = searchingAngle + angularInterval; // Next range is the upper half

						// Update maxSight
						float searchedDistance = XZDistance(centerPosition, hitSearched.point);
						if (searchedDistance >= maxSight)
						{
							maxSight = Mathf.Clamp(searchedDistance, 0.0f, generationInfo.samplingRange);
						}
					}
					else
					{
						searchingAngle = searchingAngle - angularInterval; // Next range is the lower half
					}
				}

				break;
			}
		}
		float distanceRatio = maxSight == 0.0f ? 1.0f : maxSight / generationInfo.samplingRange;
		return distanceRatio;
	}


	/// <summary>
	/// Bakes a FOV map using the provided settings and FOVManager transform
	/// </summary>
	/// <param name="settings">The FOV bake settings to use</param>
	/// <param name="fovManagerTransform">The transform of the FOVManager (used as the plane)</param>
	/// <param name="durationSeconds">Output parameter containing the baking duration in seconds</param>
	/// <returns>True if baking was successful, false otherwise</returns>
	public static bool BakeFOVMap(FOVBakeSettings settings, Transform fovManagerTransform, out double durationSeconds)
	{
		if (settings == null || fovManagerTransform == null)
		{
			Debug.LogError("FOVMapGenerator: Settings or FOVManager transform is null");
			durationSeconds = 0;
			return false;
		}

		double startTime = Time.realtimeSinceStartup;

		// Create generation info with the FOVManager's transform as the plane
		FOVMapGenerationInfo generationInfo = settings.ToGenerationInfo(fovManagerTransform);

		bool isSuccessful = CreateFOVMap
		(
			generationInfo,
			(current, total) =>
			{
				string stage = "";
				if (current <= 20) stage = "Ground Detection";
				else if (current <= 70) stage = "Direction Sampling";
				else if (current <= 95) stage = "Binary Search Refinement";
				else stage = "Creating Texture";
				
				return EditorUtility.DisplayCancelableProgressBar("Baking FOV Map", $"{stage} - {current}%", (float)current / total);
			}
		);

		EditorUtility.ClearProgressBar();

		double endTime = Time.realtimeSinceStartup;
		durationSeconds = endTime - startTime;

		Debug.Log($"FOVMapGenerator: FOV map generation {(isSuccessful ? "completed successfully" : "FAILED")} after {durationSeconds} seconds ({durationSeconds / 60:F2} minutes).");
		
		if (isSuccessful)
		{
			// Auto-assign the generated FOV map to settings using EditorApplication.delayCall
			// This defers the asset loading until after the current frame, avoiding domain reload
			string assetPath = $"Assets/{settings.path}/{settings.fileName}.asset";
			EditorApplication.delayCall += () => {
				Texture2DArray generatedFOVMap = AssetDatabase.LoadAssetAtPath<Texture2DArray>(assetPath);
				if (generatedFOVMap != null)
				{
					settings.FOVMapArray = generatedFOVMap;
					EditorUtility.SetDirty(settings);
					Debug.Log($"FOV map auto-assigned to settings: {assetPath}");
				}
			};
		}

		return isSuccessful;
	}

	/// <summary>
	/// Bakes a FOV map and shows a dialog with the result
	/// </summary>
	/// <param name="settings">The FOV bake settings to use</param>
	/// <param name="fovManagerTransform">The transform of the FOVManager (used as the plane)</param>
	public static void BakeFOVMapWithDialog(FOVBakeSettings settings, Transform fovManagerTransform)
	{
		if (settings == null)
		{
			EditorUtility.DisplayDialog("FOV Mapping Error", "FOVBakeSettings not assigned! Please assign a FOVBakeSettings asset.", "OK");
			return;
		}

		if (fovManagerTransform == null)
		{
			EditorUtility.DisplayDialog("FOV Mapping Error", "FOVManager transform is null!", "OK");
			return;
		}

		bool isSuccessful = BakeFOVMap(settings, fovManagerTransform, out double durationSeconds);

		if (isSuccessful)
		{
			EditorUtility.DisplayDialog("FOV Mapping", $"FOV map baking completed successfully in {(int)durationSeconds} seconds ({durationSeconds / 60:F2} minutes)!", "OK");
		}
		else
		{
			EditorUtility.DisplayDialog("FOV Mapping", $"FOV map baking failed after {(int)durationSeconds} seconds ({durationSeconds / 60:F2} minutes). Check the console for details.", "OK");
		}
	}
}
}