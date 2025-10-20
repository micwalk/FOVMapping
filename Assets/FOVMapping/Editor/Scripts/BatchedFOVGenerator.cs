using System;
using System.Linq;
using Unity.Collections;
using Unity.Jobs;
using UnityEditor;
using UnityEngine;

namespace FOVMapping 
{
/// <summary>
/// Batched FOV map generation strategy
///
/// Intended to have the exact same behavior as SingleThreadedFOVGenerator, but with faster performance.
/// 
/// This implementation uses Unity's RaycastCommand API for batched raycasting, allowing
/// massive parallelization across all CPU cores. All cells executing the same iteration
/// step are batched together, eliminating the synchronous Physics.Raycast bottleneck.
/// </summary>
public sealed class BatchedFOVGenerator : IFOVGenerator
{
    public Color[][] Generate(FOVMapGenerationInfo generationInfo, Func<int, int, bool> progressAction)
    {
        return GenerateFOVMap_Batched(generationInfo, progressAction);
    }
    
    public string GetProgressStage(int progressPercent)
    {
        if (progressPercent <= 20) return "Ground Detection";
        else if (progressPercent <= 70) return "Direction Sampling";
        else if (progressPercent <= 95) return "Binary Search Refinement";
        else return "Creating Texture";
    }

    private static Color[][] GenerateFOVMap_Batched(FOVMapGenerationInfo generationInfo, Func<int, int, bool> progressAction)
    {
        // Basic checks
        bool checkPassed = generationInfo.CheckSettings();

        if (progressAction == null)
        {
            Debug.LogError("progressAction must be pased.");
            checkPassed = false;
        }

        if (!checkPassed) return null;

        // STAGE 1: Ground Height Detection using batched raycasts
        Debug.Log("FOVMapGenerator: Starting Stage 1 - Ground Height Detection");
        if (progressAction.Invoke(0, 100)) return null; // 0% - Starting ground detection
    
        SemiBatchedFOVGenerator.GroundHeightData[] groundData = SemiBatchedFOVGenerator.ProcessGroundRaycastsInBatches(generationInfo, progressAction);
    
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
    
        int directionsPerSquare = FOVMapGenerator.CHANNELS_PER_TEXEL * generationInfo.layerCount;
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
                    int layerIdx = directionIdx / FOVMapGenerator.CHANNELS_PER_TEXEL;
                    int channelIdx = directionIdx % FOVMapGenerator.CHANNELS_PER_TEXEL;
                
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

        return FOVMapTexels;
    }

    // Helper methods for batched generation

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

    #region Helper Methods for Batched Raycasting

    /// <summary>
    /// Calculates the optimal batch size based on settings and available memory
    /// </summary>
    private static int CalculateOptimalBatchSize(FOVMapGenerationInfo generationInfo, int totalRays)
    {
        return Mathf.Min(generationInfo.maxBatchSize, totalRays);
    }

    /// <summary>
    /// Processes direction sampling raycasts in configurable batches
    /// Ensures that each state's full set of samples is processed within the same batch
    /// </summary>
    private static DirectionSamplingState[] ProcessDirectionSamplingInBatches(FOVMapGenerationInfo generationInfo, SemiBatchedFOVGenerator.GroundHeightData[] groundData, Func<int, int, bool> progressAction)
    {
        int directionsPerSquare = FOVMapGenerator.CHANNELS_PER_TEXEL * generationInfo.layerCount;
        int totalStates = generationInfo.FOVMapWidth * generationInfo.FOVMapHeight * directionsPerSquare;
        DirectionSamplingState[] samplingStates = new DirectionSamplingState[totalStates];

        // Initialize all states first
        int stateIndexInit = 0;
        for (int squareZ = 0; squareZ < generationInfo.FOVMapHeight; ++squareZ)
        {
            for (int squareX = 0; squareX < generationInfo.FOVMapWidth; ++squareX)
            {
                SemiBatchedFOVGenerator.GroundHeightData ground = groundData[generationInfo.CellIndex(squareX, squareZ)];

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

                SemiBatchedFOVGenerator.GroundHeightData ground = groundData[state.cellZ * generationInfo.FOVMapWidth + state.cellX];
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
                SemiBatchedFOVGenerator.GroundHeightData ground = groundData[state.cellZ * generationInfo.FOVMapWidth + state.cellX];

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
                        float blockedDistance = XZDistance(ground.centerPosition, hit.point);
                        if (blockedDistance > maxSight)
                        {
                            maxSight = Mathf.Clamp(blockedDistance, 0.0f, generationInfo.samplingRange);
                        }

                        // If the surface is almost vertical and high enough, stop sampling here
                        float samplingAngle = -generationInfo.samplingAngle / 2.0f + samplingIdx * anglePerSample;
                        if (Vector3.Angle(hit.normal, Vector3.up) >= generationInfo.blockingSurfaceAngleThreshold && samplingAngle >= generationInfo.blockedRayAngleThreshold)
                        {
                            // CRITICAL: We must consume remaining results to keep resultIdx aligned
                            // Skip processing but still increment resultIdx for remaining samples
                            resultIdx += (generationInfo.samplesPerDirection - samplingIdx - 1);
                            break;
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
                        // CRITICAL: We must consume remaining results to keep resultIdx aligned
                        // Skip processing but still increment resultIdx for remaining samples
                        resultIdx += (generationInfo.samplesPerDirection - samplingIdx - 1);
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
    /// Helper function to calculate XZ distance (horizontal distance only)
    /// </summary>
    private static float XZDistance(Vector3 v1, Vector3 v2)
    {
        v1.y = 0.0f;
        v2.y = 0.0f;
        return Vector3.Distance(v1, v2);
    }

    /// <summary>
    /// Processes binary search synchronously for a single direction sampling state
    /// Based on the original FindNearestObstacle algorithm
    /// </summary>
    private static void ProcessBinarySearchSynchronous(DirectionSamplingState state, FOVMapGenerationInfo generationInfo, SemiBatchedFOVGenerator.GroundHeightData[] groundData)
    {
        // Get ground data for this cell
        int cellIndex = state.cellZ * generationInfo.FOVMapWidth + state.cellX;
        SemiBatchedFOVGenerator.GroundHeightData ground = groundData[cellIndex];
        Vector3 centerPosition = ground.centerPosition;
    
        // Calculate direction
        float anglePerDirection = 360.0f / (FOVMapGenerator.CHANNELS_PER_TEXEL * generationInfo.layerCount);
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
                float searchedDistance = XZDistance(centerPosition, hitSearched.point);
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

    #endregion
}
}
