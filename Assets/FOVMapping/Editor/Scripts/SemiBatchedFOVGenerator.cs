using System;
using System.Linq;
using Sirenix.OdinInspector.Editor;
using Unity.Collections;
using Unity.Jobs;
using Unity.Profiling;
using UnityEditor;
using UnityEngine;

using UnityEngine.Profiling;


namespace FOVMapping 
{
/// <summary>
/// Semi-batched FOV map generation strategy
///
/// This implementation combines the benefits of batching for ground detection
/// with the proven reliability of single-threaded direction sampling and binary search.
/// 
/// STAGE 1: BATCHED GROUND HEIGHT DETECTION
/// - Uses Unity's RaycastCommand API for parallel ground detection
/// - Batches all ground detection raycasts for maximum performance
/// - One raycast per grid cell (FOVMapWidth × FOVMapHeight total rays)
/// 
/// STAGE 2: SINGLE-THREADED DIRECTION SAMPLING
/// - Uses the proven single-threaded approach for direction sampling
/// - Ensures exact compatibility with the original algorithm
/// - Processes each cell-direction combination sequentially
/// 
/// STAGE 3: SINGLE-THREADED BINARY SEARCH
/// - Uses the proven single-threaded binary search for edge refinement
/// - Ensures exact compatibility with the original algorithm
/// - Processes each binary search sequentially
/// 
/// This approach provides significant performance improvement for ground detection
/// while maintaining the exact behavior of the original algorithm for the complex
/// direction sampling and binary search phases.
/// </summary>
public sealed class SemiBatchedFOVGenerator : IFOVGenerator
{
    public Color[][] Generate(FOVMapGenerationInfo generationInfo, Func<int, int, bool> progressAction)
    {
        return GenerateFOVMap_Wavefront(generationInfo, progressAction);
    }
    
    public string GetProgressStage(int progressPercent)
    {
        if (progressPercent <= 20) return "Ground Detection";
        else if (progressPercent <= 95) return "Direction Sampling & Binary Search";
        else return "Creating Texture";
    }

    private static Color[][] GenerateFOVMap_SemiBatched(FOVMapGenerationInfo generationInfo, Func<int, int, bool> progressAction)
    {
        // Basic checks
        bool checkPassed = generationInfo.CheckSettings();

        if (progressAction == null)
        {
            Debug.LogError("progressAction must be passed.");
            checkPassed = false;
        }

        if (!checkPassed) return null;

        // Set variables and constants
        const float MAX_HEIGHT = 5000.0f;

        float planeSizeX = generationInfo.plane.localScale.x;
        float planeSizeZ = generationInfo.plane.localScale.z;

        float squareSizeX = planeSizeX / generationInfo.FOVMapWidth;
        float squareSizeZ = planeSizeZ / generationInfo.FOVMapHeight;

        int directionsPerSquare = FOVMapGenerator.CHANNELS_PER_TEXEL * generationInfo.layerCount;

        float anglePerDirection = 360.0f / directionsPerSquare;
        float anglePerSample = generationInfo.samplingAngle / (generationInfo.samplesPerDirection - 1);

        // Create an array of FOV maps
        Color[][] FOVMapTexels = Enumerable.Range(0, generationInfo.layerCount).Select(_ => new Color[generationInfo.FOVMapWidth * generationInfo.FOVMapHeight]).ToArray();

        // STAGE 1: BATCHED GROUND HEIGHT DETECTION
        Debug.Log("SemiBatchedFOVGenerator: Starting Stage 1 - Batched Ground Height Detection");
        if (progressAction.Invoke(0, 100)) return null; // 0% - Starting ground detection

        GroundHeightData[] groundData = ProcessGroundRaycastsInBatches(generationInfo, progressAction);

        if (progressAction.Invoke(20, 100)) return null; // 20% - Ground detection complete

        // STAGE 2 & 3: SINGLE-THREADED DIRECTION SAMPLING AND BINARY SEARCH
        Debug.Log("SemiBatchedFOVGenerator: Starting Stage 2 & 3 - Single-threaded Direction Sampling and Binary Search");

        for (int squareZ = 0; squareZ < generationInfo.FOVMapHeight; ++squareZ)
        {
            for (int squareX = 0; squareX < generationInfo.FOVMapWidth; ++squareX)
            {
                GroundHeightData ground = groundData[generationInfo.CellIndex(squareX, squareZ)];

                if (ground.hasGround)
                {
                    // For all possible directions at this square
                    for (int directionIdx = 0; directionIdx < directionsPerSquare; ++directionIdx)
                    {
                        // Sample a distance to an obstacle using the proven single-threaded approach
                        float angleToward = Vector3.SignedAngle(generationInfo.plane.right, Vector3.right, Vector3.up) + directionIdx * anglePerDirection;
                        
                        var distanceRatio = FindNearestObstacle(generationInfo, angleToward, anglePerSample, ground.centerPosition);

                        // Find the location to store
                        int layerIdx = directionIdx / FOVMapGenerator.CHANNELS_PER_TEXEL;
                        int channelIdx = directionIdx % FOVMapGenerator.CHANNELS_PER_TEXEL;

                        // Store
                        FOVMapTexels[layerIdx][squareZ * generationInfo.FOVMapWidth + squareX][channelIdx] = distanceRatio;
                    }
                }
                else // No level found
                {
                    // Fill all the layers with white
                    for (int layerIdx = 0; layerIdx < generationInfo.layerCount; ++layerIdx)
                    {
                        FOVMapTexels[layerIdx][squareZ * generationInfo.FOVMapWidth + squareX] = Color.white;
                    }
                }
            }

            // Update progress (20% to 95% for direction sampling and binary search)
            int progressPercent = 20 + (squareZ * 75) / generationInfo.FOVMapHeight;
            if (progressAction.Invoke(progressPercent, 100)) return null;
        }

        if (progressAction.Invoke(99, 100)) return null; // 99% - Processing complete

        return FOVMapTexels;
    }

    /// <summary>
    /// Ground height data for a grid cell
    /// </summary>
    public struct GroundHeightData
    {
        public Vector3 centerPosition;
        public float height;
        public bool hasGround;
    }

    /// <summary>
    /// Processes ground detection raycasts in batches using RaycastCommand
    /// </summary>
    public static GroundHeightData[] ProcessGroundRaycastsInBatches(FOVMapGenerationInfo generationInfo, Func<int, int, bool> progressAction)
    {
        const float MAX_HEIGHT = 5000.0f;
        int totalCells = generationInfo.FOVMapWidth * generationInfo.FOVMapHeight;
        int batchSize = Mathf.Min(generationInfo.maxBatchSize, totalCells);

        GroundHeightData[] groundData = new GroundHeightData[totalCells];

        float planeSizeX = generationInfo.plane.localScale.x;
        float planeSizeZ = generationInfo.plane.localScale.z;

        int totalBatches = (totalCells + batchSize - 1) / batchSize; // Ceiling division

        NativeArray<RaycastCommand> commandBuffer = new NativeArray<RaycastCommand>(batchSize, Allocator.TempJob);
        NativeArray<RaycastHit> resultBuffer = new NativeArray<RaycastHit>(batchSize, Allocator.TempJob);
        
        Debug.Log($"ProcessGroundRaycastsInBatches: Raycast Count (Cell Count): {totalCells} Batch size: {batchSize}, Total batches: {totalBatches}");
        
        for (int startIndex = 0; startIndex < totalCells; startIndex += batchSize)
        {
            int currentBatchSize = Mathf.Min(batchSize, totalCells - startIndex);
            int currentBatch = startIndex / batchSize;

            // Update progress (0% to 20% for ground detection)
            int progressPercent = 0 + (currentBatch * 20) / totalBatches;
            if (progressAction.Invoke(progressPercent, 100)) return null;

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

                commandBuffer[i] = new RaycastCommand
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

            //Fill extras with empty commands
            if (currentBatchSize < commandBuffer.Length) {
                for (int i = currentBatchSize; i < commandBuffer.Length; ++i) {
                    commandBuffer[i] = new RaycastCommand();
                }
            }

            // Execute batch
            JobHandle handle = RaycastCommand.ScheduleBatch(commandBuffer, resultBuffer, 1, 1);
            
            // Process results
            handle.Complete(); // Await completion
            for (int i = 0; i < currentBatchSize; ++i)
            {
                int globalIndex = startIndex + i;
                RaycastHit hit = resultBuffer[i];

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
            
        }
        
        // Cleanup
        commandBuffer.Dispose();
        resultBuffer.Dispose();

        return groundData;
    }

    /// <summary>
    /// Find nearest obstacle using the proven single-threaded algorithm
    /// This is a direct copy from SingleThreadedFOVGenerator to ensure exact compatibility
    /// </summary>
    private static float FindNearestObstacle(FOVMapGenerationInfo generationInfo, float angleToward, float anglePerSample, Vector3 centerPosition) 
    {
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

    private static Color[][] GenerateFOVMap_Wavefront(FOVMapGenerationInfo generationInfo, Func<int, int, bool> progressAction) {
        // Basic checks
        bool checkPassed = generationInfo.CheckSettings();

        if (progressAction == null) {
            Debug.LogError("progressAction must be passed.");
            checkPassed = false;
        }

        if (!checkPassed) return null;

        // Set variables and constants
        const float MAX_HEIGHT = 5000.0f;

        float planeSizeX = generationInfo.plane.localScale.x;
        float planeSizeZ = generationInfo.plane.localScale.z;

        float squareSizeX = planeSizeX / generationInfo.FOVMapWidth;
        float squareSizeZ = planeSizeZ / generationInfo.FOVMapHeight;

        int directionsPerSquare = FOVMapGenerator.CHANNELS_PER_TEXEL * generationInfo.layerCount;

        float anglePerDirection = 360.0f / directionsPerSquare;
        float anglePerSample = generationInfo.samplingAngle / (generationInfo.samplesPerDirection - 1);

        // Create an array of FOV maps
        Color[][] FOVMapTexels = Enumerable.Range(0, generationInfo.layerCount)
            .Select(_ => new Color[generationInfo.FOVMapWidth * generationInfo.FOVMapHeight]).ToArray();

        // STAGE 1: BATCHED GROUND HEIGHT DETECTION
        Debug.Log("SemiBatchedFOVGenerator: Starting Stage 1 - Batched Ground Height Detection");
        if (progressAction.Invoke(0, 100)) return null; // 0% - Starting ground detection

        GroundHeightData[] groundData = ProcessGroundRaycastsInBatches(generationInfo, progressAction);

        if (progressAction.Invoke(20, 100)) return null; // 20% - Ground detection complete

        // STAGE 2 & 3: SINGLE-THREADED DIRECTION SAMPLING AND BINARY SEARCH
        Debug.Log("SemiBatchedFOVGenerator: Starting Stage 2 & 3 - Single-threaded Direction Sampling and Binary Search");

        RunDirectionsWavefront(generationInfo, groundData, FOVMapTexels, progressAction);

        return FOVMapTexels;
    }

    enum DirPhase : byte { InitSample, Sampling, BinarySearch, Done }

    struct DirState
    {
        public DirPhase phase;
        public int sampleIdx;
        public bool obstacleHit;
        public float lastHitAngle;    // deg
        public float lastMissAngle;   // deg
        public int bsIter;
        public float maxSight;        // world XZ distance (clamped later)
    }

    static Vector3 DirectionFromAngleDeg(float angleDeg)
    {
        float r = angleDeg * Mathf.Deg2Rad;
        return new Vector3(Mathf.Cos(r), 0f, Mathf.Sin(r));
    }

    static float XZDistance(Vector3 a, Vector3 b)
    {
        a.y = 0f; b.y = 0f;
        return Vector3.Distance(a, b);
    }

    
// Markers
    static readonly ProfilerMarker kWaveLoop    = new("FOV/WaveLoop");
    static readonly ProfilerMarker kGather      = new("FOV/Gather");
    static readonly ProfilerMarker kRaycast    = new("FOV/Raycast");
    static readonly ProfilerMarker kConsume     = new("FOV/Consume");

// // Counters (appear in Profiler Counters panel)
//     static ProfilerCounter<int> cWave        = new(ProfilerCategory.Scripts, "FOV/Wave Index", ProfilerMarkerDataUnit.Count);
//     static ProfilerCounter<int> cEmitted     = new(ProfilerCategory.Scripts, "FOV/Rays Emitted This Wave", ProfilerMarkerDataUnit.Count);
//     static ProfilerCounter<int> cDone        = new(ProfilerCategory.Scripts, "FOV/Dirs Done", ProfilerMarkerDataUnit.Count);
//     static ProfilerCounter<int> cFillRatioPct= new(ProfilerCategory.Scripts, "FOV/Batch Fill %", ProfilerMarkerDataUnit.Percent);
//
// // optional: split sampling vs binary counts
//     static ProfilerCounter<int> cSamplingQ   = new(ProfilerCategory.Scripts, "FOV/Sampling Pending", ProfilerMarkerDataUnit.Count);
//     static ProfilerCounter<int> cBinaryQ     = new(ProfilerCategory.Scripts, "FOV/Binary Pending", ProfilerMarkerDataUnit.Count);
    static void RunDirectionsWavefront(FOVMapGenerationInfo g, GroundHeightData[] ground, Color[][] FOVMapTexels, Func<int,int,bool> progressAction)
    {
        int cells  = g.CellCount;

        int directionsPerSquare = FOVMapGenerator.CHANNELS_PER_TEXEL * g.layerCount;
        float anglePerDirection = 360f / directionsPerSquare;
        float anglePerSample    = g.samplingAngle / (g.samplesPerDirection - 1);
        const float RAY_DISTANCE = 1000f;

        // Per (cell,dir) state
        var dirStates = new DirState[cells * directionsPerSquare];

        // Initialize phases (skip cells without ground)
        for (int cell = 0; cell < cells; cell++)
        {
            bool hasGround = ground[cell].hasGround;
            for (int d = 0; d < directionsPerSquare; d++)
            {
                int idx = cell * directionsPerSquare + d;
                ref var st = ref dirStates[idx];
                if (!hasGround)
                {
                    st.phase = DirPhase.Done; // will fill white later
                }
                else
                {
                    st.phase = DirPhase.InitSample;
                    st.sampleIdx = 0;
                    st.obstacleHit = false;
                    st.maxSight = 0f;
                    st.bsIter = 0;
                    st.lastHitAngle = float.NaN;
                    st.lastMissAngle = float.NaN;
                }
            }
        }

        // Precompute base horizontal angle per direction (world-aligned)
        var baseDirAngle = new float[directionsPerSquare];
        float planeRightYaw = Vector3.SignedAngle(g.plane.right, Vector3.right, Vector3.up);
        for (int d = 0; d < directionsPerSquare; d++)
            baseDirAngle[d] = planeRightYaw + d * anglePerDirection;

        // Wave loop
        int doneCount = 0;
        int wave = 0;
        int totalDirs = cells * directionsPerSquare;

        // (Optional) chunking to keep Editor responsive
        int batchSize = Mathf.Max(1, g.maxBatchSize);

        var cmdBuffer  = new NativeArray<RaycastCommand>(batchSize, Allocator.TempJob);
        var mapBack    = new NativeArray<(int cell, int dir, byte tag)> (batchSize, Allocator.TempJob); // tag: 0=sample,1=binsearch,255=Filler
        var hitResultBuffer = new NativeArray<RaycastHit>(batchSize, Allocator.TempJob);

        while (doneCount < totalDirs) {
            kWaveLoop.Begin();
            int bufferIndex = 0;
            
            kGather.Begin();
            // 1) Gather: emit at most one ray per active (cell,dir) this wave.
            for (int cell = 0; cell < cells && bufferIndex < cmdBuffer.Length; cell++)
            {
                if (!ground[cell].hasGround) continue;
                Vector3 center = ground[cell].centerPosition;

                for (int d = 0; d < directionsPerSquare && bufferIndex < cmdBuffer.Length; d++)
                {
                    int idx = cell * directionsPerSquare + d;
                    ref var st = ref dirStates[idx];
                    if (st.phase == DirPhase.Done) continue;

                    // Horizontal toward vector
                    float baseYaw = baseDirAngle[d];
                    Vector3 samplingDir = DirectionFromAngleDeg(baseYaw);

                    // Decide which vertical angle we’re casting this wave
                    switch (st.phase)
                    {
                        case DirPhase.InitSample:
                        case DirPhase.Sampling:
                        {
                            // same logic as your single-threaded loop
                            float samplingAngle = -g.samplingAngle / 2f + st.sampleIdx * anglePerSample;
                            Vector3 rayDir = samplingDir;
                            rayDir.y = rayDir.magnitude * Mathf.Tan(samplingAngle * Mathf.Deg2Rad);

                            var cmd = new RaycastCommand
                            {
                                from = center,
                                direction = rayDir.normalized,
                                distance = RAY_DISTANCE,
                                queryParameters = new QueryParameters
                                {
                                    layerMask = g.levelLayer,
                                    hitTriggers = QueryTriggerInteraction.Ignore
                                }
                            };
                            cmdBuffer[bufferIndex] = cmd;
                            mapBack[bufferIndex] = (cell, d, 0); //0=sampling
                            bufferIndex++;
                            st.phase = DirPhase.Sampling; // ensure we mark it as sampling
                            break;
                        }

                        case DirPhase.BinarySearch:
                        {
                            // midpoint between last hit/miss
                            float searchAngle = 0.5f * (st.lastHitAngle + st.lastMissAngle);
                            Vector3 rayDir = samplingDir;
                            rayDir.y = rayDir.magnitude * Mathf.Tan(searchAngle * Mathf.Deg2Rad);

                            var cmd = new RaycastCommand
                            {
                                from = center,
                                direction = rayDir.normalized,
                                distance = RAY_DISTANCE,
                                queryParameters = new QueryParameters
                                {
                                    layerMask = g.levelLayer,
                                    hitTriggers = QueryTriggerInteraction.Ignore
                                }
                            };
                            cmdBuffer[bufferIndex] = cmd;
                            mapBack[bufferIndex] = (cell, d, 1); //1 = binsearch
                            bufferIndex++;
                            break;
                        }
                    }
                }
            }
            

            // If no rays emitted this wave, finalize remaining directions (e.g., empty cells)
            if (bufferIndex == 0)
            {
                // finalize any lingering states that need it
                for (int cell = 0; cell < cells; cell++)
                {
                    if (!ground[cell].hasGround)
                    {
                        // fill white
                        for (int layer = 0; layer < g.layerCount; layer++)
                            FOVMapTexels[layer][cell] = Color.white;
                    }
                    else
                    {
                        for (int d = 0; d < directionsPerSquare; d++)
                        {
                            int idx = cell * directionsPerSquare + d;
                            ref var st = ref dirStates[idx];
                            if (st.phase != DirPhase.Done)
                            {
                                // if we somehow didn’t finish, assume maxSight==0 → white channel
                                WriteChannel(g, FOVMapTexels, cell, d, st.maxSight, g.samplingRange);
                                st.phase = DirPhase.Done;
                                doneCount++;
                            }
                        }
                    }
                }
                break;
            }
            kGather.End();

            
            // 2) Execute Batch
            kRaycast.Begin();
            //First, if we didn't fill the whole array, fill with default commands
            int thisBatchSize = bufferIndex;
            while (bufferIndex < cmdBuffer.Length) {
                cmdBuffer[bufferIndex] = new RaycastCommand();
                mapBack[bufferIndex] = (-1, -1, 255);
                bufferIndex++;
                    
            }
            
            var handle = RaycastCommand.ScheduleBatch(cmdBuffer, hitResultBuffer, 1);
            handle.Complete();
            kRaycast.End();
            
            // 3) Consume
            kConsume.Begin();
            for (int i = 0; i < thisBatchSize; i++)
            {
                var (cell, d, tag) = mapBack[i];
                if(tag == 255) continue; // Though, this shouldnt happen.
                int idx = cell * directionsPerSquare + d;
                ref var st = ref dirStates[idx];

                Vector3 center = ground[cell].centerPosition;
                RaycastHit hit = hitResultBuffer[i];

                if (tag == 0) // Sampling phase
                {
                    float samplingAngle = -g.samplingAngle / 2f + st.sampleIdx * anglePerSample;

                    if (hit.collider != null)
                    {
                        st.obstacleHit = true;
                        float blocked = XZDistance(center, hit.point);
                        if (blocked > st.maxSight)
                            st.maxSight = Mathf.Clamp(blocked, 0f, g.samplingRange);

                        // vertical/steep check
                        if (Vector3.Angle(hit.normal, Vector3.up) >= g.blockingSurfaceAngleThreshold &&
                            samplingAngle >= g.blockedRayAngleThreshold)
                        {
                            FinalizeDirection(g, FOVMapTexels, cell, d, ref st); doneCount++;
                            continue;
                        }

                        // prepare for potential binary search if next higher sample misses
                        st.lastHitAngle = samplingAngle;
                    }
                    else
                    {
                        // special case: any sample below eye-line misses implies max sight
                        if (st.sampleIdx <= (g.samplesPerDirection + 2 - 1) / 2)
                        {
                            st.maxSight = g.samplingRange;
                            FinalizeDirection(g, FOVMapTexels, cell, d, ref st); doneCount++;
                            continue;
                        }

                        // if we had a prior hit and now miss, enter binary search
                        if (st.obstacleHit && !float.IsNaN(st.lastHitAngle))
                        {
                            st.lastMissAngle = samplingAngle;
                            st.phase = DirPhase.BinarySearch;
                            st.bsIter = 0;
                            continue;
                        }
                    }

                    // advance to next sample if still sampling
                    if (st.phase == DirPhase.Sampling)
                    {
                        st.sampleIdx++;
                        if (st.sampleIdx >= g.samplesPerDirection)
                        {
                            // no binary search needed; finalize with whatever maxSight we have
                            FinalizeDirection(g, FOVMapTexels, cell, d, ref st); doneCount++;
                        }
                    }
                }
                else // Binary search phase
                {
                    float searchAngle = 0.5f * (st.lastHitAngle + st.lastMissAngle);

                    if (hit.collider != null)
                    {
                        // move split upward
                        st.lastHitAngle = searchAngle;
                        float dist = XZDistance(center, hit.point);
                        if (dist > st.maxSight)
                            st.maxSight = Mathf.Clamp(dist, 0f, g.samplingRange);
                    }
                    else
                    {
                        // move split downward
                        st.lastMissAngle = searchAngle;
                    }

                    st.bsIter++;
                    if (st.bsIter >= g.binarySearchCount)
                    {
                        FinalizeDirection(g, FOVMapTexels, cell, d, ref st); doneCount++;
                    }
                }
            }
            kConsume.End();

            // Progress (map 20%..95%)
            int pct = 20 + Mathf.RoundToInt(75f * (doneCount / (float)totalDirs));
            if (progressAction.Invoke(Mathf.Clamp(pct, 20, 95), 100)) 
            {
                kWaveLoop.End();
                break;
            }

            wave++;
            
            // Debug.Log($"Finished wave {wave}. Finished {doneCount} of {totalDirs} total directions");
            // (Optional) yield a frame here if you want editor responsiveness:
            // EditorApplication.QueuePlayerLoopUpdate();  // or await/yield elsewhere
             
             kWaveLoop.End();
        }
        
        // Release memory
        cmdBuffer.Dispose();
        mapBack.Dispose();
        hitResultBuffer.Dispose();
        
        //No return -- FOVMapTexels has been updated
    }

    static void FinalizeDirection(
        FOVMapGenerationInfo g, Color[][] texels, int cell, int d,
        ref DirState st)
    {
        WriteChannel(g, texels, cell, d, st.maxSight, g.samplingRange);
        st.phase = DirPhase.Done;
    }

    static void WriteChannel(
        FOVMapGenerationInfo g, Color[][] texels, int cell, int directionIdx,
        float maxSight, float samplingRange)
    {
        int layerIdx   = directionIdx / FOVMapGenerator.CHANNELS_PER_TEXEL;
        int channelIdx = directionIdx % FOVMapGenerator.CHANNELS_PER_TEXEL;

        float ratio = (maxSight == 0f) ? 1f : Mathf.Clamp01(maxSight / samplingRange);
        var c = texels[layerIdx][cell];
        c[channelIdx] = ratio;
        texels[layerIdx][cell] = c;
    }

}
}
