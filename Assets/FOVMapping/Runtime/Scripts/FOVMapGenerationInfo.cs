using UnityEngine;
using FOVMapping;

namespace FOVMapping
{
    [System.Serializable]
    public class FOVMapGenerationInfo
    {
        [Tooltip("(Essential) Path to save the generated FOV map")] 
        public string path = "FOVMapping/FOVMaps";
        
        [Tooltip("(Essential) Name of the FOV map file")] 
        public string fileName = "FOVMap1024";
        
        [Tooltip("(Essential) Plane for FOV mapping")] 
        public Transform plane;
        
        [Tooltip("(Essential) Layer of the level to be sampled")] 
        public LayerMask levelLayer;
        
        [Tooltip("Width of the generated FOV map")] 
        public int FOVMapWidth = 1024;
        
        [Tooltip("Height of the generated FOV map")] 
        public int FOVMapHeight = 1024;
        
        [Tooltip("Number of layers in the generated FOV map")] 
        public int layerCount = 90;
        
        [Tooltip("Height of the 'sampling eye'")] 
        public float eyeHeight = 1.8f;
        
        [Tooltip("Maximum sampling range; sight system does not work beyond this boundary")] 
        public float samplingRange = 50.0f;
        
        [Tooltip("(Advanced) Vertical angular range from the sampling eye")] 
        public float samplingAngle = 140.0f;
        
        [Tooltip("(Advanced) How many rays will be fired toward a direction at a location?")] 
        public int samplesPerDirection = 9;
        
        [Tooltip("(Advanced) How many iterations for the binary search to find an edge?")] 
        public int binarySearchCount = 10;
        
        [Tooltip("(Advanced) Surfaces steeper than this angle are considered vertical and there will be no further sampling toward the direction at the location.")] 
        public float blockingSurfaceAngleThreshold = 85.0f;
        
        [Tooltip("(Advanced) Surfaces located below this vertical angle are never considered vertical.")] 
        public float blockedRayAngleThreshold = 0.0f;
        
        [Tooltip("(Advanced) Maximum number of raycasts to process in a single batch")] 
        public int maxBatchSize = 500;
        
        [Tooltip("Algorithm to use for baking")]
        public BakeAlgorithm bakeAlgorithm = BakeAlgorithm.SingleThreaded;

        public bool CheckSettings() {
            bool checkPassed = true;
            
            if (plane == null)
            {
                Debug.LogError("No FOW plane has been assigned.");
                checkPassed = false;
            }

            if (string.IsNullOrEmpty(path) || string.IsNullOrEmpty(fileName))
            {
                Debug.LogError("Either path or file name have not been assigned.");
                checkPassed = false;
            }

            if (FOVMapWidth == 0 || FOVMapHeight == 0)
            {
                Debug.LogError("Incorrect texture size.");
                checkPassed = false;
            }

            if (samplingRange <= 0.0f)
            {
                Debug.LogError("Sampling range must be greater than zero.");
                checkPassed = false;
            }

            if (levelLayer == 0)
            {
                Debug.LogError("Level layer must be non-zero.");
                checkPassed = false;
            }

            return checkPassed;
        }

        public int CellCount => FOVMapWidth * FOVMapHeight;
        public int CellIndex(int squareX, int squareZ) => squareZ * FOVMapWidth + squareX;
    }
}
