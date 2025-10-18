using UnityEditor;
using UnityEngine;
using FOVMapping;

namespace FOVMapping
{
    [CustomEditor(typeof(FOVManager))]
    public class FOVManagerEditor : Editor
    {
        private SerializedProperty settingsProperty;
        private SerializedProperty fovMapArrayLegacyProperty;

        private void OnEnable()
        {
            settingsProperty = serializedObject.FindProperty("settings");
            fovMapArrayLegacyProperty = serializedObject.FindProperty("FOVMapArray_legacy");
        }

        public override void OnInspectorGUI()
        {
            FOVManager fovManager = (FOVManager)target;
            serializedObject.Update();

            // Draw all properties except the ones we want to handle specially
            SerializedProperty property = serializedObject.GetIterator();
            property.NextVisible(true);

            while (property.NextVisible(false))
            {
                // Skip the legacy FOVMapArray field if settings are populated
                if (property.name == "FOVMapArray_legacy" && fovManager.Settings != null)
                {
                    continue;
                }

                // Skip the settings field since we draw it manually in the baking section
                if (property.name == "settings")
                {
                    continue;
                }

                EditorGUILayout.PropertyField(property, true);
            }

            // Add some space
            EditorGUILayout.Space(10);

            // Show inline settings editor if settings are populated
            EditorGUILayout.LabelField("FOV Map Baking", EditorStyles.boldLabel);
            
            // Draw the settings property field
            EditorGUILayout.PropertyField(settingsProperty, true);
            
            if (fovManager.Settings != null)
            {
                EditorGUILayout.LabelField("FOV Bake Settings", EditorStyles.boldLabel);
                EditorGUI.indentLevel++;
                
                SerializedObject settingsSerializedObject = new SerializedObject(fovManager.Settings);
                SerializedProperty settingsIterator = settingsSerializedObject.GetIterator();
                settingsIterator.NextVisible(true);
                
                while (settingsIterator.NextVisible(false))
                {
                    EditorGUILayout.PropertyField(settingsIterator, true);
                }
                settingsSerializedObject.ApplyModifiedProperties();
                EditorGUI.indentLevel--;
                
                EditorGUILayout.Space(10);
            }
            
            //Extra stuff to help display info / migration re the new ScriptableObject-based workflow
            if (fovManager.Settings == null)
            {
                // Check if legacy field is populated
                if (fovManager.LegacyFOVMapArray != null)
                {
                    EditorGUILayout.HelpBox("Attention: You are using the legacy FOVMapArray field. Please switch to the new ScriptableObject-based workflow for better functionality and future compatibility. Create a FOVBakeSettings asset and assign it to the Settings field above.", MessageType.Warning);
                }
                else
                {
                    EditorGUILayout.HelpBox("FOVBakeSettings not assigned! Please assign a FOVBakeSettings asset.", MessageType.Warning);
                }
            }
            else
            {
                // If settings are assigned, clear the legacy field
                if (fovManager.LegacyFOVMapArray != null)
                {
                    fovMapArrayLegacyProperty.objectReferenceValue = null;
                    EditorUtility.SetDirty(fovManager);
                }

                if (GUILayout.Button("Bake FOV Map", GUILayout.Height(30)))
                {
                    FOVMapGenerator.BakeFOVMapWithDialog(fovManager.Settings, fovManager.transform);
                }
            }

            serializedObject.ApplyModifiedProperties();
        }
    }
}
