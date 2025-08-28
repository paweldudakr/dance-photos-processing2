import mediapipe as mp
import traceback

try:
    print("Attempting to initialize SelfieSegmentation...")
    segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
    print("SelfieSegmentation initialized successfully.")
except Exception as e:
    print(f"An error occurred: {e}")
    traceback.print_exc()