import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

print("Loading YOLO11 Large model...")
# 1. Initialize the Detection Model
detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8', 
    model_path='yolo11l.pt',
    confidence_threshold=0.15, # Lower threshold to catch the dense crowd in the back
    device='cpu', # Change to 'cuda:0' if you have an NVIDIA GPU
)

print("Slicing image and running inference... (This may take a minute)")
# 2. Perform Sliced Inference ("The Magnifying Glass")
result = get_sliced_prediction(
    image="stadium_crowd.jpg",
    detection_model=detection_model,
    slice_height=400,             # Smaller slices = tighter zoom on tiny heads
    slice_width=400,
    overlap_height_ratio=0.25,    # 25% overlap ensures no one is missed on the edges
    overlap_width_ratio=0.25,
    perform_standard_pred=False   # Skip full-image prediction to avoid "cell phone" confusion
)

# 3. Filter Results (Keep ONLY People)
full_predictions = result.object_prediction_list

# In the COCO dataset, we only care about the 'person' category
only_people = [p for p in full_predictions if p.category.name == 'person']

# We overwrite the original list so the final image ONLY draws boxes around people
result.object_prediction_list = only_people

# 4. Terminal Analytics Report
print("\n" + "="*40)
print("📊 FINAL ELITE STADIUM REPORT")
print("="*40)
print(f"Total Objects Found  : {len(full_predictions)}")
print(f"Confirmed People     : {len(only_people)}")
print("="*40 + "\n")

# 5. Save Visualization
result.export_visuals(export_dir=".", file_name="elite_prediction_final")
print("Success! Check your folder for 'elite_prediction_final.png'.")