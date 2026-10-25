"""
Omni-Track Slicing Aided Hyper Inference (SAHI) Static Crowd Analysis

Splits high-resolution stadium imagery into overlapping grid slices, runs YOLO
inference across slices to capture micro-scale human targets, and merges predictions.
"""

import argparse
from pathlib import Path
import sys
from typing import List
import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for SAHI crowd inference."""
    parser = argparse.ArgumentParser(description="Omni-Track Slicing Aided Hyper Inference (SAHI) Static Crowd Analysis")
    parser.add_argument("--image", "-i", default="stadium_crowd.jpg", help="Path to input high-resolution crowd image")
    parser.add_argument("--model", "-m", default="yolov8l.pt", help="YOLO model weights path or name (e.g. yolov8l.pt, yolo11l.pt)")
    parser.add_argument("--device", "-d", default="cpu", help="Device to run inference on ('cpu' or 'cuda:0')")
    parser.add_argument("--conf", type=float, default=0.15, help="Confidence threshold (default: 0.15)")
    parser.add_argument("--slice-size", type=int, default=400, help="SAHI slice grid size in pixels (default: 400)")
    parser.add_argument("--out-dir", default=".", help="Output directory for visual report export")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)

    # Validate image file existence prior to loading large neural network weights
    if not image_path.is_file():
        print(f"[!] Error: Image file '{image_path}' not found. Please specify a valid image path with --image.")
        sys.exit(1)

    print(f"Loading object detection model ({args.model}) on {args.device}...")
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=args.model,
        confidence_threshold=args.conf,
        device=args.device,
    )

    print(f"Slicing image '{image_path}' and running SAHI inference...")
    result = get_sliced_prediction(
        image=str(image_path),
        detection_model=detection_model,
        slice_height=args.slice_size,
        slice_width=args.slice_size,
        overlap_height_ratio=0.25,
        overlap_width_ratio=0.25,
        perform_standard_pred=False,
    )

    full_predictions = result.object_prediction_list
    only_people = [p for p in full_predictions if p.category.name == "person"]
    result.object_prediction_list = only_people

    print("\n" + "=" * 40)
    print("📊 FINAL ELITE STADIUM REPORT")
    print("=" * 40)
    print(f"Total Objects Found  : {len(full_predictions)}")
    print(f"Confirmed People     : {len(only_people)}")
    print("=" * 40 + "\n")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result.export_visuals(export_dir=str(out_dir), file_name="elite_prediction_final")
    print(f"Success! Check '{out_dir}' for 'elite_prediction_final.png'.")


if __name__ == "__main__":
    main()
