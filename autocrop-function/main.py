import argparse
import glob
import json
import os
import cv2
import numpy as np
from PIL import Image, ExifTags
from ultralytics import YOLO
import mediapipe as mp
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from google.cloud import vision
import piexif
from tqdm import tqdm
import warnings
from io import BytesIO


# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.cluster._kmeans")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.activation")

# --- Global Initializations ---
YOLO_MODEL = None
SELFIE_SEGMENTER = None
VISION_CLIENT = None

def initialize_models():
    """
    Loads and initializes all the required AI models.
    """
    global YOLO_MODEL, SELFIE_SEGMENTER, VISION_CLIENT
    print("INFO: Initializing AI models...")
    try:
        if not YOLO_MODEL:
            print("  - Loading YOLOv8 model...")
            YOLO_MODEL = YOLO('yolov8n-seg.pt')
        if not SELFIE_SEGMENTER:
            print("  - Loading MediaPipe Selfie Segmentation model...")
            SELFIE_SEGMENTER = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
        if not VISION_CLIENT:
            print("  - Initializing Google Cloud Vision client...")
            VISION_CLIENT = vision.ImageAnnotatorClient()
        print("INFO: All models initialized successfully.")
        return True
    except Exception as e:
        print(f"CRITICAL: Failed to initialize one or more models: {e}")
        return False

# --- Utility Functions ---
def calculate_iou(box_a, box_b):
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])
    inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)
    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    denominator = float(box_a_area + box_b_area - inter_area)
    return inter_area / denominator if denominator > 0 else 0

def hex_to_rgb(hex_str):
    hex_str = hex_str.lstrip('#')
    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_lab(rgb_tuple):
    rgb_np = np.uint8([[rgb_tuple]])
    return cv2.cvtColor(rgb_np, cv2.COLOR_RGB2LAB)[0][0].astype(float)

def find_closest_color(lab_color, palette_lab_map, palette_names_list, ignore_lightness=False):
    if not palette_names_list:
        return "UnknownPaletteEmpty"
    palette_lab_values = np.array(list(palette_lab_map.values()))
    if ignore_lightness:
        distances = cdist([lab_color[1:]], palette_lab_values[:, 1:])
    else:
        distances = cdist([lab_color], palette_lab_values)
    return palette_names_list[np.argmin(distances)]

def load_palette(palette_path):
    palette_data_lab = {}
    palette_names = []
    try:
        with open(palette_path, 'r') as f:
            palette_data = json.load(f)
        for name, color_val in palette_data.items():
            palette_names.append(name)
            if isinstance(color_val, str):
                palette_data_lab[name] = rgb_to_lab(hex_to_rgb(color_val))
            elif isinstance(color_val, list) and len(color_val) == 3:
                palette_data_lab[name] = np.array(color_val, dtype=float)
        print(f"Loaded palette from: {palette_path}")
        return palette_data_lab, palette_names
    except Exception as e:
        print(f"CRITICAL: Could not load palette '{palette_path}': {e}")
        return {}, []

# --- Core Processing Functions ---

def get_professional_crop(person_detections, img_w, img_h, target_aspect_ratio=3/2):
    if not person_detections:
        return None
    sorted_detections = sorted(person_detections, key=lambda p: p['area'], reverse=True)
    subjects = sorted_detections[:2] if len(sorted_detections) >= 2 else sorted_detections[:1]
    min_x = min(s['bbox'][0] for s in subjects)
    min_y = min(s['bbox'][1] for s in subjects)
    max_x = max(s['bbox'][2] for s in subjects)
    max_y = max(s['bbox'][3] for s in subjects)
    box_w, box_h = max_x - min_x, max_y - min_y
    center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2
    padded_w, padded_h = box_w * 1.5, box_h * 1.7
    if padded_w / padded_h > target_aspect_ratio:
        new_h = padded_w / target_aspect_ratio
        new_w = padded_w
    else:
        new_w = padded_h * target_aspect_ratio
        new_h = padded_h
    crop_x1, crop_y1 = int(center_x - new_w / 2), int(center_y - new_h / 2)
    crop_x2, crop_y2 = int(center_x + new_w / 2), int(center_y + new_h / 2)
    crop_x1, crop_y1 = max(0, crop_x1), max(0, crop_y1)
    crop_x2, crop_y2 = min(img_w, crop_x2), min(img_h, crop_y2)
    return (crop_x1, crop_y1, crop_x2, crop_y2)

def localize_objects(image_rgb_full_np, vision_client):
    print("INFO: Starting object localization using Google Vision API...")
    try:
        success, encoded_image = cv2.imencode('.jpg', cv2.cvtColor(image_rgb_full_np, cv2.COLOR_RGB2BGR))
        content = encoded_image.tobytes()
        gcv_image = vision.Image(content=content)
        response = vision_client.object_localization(image=gcv_image)
        return response.localized_object_annotations
    except Exception as e:
        print(f"  ERROR: Google Vision API object_localization call failed: {e}")
        return []

def get_color_from_vision_api(target_bbox, all_objects, object_names, image_rgb_full_np, vision_client, palette_lab_map, palette_names_list, ignore_lightness=False):
    if not target_bbox:
        return "Unknown (no target)"
    relevant_objects = [obj for obj in all_objects if obj.name in object_names]
    if not relevant_objects:
        return f"Unknown (no {object_names[0]} detected)"

    img_h, img_w = image_rgb_full_np.shape[:2]
    
    def get_obj_bbox(obj):
        verts = obj.bounding_poly.normalized_vertices
        return (int(verts[0].x * img_w), int(verts[0].y * img_h), int(verts[2].x * img_w), int(verts[2].y * img_h))

    best_match_obj = max(relevant_objects, key=lambda obj: calculate_iou(target_bbox, get_obj_bbox(obj)), default=None)

    if not best_match_obj:
        return f"Unknown (no matching {object_names[0]})"

    x1, y1, x2, y2 = get_obj_bbox(best_match_obj)
    if x1 >= x2 or y1 >= y2:
        return "Unknown (invalid bbox)"

    cropped_np = image_rgb_full_np[y1:y2, x1:x2]

    try:
        success, encoded_image = cv2.imencode('.jpg', cv2.cvtColor(cropped_np, cv2.COLOR_RGB2BGR))
        content = encoded_image.tobytes()
        response = vision_client.image_properties(image=vision.Image(content=content))
        dominant_color = max(response.image_properties_annotation.dominant_colors.colors, key=lambda c: c.pixel_fraction)
        rgb_tuple = (dominant_color.color.red, dominant_color.color.green, dominant_color.color.blue)
        dominant_lab_color = rgb_to_lab(rgb_tuple)
        return find_closest_color(dominant_lab_color, palette_lab_map, palette_names_list, ignore_lightness)
    except Exception as e:
        print(f"  ERROR: Vision API image_properties call failed: {e}")
        return "Unknown (Vision API error)"

def _do_actual_processing(image_bytes_input, original_filename):
    print(f"INFO: Processing: {original_filename}")
    if not all([YOLO_MODEL, VISION_CLIENT, SELFIE_SEGMENTER]):
        initialize_models()

    dress_palette_lab, dress_palette_names = load_palette("autocrop-function/default_palette.json")
    hair_palette_lab, hair_palette_names = load_palette("autocrop-function/hair_palette.json")

    try:
        img_pil = Image.open(BytesIO(image_bytes_input)).convert('RGB')
        image_rgb_full_np = np.array(img_pil)
        img_h, img_w = image_rgb_full_np.shape[:2]
        original_exif_bytes = img_pil.info.get('exif', b'')
    except Exception as e:
        return {"file": original_filename, "error": f"Image loading failed: {e}"}, None, None

    all_objects = localize_objects(image_rgb_full_np, VISION_CLIENT)
    person_detections = []
    for obj in [o for o in all_objects if o.name == 'Person']:
        verts = obj.bounding_poly.normalized_vertices
        bbox = (int(verts[0].x * img_w), int(verts[0].y * img_h), int(verts[2].x * img_w), int(verts[2].y * img_h))
        person_detections.append({"bbox": bbox, "area": (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])})

    sorted_persons = sorted(person_detections, key=lambda p: p['area'], reverse=True)
    top_two_persons = sorted_persons[:2]

    main_dancer_details = top_two_persons[0] if top_two_persons else None
    main_dancer_dress_color = "Unknown"
    if main_dancer_details:
        main_dancer_dress_color = get_color_from_vision_api(main_dancer_details['bbox'], all_objects, ["Dress", "Skirt", "Clothing"], image_rgb_full_np, VISION_CLIENT, dress_palette_lab, dress_palette_names, ignore_lightness=True)

    all_hair_colors = []
    for person in top_two_persons:
        hair_color = get_color_from_vision_api(person['bbox'], all_objects, ["Hair"], image_rgb_full_np, VISION_CLIENT, hair_palette_lab, hair_palette_names, ignore_lightness=True)
        all_hair_colors.append(hair_color)

    # Final Image Generation
    image_with_boxes = image_rgb_full_np.copy()
    
    for person_det in top_two_persons:
        x1, y1, x2, y2 = person_det['bbox']
        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 0, 255), 3) # Blue box

    if main_dancer_details:
        x1, y1, x2, y2 = main_dancer_details['bbox']
        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 6) # Thicker green box

        hair_info = all_hair_colors[0] if all_hair_colors else "Unknown"
        info_text = f"Dress: {main_dancer_dress_color}, Hair: {hair_info}"
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        font_color = (255, 255, 255)
        thickness = 2
        text_y = y1 - 15 if y1 > 40 else y2 + 40
        
        (text_width, text_height), _ = cv2.getTextSize(info_text, font, font_scale, thickness)
        cv2.rectangle(image_with_boxes, (x1, text_y - text_height - 10), (x1 + text_width + 10, text_y + 5), (0,0,0), -1)
        cv2.putText(image_with_boxes, info_text, (x1 + 5, text_y - 5), font, font_scale, font_color, thickness)

    img_pil = Image.fromarray(image_with_boxes)
    print(f"INFO: Calculating professional crop for source image size {img_w}x{img_h}...")
    crop_coords = get_professional_crop(top_two_persons, img_w, img_h)
    if crop_coords:
        print(f"  - Applying crop with coordinates: {crop_coords}")
        img_pil = img_pil.crop(crop_coords)
    else:
        print("  - No people detected or invalid crop, skipping crop.")

    # EXIF Data
    result_data_json = {
        "file": original_filename,
        "dress_color": main_dancer_dress_color,
        "hair_colors": all_hair_colors if all_hair_colors else ["Unknown"],
        "bib_number": "N/A"
    }
    modified_exif_bytes = original_exif_bytes
    try:
        print("INFO: Attempting to modify EXIF data...")
        exif_dict = piexif.load(original_exif_bytes)
        
        if "GPS" in exif_dict:
            del exif_dict["GPS"]
            print("  - Removed GPS data from EXIF to prevent corruption.")

        # Write full JSON data to UserComment (for machines)
        user_comment_str = json.dumps(result_data_json, ensure_ascii=False)
        exif_dict["Exif"][piexif.ExifIFD.UserComment] = b'UNDEFINED\x00' + user_comment_str.encode('utf-8', 'replace')

        # Write human-readable summary to ImageDescription (for apps like Microsoft Photos)
        description_str = f"Dress: {main_dancer_dress_color}, Hair: {",".join(all_hair_colors)}"
        exif_dict["0th"][piexif.ImageIFD.ImageDescription] = description_str.encode("utf-8")

        modified_exif_bytes = piexif.dump(exif_dict)
        print("  - EXIF data successfully prepared for both UserComment and ImageDescription.")
    except Exception as e:
        print(f"  WARNING: Could not modify EXIF data due to error: {e}. Original EXIF will be used.")
        modified_exif_bytes = original_exif_bytes

    # Save Image
    output_image_stream = BytesIO()
    try:
        print("INFO: Saving image to memory stream...")
        img_pil.save(output_image_stream, format='JPEG', exif=modified_exif_bytes)
        print("  - Image saved successfully with EXIF data.")
        return result_data_json, output_image_stream.getvalue(), 'JPEG'
    except Exception as e:
        print(f"  CRITICAL ERROR: Failed to save image with EXIF: {e}")
        try:
            print("  - Last resort: attempting to save image without any EXIF data.")
            output_image_stream_no_exif = BytesIO()
            img_pil.save(output_image_stream_no_exif, format='JPEG')
            return result_data_json, output_image_stream_no_exif.getvalue(), 'JPEG'
        except Exception as e_final:
            print(f"  FATAL: Could not save image even without EXIF: {e_final}")
            return result_data_json, None, None
