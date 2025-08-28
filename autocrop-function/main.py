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


# Suppress specific warnings if necessary (e.g., from KMeans or other libraries)
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.cluster._kmeans")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.activation")

# --- Global Initializations ---
# Will be initialized by initialize_models() to allow for potential model download messages
YOLO_MODEL = None
SELFIE_SEGMENTER = None
VISION_CLIENT = None
DEFAULT_PALETTE_NAME = "palette.json"


def initialize_models():
    """
    Loads and initializes all the required AI models.
    This function should be called once at the start of the application.
    """
    global YOLO_MODEL, SELFIE_SEGMENTER, VISION_CLIENT
    print("INFO: Initializing AI models...")
    try:
        # 1. Initialize YOLO model
        # Using yolov8n-seg.pt for person detection and segmentation
        if not YOLO_MODEL:
            print("  - Loading YOLOv8 model (yolov8n-seg.pt)...")
            YOLO_MODEL = YOLO('yolov8n-seg.pt')
            print("  - YOLO model loaded successfully.")

        # 2. Initialize MediaPipe Selfie Segmenter
        if not SELFIE_SEGMENTER:
            print("  - Loading MediaPipe Selfie Segmentation model...")
            SELFIE_SEGMENTER = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1) # 0 for general, 1 for landscape
            print("  - MediaPipe model loaded successfully.")

        # 3. Initialize Google Cloud Vision Client
        if not VISION_CLIENT:
            print("  - Initializing Google Cloud Vision client...")
            VISION_CLIENT = vision.ImageAnnotatorClient()
            print("  - Google Cloud Vision client initialized successfully.")
        
        print("INFO: All models initialized successfully.")
        return True

    except Exception as e:
        print(f"CRITICAL: Failed to initialize one or more models: {e}")
        import traceback
        traceback.print_exc()
        return False


# --- Utility Functions ---
def calculate_iou(box_a, box_b):
    """Calculate the Intersection over Union (IoU) of two bounding boxes."""
    # Determine the coordinates of the intersection rectangle
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    # Compute the area of intersection
    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

    # Compute the area of both the prediction and ground-truth rectangles
    box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)

    # Compute the IoU
    iou = inter_area / float(box_a_area + box_b_area - inter_area)

    return iou

def hex_to_rgb(hex_str):
    hex_str = hex_str.lstrip('#')
    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_lab(rgb_tuple):
    rgb_np = np.uint8([[rgb_tuple]])
    lab_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2LAB)
    return lab_np[0][0].astype(float)

def load_palette(palette_path=None):
    palette_data_rgb = {}
    palette_data_lab = {}
    palette_names = []

    if palette_path and os.path.exists(palette_path):
        try:
            with open(palette_path, 'r') as f:
                palette_data_rgb = json.load(f)
            print(f"Loaded custom palette from: {palette_path}")
        except Exception as e:
            print(f"Warning: Could not load custom palette '{palette_path}': {e}. Using default.")
            palette_path = None

    if not palette_path or not palette_data_rgb:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_palette_file = os.path.join(script_dir, DEFAULT_PALETTE_NAME)
        try:
            with open(default_palette_file, 'r') as f:
                palette_data_lab_direct = json.load(f)
            for name, lab_vals in palette_data_lab_direct.items():
                palette_names.append(name)
                palette_data_lab[name] = np.array(lab_vals, dtype=float)
            print(f"Loaded default palette with LAB values: {default_palette_file}")
            return palette_data_lab, palette_names
        except Exception as e:
            print(f"CRITICAL: Could not load default palette '{default_palette_file}': {e}")
            return {}, []

    for name, color_val in palette_data_rgb.items():
        palette_names.append(name)
        if isinstance(color_val, str):
            rgb = hex_to_rgb(color_val)
        elif isinstance(color_val, list) and len(color_val) == 3:
            rgb = tuple(color_val)
        else:
            print(f"Warning: Invalid color format for '{name}' in custom palette. Skipping.")
            palette_data_lab[name] = np.array([0,0,0], dtype=float)
            continue
        palette_data_lab[name] = rgb_to_lab(rgb)
    
    return palette_data_lab, palette_names

def extract_dominant_color(image_rgb_full, mask_full, palette_lab_map, palette_names_list, k=3):
    if image_rgb_full is None or mask_full is None or not palette_names_list:
        return "Unknown"
    
    active_pixels_rgb = image_rgb_full[mask_full > 0]

    if active_pixels_rgb.shape[0] < k:
        if active_pixels_rgb.shape[0] > 0:
            avg_rgb = np.mean(active_pixels_rgb, axis=0).astype(np.uint8)
            dominant_lab_color = rgb_to_lab(tuple(avg_rgb))
        else:
            return "Unknown"
    else:
        pixels_for_lab = active_pixels_rgb.reshape(-1, 1, 3).astype(np.uint8)
        pixels_lab = cv2.cvtColor(pixels_for_lab, cv2.COLOR_RGB2LAB).reshape(-1, 3)

        try:
            kmeans = KMeans(n_clusters=k, n_init='auto', random_state=0, algorithm='lloyd')
            kmeans.fit(pixels_lab)
            unique_labels, counts = np.unique(kmeans.labels_, return_counts=True)
            dominant_cluster_index = unique_labels[np.argmax(counts)]
            dominant_lab_color = kmeans.cluster_centers_[dominant_cluster_index]
        except Exception as e:
            print(f"  Warning: K-means failed ({e}), using average color of masked region.")
            if active_pixels_rgb.shape[0] > 0:
                avg_rgb = np.mean(active_pixels_rgb, axis=0).astype(np.uint8)
                dominant_lab_color = rgb_to_lab(tuple(avg_rgb))
            else:
                return "Unknown"

    palette_lab_values_array = np.array(list(palette_lab_map.values()))
    
    if palette_lab_values_array.size == 0:
        return "UnknownPaletteEmpty"

    distances = cdist([dominant_lab_color], palette_lab_values_array)
    closest_color_index = np.argmin(distances)
    return palette_names_list[closest_color_index]

# --- Detection, Segmentation, and Recognition Functions ---

def get_main_dancer_info(yolo_person_detections, image_cv_rgb, vision_client):
    main_dancer_candidate = None
    largest_area = 0
    best_bib_info = {"text": "UNREADABLE", "confidence": 0.0}

    for det in yolo_person_detections:
        x1, y1, x2, y2 = det['bbox']
        area = det['area']
        
        bbox_height = y2 - y1
        torso_y_start = y1 + int(0.30 * bbox_height)
        torso_y_end = y1 + int(0.80 * bbox_height)
        torso_x_start = x1
        torso_x_end = x2

        if torso_y_start >= torso_y_end or torso_x_start >= torso_x_end:
            continue

        torso_crop_bgr = image_cv_rgb[torso_y_start:torso_y_end, torso_x_start:torso_x_end]
        if torso_crop_bgr.size == 0:
            continue
        
        torso_gray = cv2.cvtColor(torso_crop_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_torso = clahe.apply(torso_gray)

        success, encoded_image = cv2.imencode('.png', enhanced_torso)
        if not success:
            continue
        content = encoded_image.tobytes()
        
        image = vision.Image(content=content)
        response = vision_client.text_detection(image=image)
        texts = response.text_annotations

        current_person_best_bib_text = None
        
        if texts:
            for text in texts:
                cleaned_text = text.description.strip()
                if cleaned_text.isdigit() and 1 <= len(cleaned_text) <= 4:
                    current_person_best_bib_text = cleaned_text
                    break

        if current_person_best_bib_text:
            if area > largest_area:
                largest_area = area
                main_dancer_candidate = det
                best_bib_info = {"text": current_person_best_bib_text, "confidence": 0.9}

    if main_dancer_candidate:
        main_dancer_candidate['bib_number'] = best_bib_info['text']
        main_dancer_candidate['bib_confidence'] = best_bib_info['confidence']
        return main_dancer_candidate
    return None

def get_hair_region_mask(person_yolo_bbox, image_rgb_full, selfie_segmenter):
    x1, y1, x2, y2 = person_yolo_bbox
    img_h, img_w = image_rgb_full.shape[:2]

    bbox_h = y2 - y1
    head_y1 = y1
    head_y2 = min(y2, y1 + int(0.30 * bbox_h))
    head_x1 = x1
    head_x2 = x2

    if head_y1 >= head_y2 or head_x1 >= head_x2:
        return None 
    
    head_crop_rgb = image_rgb_full[head_y1:head_y2, head_x1:head_x2]

    if head_crop_rgb.size == 0:
        return None

    try:
        mp_results = selfie_segmenter.process(head_crop_rgb)
        soft_mp_mask_on_crop = mp_results.segmentation_mask
        binary_mp_mask_on_crop = (soft_mp_mask_on_crop > 0.75).astype(np.uint8)
    except Exception as e:
        print(f"  Warning: MediaPipe Selfie Segmentation failed for a head crop: {e}")
        return None

    hair_mask_full_img = np.zeros((img_h, img_w), dtype=np.uint8)
    hair_mask_full_img[head_y1:head_y2, head_x1:head_x2] = binary_mp_mask_on_crop
    
    return hair_mask_full_img


def people_detection(image_rgb_full_np, img_h, img_w, vision_client, selfie_segmenter):
    """Detects people using Google Vision API and generates segmentation masks."""
    print("INFO: Starting people detection using Google Vision API...")
    detections = []

    try:
        success, encoded_image = cv2.imencode('.jpg', cv2.cvtColor(image_rgb_full_np, cv2.COLOR_RGB2BGR))
        content = encoded_image.tobytes()
        gcv_image = vision.Image(content=content)
        response = vision_client.object_localization(image=gcv_image)
        gcv_objects = response.localized_object_annotations
    except Exception as e:
        print(f"  ERROR: Google Vision API call failed: {e}")
        return []

    # Filter for 'Person' objects first
    person_objects = [obj for obj in gcv_objects if obj.name == 'Person']
    print(f"INFO: Google Vision found {len(person_objects)} person object(s).")

    for obj in person_objects:
        box_norm = obj.bounding_poly.normalized_vertices
        x1 = int(box_norm[0].x * img_w)
        y1 = int(box_norm[0].y * img_h)
        x2 = int(box_norm[2].x * img_w)
        y2 = int(box_norm[2].y * img_h)
        bbox = (x1, y1, x2, y2)

        person_mask = None
        if selfie_segmenter and x1 < x2 and y1 < y2:
            person_crop_rgb = image_rgb_full_np[y1:y2, x1:x2]
            try:
                mp_results = selfie_segmenter.process(person_crop_rgb)
                # Create a binary mask from the segmentation output
                binary_mask_crop = (mp_results.segmentation_mask > 0.75).astype(np.uint8)
                
                # Place the cropped mask onto a full-sized blank mask
                person_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                person_mask[y1:y2, x1:x2] = binary_mask_crop
            except Exception as e:
                print(f"  Warning: Selfie Segmentation failed for person bbox {bbox}: {e}")
        
        detections.append({
            "bbox": bbox,
            "mask": person_mask,
            "area": (x2 - x1) * (y2 - y1),
            "confidence": obj.score
        })
    
    print(f"INFO: Processed {len(detections)} person(s) with segmentation.")
    return detections


def identify_main_dancer(yolo_detections_filtered, image_bgr_full_np, original_filename, main_dancer_details = None, bib_number_val = "UNREADABLE"):
    print("Identify Main Dancer (largest person bbox with a visible digit-only start number)")

    if yolo_detections_filtered and VISION_CLIENT:
        main_dancer_details = get_main_dancer_info(yolo_detections_filtered, image_bgr_full_np, VISION_CLIENT)
        if main_dancer_details:
            bib_number_val = main_dancer_details.get("bib_number", "UNREADABLE")
            print(f"  INFO: Główny tancerz zidentyfikowany dla {original_filename}. Numer: {bib_number_val} (Pewność: {main_dancer_details.get('bib_confidence',0):.2f})")
        else:
            print(f"  INFO: Główny tancerz nie mógł zostać zidentyfikowany (brak czytelnego numeru) w {original_filename}.")
    elif not VISION_CLIENT and yolo_detections_filtered:
         print(f"  OSTRZEŻENIE: Klient Google Vision niedostępny. Nie można zidentyfikować głównego tancerza na podstawie numeru dla {original_filename}.")
         if yolo_detections_filtered:
            largest_person = max(yolo_detections_filtered, key=lambda p: p['area'])
            main_dancer_details = largest_person
            print(f"  INFO: Używam największej osoby jako głównego tancerza (bez potwierdzenia numeru startowego) dla {original_filename}.")
    
    return main_dancer_details, bib_number_val


# ... (Keep global model initializations: YOLO_MODEL, SELFIE_SEGMENTER, VISION_CLIENT) ...
# ... (Keep color utility functions: hex_to_rgb, rgb_to_lab, load_palette, extract_dominant_color) ...
# ... (Keep detection helper functions: get_main_dancer_info, get_hair_region_mask from response #22 or #24) ...
# Ensure these helper functions (get_main_dancer_info, get_hair_region_mask, extract_dominant_color)
# correctly use the global YOLO_MODEL, SELFIE_SEGMENTER, VISION_CLIENT.

def _do_actual_processing(image_bytes_input, original_filename, palette_lab_map="", palette_names_list=""):
    """
    Processes a single image (from bytes) according to the detailed requirements.
    """
    print(f"INFO: _do_actual_processing starting for: {original_filename}")
    global YOLO_MODEL, SELFIE_SEGMENTER, VISION_CLIENT

    if not YOLO_MODEL or not VISION_CLIENT:
        print("INFO: Jeden lub więcej krytycznych modeli nie jest załadowany. Próba inicjalizacji...")
        initialize_models()

    if not YOLO_MODEL: 
        print(f"  KRYTYCZNY BŁĄD (_do_actual_processing): Model YOLO nie jest załadowany dla {original_filename}. Przerywam przetwarzanie tego pliku.")
        return {"file": original_filename, "error": "YOLO model not loaded"}, None, None
    
    if not VISION_CLIENT:
        print(f"  KRYTYCZNY BŁĄD (_do_actual_processing): Klient Google Vision nie jest załadowany dla {original_filename}. Nie można odczytać numeru startowego.")
        return {"file": original_filename, "error": "Google Vision client not loaded"}, None, None
    
    print("INFO: Ładowanie palety kolorów dla test_runner...")
    palette_lab_map, palette_names_list = load_palette("palette.json")
    if not palette_names_list:
        print("BŁĄD KRYTYCZNY: Nie udało się załadować palety kolorów w test_runner. Zatrzymuję.")
        exit()

    if not palette_lab_map or not palette_names_list:
        print(f"  KRYTYCZNY BŁĄD (_do_actual_processing): Paleta kolorów niezaładowana dla {original_filename}. Przerywam przetwarzanie tego pliku.")
        return {"file": original_filename, "error": "Color palette not loaded"}, None, None

    try:
        img_pil = Image.open(BytesIO(image_bytes_input)).convert('RGB')
        image_rgb_full_np = np.array(img_pil)
        image_bgr_full_np = cv2.cvtColor(image_rgb_full_np, cv2.COLOR_RGB2BGR)
        img_h, img_w = image_rgb_full_np.shape[:2]
        original_exif_bytes = img_pil.info.get('exif', b'')
        print(f"Obraz {original_filename} załadowany z bajtów. Format: {img_pil.format}, Rozmiar: {img_w}x{img_h}")
    except Exception as e:
        print(f"  BŁĄD: Nie udało się załadować obrazu {original_filename} z bajtów: {e}")
        return {"file": original_filename, "error": f"Image loading failed: {e}"}, None, None

    person_detections = people_detection(image_rgb_full_np, img_h, img_w, VISION_CLIENT, SELFIE_SEGMENTER)
    #yolo_detections_filtered = people_detection(image_rgb_full_np, img_h, img_w, VISION_CLIENT)

    main_dancer_details = None
    bib_number_val = "UNREADABLE"
    main_dancer_details, bib_number_val = identify_main_dancer(person_detections, image_bgr_full_np, original_filename, main_dancer_details, bib_number_val)

    print("Dominant Dress Color of the Main Dancer")
    main_dancer_dress_color = "Unknown"
    if main_dancer_details:
        md_bbox = main_dancer_details["bbox"]
        md_yolo_seg_mask = main_dancer_details["mask"]

        bbox_h = md_bbox[3] - md_bbox[1]
        
        head_arms_exclusion_mask = np.ones_like(md_yolo_seg_mask, dtype=bool)
        exclusion_zone_y_end = md_bbox[1] + int(0.35 * bbox_h)
        
        head_arms_exclusion_mask[md_bbox[1]:exclusion_zone_y_end, md_bbox[0]:md_bbox[2]] = False
        
        clothing_area_mask = np.zeros_like(md_yolo_seg_mask, dtype=np.uint8)
        clothing_y_start = md_bbox[1] + int(0.35 * bbox_h)
        clothing_y_end = md_bbox[3]
        clothing_x_start = md_bbox[0]
        clothing_x_end = md_bbox[2]

        if clothing_y_start < clothing_y_end and clothing_x_start < clothing_x_end:
            clothing_area_mask[clothing_y_start:clothing_y_end, clothing_x_start:clothing_x_end] = 1
        
        dress_mask_final = np.where((md_yolo_seg_mask > 0) & (clothing_area_mask > 0), 1, 0).astype(np.uint8)

        if np.sum(dress_mask_final) > 0:
            main_dancer_dress_color = extract_dominant_color(image_rgb_full_np, dress_mask_final, palette_lab_map, palette_names_list)
            print(f"  INFO: Kolor sukni głównego tancerza ({original_filename}): {main_dancer_dress_color}")
        else:
            main_dancer_dress_color = "Unknown (pusta maska sukni)"
            print(f"  INFO: Maska sukni głównego tancerza była pusta dla {original_filename}.")
    else:
        main_dancer_dress_color = "Unknown (brak głównego tancerza)"


    print("Hair Color of All Visible People")
    all_hair_colors = []
    if person_detections:
        if SELFIE_SEGMENTER:
            for i, person_det in enumerate(person_detections):
                hair_mask = get_hair_region_mask(person_det["bbox"], image_rgb_full_np, SELFIE_SEGMENTER)
                if hair_mask is not None and np.sum(hair_mask) > 0:
                    hair_color = extract_dominant_color(image_rgb_full_np, hair_mask, palette_lab_map, palette_names_list)
                    all_hair_colors.append(hair_color)
                else:
                    all_hair_colors.append("Unknown (brak maski włosów)")
            print(f"  INFO: Wykryte kolory włosów dla {original_filename}: {all_hair_colors}")
        else:
            all_hair_colors = ["Unknown (SelfieSegmenter niedostępny)"] * len(person_detections)
            print(f"  OSTRZEŻENIE: SelfieSegmenter niedostępny, pomijanie detekcji koloru włosów dla {original_filename}.")
    else:
        all_hair_colors.append("Unknown (brak osób)")


    # --- Draw Bounding Boxes on the image ---
    image_with_boxes = image_rgb_full_np.copy()
    
    for person_det in person_detections:
        x1, y1, x2, y2 = person_det['bbox']
        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 0, 255), 2)

    if main_dancer_details:
        x1, y1, x2, y2 = main_dancer_details['bbox']
        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 3)
        bib_text = main_dancer_details.get("bib_number", "N/A")
        cv2.putText(image_with_boxes, f"Bib: {bib_text}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,255,0), 2)

    img_pil = Image.fromarray(image_with_boxes)


    print("Prepare Output") 
    result_data_json = {
        "file": original_filename,
        "dress_color": main_dancer_dress_color,
        "hair_colors": all_hair_colors if all_hair_colors else ["Unknown"],
        "bib_number": bib_number_val
    }

    print("Prepare modified EXIF bytes")
    modified_exif_bytes = original_exif_bytes 
    try:
        exif_dict = piexif.load(original_exif_bytes) if original_exif_bytes else \
                    {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
        
        user_comment_str = json.dumps(result_data_json, ensure_ascii=False)
        if "Exif" not in exif_dict: exif_dict["Exif"] = {}
        exif_dict["Exif"][piexif.ExifIFD.UserComment] = b'UNDEFINED\x00' + user_comment_str.encode('utf-8', 'replace')
        
        description_str = f"Dress: {main_dancer_dress_color}, Bib: {bib_number_val}, Hairs: {';'.join(all_hair_colors)}"
        if "0th" not in exif_dict: exif_dict["0th"] = {}
        exif_dict["0th"][piexif.ImageIFD.ImageDescription] = description_str.encode('utf-8', 'replace')
        
        modified_exif_bytes = piexif.dump(exif_dict)
    except Exception as e:
        print(f"  OSTRZEŻENIE: Błąd podczas przygotowywania EXIF dla {original_filename}: {e}.")

    print("Save the image (PIL object) with new EXIF to an in-memory byte stream")  
    output_image_stream = BytesIO()
    final_processed_image_bytes = None
    save_image_format = None
    try:
        save_image_format = img_pil.format if img_pil.format and img_pil.format.upper() in ['JPEG', 'PNG'] else 'JPEG'
        
        if save_image_format.upper() != 'JPEG' and modified_exif_bytes != original_exif_bytes:
            print(f"  OSTRZEŻENIE: Zapisuję jako JPEG (zamiast {save_image_format}) aby zachować zmodyfikowane dane EXIF dla {original_filename}.")
            save_image_format = 'JPEG'

        img_pil.save(output_image_stream, format=save_image_format, exif=modified_exif_bytes)
        final_processed_image_bytes = output_image_stream.getvalue()
        print(f"Obraz {original_filename} ({save_image_format}) przygotowany w pamięci z nowymi EXIF.")
    except Exception as e:
        print(f"  BŁĄD: Nie udało się zapisać obrazu {original_filename} z nowymi EXIF do strumienia: {e}")
        try:
            print(f"Próba zapisu {original_filename} bez danych EXIF (po błędzie).")
            output_image_stream_no_exif = BytesIO()
            img_pil.save(output_image_stream_no_exif, format=img_pil.format or 'JPEG')
            final_processed_image_bytes = output_image_stream_no_exif.getvalue()
        except Exception as e_fallback:
            print(f"  BŁĄD KRYTYCZNY: Nie udało się zapisać obrazu {original_filename} nawet bez EXIF: {e_fallback}")
            
    return result_data_json, final_processed_image_bytes, save_image_format

