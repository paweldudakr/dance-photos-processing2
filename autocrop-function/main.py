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
import easyocr
import piexif
from tqdm import tqdm
import warnings


# Suppress specific warnings if necessary (e.g., from KMeans or other libraries)
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.cluster._kmeans")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.activation")

# --- Global Initializations ---
# Will be initialized in main() to allow for potential model wnload messages
YOLO_MODEL = None
SELFIE_SEGMENTER = None
EASYOCR_READER = None
DEFAULT_PALETTE_NAME = "default_palette.json"

# --- Color Utility Functions ---
def hex_to_rgb(hex_str):
    hex_str = hex_str.lstrip('#')
    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_lab(rgb_tuple):
    rgb_np = np.uint8([[rgb_tuple]])
    lab_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2LAB)
    return lab_np[0][0].astype(float) # L: 0-100, a,b: -128 to 127 approx

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
            palette_path = None # Fallback to default

    if not palette_path or not palette_data_rgb: # Load default if custom failed or not provided
        # Assuming default_palette.json is in the same directory as the script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_palette_file = os.path.join(script_dir, DEFAULT_PALETTE_NAME)
        try:
            with open(default_palette_file, 'r') as f:
                # The default palette should already have LAB values
                palette_data_lab_direct = json.load(f)
            # Convert to the structure we need
            for name, lab_vals in palette_data_lab_direct.items():
                palette_names.append(name)
                palette_data_lab[name] = np.array(lab_vals, dtype=float)
            print(f"Loaded default palette with LAB values: {default_palette_file}")
            return palette_data_lab, palette_names
        except Exception as e:
            print(f"CRITICAL: Could not load default palette '{default_palette_file}': {e}")
            print("Please ensure 'default_palette.json' exists and is correctly formatted with LAB values.")
            return {}, [] # Return empty if default also fails

    # If custom palette was loaded (as RGB hex or RGB tuples)
    for name, color_val in palette_data_rgb.items():
        palette_names.append(name)
        if isinstance(color_val, str): # Assuming hex string
            rgb = hex_to_rgb(color_val)
        elif isinstance(color_val, list) and len(color_val) == 3: # Assuming [R, G, B] list
            rgb = tuple(color_val)
        else:
            print(f"Warning: Invalid color format for '{name}' in custom palette. Skipping.")
            # Add a placeholder or handle error appropriately
            palette_data_lab[name] = np.array([0,0,0], dtype=float) # Default to black on error
            continue
        palette_data_lab[name] = rgb_to_lab(rgb)
    
    return palette_data_lab, palette_names


def extract_dominant_color(image_rgb_full, mask_full, palette_lab_map, palette_names_list, k=3):
    if image_rgb_full is None or mask_full is None or not palette_names_list:
        return "Unknown"
    
    # Ensure mask is boolean or 0/1
    active_pixels_rgb = image_rgb_full[mask_full > 0]

    if active_pixels_rgb.shape[0] < k:  # Not enough pixels for k-means
        if active_pixels_rgb.shape[0] > 0: # Use average color if too few for k-means
            avg_rgb = np.mean(active_pixels_rgb, axis=0).astype(np.uint8)
            dominant_lab_color = rgb_to_lab(tuple(avg_rgb))
        else:
            return "Unknown"
    else:
        # Reshape for OpenCV: (num_pixels, 1, 3)
        pixels_for_lab = active_pixels_rgb.reshape(-1, 1, 3).astype(np.uint8)
        pixels_lab = cv2.cvtColor(pixels_for_lab, cv2.COLOR_RGB2LAB).reshape(-1, 3)

        try:
            kmeans = KMeans(n_clusters=k, n_init='auto', random_state=0, algorithm='lloyd')
            kmeans.fit(pixels_lab)
            unique_labels, counts = np.unique(kmeans.labels_, return_counts=True)
            dominant_cluster_index = unique_labels[np.argmax(counts)]
            dominant_lab_color = kmeans.cluster_centers_[dominant_cluster_index]
        except Exception as e:
            # Fallback if k-means fails (e.g. not enough distinct colors)
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

def get_main_dancer_info(yolo_person_detections, image_cv_rgb, easyocr_reader):
    main_dancer_candidate = None
    largest_area = 0
    best_bib_info = {"text": "UNREADABLE", "confidence": 0.0}

    for det in yolo_person_detections:
        x1, y1, x2, y2 = det['bbox']
        area = det['area']
        
        # Define torso region (mid 30%–80% vertically)
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
        
        torso_gray = cv2.cvtColor(torso_crop_bgr, cv2.COLOR_RGB2GRAY)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_torso = clahe.apply(torso_gray)

        # Run EasyOCR
        ocr_results = easyocr_reader.readtext(enhanced_torso, allowlist='0123456789', detail=1)
        
        current_person_best_bib_text = None
        current_person_best_bib_conf = 0.0

        for (_, text, conf) in ocr_results:
            if text.isdigit() and 1 <= len(text) <= 4 and conf >= 0.4:
                if conf > current_person_best_bib_conf:
                    current_person_best_bib_text = text
                    current_person_best_bib_conf = conf
        
        if current_person_best_bib_text: # This person has a readable number
            if area > largest_area:
                largest_area = area
                main_dancer_candidate = det # Store full detection (bbox, mask, area, conf)
                best_bib_info = {"text": current_person_best_bib_text, "confidence": current_person_best_bib_conf}

    if main_dancer_candidate:
        main_dancer_candidate['bib_number'] = best_bib_info['text']
        main_dancer_candidate['bib_confidence'] = best_bib_info['confidence']
        return main_dancer_candidate
    return None


def get_hair_region_mask(person_yolo_bbox, image_rgb_full, selfie_segmenter):
    x1, y1, x2, y2 = person_yolo_bbox
    img_h, img_w = image_rgb_full.shape[:2]

    # Estimate head region: top 30% of the person's bounding box
    bbox_h = y2 - y1
    head_y1 = y1
    head_y2 = min(y2, y1 + int(0.30 * bbox_h)) # Cap at bottom of original bbox
    head_x1 = x1
    head_x2 = x2

    # Ensure valid crop dimensions
    if head_y1 >= head_y2 or head_x1 >= head_x2:
        return None 
    
    head_crop_rgb = image_rgb_full[head_y1:head_y2, head_x1:head_x2]

    if head_crop_rgb.size == 0:
        return None

    # MediaPipe Selfie Segmentation
    try:
        mp_results = selfie_segmenter.process(head_crop_rgb)
        # Soft mask (0.0 to 1.0)
        soft_mp_mask_on_crop = mp_results.segmentation_mask
        # Threshold to binary mask
        binary_mp_mask_on_crop = (soft_mp_mask_on_crop > 0.75).astype(np.uint8) # Threshold tunable
    except Exception as e:
        print(f"  Warning: MediaPipe Selfie Segmentation failed for a head crop: {e}")
        return None

    # Create a full-size mask and place the head mask onto it
    hair_mask_full_img = np.zeros((img_h, img_w), dtype=np.uint8)
    hair_mask_full_img[head_y1:head_y2, head_x1:head_x2] = binary_mp_mask_on_crop
    
    return hair_mask_full_img


def process_image(image_path, palette_lab_map, palette_names_list):
    global YOLO_MODEL, SELFIE_SEGMENTER, EASYOCR_READER # Allow access to global models

    try:
        # Load image using Pillow (keeps EXIF, handles color profiles better)
        # Convert to RGB format for consistency
        img_pil = Image.open(image_path).convert('RGB')
        image_rgb_full_np = np.array(img_pil)
        image_bgr_full_np = cv2.cvtColor(image_rgb_full_np, cv2.COLOR_RGB2BGR) # For EasyOCR if needed, and some cv2 ops
        img_h, img_w = image_rgb_full_np.shape[:2]
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

    # --- Detection & Segmentation Pipeline ---
    yolo_detections_filtered = []
    if YOLO_MODEL:
        try:
            yolo_results = YOLO_MODEL(image_rgb_full_np, verbose=False) # Pass NumPy array
            if yolo_results and yolo_results[0].boxes is not None and yolo_results[0].masks is not None:
                person_class_id = 0 # Typically 'person' in COCO
                
                boxes = yolo_results[0].boxes.cpu().numpy() # BBoxes (x1,y1,x2,y2,conf,cls)
                masks_data = yolo_results[0].masks.data.cpu().numpy() # Segmentation masks (N, H, W)
                
                for i in range(len(boxes)):
                    if int(boxes[i,5]) == person_class_id: # Filter for persons
                        bbox_coords = boxes[i,:4].astype(int)
                        x1,y1,x2,y2 = bbox_coords
                        mask_np = cv2.resize(masks_data[i], (img_w, img_h), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
                        yolo_detections_filtered.append({
                            "bbox": tuple(bbox_coords),
                            "mask": mask_np, # Full image size mask for this person
                            "area": (x2-x1)*(y2-y1),
                            "confidence": float(boxes[i,4])
                        })
        except Exception as e:
            print(f"  Error during YOLO detection on {image_path}: {e}")
    
    if not yolo_detections_filtered:
        print(f"  No persons detected by YOLO in {image_path}.")
        # Fallback for hair colors if no persons detected by YOLO
        all_hair_colors = ["Unknown"] * len(yolo_detections_filtered) # Will be empty
    else:
        print(f"  YOLO detected {len(yolo_detections_filtered)} person(s) in {image_path}.")


    # Identify Main Dancer and their bib number
    main_dancer_details = None
    bib_number_val = "UNREADABLE"
    if yolo_detections_filtered and EASYOCR_READER:
        main_dancer_details = get_main_dancer_info(yolo_detections_filtered, image_bgr_full_np, EASYOCR_READER) # EasyOCR often prefers BGR
        if main_dancer_details:
            bib_number_val = main_dancer_details.get("bib_number", "UNREADABLE")
            print(f"  Main dancer identified. Bib: {bib_number_val} (Conf: {main_dancer_details.get('bib_confidence', 0):.2f})")
        else:
            print(f"  Main dancer could not be identified (no readable bib number on largest persons).")

    # --- Dress Color for Main Dancer ---
    main_dancer_dress_color = "Unknown"
    if main_dancer_details:
        md_bbox = main_dancer_details["bbox"]
        md_yolo_mask = main_dancer_details["mask"] # This is the segmentation mask for the main dancer

        # Define dress region: exclude top 35% of bbox (head/shoulders/arms)
        bbox_h = md_bbox[3] - md_bbox[1]
        # Create a mask to exclude the top part
        exclusion_start_y = md_bbox[1] # y_start of person bbox
        exclusion_end_y = md_bbox[1] + int(0.35 * bbox_h) # end of exclusion zone
        
        # Create a boolean mask for the clothing area (lower 65% of person)
        clothing_focus_mask = np.ones_like(md_yolo_mask, dtype=bool)
        clothing_focus_mask[exclusion_start_y:exclusion_end_y, md_bbox[0]:md_bbox[2]] = False # Exclude top part within bbox
                                                                                             # More accurately, just set rows to False
        clothing_focus_mask_rows = np.ones((img_h, img_w), dtype=bool)
        clothing_focus_mask_rows[exclusion_start_y:exclusion_end_y, :] = False


        # Final dress mask is intersection of person's segmentation and the clothing focus area
        # dress_mask_final = md_yolo_mask & clothing_focus_mask # Bitwise AND for boolean/uint8
        dress_mask_final = np.where((md_yolo_mask > 0) & (clothing_focus_mask_rows), 1, 0).astype(np.uint8)


        if np.sum(dress_mask_final) > 0: # If there are any pixels in the dress mask
            main_dancer_dress_color = extract_dominant_color(image_rgb_full_np, dress_mask_final, palette_lab_map, palette_names_list)
            print(f"  Main dancer dress color: {main_dancer_dress_color}")
        else:
            print(f"  Main dancer dress mask was empty after exclusions.")
            main_dancer_dress_color = "Unknown (empty mask)"


    # --- Hair Color for All Visible People ---
    all_hair_colors = []
    if yolo_detections_filtered and SELFIE_SEGMENTER:
        for i, person_det in enumerate(yolo_detections_filtered):
            hair_mask = get_hair_region_mask(person_det["bbox"], image_rgb_full_np, SELFIE_SEGMENTER)
            if hair_mask is not None and np.sum(hair_mask) > 0:
                hair_color = extract_dominant_color(image_rgb_full_np, hair_mask, palette_lab_map, palette_names_list)
                all_hair_colors.append(hair_color)
                print(f"  Person {i+1} hair color: {hair_color}")
            else:
                all_hair_colors.append("Unknown (no hair mask)")
                print(f"  Person {i+1} hair mask could not be generated or was empty.")
    elif not yolo_detections_filtered:
        all_hair_colors.append("Unknown (no people)") # Should be empty if no people
    else: # No selfie segmenter
        all_hair_colors = ["Not Processed (SelfieSeg off)"] * len(yolo_detections_filtered)


    # --- Output ---
    filename = os.path.basename(image_path)
    result_data = {
        "file": filename,
        "dress_color": main_dancer_dress_color,
        "hair_colors": all_hair_colors if all_hair_colors else ["Unknown"], # Ensure it's not empty
        "bib_number": bib_number_val
    }

    # Generate and insert EXIF properties
    try:
        exif_dict = {}
        try:
            # Try to load existing EXIF from original PIL image
            if img_pil.info.get('exif'):
                exif_dict = piexif.load(img_pil.info['exif'])
            else: # Create basic structure if no EXIF
                exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
        except Exception as exif_load_err:
            print(f"  Warning: Could not load existing EXIF for {filename}: {exif_load_err}. Creating new EXIF dict.")
            exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}


        user_comment_str = json.dumps(result_data) # Store the full result dict
        if "Exif" not in exif_dict: exif_dict["Exif"] = {}
        exif_dict["Exif"][piexif.ExifIFD.UserComment] = b'UNDEFINED\x00' + user_comment_str.encode('utf-8', 'replace')
        
        # Add some individual tags as well if possible (optional, UserComment is primary)
        if "0th" not in exif_dict: exif_dict["0th"] = {}
        # piexif.ImageIFD.ImageDescription expects bytes
        description_str = f"Dress: {main_dancer_dress_color}, Bib: {bib_number_val}, Hairs: {';'.join(all_hair_colors)}"
        exif_dict["0th"][piexif.ImageIFD.ImageDescription] = description_str.encode('utf-8', 'replace')

        exif_bytes = piexif.dump(exif_dict)
        
        # Save the original image with new EXIF (overwrites original if no new path given)
        # To avoid overwriting, create a new path or use piexif.insert with new filename
        # For this requirement, let's assume we modify the original file.
        # Pillow save is safer to preserve image integrity and color profiles.
        img_pil.save(image_path, exif=exif_bytes)
        print(f"  EXIF data updated for {filename}")

    except Exception as e:
        print(f"  Error writing EXIF data for {image_path}: {e}")

    return result_data


def _do_actual_processing(image_bytes_original, original_file_name):
    global YOLO_MODEL, SELFIE_SEGMENTER, EASYOCR_READER # Declare use of global vars

    parser = argparse.ArgumentParser(description="Process ballroom dance photos to extract metadata.")
    parser.add_argument("input_dir", help="Directory containing input images.")
    parser.add_argument("--palette", help="Path to a custom JSON color palette file (RGB hex or list values). If not provided, uses default_palette.json (LAB values).")
    args = parser.parse_args()

    print("Initializing models...")
    try:
        YOLO_MODEL = YOLO('yolov8n-seg.pt') # Segmentation model for person masks
        print("  YOLOv8n-seg model loaded.")
    except Exception as e:
        print(f"Error loading YOLO model: {e}. Person detection/segmentation will be skipped.")
        YOLO_MODEL = None
        
    try:
        SELFIE_SEGMENTER = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0)
        print("  MediaPipe SelfieSegmentation model loaded.")
    except Exception as e:
        print(f"Error loading MediaPipe SelfieSegmentation model: {e}. Hair segmentation will be skipped.")
        SELFIE_SEGMENTER = None

    try:
        EASYOCR_READER = easyocr.Reader(['en'], gpu=False) # gpu=False for wider compatibility, can be True if GPU available
        print("  EasyOCR reader loaded.")
    except Exception as e:
        print(f"Error loading EasyOCR reader: {e}. Bib number recognition will be skipped.")
        EASYOCR_READER = None
    
    print("Loading color palette...")
    palette_lab_map, palette_names_list = load_palette("default_palette.json")
    if not palette_names_list:
        print("CRITICAL: Color palette could not be loaded. Dominant color extraction will fail.")
        # Optionally, exit or provide a minimal fallback palette here
        # For now, extract_dominant_color will return "Unknown"

    image_extensions = ('*.jpg', '*.jpeg', '*.png')
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(args.input_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(args.input_dir, ext.upper())))
    
    if not image_paths:
        print(f"No images found in {args.input_dir}")
        return

    print(f"Found {len(image_paths)} images to process.")
    all_results = []
    for img_path in tqdm(image_paths, desc="Processing images"):
        print(f"\nProcessing: {img_path}")
        result = process_image(img_path, palette_lab_map, palette_names_list)
        if result:
            print(json.dumps(result)) # Print JSON line for each image
            all_results.append(result)
    
    # Optionally, save all results to a single JSON file
    # output_summary_file = os.path.join(args.input_dir, "processing_summary.json")
    # with open(output_summary_file, 'w') as f_summary:
    #     json.dump(all_results, f_summary, indent=2)
    # print(f"\nSummary saved to {output_summary_file}")

    if SELFIE_SEGMENTER:
        SELFIE_SEGMENTER.close() # Clean up MediaPipe model


# --- Główna Funkcja Cloud Function (Entry Point) ---
# Definicja process_image_crop_and_metadata(event, context) pozostaje taka sama jak w poprzedniej odpowiedzi,
# wywołuje _do_actual_processing.
# Skopiuj tutaj pełną definicję tej funkcji z poprzedniej odpowiedzi.
def process_image_crop_and_metadata(event, context):
    bucket_name = event['bucket']
    file_name = event['name']
    print(f"Otrzymano zdarzenie dla pliku: {file_name} z bucketu: {bucket_name} (ID: {context.event_id})")

    source_bucket = storage_client.bucket(bucket_name)
    source_blob = source_bucket.blob(file_name)
    
    try:
        image_bytes_original = source_blob.download_as_bytes()
        original_content_type = source_blob.content_type 
        print(f"Pobrano plik {file_name} ({len(image_bytes_original)} bajtów) z GCS.")
    except Exception as e:
        print(f"Błąd podczas pobierania obrazu {file_name} z GCS: {e}")
        return 

    processed_image_bytes, save_format = _do_actual_processing(image_bytes_original, file_name)

    if not processed_image_bytes:
        print(f"Przetwarzanie obrazu {file_name} nie powiodło się. Nic nie zostanie zapisane.")
        return

    project_id = os.environ.get('GCP_PROJECT', None)
    if not project_id:
        if os.environ.get("FUNCTIONS_EMULATOR") == "true":
             project_id = "pdudaphotos-local-test" 
        else: 
            project_id = "pdudaphotos" 
            print(f"OSTRZEŻENIE: Zmienna środowiskowa GCP_PROJECT nie jest ustawiona. Używam domyślnego '{project_id}'.")
    
    target_bucket_name = f"{project_id}-cropped-photos" 
    base, ext = os.path.splitext(file_name)
    target_blob_name = f"processed_yolo_{base}.{save_format.lower()}" # Zmieniona nazwa dla odróżnienia

    target_bucket = storage_client.bucket(target_bucket_name)
    target_blob = target_bucket.blob(target_blob_name)

    try:
        upload_content_type = f'image/{save_format.lower()}' if save_format else original_content_type
        target_blob.upload_from_string(processed_image_bytes, content_type=upload_content_type)
        print(f"Przetworzony obraz ({len(processed_image_bytes)} bajtów) zapisany jako: gs://{target_bucket_name}/{target_blob_name}")
    except Exception as e:
        print(f"Błąd podczas zapisywania przetworzonego obrazu {target_blob_name} do GCS: {e}")