import argparse
import glob
import json
import os
import cv2
import numpy as np
from PIL import Image, ExifTags, ImageOps
import piexif
from tqdm import tqdm
import warnings
from io import BytesIO
import google.generativeai as genai

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.cluster._kmeans")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.activation")

# --- Global Initializations ---
GEMINI_API_KEY_CONFIGURED = False

def initialize_models():
    """
    Checks for the Gemini API key.
    """
    global GEMINI_API_KEY_CONFIGURED
    print("INFO: Checking for Gemini API key...")
    if os.environ.get("GEMINI_API_KEY"):
        print("INFO: Gemini API key found in environment variables.")
        GEMINI_API_KEY_CONFIGURED = True
        return True
    else:
        print("CRITICAL: GEMINI_API_KEY environment variable not set.")
        print("Please get your API key from Google AI Studio (https://aistudio.google.com/app/apikey) and set it.")
        GEMINI_API_KEY_CONFIGURED = False
        return False

# --- Utility Functions ---
def load_palette_names(palette_path):
    try:
        with open(palette_path, 'r') as f:
            palette_data = json.load(f)
        print(f"Loaded palette from: {palette_path}")
        return list(palette_data.keys())
    except Exception as e:
        print(f"CRITICAL: Could not load palette '{palette_path}': {e}")
        return []

# --- Core Processing Functions ---
def analyze_image_with_gemini(image_bytes, dress_palette_names, hair_palette_names):
    """
    Analyzes the image using the Gemini API to extract dancer info and a recommended crop.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: Gemini API key not found.")
        return None
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    
    image_part = {"mime_type": "image/jpeg", "data": image_bytes}
    
    prompt = f"""
Analyze the attached image of dancers. Your task is to identify the two largest people, their clothing/hair colors, and recommend a professional crop.

The available dress colors are: {dress_palette_names}.
The available hair colors are: {hair_palette_names}.

Additionally, recommend a professional crop for this image. The crop should follow aesthetic principles like the Rule of Thirds, focus on the main subjects, and have a standard aspect ratio of approximately 3:2.

Respond with a single, valid JSON object only, with no other text or markdown formatting. The JSON object should have the following structure:

{{
  "people": [
    {{
      "is_main_dancer": true,
      "dress_color": "<color_from_dress_palette>",
      "hair_color": "<color_from_hair_palette>",
      "bounding_box": {{ "x1": <float>, "y1": <float>, "x2": <float>, "y2": <float> }}
    }}
  ],
  "recommended_crop": {{ "x1": <float>, "y1": <float>, "x2": <float>, "y2": <float> }}
}}

All coordinates must be normalized to the range [0.0, 1.0]. The origin (0,0) is the top-left corner. The first person in the list should be the largest/main dancer. If you cannot identify any people, respond with an empty JSON object {{}}.
"""
    
    try:
        print("INFO: Sending request to Gemini API...")
        response = model.generate_content([prompt, image_part])
        
        response_text = response.text.strip()
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        if json_start == -1 or json_end == 0:
            print(f"ERROR: No JSON object found in Gemini response: {response_text}")
            return None
            
        json_string = response_text[json_start:json_end]
        
        print(f"INFO: Received response from Gemini: {json_string}")
        return json.loads(json_string)
        
    except Exception as e:
        print(f"ERROR: Gemini API call failed: {e}")
        return None

def _do_actual_processing(image_bytes_input, original_filename):
    print(f"INFO: Processing: {original_filename}")
    if not GEMINI_API_KEY_CONFIGURED:
        if not initialize_models():
            return {"file": original_filename, "error": "Gemini API key not configured"}, None, None

    dress_palette_names = load_palette_names("autocrop-function/default_palette.json")
    hair_palette_names = load_palette_names("autocrop-function/hair_palette.json")

    try:
        img_pil_original = Image.open(BytesIO(image_bytes_input))
        original_exif_bytes = img_pil_original.info.get('exif', b'')

        print("INFO: Normalizing image orientation based on EXIF data...")
        img_pil = ImageOps.exif_transpose(img_pil_original)
        
        image_rgb_full_np = np.array(img_pil.convert('RGB'))
        img_h, img_w = image_rgb_full_np.shape[:2]
        print(f"INFO: Source image resolution (after orientation correction): {img_w}x{img_h}")
        
        buffer = BytesIO()
        img_pil.save(buffer, format="JPEG")
        normalized_image_bytes = buffer.getvalue()

    except Exception as e:
        return {"file": original_filename, "error": f"Image loading failed: {e}"}, None, None

    gemini_result = analyze_image_with_gemini(normalized_image_bytes, dress_palette_names, hair_palette_names)

    main_dancer_dress_color = "Unknown"
    all_hair_colors = []
    main_dancer_bbox = None

    if gemini_result and "people" in gemini_result and gemini_result["people"]:
        people_data = gemini_result["people"]
        for person_data in people_data:
            bbox_data = person_data.get("bounding_box", {})
            # Convert normalized coordinates to pixels
            x1 = int(bbox_data.get("x1", 0) * img_w)
            y1 = int(bbox_data.get("y1", 0) * img_h)
            x2 = int(bbox_data.get("x2", 0) * img_w)
            y2 = int(bbox_data.get("y2", 0) * img_h)
            bbox = (x1, y1, x2, y2)

            all_hair_colors.append(person_data.get("hair_color", "Unknown"))
            if person_data.get("is_main_dancer"):
                main_dancer_dress_color = person_data.get("dress_color", "Unknown")
                main_dancer_bbox = bbox
    
    # Final Image Generation & EXIF
    image_with_boxes = image_rgb_full_np.copy()
    if main_dancer_bbox:
        x1, y1, x2, y2 = main_dancer_bbox
        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 6)
        hair_info = all_hair_colors[0] if all_hair_colors else "Unknown"
        info_text = f"Dress: {main_dancer_dress_color}, Hair: {hair_info}"
        cv2.putText(image_with_boxes, info_text, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(image_with_boxes, info_text, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2, cv2.LINE_AA)

    img_pil_to_crop = Image.fromarray(image_with_boxes)
    
    crop_data = gemini_result.get("recommended_crop") if gemini_result else None
    if crop_data:
        # Convert normalized coordinates to pixels
        x1 = int(crop_data.get("x1", 0) * img_w)
        y1 = int(crop_data.get("y1", 0) * img_h)
        x2 = int(crop_data.get("x2", 0) * img_w)
        y2 = int(crop_data.get("y2", 0) * img_h)
        crop_coords = (x1, y1, x2, y2)
        
        print(f"INFO: Applying Gemini's recommended crop: {crop_coords}")
        final_img_pil = img_pil_to_crop.crop(crop_coords)
    else:
        print("WARNING: Gemini did not return a recommended crop. Image will not be cropped.")
        final_img_pil = img_pil_to_crop

    result_data_json = {"file": original_filename, "dress_color": main_dancer_dress_color, "hair_colors": all_hair_colors, "bib_number": "N/A"}
    modified_exif_bytes = original_exif_bytes
    try:
        exif_dict = piexif.load(original_exif_bytes)
        if "GPS" in exif_dict: del exif_dict["GPS"]
        if piexif.ImageIFD.Orientation in exif_dict["0th"]:
            del exif_dict["0th"][piexif.ImageIFD.Orientation]
            print("  - Removed Orientation tag from EXIF after applying it.")

        exif_dict["Exif"][piexif.ExifIFD.UserComment] = json.dumps(result_data_json, ensure_ascii=False).encode("utf-8")
        exif_dict["0th"][piexif.ImageIFD.ImageDescription] = f"Dress: {main_dancer_dress_color}, Hair: {",".join(all_hair_colors)}".encode("utf-8")
        modified_exif_bytes = piexif.dump(exif_dict)
    except Exception as e:
        print(f"  WARNING: Could not modify EXIF data: {e}.")

    output_image_stream = BytesIO()
    try:
        final_img_pil.save(output_image_stream, format='JPEG', exif=modified_exif_bytes)
        return result_data_json, output_image_stream.getvalue(), 'JPEG'
    except Exception as e:
        print(f"  CRITICAL ERROR: Failed to save image: {e}")
        return result_data_json, None, None
