# main.py
import os
from io import BytesIO
from google.cloud import storage
from google.cloud import vision
from PIL import Image, ImageDraw # ImageDraw może być przydatny do debugowania wizualnego
import piexif
import json
from ultralytics import YOLO # NOWY IMPORT
import numpy as np # Często przydatny przy pracy z bounding boxami

# --- Klienty Google Cloud (bez zmian) ---
storage_client = storage.Client()
vision_client = vision.ImageAnnotatorClient() # Dla analizy AI metadanych (kolor, tekst)

# --- Konfiguracja Kadrowania YOLO ---
YOLO_MODEL_NAME = 'yolov8n-pose.pt' # Model do detekcji osób i ich póz
try:
    # Załaduj model YOLO globalnie, aby był ładowany tylko raz przy starcie instancji funkcji (cold start)
    # Automatycznie pobierze wagi, jeśli ich nie ma.
    yolo_model = YOLO(YOLO_MODEL_NAME)
    print(f"Pomyślnie załadowano model YOLO: {YOLO_MODEL_NAME}")
except Exception as e:
    print(f"KRYTYCZNY BŁĄD: Nie udało się załadować modelu YOLO {YOLO_MODEL_NAME}: {e}")
    yolo_model = None # Ustaw na None, aby można było sprawdzić i obsłużyć błąd

CROP_PADDING_PERCENTAGE = 0.10  # 10% marginesu (konfigurowalne)
TARGET_ASPECT_RATIO_W_H = 3.0 / 4.0  # Portret 3:4 (szerokość do wysokości)
MIN_PROXIMITY_PX = 25.0 # Maksymalna odległość między bounding boxami pary

# --- Nowe Funkcje Pomocnicze dla Detekcji i Kadrowania YOLO ---

def _yolo_detect_people(image_pil):
    """Wykrywa osoby na obrazie za pomocą YOLO i zwraca listę detekcji."""
    if not yolo_model:
        print("Model YOLO nie jest załadowany, pomijanie detekcji.")
        return []

    detected_persons = []
    try:
        results = yolo_model(image_pil, verbose=False) # verbose=False aby uniknąć logów z YOLO
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Sprawdź, czy klasa to 'person' (ID 0 w COCO dataset, którego YOLO używa)
                if int(box.cls) == 0: # 0 to zazwyczaj klasa 'person'
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf)
                    # Możesz dodać próg ufności, np. if confidence > 0.5:
                    detected_persons.append({
                        "bbox": (x1, y1, x2, y2),
                        "area": (x2 - x1) * (y2 - y1),
                        "confidence": confidence
                        # Jeśli potrzebujesz keypointów: r.keypoints (jeśli model to *-pose)
                    })
        print(f"YOLO wykryło {len(detected_persons)} osób.")
    except Exception as e:
        print(f"Błąd podczas detekcji YOLO: {e}")
    return detected_persons

def _calculate_box_distance(box1, box2):
    """Oblicza minimalną odległość między krawędziami dwóch prostokątów."""
    # box = (x1, y1, x2, y2)
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Sprawdzenie, czy się przecinają
    if not (x2_1 < x1_2 or x2_2 < x1_1 or y2_1 < y1_2 or y2_2 < y1_1):
        return 0.0  # Przecinają się, odległość 0

    # Odległość w osi X
    if x2_1 < x1_2: # box1 jest na lewo od box2
        dist_x = x1_2 - x2_1
    elif x2_2 < x1_1: # box2 jest na lewo od box1
        dist_x = x1_1 - x2_2
    else: # nakładanie się w osi X
        dist_x = 0

    # Odległość w osi Y
    if y2_1 < y1_2: # box1 jest powyżej box2
        dist_y = y1_2 - y2_1
    elif y2_2 < y1_1: # box2 jest powyżej box1
        dist_y = y1_1 - y2_2
    else: # nakładanie się w osi Y
        dist_y = 0
    
    if dist_x == 0: return float(dist_y)
    if dist_y == 0: return float(dist_x)
    return np.sqrt(dist_x**2 + dist_y**2)


def _find_main_dancing_couple(persons_data):
    """Znajduje główną parę taneczną."""
    if len(persons_data) < 2:
        return None # Nie ma wystarczająco dużo osób, aby utworzyć parę

    # Sortuj osoby wg pola powierzchni malejąco, aby dać priorytet większym
    sorted_persons = sorted(persons_data, key=lambda p: p["area"], reverse=True)
    
    best_couple = None
    max_combined_area = 0

    for i in range(len(sorted_persons)):
        for j in range(i + 1, len(sorted_persons)):
            person1 = sorted_persons[i]
            person2 = sorted_persons[j]

            distance = _calculate_box_distance(person1["bbox"], person2["bbox"])

            if distance <= MIN_PROXIMITY_PX: # Są w kontakcie lub blisko
                # Sprawdźmy czy to "największa" taka para
                current_combined_area = person1["area"] + person2["area"]
                if current_combined_area > max_combined_area:
                    max_combined_area = current_combined_area
                    best_couple = (person1, person2)
    
    if best_couple:
        print(f"Znaleziono parę: osoba 1 (pole: {best_couple[0]['area']}), osoba 2 (pole: {best_couple[1]['area']})")
    else:
        print("Nie znaleziono pary spełniającej kryteria bliskości.")
    return best_couple


def _compute_final_crop_box(box_person1, box_person2, img_width, img_height):
    """Oblicza finalny prostokąt kadrowania dla pary."""
    # 1. Utwórz bounding box obejmujący obie osoby
    x1_u = min(box_person1[0], box_person2[0])
    y1_u = min(box_person1[1], box_person2[1])
    x2_u = max(box_person1[2], box_person2[2])
    y2_u = max(box_person1[3], box_person2[3])
    
    union_width = x2_u - x1_u
    union_height = y2_u - y1_u

    # 2. Dodaj padding
    padding_w = union_width * CROP_PADDING_PERCENTAGE
    padding_h = union_height * CROP_PADDING_PERCENTAGE

    x1_p = x1_u - padding_w / 2
    y1_p = y1_u - padding_h / 2
    x2_p = x2_u + padding_w / 2
    y2_p = y2_u + padding_h / 2
    
    padded_width = x2_p - x1_p
    padded_height = y2_p - y1_p

    if padded_width <=0 or padded_height <=0:
        print("Nieprawidłowe wymiary po dodaniu paddingu, zwracam oryginalne wymiary pary.")
        return int(x1_u), int(y1_u), int(x2_u), int(y2_u)


    # 3. Wymuś proporcje 3:4 (portret)
    current_aspect_ratio = padded_width / padded_height
    
    final_x1, final_y1, final_x2, final_y2 = x1_p, y1_p, x2_p, y2_p

    if abs(current_aspect_ratio - TARGET_ASPECT_RATIO_W_H) > 0.01: # Jeśli różni się znacząco
        if current_aspect_ratio > TARGET_ASPECT_RATIO_W_H: # Box jest za szeroki (lub za niski) -> zwiększ wysokość
            new_height = padded_width / TARGET_ASPECT_RATIO_W_H
            delta_h = (new_height - padded_height) / 2
            final_y1 -= delta_h
            final_y2 += delta_h
        else: # Box jest za wysoki (lub za wąski) -> zwiększ szerokość
            new_width = padded_height * TARGET_ASPECT_RATIO_W_H
            delta_w = (new_width - padded_width) / 2
            final_x1 -= delta_w
            final_x2 += delta_w

    # 4. Upewnij się, że box pozostaje w granicach obrazu
    final_x1 = max(0, final_x1)
    final_y1 = max(0, final_y1)
    final_x2 = min(img_width, final_x2)
    final_y2 = min(img_height, final_y2)

    if final_x2 <= final_x1 or final_y2 <= final_y1:
        print("Nieprawidłowe wymiary po korekcie proporcji i przycięciu do granic. Zwracam box po paddingu.")
        # Wróć do boxa tylko z paddingiem, ale przyciętego do granic
        safe_x1_p = max(0, x1_p)
        safe_y1_p = max(0, y1_p)
        safe_x2_p = min(img_width, x2_p)
        safe_y2_p = min(img_height, y2_p)
        if safe_x2_p > safe_x1_p and safe_y2_p > safe_y1_p:
            return int(safe_x1_p), int(safe_y1_p), int(safe_x2_p), int(safe_y2_p)
        else: # Ostateczny fallback
            return int(x1_u), int(y1_u), int(x2_u), int(y2_u)


    return int(final_x1), int(final_y1), int(final_x2), int(final_y2)


# --- Funkcje Analizy AI i Zapisu EXIF (z poprzedniej wersji, BEZ ZMIAN) ---
# def analyze_photo_for_ai_metadata(image_bytes, pre_detected_people, pre_detected_faces): ...
# def add_metadata_to_exif(image_bytes_jpeg, custom_metadata_dict): ...
# (Te funkcje zostały dostarczone w poprzednich odpowiedziach i są tutaj pominięte dla zwięzłości,
#  ale powinny być obecne w Twoim main.py)
# Skopiuj tutaj pełne definicje tych funkcji z poprzedniej odpowiedzi.
def analyze_photo_for_ai_metadata(image_bytes, pre_detected_people, pre_detected_faces):
    print("Rozpoczynanie analizy AI dla metadanych...")
    metadata_results = {
        "detected_dress_color_suggestion": "Nieznany",
        "detected_hair_color_suggestion": "Nieznany",
        "detected_competitor_number_suggestion": "Nieznany",
        "ai_confidence_notes": []
    }
    image_vision = vision.Image(content=image_bytes)
    metadata_results["ai_confidence_notes"].append(f"Analiza bazuje na {len(pre_detected_people)} os., {len(pre_detected_faces)} tw.")
    try:
        texts_response = vision_client.text_detection(image=image_vision)
        all_detected_text_blocks = texts_response.text_annotations
        potential_numbers = []
        if all_detected_text_blocks:
            print(f"Znaleziono {len(all_detected_text_blocks) -1 } bloków tekstu.")
            for text_block in all_detected_text_blocks[1:]:
                desc = text_block.description.strip()
                if desc.isdigit() and 1 <= len(desc) <= 3:
                    potential_numbers.append(desc)
            if potential_numbers:
                metadata_results["detected_competitor_number_suggestion"] = ", ".join(sorted(list(set(potential_numbers))))
            else:
                 metadata_results["ai_confidence_notes"].append("Nie znaleziono tekstu wyglądającego jak numer startowy.")
        else:
            metadata_results["ai_confidence_notes"].append("Nie wykryto żadnego tekstu.")
    except Exception as e:
        print(f"Błąd podczas detekcji tekstu: {e}")
        metadata_results["ai_confidence_notes"].append(f"Błąd OCR: {e}")
    try:
        props_response = vision_client.image_properties(image=image_vision)
        dominant_colors = props_response.image_properties_annotation.dominant_colors.colors
        if dominant_colors:
            print(f"Znaleziono {len(dominant_colors)} dominujących kolorów.")
            top_color_rgb = dominant_colors[0].color
            color_suggestion_rgb_str = f"RGB({int(top_color_rgb.red)}, {int(top_color_rgb.green)}, {int(top_color_rgb.blue)})"
            metadata_results["detected_dress_color_suggestion"] = f"Dominujący kolor obrazu: {color_suggestion_rgb_str}"
            metadata_results["detected_hair_color_suggestion"] = f"Dominujący kolor obrazu: {color_suggestion_rgb_str}"
            metadata_results["ai_confidence_notes"].append("Sugestie kolorów oparte na globalnych dominujących kolorach obrazu.")
            metadata_results["ai_confidence_notes"].append("UWAGA: Dla dokładnego koloru sukni/włosów wymagana jest analiza specyficznych regionów obrazu.")
        else:
            metadata_results["ai_confidence_notes"].append("Nie udało się wykryć dominujących kolorów.")
    except Exception as e:
        print(f"Błąd podczas analizy właściwości obrazu (kolorów): {e}")
        metadata_results["ai_confidence_notes"].append(f"Błąd analizy kolorów: {e}")
    print(f"Zakończono analizę AI. Sugestie: {metadata_results}")
    return metadata_results

def add_metadata_to_exif(image_bytes_jpeg, custom_metadata_dict):
    try:
        exif_dict = piexif.load(image_bytes_jpeg)
    except piexif.InvalidImageDataError:
        print("Brak danych EXIF lub obraz nie jest JPEG. Tworzenie nowego słownika EXIF.")
        return image_bytes_jpeg 
    except Exception as e:
        print(f"Nieoczekiwany błąd podczas ładowania EXIF: {e}. Zwracanie oryginalnych bajtów.")
        return image_bytes_jpeg
    user_comment_str = json.dumps(custom_metadata_dict, ensure_ascii=False)
    if "Exif" not in exif_dict:
        exif_dict["Exif"] = {}
    exif_dict["Exif"][piexif.ExifIFD.UserComment] = b'UNDEFINED\x00' + user_comment_str.encode('utf-8')
    try:
        exif_bytes_dump = piexif.dump(exif_dict)
        new_image_bytes = piexif.insert(exif_bytes_dump, image_bytes_jpeg)
        print("Pomyślnie dodano/zaktualizowano metadane EXIF (UserComment).")
        return new_image_bytes
    except Exception as e:
        print(f"Błąd podczas zapisu lub wstawiania danych EXIF: {e}")
        return image_bytes_jpeg


# --- Główna funkcja przetwarzania, teraz używająca YOLO do kadrowania ---
def _do_actual_processing(image_bytes_original, original_file_name):
    is_jpeg = False
    original_exif = b''
    img_width, img_height = 0, 0
    save_format = 'JPEG'

    try:
        image_pil_original = Image.open(BytesIO(image_bytes_original))
        img_width, img_height = image_pil_original.size
        original_exif = image_pil_original.info.get('exif', b'')
        if image_pil_original.format and 'JPEG' in image_pil_original.format.upper():
            is_jpeg = True
        elif image_pil_original.format and 'PNG' in image_pil_original.format.upper():
            save_format = 'PNG'
            is_jpeg = False
        else:
            print(f"Oryginalny format to {image_pil_original.format}. Domyślnie konwertuję na JPEG.")
            is_jpeg = True # Bo będziemy zapisywać jako JPEG
        print(f"Obraz {original_file_name} załadowany. Format: {image_pil_original.format}, Rozmiar: {img_width}x{img_height}, is_jpeg (dla zapisu EXIF): {is_jpeg}")
    except Exception as e:
        print(f"Krytyczny błąd podczas otwierania obrazu {original_file_name} z bajtów: {e}")
        return None, None

    # 1. NOWA Detekcja osób za pomocą YOLO
    detected_persons_yolo = _yolo_detect_people(image_pil_original)

    # 2. NOWA Logika kadrowania oparta na YOLO
    cropped_image_pil = image_pil_original.copy() # Domyślnie kopia oryginału
    crop_applied_yolo = False

    if yolo_model is None:
        print("Model YOLO niezaładowany, pomijanie kadrowania YOLO.")
    elif detected_persons_yolo:
        main_couple = _find_main_dancing_couple(detected_persons_yolo)
        if main_couple:
            person1_box = main_couple[0]["bbox"]
            person2_box = main_couple[1]["bbox"]
            crop_box = _compute_final_crop_box(person1_box, person2_box, img_width, img_height)
            
            if crop_box:
                x1, y1, x2, y2 = crop_box
                if x1 < x2 and y1 < y2:
                    print(f"YOLO Kadrowanie dla {original_file_name} do: ({x1}, {y1}, {x2}, {y2})")
                    cropped_image_pil = image_pil_original.crop((x1, y1, x2, y2))
                    crop_applied_yolo = True
                else:
                    print(f"Nieprawidłowe wymiary kadru YOLO dla {original_file_name}. Pomijanie kadrowania.")
            else:
                print(f"Nie udało się obliczyć ramki kadrowania YOLO dla {original_file_name}.")
        else: # Nie znaleziono pary, ale są osoby - można by tu dodać logikę dla pojedynczej osoby
            if len(detected_persons_yolo) == 1:
                print(f"Znaleziono tylko jedną osobę przez YOLO. Kadrowanie tej osoby.")
                # Proste kadrowanie pojedynczej osoby z paddingiem i proporcjami
                # Użyj _compute_final_crop_box przekazując box tej osoby jako person1 i person2
                # lub stwórz uproszczoną funkcję kadrowania dla pojedynczej osoby.
                # Dla uproszczenia, na razie pomijamy specjalne kadrowanie dla jednej osoby.
                person_box = detected_persons_yolo[0]["bbox"]
                # Stwórzmy tymczasowy crop_box dla jednej osoby
                temp_x1, temp_y1, temp_x2, temp_y2 = person_box
                temp_w, temp_h = temp_x2-temp_x1, temp_y2-temp_y1
                
                padding_w_single = temp_w * CROP_PADDING_PERCENTAGE
                padding_h_single = temp_h * CROP_PADDING_PERCENTAGE

                x1_s = max(0, temp_x1 - padding_w_single / 2)
                y1_s = max(0, temp_y1 - padding_h_single / 2)
                x2_s = min(img_width, temp_x2 + padding_w_single / 2)
                y2_s = min(img_height, temp_y2 + padding_h_single / 2)
                
                # Wymuś proporcje
                # (Ta logika jest uproszczona, powinna być w osobnej funkcji)
                # ... pominięto dla zwięzłości, można zaadaptować _compute_final_crop_box ...
                # Na razie użyjemy boxa z paddingiem bez wymuszania proporcji dla jednej osoby
                if x1_s < x2_s and y1_s < y2_s:
                    print(f"YOLO Kadrowanie dla {original_file_name} (1 os.) do: ({int(x1_s)}, {int(y1_s)}, {int(x2_s)}, {int(y2_s)})")
                    cropped_image_pil = image_pil_original.crop((int(x1_s), int(y1_s), int(x2_s), int(y2_s)))
                    crop_applied_yolo = True
                else:
                     print("Nie udało się wykonać kadrowania dla pojedynczej osoby.")
            else:
                print(f"Nie znaleziono głównej pary przez YOLO dla {original_file_name}.")
    else:
        print(f"YOLO nie wykryło żadnych osób dla {original_file_name}. Używanie oryginalnego obrazu.")

    if not crop_applied_yolo: # Jeśli kadrowanie YOLO się nie udało, upewnij się, że pracujemy na kopii
        cropped_image_pil = image_pil_original.copy()

    # 3. Analiza AI dla metadanych (na oryginalnym obrazie)
    ai_metadata = {}
    # Przekazujemy puste listy dla pre_detected_people/faces, ponieważ analiza AI
    # jest teraz niezależna od detekcji używanej do kadrowania (YOLO vs Vision AI)
    # lub można by przekazać tu detekcje YOLO, jeśli `analyze_photo_for_ai_metadata` zostałaby dostosowana
    # do ich użycia (np. do bardziej celowanej analizy kolorów/tekstu w regionach wykrytych przez YOLO).
    # Na razie zostawmy ją tak, jak była, działającą na całym obrazie dla tekstów/kolorów.
    
    # Jeśli chcemy, aby AI metadata (OCR, kolory) działało na *wykadrowanym* obrazie:
    # temp_cropped_bytes_stream = BytesIO()
    # cropped_image_pil.save(temp_cropped_bytes_stream, format='JPEG') # Zapisz wykadrowany do analizy
    # image_bytes_for_ai_analysis = temp_cropped_bytes_stream.getvalue()
    # ai_metadata = analyze_photo_for_ai_metadata(image_bytes_for_ai_analysis, [], [])
    # Ale dla spójności i pełnego kontekstu, lepiej analizować oryginał, chyba że celem jest analiza tylko kadru.
    # Na razie zostawmy analizę na oryginalnych bajtach:
    
    if save_format.upper() == 'JPEG':
        ai_metadata = analyze_photo_for_ai_metadata(image_bytes_original, [], []) # Użyj oryginalnych bajtów
    else:
        print(f"Pominięto analizę AI dla {original_file_name}, ponieważ finalny format nie będzie JPEG ({save_format}).")


    # 4. Przygotuj finalne bajty obrazu (wykadrowanego) do zapisu
    final_image_stream = BytesIO()
    final_exif_bytes_for_save = original_exif

    if save_format.upper() == 'JPEG' and ai_metadata:
        print(f"Przygotowywanie metadanych AI EXIF dla {original_file_name}: {ai_metadata}")
        try:
            exif_dict = piexif.load(original_exif) if original_exif else {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
            user_comment_str = json.dumps(ai_metadata, ensure_ascii=False)
            if "Exif" not in exif_dict: exif_dict["Exif"] = {}
            exif_dict["Exif"][piexif.ExifIFD.UserComment] = b'UNDEFINED\x00' + user_comment_str.encode('utf-8')
            # Dodanie ImageDescription (opcjonalne, dla lepszej widoczności w niektórych viewerach)
            if "0th" not in exif_dict: exif_dict["0th"] = {}
            try:
                exif_dict["0th"][piexif.ImageIFD.ImageDescription] = user_comment_str.encode('utf-8')
            except Exception as e_img_desc:
                print(f"DEBUG: Nie udało się dodać ImageDescription: {e_img_desc}")

            final_exif_bytes_for_save = piexif.dump(exif_dict)
            print(f"Pomyślnie przygotowano nowe dane EXIF dla {original_file_name}.")
        except Exception as e_prepare_exif:
            print(f"Błąd podczas przygotowywania danych EXIF dla {original_file_name}: {e_prepare_exif}.")
            # final_exif_bytes_for_save pozostaje original_exif
            
    try:
        if save_format.upper() == 'JPEG':
            cropped_image_pil.save(final_image_stream, format='JPEG', quality=90, exif=final_exif_bytes_for_save)
        else:
            cropped_image_pil.save(final_image_stream, format=save_format)
        print(f"Obraz {original_file_name} przygotowany do zapisu w formacie {save_format}.")
    except Exception as e:
        print(f"Krytyczny błąd zapisu obrazu {original_file_name}: {e}.")
        return None, None

    image_bytes_for_upload = final_image_stream.getvalue()
    return image_bytes_for_upload, save_format

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


