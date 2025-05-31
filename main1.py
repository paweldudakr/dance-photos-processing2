import os
from io import BytesIO
from google.cloud import storage
from google.cloud import vision
from PIL import Image, ImageDraw

# --- Konfiguracja ---
# Odczytaj ID projektu z zmiennej środowiskowej ustawionej podczas deploymentu
# lub ustaw domyślnie, jeśli funkcja nie jest uruchamiana w GCP.
PROJECT_ID = os.environ.get('GCP_PROJECT', 'pdudaphotos')
TARGET_BUCKET_NAME = f"{PROJECT_ID}-cropped-photos" # Automatycznie ustawia nazwę bucketa docelowego

# Klienty usług Google Cloud
storage_client = storage.Client()
vision_client = vision.ImageAnnotatorClient()

# --- Logika kadrowania ---
# Możesz dostosować te wartości
TARGET_ASPECT_RATIO = 16 / 9  # np. 16:9, 4:3, 1/1 (kwadrat)
PADDING_FACTOR = 0.1  # 10% dodatkowego marginesu wokół wykrytych osób

def get_combined_bounding_box(people_annotations, img_width, img_height):
    """Łączy wszystkie wykryte osoby w jeden prostokąt kadrujący."""
    if not people_annotations:
        return None

    min_x = img_width
    min_y = img_height
    max_x = 0
    max_y = 0

    for person in people_annotations:
        for vertex in person.bounding_poly.vertices:
            min_x = min(min_x, vertex.x)
            min_y = min(min_y, vertex.y)
            max_x = max(max_x, vertex.x)
            max_y = max(max_y, vertex.y)
    
    # Dodaj padding
    box_width = max_x - min_x
    box_height = max_y - min_y
    
    padding_x = box_width * PADDING_FACTOR
    padding_y = box_height * PADDING_FACTOR
    
    min_x = max(0, min_x - padding_x / 2)
    min_y = max(0, min_y - padding_y / 2)
    max_x = min(img_width, max_x + padding_x / 2)
    max_y = min(img_height, max_y + padding_y / 2)

    return int(min_x), int(min_y), int(max_x), int(max_y)

def adjust_crop_to_aspect_ratio(crop_box, img_width, img_height, aspect_ratio):
    """Dostosowuje prostokąt kadrowania do zadanego współczynnika proporcji."""
    x1, y1, x2, y2 = crop_box
    crop_width = x2 - x1
    crop_height = y2 - y1

    current_aspect_ratio = crop_width / crop_height

    if current_aspect_ratio > aspect_ratio: # Kadrowanie jest zbyt szerokie
        new_width = crop_height * aspect_ratio
        diff_width = crop_width - new_width
        x1 += diff_width / 2
        x2 -= diff_width / 2
    elif current_aspect_ratio < aspect_ratio: # Kadrowanie jest zbyt wysokie
        new_height = crop_width / aspect_ratio
        diff_height = crop_height - new_height
        y1 += diff_height / 2
        y2 -= diff_height / 2
        
    # Upewnij się, że nie wychodzimy poza granice obrazu
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_width, x2)
    y2 = min(img_height, y2)

    return int(x1), int(y1), int(x2), int(max_y)


def crop_image_event(event, context):
    """
    Funkcja wyzwalana przez zdarzenie w GCS (nowy plik).
    Pobiera obraz, wykrywa osoby, kadruje i zapisuje.
    """
    bucket_name = event['bucket']
    file_name = event['name']

    print(f"Otrzymano plik: {file_name} z bucketu: {bucket_name}.")

    # 1. Pobierz obraz z GCS
    source_bucket = storage_client.bucket(bucket_name)
    source_blob = source_bucket.blob(file_name)
    
    try:
        image_bytes = source_blob.download_as_bytes()
        image_pil = Image.open(BytesIO(image_bytes))
        img_width, img_height = image_pil.size
    except Exception as e:
        print(f"Błąd podczas pobierania lub otwierania obrazu {file_name}: {e}")
        return

    print(f"Obraz {file_name} załadowany. Rozmiar: {img_width}x{img_height}")

    # 2. Wyślij do Vision AI w celu detekcji osób
    image_vision = vision.Image(content=image_bytes)
    try:
        response = vision_client.object_localization(image=image_vision)
        localized_object_annotations = response.localized_object_annotations
    except Exception as e:
        print(f"Błąd podczas wywołania Vision API dla {file_name}: {e}")
        return

    people_annotations = [obj for obj in localized_object_annotations if obj.name == "Person"]

    if not people_annotations:
        print(f"Nie znaleziono osób na zdjęciu: {file_name}. Kopiowanie oryginału.")
        # Opcjonalnie: skopiuj oryginalny plik, jeśli nie ma osób, lub zignoruj
        target_bucket = storage_client.bucket(TARGET_BUCKET_NAME)
        target_blob = target_bucket.blob(file_name)
        target_blob.upload_from_string(image_bytes, content_type=source_blob.content_type)
        print(f"Oryginalny plik {file_name} skopiowany do {TARGET_BUCKET_NAME}")
        return

    print(f"Znaleziono {len(people_annotations)} osób na zdjęciu {file_name}.")

    # 3. Określ obszar kadrowania
    crop_box = get_combined_bounding_box(people_annotations, img_width, img_height)
    if not crop_box:
         print(f"Nie udało się ustalić ramki kadrowania dla {file_name}. Kopiowanie oryginału.")
         target_bucket = storage_client.bucket(TARGET_BUCKET_NAME)
         target_blob = target_bucket.blob(file_name)
         target_blob.upload_from_string(image_bytes, content_type=source_blob.content_type)
         return

    # 4. Dostosuj kadr do proporcji (opcjonalnie)
    if TARGET_ASPECT_RATIO:
        crop_box = adjust_crop_to_aspect_ratio(crop_box, img_width, img_height, TARGET_ASPECT_RATIO)

    x1, y1, x2, y2 = crop_box
    if x1 >= x2 or y1 >= y2:
        print(f"Nieprawidłowe wymiary kadru dla {file_name} ({x1},{y1},{x2},{y2}). Kopiowanie oryginału.")
        target_bucket = storage_client.bucket(TARGET_BUCKET_NAME)
        target_blob = target_bucket.blob(file_name)
        target_blob.upload_from_string(image_bytes, content_type=source_blob.content_type)
        return
        
    print(f"Obszar kadrowania dla {file_name}: {crop_box}")

    # 5. Kadruj obraz używając Pillow
    cropped_image_pil = image_pil.crop(crop_box)

    # Zapisz przetworzony obraz do strumienia bajtów
    output_image_stream = BytesIO()
    # Zachowaj format oryginalnego obrazu, jeśli to możliwe, lub użyj np. JPEG/PNG
    image_format = Image.open(BytesIO(image_bytes)).format or 'JPEG'
    if image_format.upper() == 'JPEG':
        cropped_image_pil.save(output_image_stream, format='JPEG', quality=90)
    elif image_format.upper() == 'PNG':
         cropped_image_pil.save(output_image_stream, format='PNG')
    else: # Domyślnie
        cropped_image_pil.save(output_image_stream, format='JPEG', quality=90)
    output_image_stream.seek(0)

    # 6. Zapisz wykadrowany obraz do docelowego bucketa GCS
    target_bucket = storage_client.bucket(TARGET_BUCKET_NAME)
    # Możesz chcieć zmodyfikować nazwę pliku, np. dodając "_cropped"
    target_blob_name = f"cropped_{file_name}" 
    target_blob = target_bucket.blob(target_blob_name)

    try:
        target_blob.upload_from_file(output_image_stream, content_type=source_blob.content_type) # Użyj content_type oryginału
        print(f"Wykadrowany obraz zapisany jako: gs://{TARGET_BUCKET_NAME}/{target_blob_name}")
    except Exception as e:
        print(f"Błąd podczas zapisywania wykadrowanego obrazu {target_blob_name}: {e}")