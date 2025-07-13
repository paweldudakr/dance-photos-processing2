import os
from io import BytesIO
from google.cloud import storage
from google.cloud import vision
from PIL import Image, ImageDraw # Pillow jest już do kadrowania, może pomóc przy regionach
import piexif
import json # Do strukturyzowania własnych danych EXIF

storage_client = storage.Client()
vision_client = vision.ImageAnnotatorClient()

# --- Funkcje pomocnicze dla AI (uproszczone) ---

def analyze_photo_for_ai_metadata(image_bytes):
    """
    Próbuje wykryć kolor sukni, włosów i numer startowy używając Vision AI.
    Zwraca słownik z wykrytymi metadanymi.
    To jest wersja 'best-effort' i jej dokładność będzie ograniczona.
    """
    print("Rozpoczynanie analizy AI dla metadanych...")
    metadata_results = {
        "detected_dress_color_suggestion": "Nieznany",
        "detected_hair_color_suggestion": "Nieznany",
        "detected_competitor_number_suggestion": "Nieznany",
        "ai_confidence_notes": []
    }
    image_vision = vision.Image(content=image_bytes)

    # 1. Detekcja obiektów (osób) i twarzy
    people_annotations = []
    face_annotations_data = []
    try:
        objects_response = vision_client.object_localization(image=image_vision)
        people_annotations = [obj for obj in objects_response.localized_object_annotations if obj.name == "Person"]
        
        faces_response = vision_client.face_detection(image=image_vision)
        face_annotations_data = faces_response.face_annotations
        
        print(f"Znaleziono {len(people_annotations)} osób, {len(face_annotations_data)} twarzy.")
    except Exception as e:
        print(f"Błąd podczas detekcji obiektów/twarzy: {e}")
        metadata_results["ai_confidence_notes"].append(f"Błąd detekcji obiektów: {e}")


    # 2. Detekcja Tekstu (OCR) dla numerów startowych
    potential_numbers = []
    try:
        texts_response = vision_client.text_detection(image=image_vision)
        all_detected_text_blocks = texts_response.text_annotations
        if all_detected_text_blocks:
            print(f"Znaleziono {len(all_detected_text_blocks) -1 } bloków tekstu.")
            for text_block in all_detected_text_blocks[1:]: # Pomiń pierwszy (cały tekst)
                desc = text_block.description.strip()
                # Prosta heurystyka: szukaj krótkich ciągów cyfr
                if desc.isdigit() and 1 <= len(desc) <= 3:
                    # Idealnie: sprawdzić, czy bounding box tekstu jest na plecach wykrytej osoby
                    potential_numbers.append(desc)
            if potential_numbers:
                metadata_results["detected_competitor_number_suggestion"] = ", ".join(sorted(list(set(potential_numbers))))
                metadata_results["ai_confidence_notes"].append(f"Sugerowane numery (OCR ogólny): {metadata_results['detected_competitor_number_suggestion']}")
            else:
                metadata_results["ai_confidence_notes"].append("Nie znaleziono tekstu wyglądającego jak numer startowy.")
        else:
            metadata_results["ai_confidence_notes"].append("Nie wykryto żadnego tekstu.")
    except Exception as e:
        print(f"Błąd podczas detekcji tekstu: {e}")
        metadata_results["ai_confidence_notes"].append(f"Błąd OCR: {e}")


    # 3. Detekcja kolorów (bardzo uproszczone - wymagałoby analizy regionów)
    #    Dla rzeczywistej analizy koloru sukni/włosów, należałoby:
    #    a) Wyciąć regiony (np. tułów osoby, obszar głowy) na podstawie `people_annotations` i `face_annotations_data`.
    #    b) Wykonać `image_properties` na tych wyciętych regionach.
    #    c) Odfiltrować np. odcienie skóry przy analizie koloru włosów.
    #    Poniżej jest tylko ogólna detekcja dominujących kolorów całego obrazu.
    
    try:
        props_response = vision_client.image_properties(image=image_vision)
        dominant_colors = props_response.image_properties_annotation.dominant_colors.colors
        if dominant_colors:
            print(f"Znaleziono {len(dominant_colors)} dominujących kolorów.")
            # Przykładowo, weź pierwszy dominujący kolor jako sugestię (bardzo niedokładne)
            # Tu powinna być logika mapowania RGB na nazwy kolorów
            top_color_rgb = dominant_colors[0].color
            color_suggestion_rgb_str = f"RGB({int(top_color_rgb.red)}, {int(top_color_rgb.green)}, {int(top_color_rgb.blue)})"
            
            # To są tylko sugestie dla całego obrazu, nie dla konkretnych elementów
            metadata_results["detected_dress_color_suggestion"] = f"Dominujący kolor obrazu: {color_suggestion_rgb_str}"
            metadata_results["detected_hair_color_suggestion"] = f"Dominujący kolor obrazu: {color_suggestion_rgb_str}"
            metadata_results["ai_confidence_notes"].append(f"Sugestie kolorów oparte na globalnych dominujących kolorach obrazu.")
            metadata_results["ai_confidence_notes"].append("UWAGA: Dla dokładnego koloru sukni/włosów wymagana jest analiza specyficznych regionów obrazu.")

        else:
            metadata_results["ai_confidence_notes"].append("Nie udało się wykryć dominujących kolorów.")
    except Exception as e:
        print(f"Błąd podczas analizy właściwości obrazu (kolorów): {e}")
        metadata_results["ai_confidence_notes"].append(f"Błąd analizy kolorów: {e}")

    print(f"Zakończono analizę AI. Sugestie: {metadata_results}")
    return metadata_results

def add_metadata_to_exif(image_bytes, custom_metadata_dict):
    """Dodaje własne metadane do tagu UserComment EXIF obrazu (format JPEG)."""
    try:
        # piexif działa najlepiej z obrazami JPEG
        # Spróbuj załadować istniejące EXIF
        exif_dict = piexif.load(image_bytes)
    except piexif.InvalidImageDataError: # Jeśli brak EXIF lub nie JPEG
        print("Brak istniejących danych EXIF lub nieobsługiwany format dla piexif. Tworzenie nowego słownika EXIF.")
        # Dla nie-JPEG lub obrazów bez EXIF, piexif.insert może nie zadziałać zgodnie z oczekiwaniami
        # lub może usunąć inne metadane, jeśli format nie jest w pełni wspierany.
        # Najbezpieczniej jest pracować z JPEG.
        # Jeśli to PNG, EXIF jest mniej standardowy.
        return image_bytes # Zwróć oryginalne bajty, jeśli nie można przetworzyć EXIF
    except Exception as e:
        print(f"Nieoczekiwany błąd podczas ładowania EXIF: {e}. Zwracanie oryginalnych bajtów.")
        return image_bytes

    # Przygotuj dane do zapisu w UserComment
    # Tag UserComment to piexif.ExifIFD.UserComment (ID 37510)
    # Wartość powinna być bajtami. Standardowo zawiera 8 bajtów nagłówka kodowania + komentarz.
    # np. b'ASCII\x00\x00\x00Mój komentarz' lub b'UNICODE\x00MójKomentarzUnicode' (UTF-16 LE)
    # lub b'UNDEFINED\x00MójKomentarzUTF8'
    
    user_comment_str = json.dumps(custom_metadata_dict, ensure_ascii=False) # ensure_ascii=False dla polskich znaków
    
    # Użyj kodowania UNDEFINED z tekstem w UTF-8
    # Pierwsze 8 bajtów to typ kodowania, potem sam komentarz.
    # 'UNDEFINED' często oznacza, że interpretacja jest pozostawiona aplikacji.
    exif_dict["Exif"][piexif.ExifIFD.UserComment] = b'UNDEFINED\x00' + user_comment_str.encode('utf-8')

    try:
        exif_bytes = piexif.dump(exif_dict)
        # piexif.insert() wstawia nowe EXIF do strumienia bajtów obrazu JPEG
        new_image_bytes = piexif.insert(exif_bytes, image_bytes)
        print("Pomyślnie dodano/zaktualizowano metadane EXIF (UserComment).")
        return new_image_bytes
    except Exception as e:
        print(f"Błąd podczas zapisu lub wstawiania danych EXIF: {e}")
        return image_bytes # Zwróć oryginalne bajty w razie błędu

# --- Główna funkcja Cloud Function (zmodyfikowana z poprzednich przykładów) ---
def process_image_and_add_metadata(event, context): # Nazwa funkcji może być inna
    """
    Funkcja Cloud Function wyzwalana przez GCS.
    Kadruje obraz (opcjonalnie), analizuje AI i dodaje metadane EXIF.
    """
    bucket_name = event['bucket']
    file_name = event['name']
    
    print(f"Otrzymano plik: {file_name} z bucketu: {bucket_name}.")

    # 1. Pobierz oryginalny obraz
    source_bucket = storage_client.bucket(bucket_name)
    source_blob = source_bucket.blob(file_name)
    
    try:
        image_bytes_original = source_blob.download_as_bytes()
        # Sprawdźmy, czy to JPEG, bo piexif.insert działa najlepiej z JPEG
        # Można by to sprawdzić po content_type lub magic numbers, ale uprośćmy na razie
        content_type = source_blob.content_type
        is_jpeg = content_type and 'jpeg' in content_type.lower()
        
        image_pil_original = Image.open(BytesIO(image_bytes_original))
        print(f"Obraz {file_name} załadowany. Format: {image_pil_original.format}, Rozmiar: {image_pil_original.size}")
    except Exception as e:
        print(f"Błąd podczas pobierania lub otwierania obrazu {file_name}: {e}")
        return

    # --- Tutaj logika kadrowania (jeśli potrzebna) ---
    # Załóżmy, że `processed_image_pil` to obraz po kadrowaniu (lub oryginał jeśli nie kadrowano)
    # Przykład: processed_image_pil = image_pil_original.crop(box)
    # Jeśli nie ma kadrowania:
    processed_image_pil = image_pil_original 
    # --- Koniec logiki kadrowania ---

    # Przygotuj bajty obrazu do dalszego przetwarzania (zapis EXIF)
    # Musimy zapisać obraz (nawet jeśli nie był kadrowany) do strumienia, aby mieć pewność formatu
    # i móc pracować z nim przez piexif.
    final_image_stream = BytesIO()
    save_format = 'JPEG' if is_jpeg else processed_image_pil.format # Zachowaj format jeśli nie JPEG, inaczej JPEG
    if not save_format or save_format.upper() not in ['JPEG', 'PNG']: # Domyślnie JPEG
        save_format = 'JPEG'

    if save_format.upper() == 'JPEG':
        processed_image_pil.save(final_image_stream, format='JPEG', quality=90, exif=image_pil_original.info.get('exif', b'')) # Przeniesienie oryginalnego EXIF
    else:
        # piexif jest głównie dla JPEG; dla PNG metadane EXIF są mniej standardowe
        # i piexif.insert może nie działać. W takim przypadku możemy pominąć dodawanie EXIF.
        print(f"Format obrazu to {save_format}. Pomijanie dodawania metadanych EXIF (piexif najlepiej działa z JPEG).")
        final_image_stream = BytesIO() # Stwórz nowy strumień dla spójności
        processed_image_pil.save(final_image_stream, format=save_format)
        # Ustaw is_jpeg na False, aby nie próbować dodawać EXIF później
        is_jpeg = False

    image_bytes_for_upload = final_image_stream.getvalue()


    # 2. Analiza AI (na oryginalnych bajtach lub bajtach po kadrowaniu)
    #    Jeśli kadrowanie zmienia istotnie kontekst, analiza powinna być na obrazie po kadrowaniu.
    #    Tutaj użyjemy `image_bytes_original` do analizy, ale można to zmienić.
    ai_metadata = {}
    if is_jpeg: # Wykonaj analizę i próbę zapisu EXIF tylko dla JPEG
        ai_metadata = analyze_photo_for_ai_metadata(image_bytes_original) # Analiza na oryginale
        
        # 3. Dodaj wykryte metadane do EXIF (do obrazu, który będzie uploadowany)
        if ai_metadata:
            print(f"Próba dodania metadanych EXIF: {ai_metadata}")
            image_bytes_for_upload = add_metadata_to_exif(image_bytes_for_upload, ai_metadata)
    else:
        print("Pominięto analizę AI i dodawanie EXIF, ponieważ obraz nie jest JPEG.")


    # 4. Zapisz przetworzony obraz do docelowego bucketa
    # Użyj ID projektu z zmiennej środowiskowej, jeśli dostępne (dla Cloud Functions)
    project_id = os.environ.get('GCP_PROJECT', 'pdudaphotos')
    target_bucket_name = f"{project_id}-cropped-photos" # Upewnij się, że ta nazwa jest poprawna
    
    # Możesz chcieć zmodyfikować nazwę pliku
    target_blob_name = f"processed_exif_{file_name}" 
    
    target_bucket = storage_client.bucket(target_bucket_name)
    target_blob = target_bucket.blob(target_blob_name)

    try:
        # Użyj content_type odpowiedniego dla formatu zapisu
        upload_content_type = f'image/{save_format.lower()}'
        target_blob.upload_from_string(image_bytes_for_upload, content_type=upload_content_type)
        print(f"Przetworzony obraz zapisany jako: gs://{target_bucket_name}/{target_blob_name}")
    except Exception as e:
        print(f"Błąd podczas zapisywania przetworzonego obrazu {target_blob_name}: {e}")