import os
import types
import glob # Do wyszukiwania plików pasujących do wzorca
from main import _do_actual_processing, initialize_models # Importuj funkcję wewnętrzną z main.py
from PIL import Image # Do wczytania EXIF lokalnie
import piexif 
from ultralytics import YOLO

# Will be initialized in main() to allow for potential model wnload messages
YOLO_MODEL = None
SELFIE_SEGMENTER = None
EASYOCR_READER = None
DEFAULT_PALETTE_NAME = "default_palette.json"

if __name__ == "__main__":
    # 1. Uwierzytelnianie (jeśli potrzebne dla Vision AI, nawet przy lokalnym pliku)
    # Jeśli używasz 'gcloud auth application-default login', to jest obsługiwane.
    # os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:\\sciezka\\do\\twojego\\klucza.json"
    
    # Ustawienie zmiennej GCP_PROJECT dla testów lokalnych, jeśli funkcja na niej polega
    # W funkcji _do_actual_processing nie jest bezpośrednio używana, ale może być w storage_client
    # lub vision_client jeśli nie są skonfigurowane inaczej. Lepiej ustawić.
    os.environ["GCP_PROJECT"] = "pdudaphotos" # Ustaw swój project ID

    # 2. Ścieżka do LOKALNEGO KATALOGU ze zdjęciami do testów
    input_image_directory = r"g:\Mój dysk\Photos\samples"  # ZMIEŃ NA WŁAŚCIWĄ ŚCIEŻKĘ
    # Użyj 'r' przed ścieżką (raw string) lub podwójnych ukośników '\\' w Windows.

    # 3. Ścieżka do LOKALNEGO KATALOGU, gdzie będą zapisywane przetworzone zdjęcia
    output_folder = r"g:\Mój dysk\Photos\output" # ZMIEŃ NA WŁAŚCIWĄ ŚCIEŻKĘ
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Utworzono folder wyjściowy: {output_folder}")

    if not os.path.isdir(input_image_directory):
        print(f"BŁĄD: Podana ścieżka wejściowa nie jest katalogiem lub nie istnieje: {input_image_directory}")
        exit()

    # Loads and initializes all the required AI models.
    # This function should be called once at the start of the application.
    initialize_models()

    # Wyszukaj pliki obrazów (np. jpg, png) w podanym katalogu
    # Możesz dodać więcej rozszerzeń w razie potrzeby
    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp')
    image_files_to_process = []
    for ext in image_extensions:
        image_files_to_process.extend(glob.glob(os.path.join(input_image_directory, ext)))
        image_files_to_process.extend(glob.glob(os.path.join(input_image_directory, ext.upper()))) # Dla wielkich liter

    if not image_files_to_process:
        print(f"Nie znaleziono żadnych plików obrazów w katalogu: {input_image_directory}")
        exit()

    print(f"Znaleziono {len(image_files_to_process)} obrazów do przetworzenia.")

    for local_image_path in image_files_to_process:
        original_file_name = os.path.basename(local_image_path)
        print(f"\n--- Przetwarzanie lokalnego pliku: {local_image_path} ---")

        try:
            # Wczytaj bajty lokalnego obrazu
            with open(local_image_path, 'rb') as f:
                image_bytes_input = f.read()

            # Wywołaj główną logikę przetwarzania
            # (Funkcja _do_actual_processing jest w main.py)
            processed_image_bytes, save_format = _do_actual_processing(image_bytes_input, original_file_name)

            if processed_image_bytes:
                # Zapisz przetworzony obraz lokalnie do weryfikacji
                base, input_ext = os.path.splitext(original_file_name)
                # Użyj formatu zwróconego przez funkcję przetwarzającą
                output_file_name = f"processed_local_{base}.{save_format.lower()}"
                output_path = os.path.join(output_folder, output_file_name)
                
                with open(output_path, 'wb') as f_out:
                    f_out.write(processed_image_bytes)
                print(f"Przetworzony obraz zapisano jako: {output_path}")
                
                # Opcjonalnie: sprawdź EXIF zapisanego pliku, jeśli to JPEG
                if save_format.upper() == 'JPEG':
                    try:
                        img_with_exif = Image.open(output_path)
                        # info.get('exif') może nie być obecne jeśli piexif nie dodał nic lub Pillow nie zapisał
                        exif_raw_data = img_with_exif.info.get('exif')
                        if exif_raw_data:
                            exif_data = piexif.load(exif_raw_data)
                            user_comment_bytes = exif_data.get("Exif", {}).get(piexif.ExifIFD.UserComment, b'')
                            if user_comment_bytes.startswith(b'UNDEFINED\x00'):
                                user_comment_json_str = user_comment_bytes[len(b'UNDEFINED\x00'):].decode('utf-8', errors='ignore')
                                print(f"Odczytany UserComment z {output_file_name}: {user_comment_json_str}")
                            elif user_comment_bytes: # Jeśli jest, ale nie ma nagłówka UNDEFINED
                                print(f"UserComment (raw) z {output_file_name}: {user_comment_bytes[:50]}...") # Pokaż fragment
                            else:
                                print(f"Brak UserComment w {output_file_name}.")
                        else:
                            print(f"Brak danych EXIF (info['exif']) w przetworzonym pliku: {output_file_name}")

                    except Exception as e_exif:
                        print(f"Nie udało się odczytać/zinterpretować EXIF z {output_file_name}: {e_exif}")
            else:
                print(f"Przetwarzanie obrazu {original_file_name} nie zwróciło danych (błąd w _do_actual_processing).")

        except FileNotFoundError:
            print(f"BŁĄD: Nie znaleziono pliku {local_image_path}. Pomijanie.")
        except Exception as e:
            print(f"KRYTYCZNY BŁĄD podczas przetwarzania pliku {original_file_name}: {e}")
            import traceback
            traceback.print_exc() # Wydrukuje pełny ślad stosu błędu
    
    print("\n--- Zakończono przetwarzanie wszystkich plików z katalogu. ---")