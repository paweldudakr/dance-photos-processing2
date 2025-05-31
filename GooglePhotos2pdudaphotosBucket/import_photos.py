import os
import pickle
import requests
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from google.cloud import storage

# --- Konfiguracja ---
GCS_BUCKET_NAME = 'pdudaphotos-input-photos'
CLIENT_SECRETS_FILE = 'client_secret.json'  # Plik pobrany z Google Cloud Console
SCOPES = ['https://www.googleapis.com/auth/photoslibrary.readonly'] # Zakres dostępu do API Zdjęć
TOKEN_PICKLE_FILE = 'token.pickle' # Plik do przechowywania tokenów dostępu

def authenticate_google_photos():
    """Uwierzytelnia użytkownika i zwraca obiekt usługi API Zdjęć Google."""
    creds = None
    if os.path.exists(TOKEN_PICKLE_FILE):
        with open(TOKEN_PICKLE_FILE, 'rb') as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                print(f"Nie udało się odświeżyć tokena: {e}")
                # Jeśli odświeżenie zawiedzie, poproś o ponowną autoryzację
                flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
                creds = flow.run_local_server(port=0)
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        
        with open(TOKEN_PICKLE_FILE, 'wb') as token:
            pickle.dump(creds, token)
            print(f"Zapisano dane uwierzytelniające w {TOKEN_PICKLE_FILE}")

    return build('photoslibrary', 'v1', credentials=creds, static_discovery=False)

def list_albums(service):
    """Wyświetla listę albumów użytkownika i pozwala wybrać jeden."""
    print("\nPobieranie listy albumów...")
    albums_list = []
    next_page_token = None
    try:
        while True:
            results = service.albums().list(
                pageSize=50, # Można dostosować
                pageToken=next_page_token
            ).execute()
            
            albums = results.get('albums', [])
            if not albums:
                print("Nie znaleziono albumów.")
                return None
            
            albums_list.extend(albums)
            next_page_token = results.get('nextPageToken')
            if not next_page_token:
                break
        
        print("\nDostępne albumy:")
        for i, album in enumerate(albums_list):
            print(f"{i + 1}. {album.get('title')} (ID: {album.get('id')})")

        while True:
            try:
                choice = int(input("Wybierz numer albumu do zaimportowania: "))
                if 1 <= choice <= len(albums_list):
                    return albums_list[choice - 1]
                else:
                    print("Nieprawidłowy numer.")
            except ValueError:
                print("Proszę podać liczbę.")
    except Exception as e:
        print(f"Wystąpił błąd podczas pobierania albumów: {e}")
        return None


def get_media_items_from_album(service, album_id):
    """Pobiera wszystkie elementy multimedialne (zdjęcia) z danego albumu."""
    print(f"\nPobieranie zdjęć z albumu ID: {album_id}...")
    media_items_list = []
    next_page_token = None
    try:
        while True:
            body_params = {
                'albumId': album_id,
                'pageSize': 100  # Max 100
            }
            if next_page_token:
                body_params['pageToken'] = next_page_token
            
            results = service.mediaItems().search(body=body_params).execute()
            
            items = results.get('mediaItems', [])
            if not items:
                # Sprawdź, czy to faktycznie koniec, czy może błąd bez itemów
                if not media_items_list and not next_page_token: # Jeśli nic nie pobrano i nie ma następnej strony
                     print(f"Nie znaleziono żadnych zdjęć w albumie ID: {album_id} lub album jest pusty.")
                break 
            
            # Filtruj tylko zdjęcia (pomijaj filmy, jeśli nie są potrzebne)
            for item in items:
                if 'photo' in item.get('mediaMetadata', {}):
                    media_items_list.append(item)
            
            print(f"Pobrano dotychczas {len(media_items_list)} zdjęć...") # Zaktualizowany komunikat
            next_page_token = results.get('nextPageToken')
            if not next_page_token:
                break
        
        if not media_items_list: # Dodatkowe sprawdzenie po pętli
            print(f"Nie znaleziono żadnych zdjęć w albumie ID: {album_id} (po zakończeniu pętli).")

        return media_items_list
    except Exception as e:
        print(f"Wystąpił błąd podczas pobierania zdjęć z albumu: {e}")
        import traceback
        traceback.print_exc() # Wydrukuje pełny traceback błędu
        return []


def download_and_upload_to_gcs(media_item, gcs_bucket):
    """Pobiera zdjęcie i przesyła je do Google Cloud Storage."""
    filename = media_item.get('filename')
    base_url = media_item.get('baseUrl')
    photo_id = media_item.get('id')

    if not base_url or not filename:
        print(f"Brak base_url lub nazwy pliku dla elementu ID: {photo_id}")
        return

    # Dodaj parametr '=d', aby pobrać oryginalny plik
    download_url = base_url + "=d"
    
    try:
        print(f"Pobieranie: {filename} (ID: {photo_id})")
        response = requests.get(download_url, stream=True)
        response.raise_for_status() # Rzuć wyjątkiem dla złych statusów HTTP

        # Użyj photo_id jako części nazwy pliku w GCS dla unikalności,
        # lub po prostu filename, jeśli duplikaty nie są problemem.
        # Można też dodać strukturę folderów, np. na podstawie nazwy albumu.
        blob_name = f"{photo_id}-{filename}" 
        
        blob = gcs_bucket.blob(blob_name)
        
        # Sprawdź, czy plik już istnieje, aby uniknąć ponownego przesyłania (opcjonalne)
        # if blob.exists():
        #     print(f"Plik {blob_name} już istnieje w GCS. Pomijanie.")
        #     return

        print(f"Przesyłanie {filename} do GCS jako {blob_name}...")
        blob.upload_from_file(response.raw, content_type=media_item.get('mimeType'))
        print(f"Pomyślnie przesłano {blob_name} do {gcs_bucket.name}")

    except requests.exceptions.RequestException as e:
        print(f"Błąd podczas pobierania {filename}: {e}")
    except Exception as e:
        print(f"Błąd podczas przesyłania {filename} do GCS: {e}")


if __name__ == '__main__':
    # Uwierzytelnianie w Google Photos
    photos_service = authenticate_google_photos()
    if not photos_service:
        print("Nie udało się uwierzytelnić w Google Photos. Zakończono.")
        exit()

    # Wybór albumu
    selected_album = list_albums(photos_service)
    if not selected_album:
        print("Nie wybrano albumu lub wystąpił błąd. Zakończono.")
        exit()
    
    album_id = selected_album.get('id')
    album_title = selected_album.get('title', 'NieznanyAlbum')
    print(f"\nWybrano album: '{album_title}' (ID: {album_id})")

    # Pobieranie listy zdjęć z albumu
    media_items = get_media_items_from_album(photos_service, album_id)
    if not media_items:
        print(f"Brak zdjęć w albumie '{album_title}' lub wystąpił błąd.")
        exit()
        
    print(f"Znaleziono {len(media_items)} zdjęć w albumie '{album_title}'.")

    # Inicjalizacja klienta GCS
    try:
        storage_client = storage.Client() # Użyje domyślnych danych logowania gcloud lub konta usługi
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        if not bucket.exists():
            print(f"Bucket GCS '{GCS_BUCKET_NAME}' nie istnieje! Utwórz go najpierw.")
            exit()
    except Exception as e:
        print(f"Nie udało się połączyć z Google Cloud Storage: {e}")
        print("Upewnij się, że jesteś zalogowany (`gcloud auth application-default login`) lub skonfigurowałeś konto usługi.")
        exit()

    # Pobieranie i przesyłanie każdego zdjęcia
    for item in media_items:
        download_and_upload_to_gcs(item, bucket)

    print("\nZakończono importowanie zdjęć.")