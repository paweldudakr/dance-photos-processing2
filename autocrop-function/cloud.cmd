#cloud functions deploy auto-crop-function \
#   --runtime python312 \
#   --trigger-resource gs://pdudaphotos-input-photos \
#   --trigger-event google.storage.object.finalize \
#   --entry-point crop_image_event \
#   --region europe-central2 \
#   --memory 512MB \
#   --timeout 300s \
#   --service-account pdudaphotos@appspot.gserviceaccount.com \
#   --set-env-vars GCP_PROJECT=pdudaphotos 
#


gcloud functions deploy process-image-crop-and-metadata --project pdudaphotos --runtime python312 --trigger-resource gs://pdudaphotos-input-photos --trigger-event google.cloud.storage.object.v1.finalized --entry-point process_image_crop_and_metadata --region europe-central2 --memory 1024MB --timeout 300s --service-account pdudaphotos@appspot.gserviceaccount.com --set-env-vars GCP_PROJECT=pdudaphotos --gen2 --source .


