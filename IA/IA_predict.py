import Data_Preprocessing as dp
import Prediction_Evaluation as eva

import numpy as np
import os
from google.cloud import storage
from tensorflow import keras

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROCESSING = os.path.join(BASE_PATH, 'processed_data')
TRAINING = os.path.join(BASE_PATH, 'training_data')
MODELS = os.path.join(BASE_PATH, 'models')
ORIGIN = os.path.join(BASE_PATH, 'original_data')

storage_client = storage.Client()
BUCKET_NAME = "nombre-de-tu-bucket"

def predict(body):
    remote_path = body['ubicacion_fragmento']
    patient_id = body['id_paciente']
    segment_id = body['num_fragmento']

    edf_path = os.path.join(ORIGIN, f'{patient_id}_segment_{segment_id}.edf')
    descargar_archivo_gcs(remote_path, edf_path)
    
    model = keras.models.load_model(os.path.join(MODELS, 'modelo_ia_EpilepsIA.h5'))

    npy_output_path = os.path.join(PROCESSING, f'{patient_id}_{segment_id}.npy')
    training_data_path = os.path.join(TRAINING, 'x_train.npy')
    new_data = dp.preprocess_new_eeg(edf_path, npy_output_path, training_data_path)

    predictions = eva.predict_with_model(model, new_data)
    
    epileptic_segments = np.sum(predictions > 0)
    print(f"Total de segmentos con picos epilépticos detectados: {epileptic_segments}")
    print(f"Predicciones para nuevo EEG: {predictions}")

    os.remove(edf_path)
    os.remove(npy_output_path)
    return {'id_paciente': patient_id, 
            'id_examen': body['id_examen'], 
            'num_fragmento': segment_id, 
            'total_fragmentos': body['total_fragmentos'],
            'num_picos': epileptic_segments}


def descargar_archivo_gcs(ruta_remota, ruta_local):
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(ruta_remota)
    blob.download_to_filename(ruta_local)
    print(f"Descargado: {ruta_remota} -> {ruta_local}")