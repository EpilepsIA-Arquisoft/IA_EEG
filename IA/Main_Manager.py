import Data_Preprocessing as dp
import Prediction_Evaluation as eva

import numpy as np
import os
import requests
from flask import Flask, request, jsonify
from tensorflow import keras

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROCESSING = os.path.join(BASE_PATH, 'processed_data')
TRAINING = os.path.join(BASE_PATH, 'training_data')
MODELS = os.path.join(BASE_PATH, 'models')
ORIGIN = os.path.join(BASE_PATH, 'original_data')

app = Flask(__name__)

@app.route('/ia', methods=['POST'])
def ia_task():
    edf_file = request.files['file']
    patient_id = request.form['patient_id']
    reduce_url = 'http://10.128.0.4:5002'
    
    # Guardar el archivo temporalmente
    edf_path = os.path.join(ORIGIN, f'{patient_id}_segment.edf')
    edf_file.save(edf_path)
    
    model = keras.models.load_model(os.path.join(MODELS, 'modelo_ia_EpilepsIA.h5'))

    npy_output_path = os.path.join(PROCESSING, f'{patient_id}.npy')
    training_data_path = os.path.join(TRAINING, 'x_train.npy')
    new_data = dp.preprocess_new_eeg(edf_path, npy_output_path, training_data_path)


    predictions = eva.predict_with_model(model, new_data)
    
    epileptic_segments = np.sum(predictions > 0)
    print(f"Total de segmentos con picos epilépticos detectados: {epileptic_segments}")
    print(f"Predicciones para nuevo EEG: {predictions}")

    # Enviar resultados al Reducer
    # response = requests.post(f'{reduce_url}/reduce', json={'peaks': epileptic_segments, 'patient_id': patient_id})
    
    os.remove(edf_path)
    os.remove(npy_output_path)
    return jsonify({'message': 'IA completada'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)