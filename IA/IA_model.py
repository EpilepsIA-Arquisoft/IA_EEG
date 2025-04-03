import Prediction_Evaluation as eva

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import mne  # Para cargar archivos .edf
import os

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TRAINING = os.path.join(BASE_PATH, 'training_data')
MODELS = os.path.join(BASE_PATH, 'models')

#< CONSTANTES >========================================================================================================x>
EXPECTED_CHANNELS = [
    'EEG Fp2-Ref', 'EEG Fp1-Ref', 'EEG F8-Ref', 'EEG F4-Ref', 'EEG Fz-Ref', 'EEG F3-Ref',
    'EEG F7-Ref', 'EEG A2-Ref', 'EEG T4-Ref', 'EEG C4-Ref', 'EEG C3-Ref', 'EEG T3-Ref',
    'EEG A1-Ref', 'EEG T6-Ref', 'EEG P4-Ref', 'EEG P3-Ref', 'EEG T5-Ref', 'EEG O2-Ref', 'EEG O1-Ref'
]
#<x====================================================================================================================x>


#< CONVERSIÓN DE DATOS >===============================================================================================x>
def edf_to_npy(edf_path, npy_output_path):
    data = mne.io.read_raw_edf(edf_path, preload=True)
    channels = data.info['ch_names']

    if not all(ch in channels for ch in EXPECTED_CHANNELS):
        missing_channels = [ch for ch in EXPECTED_CHANNELS if ch not in channels]
        raise ValueError(f"El archivo EDF no contiene todos los canales esperados. Faltan: {missing_channels}")

    # Reordenar canales a EXPECTED_CHANNELS
    data.pick_channels(EXPECTED_CHANNELS)
    raw_data = data.get_data()

    if data.info['sfreq'] != 500:
        data.resample(500)
        raw_data = data.get_data()

    total_samples = raw_data.shape[1]
    num_segments = total_samples // 500
    leftover = total_samples % 500  # Muestras que sobran

    segments = [raw_data[:, i*500:(i+1)*500] for i in range(num_segments)]

    if leftover > 0:
        print(f"Advertencia: Se encontraron {leftover} muestras adicionales que no forman un segmento completo de 500. Procesándolas por separado.")
        last_segment = np.zeros((19, 500))
        last_segment[:, :leftover] = raw_data[:, -leftover:]
        segments.append(last_segment)

    segments = np.array(segments)

    np.save(npy_output_path, segments)
    print(f"Archivo convertido y guardado como {npy_output_path}")
    return segments
#<x====================================================================================================================x>


#< DETECCIÓN DE PICOS >================================================================================================x>
def detect_peaks(data, height_threshold=0.5, distance=100):
    peaks_dict = {}

    for channel_idx in range(data.shape[0]):
        channel_data = data[channel_idx, :]
        peaks, properties = find_peaks(channel_data, height=height_threshold, distance=distance)
        peaks_dict[f"Channel_{channel_idx}"] = {
            "peaks": peaks,
            "heights": properties['peak_heights']
        }

    return peaks_dict
#<x====================================================================================================================x>


#< PREPARACIÓN DE DATOS >==============================================================================================x>
def prepare_data(x, y):
    x = np.array(x)
    y = np.array(y)
    x = x.reshape((x.shape[0], 19, 500, 1))  # Añadir canal para CNN
    y = keras.utils.to_categorical(y)
    return x, y
#<x====================================================================================================================x>


#< ENTRENAMIENTO DE MODELO >===========================================================================================x>
def train_model(x_train, y_train, x_val, y_val):
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(19, 500, 1)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(4, activation='softmax')  # Salida con 4 clases: normal, noc, elec, CPS
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))

    model.save(os.path.join(MODELS, 'modelo_ia_EpilepsIA.h5'))
    print("Modelo entrenado y guardado como 'modelo_ia_epilepsia.h5'")
    return model
#<x====================================================================================================================x>


#< MAIN >==============================================================================================================x>
def main():
    # Cargar datos
    x_train = np.load(os.path.join(TRAINING, 'x_train.npy'))
    y_train = np.load(os.path.join(TRAINING, 'y_train.npy'))
    x_test = np.load(os.path.join(TRAINING, 'x_test.npy'))
    y_test = np.load(os.path.join(TRAINING, 'y_test.npy'))

    # Preparar datos
    x_train, y_train = prepare_data(x_train, y_train)
    x_test, y_test = prepare_data(x_test, y_test)


    # Entrenar modelo
    train_model(x_train, y_train, x_test, y_test)

main()
#<x====================================================================================================================x>