import numpy as np
import mne

#< CONSTANTES >========================================================================================================x>
"""
    Selecciona los canales esperados en los archivos EDF.
"""
EXPECTED_CHANNELS = [
    'EEG Fp2-Ref', 'EEG Fp1-Ref', 'EEG F8-Ref', 'EEG F4-Ref', 'EEG Fz-Ref', 'EEG F3-Ref',
    'EEG F7-Ref', 'EEG A2-Ref', 'EEG T4-Ref', 'EEG C4-Ref', 'EEG C3-Ref', 'EEG T3-Ref',
    'EEG A1-Ref', 'EEG T6-Ref', 'EEG P4-Ref', 'EEG P3-Ref', 'EEG T5-Ref', 'EEG O2-Ref', 'EEG O1-Ref'
]
#<=====================================================================================================================x>


#< PREPROCESAMIENTO >==================================================================================================x>
def preprocess_new_eeg(edf_path, output_path, training_data_path):
    """
    Preprocesa un archivo EDF, lo normaliza y lo convierte a formato NPY compatible con el modelo IA.
    """
    data = mne.io.read_raw_edf(edf_path, preload=True)
    channels = data.info['ch_names']

    if not all(ch in channels for ch in EXPECTED_CHANNELS):
        missing_channels = [ch for ch in EXPECTED_CHANNELS if ch not in channels]
        raise ValueError(f"El archivo EDF no contiene todos los canales esperados. Faltan: {missing_channels}")

    # Reordenar canales y obtener datos crudos
    data.pick_channels(EXPECTED_CHANNELS)
    raw_data = data.get_data()

    if data.info['sfreq'] != 500:  # Remuestrear si es necesario
        data.resample(500)
        raw_data = data.get_data()

    total_samples = raw_data.shape[1]
    num_segments = total_samples // 500
    leftover = total_samples % 500  # Muestras restantes

    segments = [raw_data[:, i*500:(i+1)*500] for i in range(num_segments)]

    if leftover > 0:
        print(f"Advertencia: Se encontraron {leftover} muestras adicionales. Procesándolas por separado.")
        last_segment = np.zeros((19, 500))
        last_segment[:, :leftover] = raw_data[:, -leftover:]
        segments.append(last_segment)

    segments = np.array(segments)

    # Normalización basada en datos de entrenamiento
    training_data = np.load(training_data_path)
    train_max_value = np.max(training_data)
    train_min_value = np.min(training_data)

    segments = (segments - np.min(segments)) / (np.max(segments) - np.min(segments))
    segments = segments * (train_max_value - train_min_value) + train_min_value

    np.save(output_path, segments)
    print(f"Archivo preprocesado y normalizado guardado como {output_path}")
    return segments
#<=======================================================================================================================x>