import numpy as np

#< PREDICCIÓN >========================================================================================================x>
def predict_with_model(model, new_data):
    if len(new_data.shape) == 3:  # Si falta la dimensión del canal, la agregamos
        new_data = new_data.reshape((new_data.shape[0], 19, 500, 1))

    predictions = model.predict(new_data)
    predicted_labels = np.argmax(predictions, axis=1)

    return predicted_labels
#<x====================================================================================================================x>