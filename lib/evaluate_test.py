import tensorflow as tf
import numpy as np


def predict(model_json_path: str, model_weights_path: str, test_input_path: str) -> np.ndarray:
    """ Loads the model and performs the predictions using the input data. """

    # Load the model and weights
    print('Loading model')
    with open(model_json_path, 'rt') as file:
        model: tf.keras.Model = tf.keras.models.model_from_json(file.read())
    model.load_weights(model_weights_path)

    test_X = np.load(test_input_path)

    # Get probabilities of each class
    return model.predict(test_X)
