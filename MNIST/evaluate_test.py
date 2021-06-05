import csv
import tensorflow as tf
import numpy as np

# Load the model and weights
print('Loading model')
with open('./models/model.json', 'rt') as file:
    model: tf.keras.Model = tf.keras.models.model_from_json(file.read())
model.load_weights('./models/model.h5')

test_X = np.load('./preprocessed/test_X.npy')

# Get probabilities of each class
probabilities = model.predict(test_X)

# Get the classes with highest probabilities.
# Note that the index is the same as the class
predictions = np.argmax(probabilities, axis=-1)

# Save the predictions
print('Writing test labels to CSV')
with open('./predictions.csv', 'wt', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['ImageId','Label'])
    for i, prediction in enumerate(predictions):
        writer.writerow([i + 1, prediction])
