from lib import predict, write_csv
import numpy as np


def check_accuracy(input_path: str, labels_path: str):
    train_Y = np.load(labels_path)
    probabilities = predict('./models/model.json', './models/model.h5', input_path)
    probabilities = probabilities.reshape((-1,))

    threshold = 0.5
    predictions = np.zeros(probabilities.shape)
    predictions[probabilities >= threshold] = 1

    accuracy = np.mean(train_Y[train_Y == predictions])
    print(accuracy)


probabilities = predict('./models/model.json', './models/model.h5', './preprocessed/test_X.npy')
probabilities = probabilities.reshape((-1,))

threshold = 0.5
predictions = np.zeros(probabilities.shape)
predictions[probabilities >= threshold] = 1
ids = np.load('./preprocessed/test_ids.npy')

print(ids[:10])
print(predictions[:10])
write_csv('predictions.csv', zip(ids, predictions.astype(int)))


