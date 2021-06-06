import numpy as np
from lib import read_csv, transpose


def process_csv(filepath: str, has_label_column: bool) -> (np.ndarray, np.ndarray):
    """
    Processes the CSV at the given filepath and returns (X, Y).
    Y is empty if 'has_label_column' is False
    """
    expected_column_count = 784
    if has_label_column:
        expected_column_count += 1
    headers, rows = read_csv(filepath, expected_column_count)

    columns = transpose(rows)
    y_list = [int(y) for y in columns[0]] if has_label_column else []
    X_list = [[int(x) for x in column] for column in columns[1 if has_label_column else 0:]]
    X_list = transpose(X_list)

    # Reshape and normalise
    m = len(X_list)
    X = np.array(X_list).reshape((m, 28, 28, 1))
    X = X / 255

    y = np.array(y_list)
    return X, y


print('Processing training data')
train_X, train_y = process_csv('csv_data/train.csv', True)
np.save('./preprocessed/train_X', train_X)
np.save('./preprocessed/train_y', train_y)
print(f'X shape: {train_X.shape}')
print(f'Y shape: {train_y.shape}')


print('Processing test data')
test_X, _ = process_csv('csv_data/test.csv', False)
np.save('./preprocessed/test_X', test_X)
print(f'X shape: {test_X.shape}')






