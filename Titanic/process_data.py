from typing import List
import numpy as np
from lib import read_csv, transpose, mean_normalise, write_csv


def _process_sex(sex: str) -> int:
    return 1 if sex.lower() == 'male' else 2


def _process_embarked(embarked: str) -> int:
    embarked = embarked.lower()
    if not embarked:
        return 0
    if embarked == 'c':
        return 1
    if embarked == 'q':
        return 2
    if embarked == 's':
        return 3
    raise Exception(f'Unexpected embarked value {embarked}')


CABINS = ['a', 'b', 'c', 'd', 'e', 'f', 't', 'g']


def _process_cabin(cabin: str) -> int:
    if not cabin:
        return 0
    cabin = cabin.lower()
    for i, c in enumerate(CABINS):
        if c in cabin:
            return i


TITLE_TO_INT = {
    'Capt': 1, 'Master': 2, 'Ms': 3, 'Miss': 4, 'Sir': 5,
    'Mr': 6, 'Mme': 7, 'Jonkheer': 8, 'the Countess': 9, 'Dr': 10,
    'Mlle': 11, 'Mrs': 12, 'Col': 13, 'Don': 14, 'Rev': 15,
    'Major': 16, 'Lady': 17, 'Dona': 18
}


def _process_name(name: str, sex: str) -> int:
    title = name.split(',')[1].split('.')[0].strip()
    # Convert titles into Mr, Miss, Mrs or Master (might not be good idea)
    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col', 'Sir']:
        title = 'Mr'
    elif title in ['Countess', 'Mme', 'Lady', 'the Countess', 'Dona']:
        title = 'Mrs'
    elif title in ['Mlle', 'Ms']:
        title = 'Miss'
    elif title == 'Dr':
        if sex.lower() == 'male':
            title = 'Mr'
        elif sex.lower() == 'female':
            title = 'Mrs'
        else:
            raise Exception(f'Unexpected sex {sex}')
    elif title not in ['Mr', 'Mrs', 'Miss', 'Master']:
        raise Exception(f'Unexpected title {title}')
    return TITLE_TO_INT[title]


def _float_else_0(value: str) -> float:
    return float(value) if value else 0


def process_csv(filepath: str, has_label_column: bool) -> (List[int], np.ndarray, np.ndarray):
    """
    Processes the CSV at the given filepath and returns (X, Y).
    Y is empty if 'has_label_column' is False
    """
    expected_column_count = 11
    if has_label_column:
        expected_column_count += 1
    headers, rows = read_csv(filepath, expected_column_count)

    columns = transpose(rows)

    # Get the passenger IDs
    passenger_ids = columns[0]

    # Get the labels
    y_list = [int(y) for y in columns[1]] if has_label_column else []

    # Get the input from the features
    X_list = []
    features_matrix = transpose(columns[2 if has_label_column else 1:])
    column_indexes = {
        'class': 0, 'name': 1, 'sex': 2, 'age': 3, 'sibling': 4, 'parch': 5,
        'ticket': 6, 'fare': 7, 'cabin': 8, 'embarked': 9
    }
    for i in range(len(features_matrix)):
        features = features_matrix[i]

        # Convert features into numbers
        x = []
        x.append(_float_else_0(features[column_indexes['class']]))
        x.append(_process_name(features[column_indexes['name']], features[column_indexes['sex']]))
        x.append(_process_sex(features[column_indexes['sex']]))
        x.append(_float_else_0(features[column_indexes['age']]))
        x.append(_float_else_0(features[column_indexes['sibling']]))
        x.append(_float_else_0(features[column_indexes['parch']]))
        x.append(_float_else_0(features[column_indexes['fare']]))
        x.append(_process_cabin(features[column_indexes['cabin']]))
        x.append(_process_embarked(features[column_indexes['embarked']]))

        # Create new features
        # age * class
        x.append(_float_else_0(features[column_indexes['age']]) * _float_else_0(features[column_indexes['class']]))

        # number of relatives
        number_of_relatives = _float_else_0(features[column_indexes['sibling']]) + _float_else_0(features[column_indexes['parch']])
        x.append(number_of_relatives)

        # fare per person
        x.append(_float_else_0(features[column_indexes['fare']]) / (number_of_relatives + 1))

        X_list.append(x)

    X = mean_normalise(np.array(X_list))
    y = np.array(y_list)
    return passenger_ids, X, y


print('Processing training data')
_, data_X, data_Y = process_csv('csv_data/train.csv', True)

# Shuffle the data
example_count = data_X.shape[0]
np.random.seed(1)
shuffled_indexes = np.random.permutation(example_count)
data_X = data_X[shuffled_indexes]
data_Y = data_Y[shuffled_indexes]

# Split data into dev and test
dev_size = 0
train_X = data_X[dev_size:]
train_Y = data_Y[dev_size:]
dev_X = data_X[:dev_size]
dev_Y = data_Y[:dev_size]

np.save('./preprocessed/train_X', train_X)
np.save('./preprocessed/train_Y', train_Y)
print(f'train_X shape: {train_X.shape}')
print(f'train_Y shape: {train_Y.shape}')

np.save('./preprocessed/dev_X', dev_X)
np.save('./preprocessed/dev_Y', dev_Y)
print(f'dev_X shape: {dev_X.shape}')
print(f'dev_Y shape: {dev_Y.shape}')

print('Processing test data')
ids, test_X, _ = process_csv('csv_data/test.csv', False)
np.save('./preprocessed/test_ids', ids)
np.save('./preprocessed/test_X', test_X)
print(f'Number of IDs: {len(ids)}')
print(f'X shape: {test_X.shape}')





