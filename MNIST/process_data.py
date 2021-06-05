import numpy as np
import csv


def process_csv(filepath: str, has_label_column: bool) -> (np.ndarray, np.ndarray):
    X_list = []
    y_list = []
    with open(filepath, 'rt') as file:
        reader = csv.reader(file)

        # Skip the header row
        next(reader)

        for i, row in enumerate(reader):

            expected_column_count = 784
            if has_label_column:
                expected_column_count += 1

            if len(row) > expected_column_count:
                raise Exception(f'({i}) Unknown row length: {len(row)}')

            x_start_index = 0
            if has_label_column:
                # Expect label to be 0-9
                label = int(row[0])
                if label < 0 or label > 9:
                    raise Exception(f'{i} Unknown label: {label}')
                y_list.append(label)

                x_start_index = 1

            X_list.append(list(map(int, row[x_start_index:])))

        m = len(X_list)
        X = np.array(X_list).reshape((m, 28, 28, 1))
        y = np.array(y_list)
        return X, y


print('Processing training data')
train_X, train_y = process_csv('./csv/train.csv', True)
np.save('./preprocessed/train_X', train_X)
np.save('./preprocessed/train_y', train_y)
print(train_X.shape)
print(train_y.shape)


print('Processing test data')
test_X, _ = process_csv('./csv/test.csv', False)
np.save('./preprocessed/test_X', test_X)
print(test_X.shape)







