from typing import List, Iterable
import numpy as np
import csv


def read_csv(filepath: str, expected_column_count: int) -> (List[str], List[List[str]]):
    """
    Processes the CSV at the given filepath and returns the header names and the rows of the CSV.
    :param filepath - The path to the CSV file
    :param expected_column_count - The number of columns we expect in the CSV
    """
    rows = []
    with open(filepath, 'rt') as file:
        reader = csv.reader(file)

        headers = next(reader)
        for i, row in enumerate(reader):

            # Make sure the row has the expected column count
            if len(row) > expected_column_count:
                raise Exception(f'({i}) Unknown row length: {len(row)}')

            # Get the values
            rows.append(row)

        return headers, rows


def write_csv(filepath: str, data: Iterable[Iterable]):
    """ Writes the data to a CSV file. """
    with open(filepath, 'wt', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)


def mean_normalise(X: np.ndarray) -> np.ndarray:
    """
    Normalises every input feature by taking into account
    the mean and standard deviation of data.
    :param X - An array of dimensions (number of examples, number of features)
    """

    # Transpose so we can loop through all the features
    X = np.array(X).T

    # Perform normalisation
    for i in range(len(X)):
        row = X[i, :]
        X[i, :] = (row - np.mean(row)) / np.std(row)

    # Transpose data back into original form
    return X.T


def transpose(matrix: List[List]) -> List[List]:
    """ Transposes the given 2D list matrix. """
    return [list(x) for x in zip(*matrix)]









