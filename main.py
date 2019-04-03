import os

import pandas as pd

from src.cleaning import clean_data
from src.exploration import explore_data
from src.modeling import train_model, generate_submission

target_column = 'accepted'
id_column = 'row_id'


def main():
    raw_data = import_data('data/provided/train_values.csv')
    raw_labels = import_data('data/provided/train_labels.csv')

    training_data = clean_data(raw_data, raw_labels, id_column)
    explore_data(training_data, target_column)
    model = train_model(training_data, id_column, target_column)

    raw_test_data = import_data('data/provided/train_values.csv')
    generate_submission(raw_test_data, model, id_column, target_column)


def import_data(file_name):
    current_directory = os.path.dirname(__file__)
    path_to_data = os.path.normpath(os.path.join(current_directory, file_name))

    # Read the csv file and import it into pandas
    return pd.read_csv(path_to_data)


if __name__ == '__main__':
    main()
