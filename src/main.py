import os

import pandas as pd
from src.translations import translations


def main():
    mortgage_data = import_data()
    readable_data = make_readable(mortgage_data)
    print(readable_data.head())


def import_data():
    current_directory = os.path.dirname(__file__)
    path_to_data = os.path.normpath(os.path.join(current_directory, '../data/provided/train_values.csv'))

    # Read the csv file and import it into pandas
    return pd.read_csv(path_to_data)


def make_readable(raw_data):
    readable_data = raw_data
    for column_name, col_translations in translations.items():
        readable_data[column_name] = raw_data[column_name].map(col_translations)

    return readable_data


if __name__ == '__main__':
    main()
