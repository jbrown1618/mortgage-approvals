import os
import pandas as pd
# import seaborn as sb


def main():
    mortgage_data = import_data()
    print(mortgage_data.head())


def import_data():
    current_directory = os.path.dirname(__file__)
    path_to_data = os.path.normpath(os.path.join(current_directory, '../data/provided/train_values.csv'))

    # Read the csv file and import it into pandas
    return pd.read_csv(path_to_data, sep=';')


if __name__ == '__main__':
    main()