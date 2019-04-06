from src.data_types import categorical_columns, numeric_columns
from src.translations import translations


def clean_data_for_exploration(raw_data, raw_labels, id_column):
    """
    For exploration and interpretation, we would like to have data that is easier to read.
    For readability, we will replace integers for numeric columns with meaningful labels.
    """
    # Join the labels to the data
    data = raw_data.set_index(id_column).join(raw_labels.set_index(id_column))
    data = make_readable(data)
    data = coerce_data_types(data)
    data.to_csv('data/generated/cleaned_values.csv')
    return data


def make_readable(raw_data):
    """
    Use the dicts defined in translations.py to replace integers for numeric columns
    with meaningful labels.  column.map takes a dict and uses it to replace any values
    in the column that match one of the keys with the value in the dict for that key.
    """
    readable_data = raw_data
    for column_name, col_translations in translations.items():
        readable_data[column_name] = raw_data[column_name].map(col_translations)

    return readable_data


def coerce_data_types(data):
    """
    There are a few categorical columns in our data that do not have nice translations
    (too many categories!).  For these, we want to make sure they are treated as categories
    instead of as numbers, so we will use str to turn 5 into "5", for example.

    Also, some of the functions we use to explore the data work better with floats for numeric
    data, so use float to turn 5 into 5.0, for example.
    """
    coerced_data = data
    for column in categorical_columns:
        coerced_data[column] = data[column].apply(str)

    for column in numeric_columns:
        coerced_data[column] = data[column].apply(float)

    return coerced_data


def clean_data_for_modeling(raw_data):
    return raw_data
