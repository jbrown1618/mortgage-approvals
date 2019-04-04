from src.data_types import categorical_columns, numeric_columns
from src.translations import translations


def clean_data_for_exploration(raw_data, raw_labels, id_column):
    joined_data = raw_data.set_index(id_column).join(raw_labels.set_index(id_column))
    data = make_readable(joined_data)
    data = coerce_data_types(data)
    data = data.dropna()
    data.to_csv('data/generated/cleaned_values.csv')
    return data


def clean_data_for_modeling(raw_data):
    return raw_data


def make_readable(raw_data):
    readable_data = raw_data
    for column_name, col_translations in translations.items():
        readable_data[column_name] = raw_data[column_name].map(col_translations)

    return readable_data


def coerce_data_types(data):
    coerced_data = data
    for column in categorical_columns:
        coerced_data[column] = data[column].apply(str)

    for column in numeric_columns:
        coerced_data[column] = data[column].apply(float)

    return coerced_data
