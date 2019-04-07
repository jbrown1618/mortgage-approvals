from sklearn import preprocessing
from scipy import stats

from src.data_types import categorical_columns, numeric_columns
from src.defaults import categorical_defaults
from src.translations import translations

local_categorical_columns = categorical_columns.copy()
local_numeric_columns = numeric_columns.copy()


def clean_data_for_exploration(raw_data, raw_labels, id_column):
    """
    For exploration and interpretation, we would like to have data that is easier to read.
    For readability, we will replace integers for numeric columns with meaningful labels.
    """
    # Join the labels to the data
    data = raw_data.set_index(id_column).join(raw_labels.set_index(id_column))
    data = make_readable(data)
    data = coerce_data_types(data)
    data.to_csv('data/generated/readable_values.csv')
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


def clean_data_for_modeling(raw_training_data, raw_test_data):
    """
    For modeling, we have a completely different set of goals.  We want data that our chosen model can
    easily digest.  That means:

    1. Our categorical variables need to be split up into separate 0/1 variables for each category
    2. Our numeric variables need to be normalized (scaled)
    3. We have to figure out something to do with missing values
    """
    training_data = raw_training_data
    test_data = raw_test_data

    training_data, test_data = drop_bad_cols(training_data, test_data)
    training_data, test_data = replace_missing_numeric_values(training_data, test_data)
    training_data, test_data = replace_missing_categorical_values(training_data, test_data)
    training_data, test_data = normalize_numeric_columns(training_data, test_data)

    training_data['co_applicant'] = training_data['co_applicant'].map({True: 1, False: 0})
    test_data['co_applicant'] = test_data['co_applicant'].map({True: 1, False: 0})

    training_data.to_csv('data/generated/train_values_cleaned.csv')
    test_data.to_csv('data/generated/test_values_cleaned.csv')
    return training_data, test_data


def drop_bad_cols(training_data, test_data):
    bad_cols = []
    max_missing_pct = 0.1
    total = training_data.shape[0]

    for col in numeric_columns:
        non_missing = training_data[col].dropna().count()
        missing_pct = (total - non_missing) / total
        if missing_pct > max_missing_pct:
            bad_cols.append(col)
            local_numeric_columns.remove(col)

    for col in categorical_columns:
        num_missing = training_data[col][training_data[col] == -1].count()
        missing_pct = num_missing / total
        if missing_pct > max_missing_pct:
            bad_cols.append(col)
            local_categorical_columns.remove(col)

    training_data = training_data.drop(columns=bad_cols)
    test_data = test_data.drop(columns=bad_cols)

    return training_data, test_data


def replace_missing_numeric_values(training_data, test_data):
    """
    Just replace any missing numeric values with the median.  I'm sure it will be fine.  Seems legit.
    """
    for col in local_numeric_columns:
        median = training_data[col].median()
        training_data[col] = training_data[col].fillna(median)
        test_data[col] = test_data[col].fillna(median)

    return training_data, test_data


def replace_missing_categorical_values(training_data, test_data):
    """
    Some columns have an obvious value for "missing" (like "Not Provided") so we can use those.
    In other cases, just use the most common value.
    Note: for most of these, -1 counts as missing.
    """
    for col in local_categorical_columns:
        most_common = training_data[col].value_counts().index[0]
        if most_common == -1:
            most_common = training_data[col].value_counts().index[1]
        default = categorical_defaults.get(col, most_common)

        training_data[col] = training_data[col].replace(-1, default)
        test_data[col] = test_data[col].replace(-1, default)

    return training_data, test_data


def normalize_numeric_columns(training_data, test_data):
    normalish_columns = []
    other_num_columns = []

    for col in local_numeric_columns:
        n, p = stats.normaltest(training_data[col])
        if p > .05:
            normalish_columns.append(col)
        else:
            other_num_columns.append(col)

    if len(normalish_columns) > 0:
        scaler = preprocessing.StandardScaler().fit(training_data[normalish_columns])
        training_data[normalish_columns] = scaler.transform(training_data[normalish_columns])
        test_data[normalish_columns] = scaler.transform(test_data[normalish_columns])

    if len(other_num_columns) > 0:
        transformer = preprocessing.PowerTransformer(method='box-cox', standardize=True).fit(training_data[other_num_columns])
        training_data[other_num_columns] = transformer.transform(training_data[other_num_columns])
        test_data[other_num_columns] = transformer.transform(test_data[other_num_columns])

    return training_data, test_data
