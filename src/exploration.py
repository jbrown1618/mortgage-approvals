import seaborn as sns

from src.data_types import numeric_columns


def explore_data(data, target_column):
    print(data.head())
    print(data.describe())
    print(data.dtypes)

    sample = data.sample(5000)

    visualize_categorical_variables(sample, target_column)
    visualize_correlations(sample, target_column)


def visualize_categorical_variables(data, target_column):
    pass


def visualize_correlations(data, target_column):
    pair_plot = sns.pairplot(data, hue=target_column, vars=numeric_columns)
    pair_plot.savefig('../images/generated/scatter-plot-matrix.png')
