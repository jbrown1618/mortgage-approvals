import seaborn as sns
import matplotlib.pyplot as plt

from src.data_types import numeric_columns, categorical_columns


def explore_data(data, target_column):
    data.describe().to_csv('data/generated/summary.csv')

    sample = data.sample(5000)

    visualize_categorical_variables(data, target_column)
    visualize_numeric_variables(data, target_column)
    visualize_correlations(sample, target_column)


def visualize_categorical_variables(data, target_column):
    for col in categorical_columns:
        number_of_categories = data[col].nunique()
        if number_of_categories > 6:
            continue
        sns.countplot(x=data[col], hue=data[target_column])
        plt.savefig('images/generated/' + col + '.png')


def visualize_numeric_variables(data, target_column):
    for col in numeric_columns:
        grid = sns.FacetGrid(data, col=target_column)
        grid = grid.map(sns.distplot, col)
        grid.savefig('images/generated/' + col + '.png')


def visualize_correlations(data, target_column):
    pair_plot = sns.pairplot(data, hue=target_column, vars=numeric_columns)
    pair_plot.savefig('images/generated/correlations.png')
