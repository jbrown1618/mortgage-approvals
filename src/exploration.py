import seaborn as sns
import matplotlib.pyplot as plt

from src.data_types import numeric_columns, categorical_columns


def explore_data(data, target_column, create_visuals):
    data.describe().to_csv('data/generated/summary.csv')

    sample = data.sample(5000)

    if create_visuals:
        visualize_categorical_variables(sample, target_column)
        visualize_numeric_variables(sample, target_column)
        visualize_correlations(sample, target_column)
        visualize_categorical_independence(sample)
        visualize_categorical_numeric_independence(sample)


def visualize_categorical_variables(data, target_column):
    for col in categorical_columns:
        number_of_categories = data[col].nunique()
        if number_of_categories > 6:
            continue
        sns.countplot(x=data[col], hue=data[target_column])
        plt.savefig('images/generated/' + col + '.png')
        plt.close('all')


def visualize_numeric_variables(data, target_column):
    for col in numeric_columns:
        grid = sns.FacetGrid(data, col=target_column)
        grid = grid.map(sns.distplot, col)
        grid.savefig('images/generated/' + col + '.png')
        plt.close('all')


def visualize_correlations(data, target_column):
    pair_plot = sns.pairplot(data, hue=target_column, vars=numeric_columns)
    pair_plot.savefig('images/generated/correlations.png')
    plt.close('all')


def visualize_categorical_independence(data):
    for col1 in categorical_columns:
        for col2 in categorical_columns:
            if col1 == col2 or data[col1].nunique() > 6 or data[col2].nunique() > 6:
                continue

            grid = sns.FacetGrid(data, col=col2)
            grid = grid.map(sns.countplot, col1)
            grid.savefig('images/generated/' + col1 + '-VS-' + col2 + '.png')
            plt.close('all')


def visualize_categorical_numeric_independence(data):
    for cat_col in categorical_columns:
        for num_col in numeric_columns:
            if data[cat_col].nunique() > 6:
                continue

            catplot = sns.catplot(data=data, x=cat_col, y=num_col, kind='violin')
            catplot.savefig('images/generated/' + cat_col + '-VS-' + num_col + '.png')
            plt.close('all')
