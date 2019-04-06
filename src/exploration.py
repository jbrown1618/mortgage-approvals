import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

from src.data_types import numeric_columns, categorical_columns

p_value_threshold = 0.005


def explore_data(data, target_column, create_visuals):
    data.describe().to_csv('data/generated/summary.csv')

    sample = data.sample(5000)

    if create_visuals:
        visualize_categorical_variables(data, target_column)
        visualize_numeric_variables(data, target_column)
        visualize_correlations(sample, target_column)
        visualize_categorical_independence(sample)
        visualize_categorical_numeric_independence(sample)


def visualize_categorical_variables(data, target_column):
    for col in categorical_columns:
        number_of_categories = data[col].nunique()
        if number_of_categories > 6:
            continue

        countplot_data = data.dropna(subset=[col, target_column])
        sns.countplot(data=countplot_data, x=col, hue=target_column)
        plt.savefig('images/generated/features/' + col + '.png')
        plt.close('all')


def visualize_numeric_variables(data, target_column):
    for col in numeric_columns:
        grid = sns.FacetGrid(data, col=target_column)
        grid = grid.map(sns.distplot, col)
        grid.savefig('images/generated/features/' + col + '.png')
        plt.close('all')


def visualize_correlations(data, target_column):
    pairplot_data = data.dropna(subset=numeric_columns)
    pair_plot = sns.pairplot(data=pairplot_data, hue=target_column, vars=numeric_columns)
    pair_plot.savefig('images/generated/scatters/scatter-matrix.png')
    plt.close('all')

    done = []
    for col1 in numeric_columns:
        for col2 in numeric_columns:
            if col1 == col2:
                continue

            if col2 in done:
                continue

            if not has_numeric_relationship(data, col1, col2):
                continue

            sns.scatterplot(data=data, x=col1, y=col2, hue=target_column)
            plt.savefig('images/generated/scatters/' + col1 + '-VS-' + col2 + '.png')
            plt.close('all')

        done.append(col1)


def has_numeric_relationship(data, col1, col2):
    lin_reg_data = data.dropna(subset=[col1, col2])
    r, p = stats.pearsonr(lin_reg_data[col1], lin_reg_data[col2])
    return p < p_value_threshold


def visualize_categorical_independence(data):
    done = []
    for col1 in categorical_columns:
        for col2 in categorical_columns:
            if col1 == col2 or data[col1].nunique() > 6 or data[col2].nunique() > 6:
                continue

            if col2 in done:
                continue

            if not has_categorical_relationship(data, col1, col2):
                continue

            grid = sns.FacetGrid(data, col=col2)
            grid = grid.map(sns.countplot, col1)
            grid.savefig('images/generated/bars/' + col1 + '-VS-' + col2 + '.png')
            plt.close('all')

        done.append(col1)


def has_categorical_relationship(data, col1, col2):
    chi2_data = data.dropna(subset=[col1, col2])
    crosstab = pd.crosstab(chi2_data[col1], chi2_data[col2])
    chi2, p, dof, exp = stats.chi2_contingency(crosstab)
    return p < p_value_threshold


def visualize_categorical_numeric_independence(data):
    for cat_col in categorical_columns:
        for num_col in numeric_columns:
            if data[cat_col].nunique() > 6:
                continue

            if not has_numeric_categorical_relationship(data, num_col, cat_col):
                continue

            catplot_data = data.dropna(subset=[cat_col, num_col])
            catplot = sns.catplot(data=catplot_data, x=cat_col, y=num_col, kind='violin')
            catplot.savefig('images/generated/violins/' + cat_col + '-VS-' + num_col + '.png')
            plt.close('all')


def has_numeric_categorical_relationship(data, num_col, cat_col):
    groups = []

    for value in data[cat_col].unique():
        group = data[data[cat_col] == value][num_col]
        groups.append(group)

    f, p = stats.f_oneway(*groups)
    return p < p_value_threshold
