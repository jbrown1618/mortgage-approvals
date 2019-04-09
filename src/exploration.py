import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

from src.metadata import numeric_columns, categorical_columns, target_column

p_value_threshold = 0.005

all_categorical_columns = categorical_columns.copy()
all_categorical_columns.append('accepted')


def explore_data(data, create_visuals):
    # Save off a csv file that contains an overview of the data
    data.describe().to_csv('data/generated/summary.csv')

    save_correlation_information(data)

    # Some of these visualizations take a long time to generate with the full
    # data set, and we don't really care about exact counts.  5000 data points
    # should be plenty to see an accurate shape.
    sample = data.sample(5000)

    if create_visuals:
        visualize_categorical_variables(data)
        visualize_numeric_variables(data)
        visualize_correlations(sample)
        visualize_categorical_independence(sample)
        visualize_categorical_numeric_independence(sample)


def save_correlation_information(data):
    correlation_data = {}
    p_data = {}

    for col1 in numeric_columns:
        correlation_data[col1] = []
        p_data[col1] = []
        for col2 in numeric_columns:
            lin_reg_data = data.dropna(subset=[col1, col2])
            r, p = stats.pearsonr(lin_reg_data[col1], lin_reg_data[col2])
            correlation_data[col1].append(r**2)
            p_data[col1].append(p)

    correlation_df = pd.DataFrame(correlation_data, index=numeric_columns)
    p_df = pd.DataFrame(p_data, index=numeric_columns)

    correlation_df.to_csv('data/generated/correlations.csv')
    p_df.to_csv('data/generated/correlation_p_values.csv')


def visualize_categorical_variables(data):
    """
    For each categorical variable, generate a grouped bar chart to visualize
    how accepted and rejected loan applications are distributed differently.
    """
    for col in categorical_columns:
        number_of_categories = data[col].nunique()
        if number_of_categories > 6:
            continue  # skip columns with lots of different values because the bar chart will look like garbage anyway

        countplot_data = data.dropna(subset=[col, target_column])
        sns.countplot(data=countplot_data, x=col, hue=target_column)
        plt.savefig('images/generated/features/' + col + '.png')
        plt.close('all')


def visualize_numeric_variables(data):
    """
    For each numeric variable, generate side-by-side histograms to visualize how
    the distribution differs between accepted and rejected loan applications.
    """
    for col in numeric_columns:
        # Create a grid with side-by-side histograms of the numeric columns for accepted and rejected loans
        grid = sns.FacetGrid(data, col=target_column)
        grid = grid.map(sns.distplot, col)
        grid.savefig('images/generated/features/' + col + '.png')
        plt.close('all')


def visualize_correlations(data):
    """
    First generate a scatter plot matrix because that's pretty cool.  Then generate
    separate scatter plots for each pair of numeric columns for which there is a
    strong correlation.
    """
    pairplot_data = data.dropna(subset=numeric_columns)
    pair_plot = sns.pairplot(data=pairplot_data, hue=target_column, vars=numeric_columns)
    pair_plot.savefig('images/generated/scatters/scatter-matrix.png')
    plt.close('all')

    done = []
    for col1 in numeric_columns:
        for col2 in numeric_columns:
            if col1 == col2:
                continue  # skip comparing a column to itself

            if col2 in done:
                continue  # skip col1/col2 if we have already looked at col2/col1

            if not has_numeric_relationship(data, col1, col2):
                continue  # skip pairs with no correlation

            sns.scatterplot(data=data, x=col1, y=col2, hue=target_column)
            plt.savefig('images/generated/scatters/' + col1 + '-VS-' + col2 + '.png')
            plt.close('all')

        done.append(col1)  # keep track of columns we've already look at


def has_numeric_relationship(data, col1, col2):
    """
    Use a linear regression hypothesis test to see if there is a correlation
    between the two numeric columns.
    """
    lin_reg_data = data.dropna(subset=[col1, col2])
    r, p = stats.pearsonr(lin_reg_data[col1], lin_reg_data[col2])
    return p < p_value_threshold and r > 0.3


def visualize_categorical_independence(data):
    """
    For each pair of two categorical variables, see if a relationship exists.
    If it does, generate a plot.
    """
    done = []
    for col1 in all_categorical_columns:
        for col2 in all_categorical_columns:
            if col1 == col2 or data[col1].nunique() > 6 or data[col2].nunique() > 6:
                continue  # skip comparing a column to itself, or categories with more than 6 values

            if col2 in done:
                continue  # skip col1/col2 if we have already looked at col2/col1

            if not has_categorical_relationship(data, col1, col2):
                continue  # skip columns that are independent of one another

            grid = sns.FacetGrid(data, col=col2)
            grid = grid.map(sns.countplot, col1)
            grid.savefig('images/generated/bars/' + col1 + '-VS-' + col2 + '.png')
            plt.close('all')

        done.append(col1)  # keep track of columns we've already look at


def has_categorical_relationship(data, col1, col2):
    """
    Use a chi-square test to see if the values in the first categorical column
    are independent of the values in the second categorical column.
    """

    # Drop rows that have missing values in either of the columns
    chi2_data = data.dropna(subset=[col1, col2])

    # If I were writing a chi squared function, I would have it just accept
    # the two categorical columns, but unfortunately whoever wrote this had
    # a different opinion.  They want a table whose rows correspond to one
    # variable and whose columns correspond to the other variable.  Luckily
    # pd.crosstab gives us exactly what we need.
    crosstab = pd.crosstab(chi2_data[col1], chi2_data[col2])

    # See if the two variables are independent
    chi2, p, dof, exp = stats.chi2_contingency(crosstab)
    return p < p_value_threshold


def visualize_categorical_numeric_independence(data):
    """
    For each pair of one categorical column and one numeric column, see
    if a relationship exists.  If it does, generate a plot.
    """
    for cat_col in all_categorical_columns:
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
    """
    Use a one-way ANOVA test to see if the values in a numeric column are distributed
    differently depending on the value in the categorical column.
    """
    # Drop rows that have missing values for either column
    anova_data = data.dropna(subset=[num_col, cat_col])

    # For each unique value in the categorical column, put together a group of numerical values
    groups = []
    for value in anova_data[cat_col].unique():
        # The values in the numeric column where the categorical column has the given value
        group = anova_data[anova_data[cat_col] == value][num_col]
        groups.append(group)

    # See if the groups we made are independent
    f, p = stats.f_oneway(*groups)
    return p < p_value_threshold
