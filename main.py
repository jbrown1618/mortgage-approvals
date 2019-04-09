import matplotlib
import pandas as pd
from sklearn import ensemble, cluster

from src.cleaning import clean_data_for_exploration, clean_data_for_modeling
from src.exploration import explore_data
from src.modeling import apply_model, evaluate_model_from_training_data, sweep_hyperparameters
from src.questions import answer_questions_about_raw_data

# Only needed for OSX - apparently the default binaries that
# matplotlib uses to generate graphics on mac are not installed
# to begin with.  Comment out on Windows.
matplotlib.use('TkAgg')

# TODO:
# - sk FeatureSelection to subset features
#   - VarianceThreshold and recursive cross validation
# - Regularization - sweep the C parameter, but also check for a sk module
# - Try PCA for dimensionality reduction
#   - DimensionalityReduction.PCA -> explained_variance_ratio


def main():
    # Load up all the provided raw data
    print('Loading data...')
    raw_training_data = pd.read_csv('data/provided/train_values.csv')
    training_labels = pd.read_csv('data/provided/train_labels.csv')
    raw_test_data = pd.read_csv('data/provided/train_values.csv')

    # Answer questions for the assignment
    answer_questions_about_raw_data(raw_training_data, training_labels)

    # Clean and explore the data
    # I want the data in a more readable format for exploration
    # so that the labels on the visualizations make sense.
    print('Preparing data for exploration...')
    exploration_data = clean_data_for_exploration(raw_training_data.copy(), training_labels.copy())
    # Change the last arg to True if you want to generate visualizations.  This takes a significant amount of time.
    print('Exploring data...')
    explore_data(exploration_data.copy(), False)

    # Get the data into a format the machine learning model can digest
    print('Cleaning data...')
    training_data, test_data = clean_data_for_modeling(raw_training_data.copy(), raw_test_data.copy(), training_labels.copy())

    # Apply the model to the test data to generate our predictions
    # print('Trying different models and parameters...')
    # sweep_hyperparameters(training_data.copy(), training_labels.copy())

    # Choose a variable clusterer and a classifier
    classifier = ensemble.RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_split=.0001)
    clusterer = cluster.FeatureAgglomeration(n_clusters=29)

    print('Evaluating the model...')
    accuracy = evaluate_model_from_training_data(training_data.copy(), training_labels.copy(), clusterer, classifier)
    print('Accuracy: ', accuracy * 100)

    print('Generating the submission...')
    submission = apply_model(training_data.copy(), test_data.copy(), training_labels.copy(), clusterer, classifier)
    submission.to_csv('data/generated/submission.csv', index=False)


if __name__ == '__main__':
    main()
