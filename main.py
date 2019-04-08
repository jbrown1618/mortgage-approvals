import matplotlib
import pandas as pd

from src.cleaning import clean_data_for_exploration, clean_data_for_modeling
from src.exploration import explore_data
from src.modeling import apply_model
from src.questions import answer_questions_about_raw_data

# Only needed for OSX - apparently the default binaries that
# matplotlib uses to generate graphics on mac are not installed
# to begin with.  Comment out on Windows.
matplotlib.use('TkAgg')


def main():
    # Load up all the provided raw data
    raw_training_data = pd.read_csv('data/provided/train_values.csv')
    raw_training_labels = pd.read_csv('data/provided/train_labels.csv')
    raw_test_data = pd.read_csv('data/provided/train_values.csv')

    # Answer questions for the assignment
    answer_questions_about_raw_data(raw_training_data, raw_training_labels)

    # Clean and explore the data
    # I want the data in a more readable format for exploration
    # so that the labels on the visualizations make sense.
    exploration_data = clean_data_for_exploration(raw_training_data, raw_training_labels)
    # Change the last arg to True if you want to generate visualizations.  This takes a significant amount of time.
    explore_data(exploration_data, False)

    # Get the data into a format the machine learning model can digest
    training_data, test_data = clean_data_for_modeling(raw_training_data, raw_test_data, raw_training_labels)

    # Apply the model to the test data to generate our predictions
    submission = apply_model(training_data, test_data, raw_training_labels)
    submission.to_csv('data/generated/submission.csv')


if __name__ == '__main__':
    main()
