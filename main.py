import matplotlib
import pandas as pd

from src.cleaning import clean_data_for_exploration, clean_data_for_modeling
from src.exploration import explore_data
from src.modeling import train_model, generate_submission
from src.questions import answer_questions_about_raw_data

# Only needed for OSX - apparently the default binaries that
# matplotlib uses to generate graphics on mac are not installed
# to begin with.  Comment out on Windows.

matplotlib.use('TkAgg')


def main():
    # Define these as constants to avoid typos later
    target_column = 'accepted'
    id_column = 'row_id'

    # Load up all the provided raw data
    raw_training_data = pd.read_csv('data/provided/train_values.csv')
    raw_training_labels = pd.read_csv('data/provided/train_labels.csv')
    raw_test_data = pd.read_csv('data/provided/train_values.csv')

    # Answer questions for the assignment
    answer_questions_about_raw_data(raw_training_data, raw_training_labels)

    # Clean and explore the data
    # I want the data in a more readable format for exploration
    # so that the labels on the visualizations make sense.
    exploration_data = clean_data_for_exploration(raw_training_data, raw_training_labels, id_column)
    # Change the last arg to True if you want to generate visualizations.  This takes a significant amount of time.
    explore_data(exploration_data, target_column, False)

    # Train the model on the training data and labels
    training_data = clean_data_for_modeling(raw_training_data)
    model = train_model(training_data, raw_training_labels, id_column, target_column)

    # Apply the model to the test data to generate our predictions
    test_data = clean_data_for_modeling(raw_test_data)
    submission = generate_submission(test_data, model, id_column, target_column)
    submission.to_csv('data/generated/submission.csv')


if __name__ == '__main__':
    main()
