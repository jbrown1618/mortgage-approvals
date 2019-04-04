import pandas as pd


def answer_questions_about_raw_data(raw_data, raw_labels):
    questions = []
    answers = []

    questions.append('What is the minimum loan amount?')
    answers.append(raw_data['loan_amount'].min())

    questions.append('What is the maximum loan amount?')
    answers.append(raw_data['loan_amount'].max())

    questions.append('What is the mean loan amount?')
    answers.append(raw_data['loan_amount'].mean())

    questions.append('What is the standard deviation of the loan amount?')
    answers.append(raw_data['loan_amount'].std())

    questions.append('What is the median loan amount?')
    answers.append(raw_data['loan_amount'].median())

    pd.DataFrame.from_dict({
        'Question': questions,
        'Answer': answers
    }).to_csv('data/generated/questions.csv')
