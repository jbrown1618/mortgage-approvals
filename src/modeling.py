import numpy as np
import pandas as pd
from sklearn import tree, cluster, ensemble, linear_model

from src.metadata import target_column, id_column


def apply_model(training_data, test_data, training_labels, clusterer, classifier):
    test_ids = test_data[id_column]

    training_data = training_data.drop(columns=[id_column])
    test_data = test_data.drop(columns=[id_column])

    clusterer = clusterer.fit(training_data)
    training_data = clusterer.transform(training_data)
    test_data = clusterer.transform(test_data)

    classifier = classifier.fit(training_data, training_labels[target_column].values)
    predictions = classifier.predict(test_data)

    return pd.DataFrame({
        id_column: test_ids,
        target_column: predictions
    })


def evaluate_model_from_training_data(training_data, training_labels, clusterer, classifier):
    percentage_for_training = 0.8
    true_target_column = 'true_accepted'

    rows_for_training = np.random.rand(len(training_data)) < percentage_for_training

    training_subset = training_data[rows_for_training]
    training_labels_subset = training_labels[rows_for_training]
    test_subset = training_data[~rows_for_training]
    test_labels_subset = training_labels[~rows_for_training]

    predictions = apply_model(training_subset, test_subset, training_labels_subset, clusterer, classifier)
    predictions[true_target_column] = test_labels_subset[target_column]

    correct_predictions = predictions[predictions[target_column] == predictions[true_target_column]]

    # Return the accuracy
    return len(correct_predictions) / len(predictions)


def sweep_hyperparameters(training_data, training_labels):
    sweep_decision_tree(training_data, training_labels)
    sweep_logistic_regression(training_data, training_labels)
    sweep_ada_boost(training_data, training_labels)
    sweep_random_forest(training_data, training_labels)


def sweep_decision_tree(training_data, training_labels):
    rec_n_clusters = []
    rec_max_depth = []
    rec_min_samples_leaf = []
    rec_accuracies = []

    n_clusters_values = [18, 23, 25, 28]
    max_depth_values = [6, 10, 15]
    min_samples_leaf_values = [10, 50, 100]
    for n_clusters in n_clusters_values:
        for max_depth in max_depth_values:
            for min_samples_leaf in min_samples_leaf_values:
                clusterer = cluster.FeatureAgglomeration(n_clusters=n_clusters)
                classifier = tree.DecisionTreeClassifier(max_depth=max_depth,
                                                         min_samples_leaf=min_samples_leaf)
                accuracy = evaluate_model_from_training_data(training_data, training_labels, clusterer, classifier)

                print('DecisionTree', n_clusters, max_depth, min_samples_leaf, accuracy * 100)

                rec_n_clusters.append(n_clusters)
                rec_max_depth.append(max_depth)
                rec_min_samples_leaf.append(min_samples_leaf)
                rec_accuracies.append(accuracy)
    rec = pd.DataFrame({
        'n_clusters': rec_n_clusters,
        'max_depth': rec_max_depth,
        'min_samples_leaf': rec_min_samples_leaf,
        'accuracy': rec_accuracies
    })
    rec.to_csv('data/generated/sweeping_DecisionTree.csv')


def sweep_logistic_regression(training_data, training_labels):
    rec_n_clusters = []
    rec_c = []
    rec_regularizer = []
    rec_accuracies = []

    n_clusters_values = [18, 23, 25, 28]
    c_values = [0.1, 1, 10, 100, 1000]
    regularizer_values = ['l2']
    for n_clusters in n_clusters_values:
        for regularizer in regularizer_values:
            for c in c_values:
                clusterer = cluster.FeatureAgglomeration(n_clusters=n_clusters)
                classifier = linear_model.LogisticRegression(solver='sag',
                                                             C=c,
                                                             penalty=regularizer)
                accuracy = evaluate_model_from_training_data(training_data, training_labels, clusterer, classifier)

                print('LogisticRegression', n_clusters, c, regularizer, accuracy * 100)

                rec_n_clusters.append(n_clusters)
                rec_c.append(c)
                rec_regularizer.append(regularizer)
                rec_accuracies.append(accuracy)
    rec = pd.DataFrame({
        'n_clusters': rec_n_clusters,
        'c': rec_c,
        'penalty': rec_regularizer,
        'accuracy': rec_accuracies
    })
    rec.to_csv('data/generated/sweeping_LogisticRegression.csv')


def sweep_random_forest(training_data, training_labels):
    rec_n_clusters = []
    rec_n_estimators = []
    rec_max_depth = []
    rec_min_samples_split = []
    rec_accuracies = []

    n_clusters_values = [18, 23, 25, 28]
    n_estimators_values = [10, 50, 100]
    max_depth_values = [None, 3, 8, 16]
    min_samples_split_values = [.001, .01, .1]
    for n_clusters in n_clusters_values:
        for n_estimators in n_estimators_values:
            for max_depth in max_depth_values:
                for min_samples_split in min_samples_split_values:
                    clusterer = cluster.FeatureAgglomeration(n_clusters=n_clusters)
                    classifier = ensemble.RandomForestClassifier(n_jobs=-1,
                                                                 n_estimators=n_estimators,
                                                                 max_depth=max_depth,
                                                                 min_samples_split=min_samples_split)
                    accuracy = evaluate_model_from_training_data(training_data, training_labels, clusterer, classifier)

                    print('RandomForest', n_clusters, n_estimators, max_depth, min_samples_split, accuracy * 100)

                    rec_n_clusters.append(n_clusters)
                    rec_n_estimators.append(n_estimators)
                    rec_max_depth.append(max_depth)
                    rec_min_samples_split.append(min_samples_split)
                    rec_accuracies.append(accuracy)
    rec = pd.DataFrame({
        'n_clusters': rec_n_clusters,
        'n_estimators': rec_n_estimators,
        'max_depth': rec_max_depth,
        'min_samples_split': rec_min_samples_split,
        'accuracy': rec_accuracies
    })
    rec.to_csv('data/generated/sweeping_RandomForest.csv')


def sweep_ada_boost(training_data, training_labels):
    rec_n_clusters = []
    rec_n_estimators = []
    rec_learning_rate = []
    rec_base_max_depth = []
    rec_accuracies = []

    n_clusters_values = [18, 23, 25, 28]
    n_estimators_values = [10, 50, 100]
    learning_rate_values = [0.3, 0.5, 0.8, 1.0]
    base_max_depth_values = [1, 2, 3]
    for n_clusters in n_clusters_values:
        for n_estimators in n_estimators_values:
            for learning_rate in learning_rate_values:
                for base_max_depth in base_max_depth_values:
                    clusterer = cluster.FeatureAgglomeration(n_clusters=n_clusters)
                    base_classifier = tree.DecisionTreeClassifier(max_depth=base_max_depth)
                    classifier = ensemble.AdaBoostClassifier(base_estimator=base_classifier,
                                                             n_estimators=n_estimators,
                                                             learning_rate=learning_rate)
                    accuracy = evaluate_model_from_training_data(training_data, training_labels, clusterer, classifier)

                    print('AdaBoost', n_clusters, n_estimators, learning_rate, base_max_depth, accuracy * 100)

                    rec_n_clusters.append(n_clusters)
                    rec_n_estimators.append(n_estimators)
                    rec_learning_rate.append(learning_rate)
                    rec_base_max_depth.append(base_max_depth)
                    rec_accuracies.append(accuracy)
    rec = pd.DataFrame({
        'n_clusters': rec_n_clusters,
        'n_estimators': rec_n_estimators,
        'learning_rate': rec_learning_rate,
        'base_max_depth': rec_base_max_depth,
        'accuracy': rec_accuracies
    })
    rec.to_csv('data/generated/sweeping_AdaBoost.csv')
