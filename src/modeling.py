import pandas as pd
import numpy as np
from sklearn import tree, cluster, linear_model, neighbors

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
    sweeping_results = {
        'description': [],
        'accuracy': []
    }

    n_clusters_values = [13, 18, 23, 28]

    # Decision Tree
    max_depth_values = [6, 10, 15]
    min_samples_leaf_values = [10, 50, 100]
    for n_clusters in n_clusters_values:

        description = 'Decision Tree with default parameters and n_clusters={}'.format(n_clusters)
        clusterer = cluster.FeatureAgglomeration(n_clusters=n_clusters)
        classifier = tree.DecisionTreeClassifier()
        accuracy = evaluate_model_from_training_data(training_data, training_labels, clusterer, classifier)
        sweeping_results['description'].append(description)
        sweeping_results['accuracy'].append(accuracy)
        print_results(description, accuracy)

        for max_depth in max_depth_values:
            for min_samples_leaf in min_samples_leaf_values:
                description = 'Decision Tree with max_depth={}, min_samples_leaf={}, n_clusters={}'.format(max_depth, min_samples_leaf, n_clusters)
                clusterer = cluster.FeatureAgglomeration(n_clusters=n_clusters)
                classifier = tree.DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
                accuracy = evaluate_model_from_training_data(training_data, training_labels, clusterer, classifier)
                sweeping_results['description'].append(description)
                sweeping_results['accuracy'].append(accuracy)
                print_results(description, accuracy)

    # Logistic Regression
    c_values = [0.001, 0.1, 1, 10, 100, 1000]
    regularizer_values = ['l1', 'l2']
    for n_clusters in n_clusters_values:
        description = 'Logistic Regression with default parameters and n_clusters={}'.format(n_clusters)
        clusterer = cluster.FeatureAgglomeration(n_clusters=n_clusters)
        classifier = linear_model.LogisticRegression(solver='lbfgs')
        accuracy = evaluate_model_from_training_data(training_data, training_labels, clusterer, classifier)
        sweeping_results['description'].append(description)
        sweeping_results['accuracy'].append(accuracy)
        print_results(description, accuracy)

        for regularizer in regularizer_values:
            for c in c_values:
                description = 'Logistic Regression with penalty={}, C={} n_clusters={}'.format(regularizer, c, n_clusters)
                clusterer = cluster.FeatureAgglomeration(n_clusters=n_clusters)
                classifier = linear_model.LogisticRegression(solver='liblinear', C=c, penalty=regularizer)
                accuracy = evaluate_model_from_training_data(training_data, training_labels, clusterer, classifier)
                sweeping_results['description'].append(description)
                sweeping_results['accuracy'].append(accuracy)
                print_results(description, accuracy)

    # Nearest Neighbors
    algorithm_values = ['auto', 'ball_tree', 'kd_tree']
    power_values = [1, 2]
    for n_clusters in n_clusters_values[:1]:
        description = 'Nearest Neighbor with default parameters and n_clusters={}'.format(n_clusters)
        clusterer = cluster.FeatureAgglomeration(n_clusters=n_clusters)
        classifier = neighbors.KNeighborsClassifier()
        accuracy = evaluate_model_from_training_data(training_data, training_labels, clusterer, classifier)
        sweeping_results['description'].append(description)
        sweeping_results['accuracy'].append(accuracy)
        print_results(description, accuracy)

        for algorithm in algorithm_values:
            for p in power_values:
                description = 'Nearest Neighbor with algorithm={}, p={} n_clusters={}'.format(algorithm, p, n_clusters)
                clusterer = cluster.FeatureAgglomeration(n_clusters=n_clusters)
                classifier = linear_model.KNeighborsClassifier(algorithm=algorithm, p=p)
                accuracy = evaluate_model_from_training_data(training_data, training_labels, clusterer, classifier)
                sweeping_results['description'].append(description)
                sweeping_results['accuracy'].append(accuracy)
                print_results(description, accuracy)

    # Random Forest - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

    # AdaBoost - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html

    return sweeping_results


def print_results(description, accuracy):
    print('For {}, found accuracy of {:.2%}'.format(description, accuracy))
