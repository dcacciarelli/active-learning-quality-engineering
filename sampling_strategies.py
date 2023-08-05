import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import balanced_accuracy_score


def random_sampling(data, initial_size=50, reps=5, iters=10, num_instances_to_label=10):
    """
    Function to perform random sampling.
    :param data: expecting a pandas DataFrame, where the output is in a column named "y".
    :param initial_size: the number of labeled examples to be used to initialize the regularized logistic regression model.
    :param reps: number of replications (seed is changing at every replication).
    :param iters: number of sampling steps.
    :param num_instances_to_label: number of images to be sampled and labeled at each step.
    :return: a list of lists with the balanced accuracy scores obtained at each sampling step, at every replication.
    """

    scores_rep = []
    for rep in range(reps):
        np.random.seed(rep)
        # Generating train, test and unlabeled pool
        one_example_per_class = data.groupby("y").apply(lambda x: x.sample(1))
        selected_indices = set([row_number for _, row_number in one_example_per_class.index])
        remaining_examples = data.drop(selected_indices)
        unlabeled_data, train_data = train_test_split(remaining_examples, test_size=initial_size, random_state=rep)
        train_data = pd.concat([train_data, one_example_per_class])
        unlabeled_data, test_data = train_test_split(unlabeled_data, test_size=.2, random_state=rep)

        # Active learning routine
        model = LogisticRegressionCV(cv=3, penalty="l2", solver="lbfgs", max_iter=5000, random_state=rep)

        # Train the initial model on the labeled data
        train_X = train_data.drop(["y"], axis=1)
        train_y = train_data["y"]
        model.fit(train_X, train_y)

        test_X = test_data.drop(["y"], axis=1)
        test_y = test_data["y"]

        # Loop until a stopping criterion is met
        scores = []
        for iter in range(iters):

            # Select the instances with the highest uncertainty
            most_uncertain_instances = unlabeled_data.sample(n=num_instances_to_label, replace=False, random_state=rep)
            unlabeled_data = unlabeled_data.drop(most_uncertain_instances.index, axis=0)

            # Label the selected instances and add them to the labeled data
            train_data = pd.concat([train_data, most_uncertain_instances])

            # Retrain the model on the expanded labeled data
            train_X = train_data.drop(["y"], axis=1)
            train_y = train_data["y"]
            model.fit(train_X, train_y)

            # Predicting
            y_pred = model.predict(test_X)
            scores.append(balanced_accuracy_score(test_y, y_pred))

        scores_rep.append(scores)
        print("Random sampling, iteration: ", rep)

    return scores_rep


def full_model(data, initial_size=50, reps=5):
    """
    Function to show the performance that can be obtained with the full model
    :param data: expecting a pandas DataFrame, where the output is in a column named "y".
    :param initial_size: the number of labeled examples to be used to initialize the regularized logistic regression model.
    :param reps: number of replications (seed is changing at every replication).
    :param iters: number of sampling steps.
    :param num_instances_to_label: number of images to be sampled and labeled at each step.
    :return: a list of lists with the balanced accuracy scores obtained at each sampling step, at every replication.
    """
    scores_rep = []
    for rep in range(reps):
        np.random.seed(rep)
        # Generating train, test and unlabeled pool
        one_example_per_class = data.groupby("y").apply(lambda x: x.sample(1))
        selected_indices = set([row_number for _, row_number in one_example_per_class.index])
        remaining_examples = data.drop(selected_indices)
        unlabeled_data, train_data = train_test_split(remaining_examples, test_size=initial_size, random_state=rep)
        train_data = pd.concat([train_data, one_example_per_class])
        unlabeled_data, test_data = train_test_split(unlabeled_data, test_size=.2, random_state=rep)
        training_data = pd.concat([train_data, unlabeled_data])

        # Active learning routine
        model = LogisticRegressionCV(cv=3, penalty="l2", solver="lbfgs", max_iter=5000, random_state=rep)

        # Train the initial model on the labeled data
        train_X = training_data.drop(["y"], axis=1)
        train_y = training_data["y"]
        model.fit(train_X, train_y)
        test_X = test_data.drop(["y"], axis=1)
        test_y = test_data["y"]
        model.fit(train_X, train_y)
        y_pred = model.predict(test_X)
        scores_rep.append(balanced_accuracy_score(test_y, y_pred))
        print("Full model, iteration: ", rep)

    return scores_rep


def margin_sampling(data, initial_size=50, reps=5, iters=10, num_instances_to_label=10):
    """
    Function to perform margin-based active learning.
    :param data: expecting a pandas DataFrame, where the output is in a column named "y".
    :param initial_size: the number of labeled examples to be used to initialize the regularized logistic regression model.
    :param reps: number of replications (seed is changing at every replication).
    :param iters: number of sampling steps.
    :param num_instances_to_label: number of images to be sampled and labeled at each step.
    :return: a list of lists with the balanced accuracy scores obtained at each sampling step, at every replication.
    """
    scores_rep = []
    for rep in range(reps):
        np.random.seed(rep)
        # Generating train, test and unlabeled pool
        one_example_per_class = data.groupby("y").apply(lambda x: x.sample(1))
        selected_indices = set([row_number for _, row_number in one_example_per_class.index])
        remaining_examples = data.drop(selected_indices)
        unlabeled_data, train_data = train_test_split(remaining_examples, test_size=initial_size, random_state=rep)
        train_data = pd.concat([train_data, one_example_per_class])
        unlabeled_data, test_data = train_test_split(unlabeled_data, test_size=.2, random_state=rep)

        # Active learning routine
        model = LogisticRegressionCV(cv=3, penalty="l2", solver="lbfgs", max_iter=5000, random_state=rep)

        # Train the initial model on the labeled data
        train_X = train_data.drop(["y"], axis=1)
        train_y = train_data["y"]
        model.fit(train_X, train_y)

        test_X = test_data.drop(["y"], axis=1)
        test_y = test_data["y"]

        # Loop until a stopping criterion is met
        scores = []
        for iter in range(iters):
            # Predict all the unlabeled pool
            unlabeled_X = unlabeled_data.drop(["y"], axis=1)
            unlabeled_probs = model.predict_proba(unlabeled_X)

            # Calculate the margin of each instance
            margin = np.abs(unlabeled_probs[:, 0] - unlabeled_probs[:, 1])

            # Select the instances with the smallest margin
            smallest_margin_indices = margin.argsort()[:num_instances_to_label]
            smallest_margin_instances = unlabeled_data.iloc[smallest_margin_indices]
            unlabeled_data = unlabeled_data.drop(unlabeled_data.index[smallest_margin_indices], axis=0)

            # Label the selected instances and add them to the labeled data
            train_data = pd.concat([train_data, smallest_margin_instances])

            # Retrain the model on the expanded labeled data
            train_X = train_data.drop(["y"], axis=1)
            train_y = train_data["y"]
            model.fit(train_X, train_y)

            # Predicting
            y_pred = model.predict(test_X)
            scores.append(balanced_accuracy_score(test_y, y_pred))

        scores_rep.append(scores)
        print("Uncertainty sampling, iteration: ", rep)

    return scores_rep
