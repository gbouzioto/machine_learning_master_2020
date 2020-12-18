"""
Machine Learning Exercise 1
"""
import argparse
import os

import graphviz

import pandas as pd
from sklearn import tree, metrics
from sklearn.model_selection import GridSearchCV, train_test_split

PROBLEM_TYPE_HELP_TEXT = """
Treats the problem based on its type.Can accept 3 values: 
c: which stands for classification, 
lr: which stands for linear regression,
bc: which stands for binary classification"""

NUMBER_OF_CLASSES_HELP_TEXT = """
The number of classes for the classification problem, will be ignored if problem_type argument is not c.
Defaults to 10 classes. Notice that since there are not a lot of data in day.csv file the less classes used,
the better the prediction will be."""

PATH_TO_FILE_HELP_TEXT = """The path to the csv file that will be parsed. 
Defaults to day.csv. (assumed that it will be in the same directory as this script)"""


def _parse_user_args():
    """
    Parses the user arguments
    :returns: user arguments
    """
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-pt', '--problem_type', help=PROBLEM_TYPE_HELP_TEXT,
                            choices=["c", "lr", "bc"], required=True)
    arg_parser.add_argument('-c', '--classes', default=10, type=int, help=NUMBER_OF_CLASSES_HELP_TEXT)
    arg_parser.add_argument('-fp', '--file_path', default='day.csv', help=PATH_TO_FILE_HELP_TEXT)
    return arg_parser.parse_args()


class ExerciseBase(object):
    """Base Exercise Class"""
    DATASET_FILE_PATH = 'day.csv'

    def __init__(self, dataset_path=None):
        """
        :param dataset_path: path to the dataset file, defaults to DATASET_FILE_PATH
        """
        self.dataset_path = dataset_path if dataset_path else self.DATASET_FILE_PATH
        self.data, self.target = self._parse_data()

    def _parse_data(self):
        """
        Data inside the file:
        - instant: record index
        - dteday : date
        - season : season (1:springer, 2:summer, 3:fall, 4:winter)
        - yr : year (0: 2011, 1:2012)
        - mnth : month ( 1 to 12)
        - holiday : weather day is holiday or not (extracted from http://dchr.dc.gov/page/holiday-schedule)
        - weekday : day of the week
        - workingday : if day is neither weekend nor holiday is 1, otherwise is 0.
        + weathersit :
            - 1: Clear, Few clouds, Partly cloudy, Partly cloudy
            - 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
            - 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
            - 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
        - temp : Normalized temperature in Celsius. The values are divided to 41 (max)
        - atemp: Normalized feeling temperature in Celsius. The values are divided to 50 (max)
        - hum: Normalized humidity. The values are divided to 100 (max)
        - windspeed: Normalized wind speed. The values are divided to 67 (max)
        - casual: count of casual users
        - registered: count of registered users
        - cnt: count of total rental bikes including both casual and registered

        Reads the dataset file. Drops the columns instance, dteday, casual and registered from the DataFrame,
        since they will add noise to the dataset:
            - instance is a simple id index that does not offer any information
            - casual + registered result in cnt, so we are basically cheating
            - dteday is already in the data in the form of yr, mnth, weekday
        Separates the data into the x (data) and y (target)

        :rtype: pd.DataFrame
        """
        df = pd.read_csv(self.dataset_path)
        target = df["cnt"]
        df = df.drop(columns=['instant', 'dteday', 'casual', 'registered', "cnt"])
        return df, target

    def __str__(self):
        return f"ExerciseBase(dataset_path={self.dataset_path})"


class Exercise1A(ExerciseBase):
    """ Question 1A """
    CLASSES = 10

    def __init__(self, dataset_path=None, classes=None):
        """
        :param dataset_path: path to the dataset file, defaults to DATASET_FILE_PATH
        :param classes: the number of classes to split the cnt value into
        """
        super(Exercise1A, self).__init__(dataset_path)
        self.classes = classes if classes else self.CLASSES

        self._best_estimator = None
        self._labels = None

    def __str__(self):
        return f"Exercise1A(dataset_path={self.dataset_path})"

    def _generate_cnt_classes(self):
        """
         Because this is a classification problem, we need to generate the number of classes.
         So the target is split into classes.
        """
        max_value = round(self.target.max(), -3)  # round to the nearest 1000
        step = max_value // self.classes
        bins = [i for i in range(0, max_value + step, step)]  # create the bins up to max value
        self._labels = [f"[{bins[i]} {bins[i + 1]}]" for i in range(0, len(bins) - 1)]  # used for visualizing the tree
        self.target = pd.cut(self.target, labels=False, precision=0, bins=bins)

    def _find_best_estimator(self, data_train, target_train):
        """
         Finds the best DecisionTreeClassifier classifier using GridSearchCV
         :rtype: tree.DecisionTreeClassifier
        """

        # the classifier
        dec_tree = tree.DecisionTreeClassifier()

        param_dict = {"criterion": ['gini', 'entropy'],
                      "max_depth": range(1, 10),
                      "min_samples_split": range(2, 10),
                      "min_samples_leaf": range(1, 5)
                      }

        clf_gs = GridSearchCV(dec_tree, param_grid=param_dict, n_jobs=4)

        # Fitting the grid search
        clf_gs.fit(data_train, target_train)

        # Viewing The Best Parameters
        print('Best Parameters: ', clf_gs.best_params_)
        print('Best Accuracy Score Achieved in Grid Search: ', clf_gs.best_score_)
        self._best_estimator = clf_gs.best_estimator_

    def visualize_tree(self):
        """
        Visualizes the created tree. It requires that the tree is trained first.
        """
        dot_data = tree.export_graphviz(self._best_estimator,
                                        feature_names=self.data.columns.values.tolist(),
                                        class_names=self._labels,
                                        filled=True, rounded=True,
                                        special_characters=True)
        # Draw graph
        graph = graphviz.Source(dot_data)
        # Save to a file in the current directory
        graph.render("exercise_1_a_decision_tree_graphivz", format='png', directory=os.getcwd())

    def train(self, test_size=0.35, random_state=0):
        """
        The training phase.
        :param test_size:
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split.
        If int, represents the absolute number of test samples.
        If None, the value is set to the complement of the train size.
        If train_size is also None, it will be set to 0.25.
        :param random_state: Controls the shuffling applied to the data before applying the split.
        Pass an int for reproducible output across multiple function calls
        :return: A tuple that with the test_data and target_test
        """
        # split the data into the training set and the test set
        data_train, data_test, target_train, target_test = train_test_split(self.data, self.target,
                                                                            stratify=self.target,
                                                                            test_size=test_size,
                                                                            random_state=random_state)
        self._find_best_estimator(data_train, target_train)

        self._best_estimator.fit(data_train, target_train)
        return data_test, target_test

    def predict(self, data_test):
        """
        Makes the prediction given the data test and target test data.

        :param data_test: The test date that the prediction will be made for
        :returns: The predicted classes, or the predict values.
        """
        return self._best_estimator.predict(data_test)

    @staticmethod
    def report(target_test, target_test_predictions):
        """
        :param target_test: that test data
        :param target_test_predictions: the prediction on test data
        :return: the same as the  metrics.classification_report
        """
        accuracy = metrics.accuracy_score(target_test, target_test_predictions)
        print('Accuracy: ', accuracy)
        return metrics.classification_report(target_test, target_test_predictions)

    @classmethod
    def create(cls, dataset_path=None, classes=None):
        """
        :param dataset_path: path to the dataset file, defaults to DATASET_FILE_PATH
        :param classes: the number of classes to split the cnt value into
        """
        exercise_a = cls(dataset_path=dataset_path, classes=classes)
        exercise_a._generate_cnt_classes()
        return exercise_a


def main_ex_1a(classes, dataset_path):
    """
    Execution of Exercise 1 a
    :param classes: number of classes to split the data into
    :param dataset_path: path to the scv file that contains the dataset
    """
    ex_1a = Exercise1A.create(classes=classes, dataset_path=dataset_path)
    data_test, target_test = ex_1a.train()
    ex_1a.visualize_tree()
    target_test_predictions = ex_1a.predict(data_test)
    print(ex_1a.report(target_test, target_test_predictions))


if __name__ == '__main__':
    args = _parse_user_args()
    if args.problem_type == 'c':
        main_ex_1a(args.classes, args.file_path)
