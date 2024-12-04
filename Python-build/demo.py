import DecisionTree as dt
import DecisionTreePP as dtp

import matplotlib.pyplot as plt
import pandas as pd
import time
# change cwd to the directory of the file
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def construct():
    kwargs = {
        "training_datafile": "../test/resources/training_symbolic.csv",
        "csv_class_column_index": "1",
        "csv_columns_for_features": {2, 3, 4, 5},
        "max_depth_desired": "5",
        "entropy_threshold": "0.1"
    }

    Dtree = dtp.DecisionTree(kwargs)
    Dtree.getTrainingData()
    Dtree.calculateFirstOrderProbabilities()
    Dtree.calculateClassPriors()

    rootNode = Dtree.constructDecisionTreeClassifier()
    rootNode.displayDecisionTree("  ")
    return


def interactiveClassification():
    return


def interractiveIntrospection():
    return


def benchmark():
    return


def evalTrainingData():
    return


# interactive demo
def display_menu():
    os.system("clear")


def main():
    construct()
    # interactiveClassification()
    # interractiveIntrospection()
    # benchmark()
    # evalTrainingData()
    return 0


if __name__ == "__main__":
    main()
