import DecisionTree as dt
import DecisionTreePP as dtp

import matplotlib.pyplot as plt
import pandas as pd
import time
# change cwd to the directory of the file
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
kwargs = {
    "training_datafile": "../test/resources/training_symbolic.csv",
    "csv_class_column_index": 1,
    "csv_columns_for_features": [2, 3, 4, 5],
    "max_depth_desired": 5,
    "entropy_threshold": 0.1
}
kwargsLarge = {
    "training_datafile": "../test/resources/training_symbolic_large1.csv",
    "csv_class_column_index": 1,
    "csv_columns_for_features": [2, 3, 4, 5],
    "max_depth_desired": 5,
    "entropy_threshold": 0.1
}
kwargsEval = {
    "training_datafile": "../test/resources/stage3cancer.csv",
    "csv_class_column_index": 2,
    "csv_columns_for_features": [3, 4, 5, 6, 7, 8],
    "max_depth_desired": 5,
    "entropy_threshold": 0.01,
    "symbolic_to_numeric_cardinality_threshold": 10,
    "csv_cleanup_needed": 1
}


def construct():
    dtp.constructDemo()
    input("Press any key to continue...")
    return


def interactiveClassification():
    dtp.interactiveClassificationDemo()
    input("Press any key to continue...")

    return


def interractiveIntrospection():
    dtp.interactiveIntrospectionDemo()
    input("Press any key to continue...")

    return


def benchmarkSmall():
    startTime = time.time()
    dtSmall = dt.DecisionTree(training_datafile=kwargs["training_datafile"],
                              csv_class_column_index=kwargs["csv_class_column_index"],
                              csv_columns_for_features=kwargs["csv_columns_for_features"],
                              max_depth_desired=kwargs["max_depth_desired"],
                              entropy_threshold=kwargs["entropy_threshold"])
    dtSmall.get_training_data()
    dtSmall.calculate_class_priors()
    dtSmall.calculate_first_order_probabilities()
    root_node = dtSmall.construct_decision_tree_classifier()
    return time.time() - startTime


def benchmarkLarge():
    startTime = time.time()
    dtLarge = dt.DecisionTree(training_datafile=kwargsLarge["training_datafile"],
                              csv_class_column_index=kwargsLarge["csv_class_column_index"],
                              csv_columns_for_features=kwargsLarge["csv_columns_for_features"],
                              max_depth_desired=kwargsLarge["max_depth_desired"],
                              entropy_threshold=kwargsLarge["entropy_threshold"])
    dtLarge.get_training_data()
    dtLarge.calculate_class_priors()
    dtLarge.calculate_first_order_probabilities()
    root_node = dtLarge.construct_decision_tree_classifier()
    return time.time() - startTime


def benchmarkClasifySmall():
    startTime = time.time()
    dtSmall = dt.DecisionTree(training_datafile=kwargs["training_datafile"],
                              csv_class_column_index=kwargs["csv_class_column_index"],
                              csv_columns_for_features=kwargs["csv_columns_for_features"],
                              max_depth_desired=kwargs["max_depth_desired"],
                              entropy_threshold=kwargs["entropy_threshold"])
    dtSmall.get_training_data()
    dtSmall.calculate_class_priors()
    dtSmall.calculate_first_order_probabilities()
    root_node = dtSmall.construct_decision_tree_classifier()
    dtSmall.classify(root_node, [
                     "exercising=never", "smoking=heavy", "fatIntake=heavy", "videoAddiction=heavy"])
    return time.time() - startTime


def benchmarkClassifyLarge():
    startTime = time.time()
    dtLarge = dt.DecisionTree(training_datafile=kwargsLarge["training_datafile"],
                              csv_class_column_index=kwargsLarge["csv_class_column_index"],
                              csv_columns_for_features=kwargsLarge["csv_columns_for_features"],
                              max_depth_desired=kwargsLarge["max_depth_desired"],
                              entropy_threshold=kwargsLarge["entropy_threshold"])
    dtLarge.get_training_data()
    dtLarge.calculate_class_priors()
    dtLarge.calculate_first_order_probabilities()
    root_node = dtLarge.construct_decision_tree_classifier()
    dtLarge.classify(root_node, [
                     "exercising=never", "smoking=heavy", "fatIntake=heavy", "videoAddiction=heavy"])
    return time.time() - startTime


def benchmark():
    benchmarkSmallTime_python, benchmarkLargeTime_python, benchmarkSmallConstructTime_python, benchmarkLargeConstructTime_python = -10, -1, -1, -1
    benchmarkSmallTime_python = benchmarkSmall()
    benchmarkLargeTime_python = benchmarkLarge()
    benchmarkSmallConstructTime_python = benchmarkClasifySmall()
    benchmarkLargeConstructTime_python = benchmarkClassifyLarge()

    # CPP implementation
    benchmarkSmallTime_cpp, benchmarkLargeTime_cpp, benchmarkSmallConstructTime_cpp, benchmarkLargeConstructTime_cpp = -1, -10, -1, -1
    benchmarkSmallTime_cpp = dtp.BenchmarkConstructSmall()
    benchmarkLargeTime_cpp = dtp.BenchmarkConstructLarge()
    benchmarkSmallConstructTime_cpp = dtp.BenchmarkClassifySmall()
    benchmarkLargeConstructTime_cpp = dtp.BenchmarkClassifyLarge()

    # print results
    os.system("clear")
    print("Test Cases: Small ~ 150 samples, Large ~ 5000 samples")
    print("Benchmark results")
    print("Test      | Case  | Python |   C++  | Speedup")
    print("------------------------------------------------")
    print("Construct | Small | {:6.3f} | {:6.3f} | {:6.1f}x".format(
        benchmarkSmallConstructTime_python,
        benchmarkSmallConstructTime_cpp,
        benchmarkSmallConstructTime_python /
        benchmarkSmallConstructTime_cpp if benchmarkSmallConstructTime_cpp != 0 else float(
            'inf')
    ))
    print("Construct | Large | {:6.3f} | {:6.3f} | {:6.1f}x".format(
        benchmarkLargeConstructTime_python,
        benchmarkLargeConstructTime_cpp,
        benchmarkLargeConstructTime_python /
        benchmarkLargeConstructTime_cpp if benchmarkLargeConstructTime_cpp != 0 else float(
            'inf')
    ))
    print("Classify  | Small | {:6.3f} | {:6.3f} | {:6.1f}x".format(
        benchmarkSmallTime_python,
        benchmarkSmallTime_cpp,
        benchmarkSmallTime_python /
        benchmarkSmallTime_cpp if benchmarkSmallTime_cpp != 0 else float(
            'inf')
    ))
    print("Classify  | Large | {:6.3f} | {:6.3f} | {:6.1f}x".format(
        benchmarkLargeTime_python,
        benchmarkLargeTime_cpp,
        benchmarkLargeTime_python /
        benchmarkLargeTime_cpp if benchmarkLargeTime_cpp != 0 else float(
            'inf')
    ))

    input("Press any key to continue...")
    return


def evalTrainingData():
    # evalPy = dt.EvalTrainingData(training_datafile=kwargsEval["training_datafile"],
    #                              csv_class_column_index=kwargsEval["csv_class_column_index"],
    #                              csv_columns_for_features=kwargsEval["csv_columns_for_features"],
    #                              max_depth_desired=kwargsEval["max_depth_desired"],
    #                              entropy_threshold=kwargsEval["entropy_threshold"],
    #                              symbolic_to_numeric_cardinality_threshold=kwargsEval[
    #                                  "symbolic_to_numeric_cardinality_threshold"],
    #                              csv_cleanup_needed=kwargsEval["csv_cleanup_needed"])

    # evalPy.get_training_data()
    # evalPy.evaluate_training_data()
    # input("Press any key to continue...")
    dtp.evalTrainingDataDemo()
    input("Press any key to continue...")
    return


# interactive demo
def display_menu():
    os.system("clear")
    print("1. Construct Decision Tree")
    print("2. Interactive Classification")
    print("3. Interactive Introspection")
    print("4. Benchmark")
    print("5. Evaluate Training Data")
    print("6. Exit")


def main():

    while True:
        display_menu()

        choice = input("Enter your choice: ")

        if choice == '1':
            construct()
        elif choice == '2':
            interactiveClassification()
        elif choice == '3':
            interractiveIntrospection()
        elif choice == '4':
            benchmark()
        elif choice == '5':
            evalTrainingData()
        elif choice == '6':
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
