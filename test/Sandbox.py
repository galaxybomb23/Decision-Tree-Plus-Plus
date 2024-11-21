import DecisionTree as dt
print("SandBox...")

eval_data = dt.EvalTrainingData(training_datafile="../test/resources/stage3cancer.csv", csv_class_column_index=2, csv_columns_for_features=[3, 4, 5, 6, 7, 8], entropy_threshold=0.01, max_depth_desired=5, symbolic_to_numeric_cardinality_threshold=10, csv_cleanup_needed=True)
eval_data.get_training_data()
eval_data.evaluate_training_data()

