import DecisionTree as dt

dtree = dt.DecisionTree( training_datafile = "test/resources/training_symbolic.csv",  
                        csv_class_column_index = 1,
                        csv_columns_for_features = [2,3,4,5],
                        max_depth_desired = 5,
                        entropy_threshold = 0.1,
                     )

dtree.get_training_data()

# ----- ProbOfFeatureValue -----
# print(dtree.probability_of_feature_value( 'smoking', 'never' ))

# ----- ProbOfFeatureValueGivenClass -----
# print(dtree.probability_of_feature_value_given_class( 'smoking', 'never', 'class=benign' ))
# print(dtree.probability_of_feature_value_given_class( 'smoking', 'never', 'class=malignant' ))
# print(dtree.probability_of_feature_value_given_class( 'smoking', 'light', 'class=benign' ))
# print(dtree.probability_of_feature_value_given_class( 'smoking', 'light', 'class=malignant' ))
# print(dtree.probability_of_feature_value_given_class( 'smoking', 'medium', 'class=benign' ))
# print(dtree.probability_of_feature_value_given_class( 'smoking', 'medium', 'class=malignant' ))
# print(dtree.probability_of_feature_value_given_class( 'smoking', 'heavy', 'class=benign' ))
# print(dtree.probability_of_feature_value_given_class( 'smoking', 'heavy', 'class=malignant' ))

# dtree.determine_data_condition()

# root_node = dtree.construct_decision_tree_classifier()

# test_sample = ['exercising=never', 'smoking=heavy', 'fatIntake=heavy', 'videoAddiction=heavy']
# classification = dtree.classify(root_node, test_sample)
# print("Classification: " + str(classification))


# NUMERIC
dtreeN = dt.DecisionTree( training_datafile = "test/resources/stage3cancer.csv",
                        csv_class_column_index = 2,
                        csv_columns_for_features = [3,4,5,6,7,8],
                        max_depth_desired = 8,
                        entropy_threshold = 0.01,
                     )

dtreeN.get_training_data()
# print(dtreeN._training_data_dict)

# ----- ProbOfFeatureValueLessThanThreshold -----
# print(dtreeN.probability_of_feature_less_than_threshold( '"age"', '47' ))
# print(dtreeN.probability_of_feature_less_than_threshold( '"age"', '50' ))
# print(dtreeN.probability_of_feature_less_than_threshold( '"age"', '100' ))

# ----- ProbOfFeatureValueLessThanThresholdGivenClass -----
print(dtreeN.probability_of_feature_less_than_threshold_given_class( '"age"', '47', '"pgstat"=1' ))

# dtreeN.calculate_first_order_probabilities()
# dtreeN.calculate_class_priors()

# root_nodeN = dtreeN.construct_decision_tree_classifier()

# test_sampleN = ["pgtime=6.1","pgstat=1","age=70","eet=1","g2=11.7","grade=3","gleason=8","ploidy=diplooid"]
# classificationN = dtreeN.classify(root_nodeN, test_sampleN)
# print("Classification: " + str(classificationN))