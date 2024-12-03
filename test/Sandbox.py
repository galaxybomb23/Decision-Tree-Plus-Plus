import DecisionTree as dt
print("SandBox...")

# # SYMBOLIC TREE #
# dtree = dt.DecisionTree(training_datafile="test/resources/training_symbolic.csv",
#                         csv_class_column_index=1,
#                         csv_columns_for_features=[2, 3, 4, 5],
#                         max_depth_desired=5,
#                         entropy_threshold=0.1,
#                         )

# dtree.get_training_data()
# dtree.calculate_class_priors()
# dtree.calculate_first_order_probabilities()
# # dtree.determine_data_condition()

# root_node = dtree.construct_decision_tree_classifier()

# # test_sample = ['exercising=never', 'smoking=heavy', 'fatIntake=heavy', 'videoAddiction=heavy']
# # classification = dtree.classify(root_node, test_sample)
# # print("Classification: " + str(classification))

# # NUMERIC TREE #
# dtreeN = dt.DecisionTree(
#     training_datafile="test/resources/stage3cancer.csv",
#     csv_class_column_index=2,
#     csv_columns_for_features=[3, 4, 5, 6, 7, 8],
#     max_depth_desired=8,
#     entropy_threshold=0.01
#     # , debug2=True
# )

# dtreeN.get_training_data()

# dtreeN.calculate_first_order_probabilities()
# dtreeN.calculate_class_priors()

training_datafile = "test/resources/stage3cancer.csv"
eval_data = dt.EvalTrainingData( 
                               training_datafile = training_datafile,
                               csv_class_column_index = 2,
                               csv_columns_for_features = [3,4,5,6,7,8],
                               entropy_threshold = 0.01,
                               max_depth_desired = 5,
                               symbolic_to_numeric_cardinality_threshold = 10,
                               csv_cleanup_needed = 1,
            )

eval_data.get_training_data()
eval_data.evaluate_training_data()

# root_nodeN = dtreeN.construct_decision_tree_classifier()

# test_sampleN = ["pgtime=6.1","pgstat=1","age=70","eet=1","g2=11.7","grade=3","gleason=8","ploidy=diplooid"]
# classificationN = dtreeN.classify(root_nodeN, test_sampleN)
# print("Classification: " + str(classificationN))

# /***************************/ ENTROPY /***************************/
# SYMBOLIC
# ----- classEntropyOnPriors -----
# print(dtree.class_entropy_on_priors())

# ----- entropyScannerForANumericFeature -----
# dtree.entropy_scanner_for_a_numeric_feature('"exercising"')
# dtree.entropy_scanner_for_a_numeric_feature('fatIntake')
# dtree.entropy_scanner_for_a_numeric_feature('smoking')
# dtree.entropy_scanner_for_a_numeric_feature('videoAddiction')

# ----- ClassEntropyForLessThanThresholdForFeature -----
# print(dtree.class_entropy_for_less_than_threshold_for_feature([ "grade=2.0", "gleason=5.0" ], "gleason", 5.0))


# NUMERIC
# ----- classEntropyOnPriors -----
# print(dtreeN.class_entropy_on_priors())

# ----- entropyScannerForANumericFeature -----
# print(dtreeN.entropy_scanner_for_a_numeric_feature('"age"'))

# ----- ClassEntropyForLessThanThresholdForFeature -----
# print("\n\nTESTS - Less Than Threshold")
# print(dtreeN.class_entropy_for_less_than_threshold_for_feature(
#     ['"grade"=2.0', '"gleason"=5.0'], '"g2"', '46.56'))
# print(dtreeN.class_entropy_for_less_than_threshold_for_feature(
#     ['"grade"=2.0', '"gleason"=5.0', '"g2"<3.84'], '"age"', '57.0'))
# print(dtreeN.class_entropy_for_less_than_threshold_for_feature(
#     ['"grade"=2.0', '"gleason"=5.0', '"g2"<3.84'], '"g2"', '3.84'))
# print(dtreeN.class_entropy_for_less_than_threshold_for_feature(
#     ['"grade"=2.0', '"gleason"=5.0', '"g2">3.84'], '"g2"', '3.84'))
# print(dtreeN.class_entropy_for_less_than_threshold_for_feature(
#     ['"grade"=2.0', '"gleason"<5.0', '"g2">3.84'], '"g2"', '46.56'))

# ----- CLassEntropyForGreaterThanThresholdForFeature -----
# print("\n\nTESTS- Greater Than")
# print(dtreeN.class_entropy_for_greater_than_threshold_for_feature(
#     ['"grade"=2.0', '"gleason"=5.0'], '"g2"', '46.56'))
# print(dtreeN.class_entropy_for_greater_than_threshold_for_feature(
#     ['"grade"=2.0', '"gleason"=5.0', '"g2"<3.84'], '"age"', '57.0'))
# print(dtreeN.class_entropy_for_greater_than_threshold_for_feature(
#     ['"grade"=2.0', '"gleason"=5.0', '"g2"<3.84'], '"g2"', '3.84'))
# print(dtreeN.class_entropy_for_greater_than_threshold_for_feature(
#     ['"grade"=2.0', '"gleason"=5.0', '"g2">3.84'], '"g2"', '3.84'))
# print(dtreeN.class_entropy_for_greater_than_threshold_for_feature(
#     ['"grade"=2.0', '"gleason"<5.0', '"g2">3.84'], '"g2"', '46.56'))

# ----- classEntropyForAGivenSequenceOfFeaturesAndValuesOrThresholds
# print("\n\nTESTS - Class Entropy For A Given Sequence Of Features And Values Or Thresholds")
# print(dtreeN.class_entropy_for_a_given_sequence_of_features_and_values_or_thresholds(
#     ['"grade"=2.0', '"gleason"=5.0', '"g2"<3.84', '"age">57.0']))
# print(dtreeN.class_entropy_for_a_given_sequence_of_features_and_values_or_thresholds(
#     ['"grade"=2.0']))
# print(dtreeN.class_entropy_for_a_given_sequence_of_features_and_values_or_thresholds(
#     ['"grade"=2.0', '"gleason"=5.0', '"g2">3.84']))
# print(dtreeN.class_entropy_for_a_given_sequence_of_features_and_values_or_thresholds(
#     ['"grade"=2.0', '"gleason"=5.0', '"g2">3.84', '"age">47.0']))
# print(dtreeN.class_entropy_for_a_given_sequence_of_features_and_values_or_thresholds(
#     ['"grade"=2.0', '"gleason">2', '"g2">3.84', '"age"<49.0']))
# print(dtreeN.class_entropy_for_a_given_sequence_of_features_and_values_or_thresholds(
#     ['"grade"=2', '"gleason"=5.0', '"g2">25.0', '"age"=65',  '"g2"<28.0']))


# /***************************/ PROBABILITY /***************************/
# SYMBOLIC
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

# ----- ProbOfASequenceOfFeaturesAndValuesOrThresholds -----
# print(dtree.probability_of_a_sequence_of_features_and_values_or_thresholds( ['exercising=never'] ))
# print(dtree.probability_of_a_sequence_of_features_and_values_or_thresholds( ['fatIntake=heavy'] ))
# print(dtree.probability_of_a_sequence_of_features_and_values_or_thresholds( ['fatIntake=low', 'smoking=heavy'] ))
# print(dtree.probability_of_a_sequence_of_features_and_values_or_thresholds( ['fatIntake=low', 'smoking=never', 'exercising=regularly'] ))
# print(dtree.probability_of_a_sequence_of_features_and_values_or_thresholds( ['fatIntake=medium', 'exercising=occasionally'] ))

# ----- ProbOfASequenceOfFeaturesAndValuesOrThresholdsGivenClass -----
# print(dtree.probability_of_a_sequence_of_features_and_values_or_thresholds_given_class( ['exercising=never'], 'class=benign' ))
# print(dtree.probability_of_a_sequence_of_features_and_values_or_thresholds_given_class( ['smoking=heavy'], 'class=malignant' ))
# print(dtree.probability_of_a_sequence_of_features_and_values_or_thresholds_given_class( ['fatIntake=heavy', 'exercising=never'], 'class=benign' ))
# print(dtree.probability_of_a_sequence_of_features_and_values_or_thresholds_given_class( ['fatIntake=heavy', 'videoAddiction=medium'], 'class=malignant' ))
# print(dtree.probability_of_a_sequence_of_features_and_values_or_thresholds_given_class( ['fatIntake=heavy', 'smoking=heavy', 'videoAddiction=none'], 'class=benign' ))
# print(dtree.probability_of_a_sequence_of_features_and_values_or_thresholds_given_class( ['fatIntake=heavy', 'smoking=heavy', 'videoAddiction=heavy', 'exercising=regularly'], 'class=benign' ))


# NUMERIC
# ----- PriorProbabilityForClass -----
# print(dtreeN.prior_probability_for_class('"pgstat"=1'))
# print(dtreeN.prior_probability_for_class('"pgstat"=0'))

# ----- ProbOfFeatureValue -----
# print(dtreeN.probability_of_feature_value( '"grade"', '2.0' ))
# print(dtreeN.probability_of_feature_value( '"grade"', '3.0' ))
# print(dtreeN.probability_of_feature_value( '"gleason"', '8.0' ))
# print(dtreeN.probability_of_feature_value( '"ploidy"', 'tetraploid' ))
# print(dtreeN.probability_of_feature_value('"age"', '63'))
# print(dtreeN.probability_of_feature_value('"age"', '64'))

# ----- ProbOfFeatureValueGivenClass -----
# print(dtreeN.probability_of_feature_value_given_class('"grade"', '2.0', '"pgstat"=1'))
# print(dtreeN.probability_of_feature_value_given_class('"grade"', '3.0', '"pgstat"=0'))
# print(dtreeN.probability_of_feature_value_given_class('"grade"', '4.0', '"pgstat"=1'))
# print(dtreeN.probability_of_feature_value_given_class('"grade"', '2.0', '"pgstat"=1'))
# print(dtreeN.probability_of_feature_value_given_class('"grade"', '3.0', '"pgstat"=1'))
# print(dtreeN.probability_of_feature_value_given_class('"gleason"', '7', '"pgstat"=1'))
# print(dtreeN.probability_of_feature_value_given_class('"ploidy"', '"diploid"', '"pgstat"=1'))
# print(dtreeN.probability_of_feature_value_given_class('"ploidy"', '"tetraploid"', '"pgstat"=1'))
# print(dtreeN.probability_of_feature_value_given_class('"eet"', '2', '"pgstat"=1'))
# print(dtreeN.probability_of_feature_value_given_class('"gleason"', '8', '"pgstat"=0'))
# print(dtreeN.probability_of_feature_value_given_class('"ploidy"', '"aneuploid"', '"pgstat"=0'))
# print(dtreeN.probability_of_feature_value_given_class('"eet"', '1', '"pgstat"=0'))
# print(dtreeN.probability_of_feature_value_given_class('"ploidy"', '"aneuploid"', '"pgstat"=0'))
# print(dtreeN.probability_of_feature_value_given_class('"grade"', '2.0', '"pgstat"=1'))
# print(dtreeN.probability_of_feature_value_given_class('"grade"', '3.0', '"pgstat"=0'))
# print(dtreeN.probability_of_feature_value_given_class('"grade"', '4.0', '"pgstat"=1'))
# print(dtreeN.probability_of_feature_value_given_class('"grade"', '2.0', '"pgstat"=1'))
# print(dtreeN.probability_of_feature_value_given_class('"grade"', '3.0', '"pgstat"=1'))
# print(dtreeN.probability_of_feature_value_given_class('"gleason"', '7', '"pgstat"=1'))
# print(dtreeN.probability_of_feature_value_given_class('"ploidy"', '"diploid"', '"pgstat"=1'))
# print(dtreeN.probability_of_feature_value_given_class('"ploidy"', '"tetraploid"', '"pgstat"=1'))
# print(dtreeN.probability_of_feature_value_given_class('"eet"', '2', '"pgstat"=1'))
# print(dtreeN.probability_of_feature_value_given_class('"gleason"', '8', '"pgstat"=0'))
# print(dtreeN.probability_of_feature_value_given_class('"ploidy"', '"aneuploid"', '"pgstat"=0'))
# print(dtreeN.probability_of_feature_value_given_class('"eet"', '1', '"pgstat"=0'))
# print(dtreeN.probability_of_feature_value_given_class('"ploidy"', '"aneuploid"', '"pgstat"=0'))
# print(dtreeN.probability_of_feature_value_given_class('"age"', '62', '"pgstat"=0'))

# ----- ProbOfFeatureValueLessThanThreshold -----
# print(dtreeN.probability_of_feature_less_than_threshold('"age"', '47'))
# print(dtreeN.probability_of_feature_less_than_threshold('"age"', '50'))
# print(dtreeN.probability_of_feature_less_than_threshold('"age"', '100'))

# ----- ProbOfFeatureValueLessThanThresholdGivenClass -----
# print(dtreeN.probability_of_feature_less_than_threshold_given_class( '"age"', '47', '"pgstat"=1' ))
# print(dtreeN.probability_of_feature_less_than_threshold_given_class( '"age"', '90', '"pgstat"=1' ))
# print(dtreeN.probability_of_feature_less_than_threshold_given_class( '"age"', '68', '"pgstat"=1' ))
# print(dtreeN.probability_of_feature_less_than_threshold_given_class( '"age"', '73', '"pgstat"=1' ))
# print(dtreeN.probability_of_feature_less_than_threshold_given_class( '"eet"', '2', '"pgstat"=1' ))
# print(dtreeN.probability_of_feature_less_than_threshold_given_class( '"g2"', '14.5', '"pgstat"=1' ))
# print(dtreeN.probability_of_feature_less_than_threshold_given_class( '"grade"', '3', '"pgstat"=1' ))
# print(dtreeN.probability_of_feature_less_than_threshold_given_class( '"gleason"', '6', '"pgstat"=1' ))

# ----- ProbOfASequenceOfFeaturesAndValuesOrThresholds -----
# print(dtreeN.probability_of_a_sequence_of_features_and_values_or_thresholds( ['"age"<47.0'] ))
# print(dtreeN.probability_of_a_sequence_of_features_and_values_or_thresholds( ['"g2"<8.640000000000052'] ))
# print(dtreeN.probability_of_a_sequence_of_features_and_values_or_thresholds( ['"grade"=2.0', '"g2">49.20000000000039'] ))
# print(dtreeN.probability_of_a_sequence_of_features_and_values_or_thresholds( ['"grade"=2.0', '"gleason"=4.0', '"g2">49.20000000000039'] ))
# print(dtreeN.probability_of_a_sequence_of_features_and_values_or_thresholds( ['"grade"=2.0', '"gleason"=5.0', '"g2"<3.840000000000012', '"ploidy"="aneuploid"'] ))
# print(dtreeN.probability_of_a_sequence_of_features_and_values_or_thresholds( ['"grade"=2.0', '"gleason"=5.0', '"g2"<3.840000000000012', '"ploidy"="tetraploid"'] ))
# print(dtreeN.probability_of_a_sequence_of_features_and_values_or_thresholds(
#  ['"grade"=2.0', '"gleason"=4.0', '"g2">3.840000000000012', '"age"<49.0', '"g2">13.440000000000092', '"g2">17.04000000000012', '"g2">49.20000000000039']))


# ----- FindBoundedIntervalsForNumericFeatures -----
# test with one feature
# print(dtreeN.find_bounded_intervals_for_numeric_features( ['"g2">51.360000000000404'] ))
# test with overlapping feature bounds
# print(dtreeN.find_bounded_intervals_for_numeric_features( ['"g2"<3.840000000000012', '"g2"<2.4'] ))
# test with multiple features
# print(dtreeN.find_bounded_intervals_for_numeric_features( ['"g2"<3.840000000000012', '"age"<63.0'] ))

# ----- ProbOfASequenceOfFeaturesAndValuesOrThresholdsGivenClass -----
# print(dtreeN.probability_of_a_sequence_of_features_and_values_or_thresholds_given_class( ['"age"<47.0'], '"pgstat"=1' ))
# print(dtreeN.probability_of_a_sequence_of_features_and_values_or_thresholds_given_class( ['"grade"=2.0', '"gleason"=5.0', '"g2"<3.840000000000012', '"age">51.0'], '"pgstat"=0' ))
# print(dtreeN.probability_of_a_sequence_of_features_and_values_or_thresholds_given_class( ['"grade"=2.0', '"gleason"=5.0', '"g2"<3.840000000000012', '"ploidy"="aneuploid"'], '"pgstat"=0' ))
# print(dtreeN.probability_of_a_sequence_of_features_and_values_or_thresholds_given_class( ['"grade"=2.0', '"g2">42.00000000000033'], '"pgstat"=0' ))
# print(dtreeN.probability_of_a_sequence_of_features_and_values_or_thresholds_given_class(
    # ['"grade"=2', '"gleason"=5.0', '"g2">25.0', '"age"=62',  '"g2"<28.0'], '"pgstat"=1'))

# ----- ProbOfAClassGivenSequenceOfFeaturesAndValuesOrThresholds -----
# print(dtreeN.probability_of_a_class_given_sequence_of_features_and_values_or_thresholds(
#     '"pgstat"=0', ['"age">47.0']), end='\n\n')
# print(dtreeN.probability_of_a_class_given_sequence_of_features_and_values_or_thresholds(
#     '"pgstat"=1', ['"age">47.0']), end='\n\n')
# print(dtreeN.probability_of_a_class_given_sequence_of_features_and_values_or_thresholds(
#     '"pgstat"=0', ['"age">47.0', '"gleason"=4.0']), end='\n\n')
# print(dtreeN.probability_of_a_class_given_sequence_of_features_and_values_or_thresholds('"pgstat"=1', [
#       '"grade"=2.0', '"gleason"=4.0', '"g2">3.840000000000012', '"age"<49.0', '"g2">13.440000000000092', '"g2">17.04000000000012', '"g2">49.20000000000039']), end='\n\n')
# print(dtreeN.probability_of_a_class_given_sequence_of_features_and_values_or_thresholds(
#     '"pgstat"=0', ['"grade"=2.0', '"g2"<36.960000000000285']), end='\n\n')
# print(dtreeN.probability_of_a_class_given_sequence_of_features_and_values_or_thresholds(
#     '"pgstat"=1', ['"grade"=2.0', '"gleason"=5.0', '"g2">25', '"age"=65', '"g2"<28']), end='\n\n')
