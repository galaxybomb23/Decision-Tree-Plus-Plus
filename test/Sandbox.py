import DecisionTree as dt
from DecisionTree import DTIntrospection as dtI

print("SandBox...")

# <============== SYMBOLIC TREE ===============>
dtreeS = dt.DecisionTree(training_datafile="test/resources/training_symbolic.csv",
                        csv_class_column_index=1,
                        csv_columns_for_features=[2, 3, 4, 5],
                        max_depth_desired=5,
                        entropy_threshold=0.1,
                        debug3=False
                        )

dtreeS.get_training_data()
dtreeS.calculate_class_priors()
dtreeS.calculate_first_order_probabilities()
# dtree.determine_data_condition()

root_node = dtreeS.construct_decision_tree_classifier()
# root_node.display_decision_tree("  ")

# <============== NUMERIC TREE ===============>
# print(f"NUMERIC TREE")
dtreeN = dt.DecisionTree(
    training_datafile="test/resources/stage3cancer.csv",
    csv_class_column_index=2,
    csv_columns_for_features=[3, 4, 5, 6, 7, 8],
    max_depth_desired=8,
    entropy_threshold=0.01
    # , debug2=True
)

dtreeN.get_training_data()
dtreeN.calculate_class_priors()
dtreeN.calculate_first_order_probabilities()

root_nodeN = dtreeN.construct_decision_tree_classifier()
# root_nodeN.display_decision_tree("  ")


# /***************************/ Classify /***************************/
# SYMBOLIC
# test_sample = ['exercising=never', 'smoking=never', 'fatIntake=heavy', 'videoAddiction=heavy']
# classification = dtreeS.classify(root_node, test_sample)
# print("Classification: " + str(classification))

# NUMERIC
# test_sample = ['"age"=65', '"eet"=2', '"g2"=6.2', '"grade"=2', '"gleason"=5', '"ploidy"=tetraploid']
# classification = dtreeN.classify(root_nodeN, test_sample)
# print("Classification:", str(classification))

# /***************************/ Introspection /***************************/
# SYMBOLIC
dtreeSI = dtI(dtreeS)

dtreeSI.initialize()

print(dtreeSI._node_serial_num_to_node_dict)


# NUMERIC
dtreeNI = dtI(dtreeN)