#ifndef DECISION_TREE_HPP
#define DECISION_TREE_HPP

// Include
#include "DecisionTreeNode.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <set>

class DecisionTree
{
public:
    DecisionTree(std::map<std::string, std::string> kwargs); // constructor
    ~DecisionTree(); // destructor

    void get_training_data();
    void calculate_first_order_probabilities();
    void show_training_data() const;

private:
    std::string _training_datafile;
    double _entropy_threshold;
    int _max_depth_desired;
    int _csv_class_column_index;
    int _symbolic_to_numeric_cardinality_threshold;
    int _number_of_histogram_bins;
    int _csv_cleanup_needed;
    int _debug1, _debug2, _debug3;

    std::vector<int> _csv_columns_for_features;
    std::vector<std::string> _class_names;
    std::map<std::string, std::vector<std::string>> _training_data_dict;
    std::map<std::string, std::vector<double>> _numeric_features_valuerange_dict;
    std::map<std::string, int> _feature_values_how_many_uniques_dict;
    std::map<std::string, std::vector<std::string>> _features_and_values_dict;
    std::map<std::string, std::set<std::string>> _features_and_unique_values_dict;
    std::map<std::string, std::string> _samples_class_label_dict;
};

#endif // DECISION_TREE_HPP