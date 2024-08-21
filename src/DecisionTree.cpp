// Include
#include "DecisionTree.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <iterator>

DecisionTree::DecisionTree(std::map<std::string, std::string> kwargs)
{
    if (kwargs.empty()) {
        throw std::invalid_argument("Missing training datafile.");
    }

    // Allowed keys for the kwargs
    std::vector<std::string> allowed_keys = {
        "training_datafile",
        "entropy_threshold",
        "max_depth_desired",
        "csv_class_column_index",
        "symbolic_to_numeric_cardinality_threshold",
        "csv_columns_for_features",
        "number_of_histogram_bins",
        "csv_cleanup_needed",
        "debug1",
        "debug2",
        "debug3"
    };

    // Set default values
    _entropy_threshold = 0.01;
    _symbolic_to_numeric_cardinality_threshold = 10;
    _csv_cleanup_needed = 0;
    _csv_columns_for_features = {};
    _debug1 = _debug2 = _debug3 = 0;
    _max_depth_desired = _csv_class_column_index = _number_of_histogram_bins = -1;
    _root_node = nullptr;
    _how_many_total_training_samples = 0;
    _probability_cache = {};
    _entropy_cache = {};
    _training_data_dict = {};
    _features_and_values_dict = {};
    _features_and_unique_values_dict = {};
    _samples_class_label_dict = {};
    _class_names = {};
    _class_priors_dict = {};
    _feature_names = {};
    _numeric_features_valuerange_dict = {};
    _sampling_points_for_numeric_feature_dict = {};
    _feature_values_how_many_uniques_dict = {};
    _prob_distribution_numeric_features_dict = {};
    _histogram_delta_dict = {};
    _num_of_histogram_bins_dict = {};

    // Check and set keyword arguments
    for (const auto& kv : kwargs) {
        const std::string& key = kv.first;
        const std::string& value = kv.second;

        if (key == "training_datafile") {
            _training_datafile = value;
        } else if (key == "entropy_threshold") {
            _entropy_threshold = std::stod(value);
        } else if (key == "max_depth_desired") {
            _max_depth_desired = std::stoi(value);
        } else if (key == "csv_class_column_index") {
            _csv_class_column_index = std::stoi(value);
        } else if (key == "csv_columns_for_features") {
            for (const auto& c : value) {
                _csv_columns_for_features.push_back(c);
            }
        } else if (key == "symbolic_to_numeric_cardinality_threshold") {
            _symbolic_to_numeric_cardinality_threshold = std::stoi(value);
        } else if (key == "number_of_histogram_bins") {
            _number_of_histogram_bins = std::stoi(value);
        } else if (key == "csv_cleanup_needed") {
            _csv_cleanup_needed = std::stoi(value);
        } else if (key == "debug1") {
            _debug1 = std::stoi(value);
        } else if (key == "debug2") {
            _debug2 = std::stoi(value);
        } else if (key == "debug3") {
            _debug3 = std::stoi(value);
        } else {
            throw std::invalid_argument(key + ": Wrong keyword used --- check spelling");
        }
    }
}

DecisionTree::~DecisionTree()
{

}

// Other functions below