#include "EvalTrainingData.hpp"

// Constructor inheriting from the DecisionTree
EvalTrainingData::EvalTrainingData(std::map<std::string, std::string> kwargs) : DecisionTree(kwargs)
{
    this->getTrainingData();
    _csvClassColumnIndex = std::stoi(kwargs["csv_class_column_index"]);
}
EvalTrainingData::~EvalTrainingData()
{
    // Destructor
}

// Method to evaluate training data
void EvalTrainingData::evaluateTrainingData()
{
    std::cout << "training data dict size: " << _trainingDataDict.size() << "\n";
    bool evalDebug = true;

    // Check if the training data file is a CSV
    if (_trainingDatafile.substr(_trainingDatafile.find_last_of(".") + 1) != "csv") {
        throw std::runtime_error("The data evaluation function can only be used for CSV files.");
    }

    std::cout << "\nWill run a 10-fold cross-validation test on your training "
                 "data...\n";
    std::map<int, std::vector<std::string>> allTrainingData;

    // Get all the training data
    for (const auto &entry : _trainingDataDict) {
        allTrainingData[entry.first] = std::vector<std::string>();
        for (const auto &feature : entry.second) {
            allTrainingData[entry.first].push_back(feature);
        }
    }

    std::vector<std::string> allSampleNames;

    // Sort samples based on some index
    for (const auto &entry : allTrainingData) {
        auto ent = std::to_string(entry.first);
        std::cout << "Sample name: " << ent << "\n";
        allSampleNames.push_back(ent);
    }

    std::sort(allSampleNames.begin(), allSampleNames.end(), [](const std::string &a, const std::string &b) {
        return std::stoi(a) < std::stoi(b);
    });

    int foldSize = static_cast<int>(0.1 * allTrainingData.size());
    std::map<int, std::map<std::string, int>> confusion_matrix;

    for (const auto &class_name : _classNames) {
        int class_index               = std::stoi(class_name);
        confusion_matrix[class_index] = std::map<std::string, int>();
        for (const auto &class_name2 : _classNames) {
            confusion_matrix[class_index][class_name2] = 0;
        }
    }
    // Perform 10-fold cross-validation
    for (int foldIndex = 0; foldIndex < 10; ++foldIndex) {
        std::vector<std::string> testing_samples(allSampleNames.begin() + foldSize * foldIndex,
                                                 allSampleNames.begin() + foldSize * (foldIndex + 1));
        std::vector<std::string> training_samples(allSampleNames.begin(),
                                                  allSampleNames.begin() + foldSize * foldIndex);
        training_samples.insert(
            training_samples.end(), allSampleNames.begin() + foldSize * (foldIndex + 1), allSampleNames.end());

        std::map<int, std::vector<std::string>> testingData, trainingData;
        for (const auto &sample : testing_samples) {
            auto samp         = std::stoi(sample);
            testingData[samp] = allTrainingData[samp];
        }
        for (const auto &sample : training_samples) {
            auto samp          = std::stoi(sample);
            trainingData[samp] = allTrainingData[samp];
        }

        _trainingDataDict = trainingData;
        _featuresAndValuesDict.clear();

        int idx = 0;
        for (const auto &item : _trainingDataDict) {
            for (const auto &feature_and_value : item.second) {
                std::string feature = _featureNames[idx % 6];
                std::string value   = feature_and_value;
                if (value != "NA") {
                    _featuresAndValuesDict[feature].push_back(value);
                }
                idx++;
            }
        }
        // Set unique values for features
        for (auto &pair : _featuresAndValuesDict) {
            std::set<std::string> unique_values(pair.second.begin(), pair.second.end());
            _featuresAndUniqueValuesDict[pair.first] =
                std::set<std::string>(unique_values.begin(), unique_values.end());
        }

        // Calculate numeric value ranges for features
        _numericFeaturesValueRangeDict.clear();

        // Assuming _numericFeaturesValueRangeDict is a map of {key -> {min, max}}
        for (const auto &feature : _featuresAndUniqueValuesDict) {
            std::set<double> numeric_values;
            for (const auto &value : feature.second) {
                try {
                    numeric_values.insert(std::stod(value));
                }
                catch (const std::invalid_argument &e) {
                    continue;
                }
            }
            if (!numeric_values.empty()) {
                std::vector<double> numeric_values_vec(numeric_values.begin(), numeric_values.end());
                std::sort(numeric_values_vec.begin(), numeric_values_vec.end());
                _numericFeaturesValueRangeDict[feature.first] = {numeric_values_vec.front(), numeric_values_vec.back()};
            }
        }

        // Probabilities and construction of decision tree root node
        this->calculateFirstOrderProbabilities();
        this->calculateClassPriors();
        auto root_node = this->constructDecisionTreeClassifier();

        for (const auto &testSampleName : testing_samples) {
            int test_sample_key = std::stoi(testSampleName);

            try {
                // Retrieve the data for the test sample safely using .at()
                const auto &test_sample_data = allTrainingData.at(test_sample_key);

                // Filter out features with '=NA' values (we prepend feature name)
                int idx = 0;
                std::vector<std::string> filtered_data;
                for (const auto &feature_value : test_sample_data) {
                    std::string modified_feature_value = _featureNames[idx % 6] + "=" + feature_value;
                    if (modified_feature_value.find("=NA") == std::string::npos) {
                        filtered_data.push_back(modified_feature_value);
                    }
                    idx++;
                }

                // print filtered data
                // for (const auto &item : filtered_data) {
                //     std::cout << "Filtered data: " << item << "\n";
                // }

                // Perform classification using the decision tree
                auto classification = this->classify(root_node, filtered_data);

                // Check if "solution_path" exists in classification
                auto solution_path = classification.at("solution_path"); // Will throw if "solution_path" is not found
                classification.erase("solution_path");

                // Sort the classes based on their probabilities
                std::vector<std::string> sorted_classes;
                for (const auto &entry : classification) {
                    sorted_classes.push_back(entry.first);
                }
                std::sort(sorted_classes.begin(),
                          sorted_classes.end(),
                          [&classification](const std::string &a, const std::string &b) {
                              return classification.at(a) > classification.at(b); // Will throw if a key is missing
                          });

                // Get the most likely class label
                std::string most_likely_class_label = sorted_classes.front();

                // Retrieve the true class label safely using .at()
                std::string true_class_label_for_test_sample =
                    this->_samplesClassLabelDict.at(test_sample_key); // Will throw if key is not found

                // Optionally print the true vs estimated class labels
                if (evalDebug) {
                    std::cout << testSampleName << ":   true_class: " << true_class_label_for_test_sample
                              << "    estimated_class: " << most_likely_class_label << "\n";
                }

                // Update the confusion matrix using .at() for inner map
                confusion_matrix.at(std::stoi(true_class_label_for_test_sample)).at(most_likely_class_label) +=
                    1; // Will throw if any key is missing
            }
            catch (const std::out_of_range &e) {
                std::cerr << "Error: Key not found in map for test sample: " << testSampleName << "\n";
                if (_trainingDataDict.find(test_sample_key) == _trainingDataDict.end()) {
                    std::cerr << "  Failed on _trainingDataDict.at(" << test_sample_key << ")\n";
                }

                else if (_samplesClassLabelDict.find(test_sample_key) == _samplesClassLabelDict.end()) {
                    std::cerr << "  Failed on _samplesClassLabelDict.at(" << test_sample_key << ")\n";
                }
                else {
                    std::cerr << "  Unknown error\n";
                }
                continue; // Skip this test sample if there is an error
            }
        }
    }
}

// methods to print information << NEEDS TO BE IMPLEMENTED >>
void EvalTrainingData::printDebugInformation(DecisionTree &trainingDT, const std::vector<std::string> &testing_samples)
{
    // if evaldebug:
    //         print("\n\nprinting samples in the testing set: " +
    //         str(testing_samples)) print("\n\nPrinting features and their
    //         values in the training set:\n") for item in
    //         sorted(_features_and_values_dict.items()):
    //             print(item[0]  + "  =>  "  + str(item[1]))
    //         print("\n\nPrinting unique values for features:\n")
    //         for item in
    //         sorted(_features_and_unique_values_dict.items()):
    //             print(item[0]  + "  =>  "  + str(item[1]))
    //         print("\n\nPrinting unique value ranges for features:\n")
    //         for item in
    //         sorted(_numeric_features_valuerange_dict.items()):
    //             print(item[0]  + "  =>  "  + str(item[1]))

    // std::cout << "\n\nPrinting samples in the testing set: " << testing_samples << "\n";
    // std::cout << "\n\nPrinting features and their values in the training set:\n";
    // for (const auto &item : _featuresAndValuesDict) {
    //     std::cout << item.first << "  =>  " << item.second << "\n";
    // }
    // std::cout << "\n\nPrinting unique values for features:\n";
    // for (const auto &item : _featuresAndUniqueValuesDict) {
    //     std::cout << item.first << "  =>  " << item.second << "\n";
    // }
    // std::cout << "\n\nPrinting unique value ranges for features:\n";
    // for (const auto &item : _numericFeaturesValueRangeDict) {
    //     std::cout << item.first << "  =>  " << item.second << "\n";
    // }
}

void EvalTrainingData::printClassificationInfo(const std::vector<std::string> &which_classes,
                                               const std::map<std::string, double> &classification,
                                               const std::string &most_likely_class_label,
                                               std::shared_ptr<DecisionTreeNode> root_node)
{
    // print("\nClassification:\n")
    //             print("     "  + str.ljust("class name", 30) +
    //             "probability") print("     ---------- -----------") for
    //             which_class in which_classes:
    //                 if which_class is not 'solution_path':
    //                     print("     "  + str.ljust(which_class, 30) +
    //                     str(classification[which_class]))
    //             print("\nSolution path in the decision tree: " +
    //             str(solution_path)) print("\nNumber of nodes created: " +
    //             str(root_node.how_many_nodes()))
    std::cout << "\nClassification:\n";
    std::cout << "     " << std::setw(30) << "class name" << "probability\n";
    std::cout << "     ----------                    -----------\n";
    for (const auto &which_class : which_classes) {
        if (which_class != "solution_path") {
            std::cout << "     " << std::setw(30) << which_class << classification.at(which_class) << "\n";
        }
    }
    std::cout << "\nSolution path in the decision tree: " << classification.at("solution_path") << "\n";
    std::cout << "\nNumber of nodes created: " << root_node->HowManyNodes() << "\n";
}
// void EvalTrainingData::displayConfusionMatrix(const std::map<int, std::map<std::string, int>> &confusion_matrix)
// {
//     //   print("\n\n       DISPLAYING THE CONFUSION MATRIX FOR THE 10-FOLD
//     //   CROSS-VALIDATION TEST:\n")
//     // matrix_header = " " * 30
//     // for class_name in self._class_names:
//     //     matrix_header += '{:^30}'.format(class_name)
//     // print("\n" + matrix_header + "\n")
//     // for row_class_name in sorted(confusion_matrix.keys()):
//     //     row_display = str.rjust(row_class_name, 30)
//     //     for col_class_name in
//     //     sorted(confusion_matrix[row_class_name].keys()):
//     //         row_display +=
//     //         '{:^30}'.format(str(confusion_matrix[row_class_name][col_class_name])
//     //         )
//     //     print(row_display + "\n")
//     // diagonal_sum, off_diagonal_sum = 0,0
//     // for row_class_name in sorted(confusion_matrix.keys()):
//     //     for col_class_name in
//     //     sorted(confusion_matrix[row_class_name].keys()):
//     //         if row_class_name == col_class_name:
//     //             diagonal_sum +=
//     //             confusion_matrix[row_class_name][col_class_name]
//     //         else:
//     //             off_diagonal_sum +=
//     //             confusion_matrix[row_class_name][col_class_name]
//     std::cout << "\n\n       DISPLAYING THE CONFUSION MATRIX FOR THE 10-FOLD "
//                  "CROSS-VALIDATION TEST:\n";
//     std::string matrix_header = std::string(30, ' ');
//     for (const auto &class_name : _classNames) {
//         matrix_header += class_name;
//     }
//     std::cout << "\n" << matrix_header << "\n";
//     for (const auto &row_class_name : _classNames) {
//         std::string row_display = std::string(30, ' ');
//         row_display += row_class_name;
//         for (const auto &col_class_name : _classNames) {
//             row_display += std::to_string(confusion_matrix.at(row_class_name).at(col_class_name));
//         }
//         std::cout << row_display << "\n";
//     }
//}
;
double
EvalTrainingData::calculateDataQualityIndex(const std::map<std::string, std::map<std::string, int>> &confusion_matrix)
{
    int diagonal_sum = 0, off_diagonal_sum = 0;
    for (const auto &row_class_name : _classNames) {
        for (const auto &col_class_name : _classNames) {
            if (row_class_name == col_class_name) {
                diagonal_sum += confusion_matrix.at(row_class_name).at(col_class_name);
            }
            else {
                off_diagonal_sum += confusion_matrix.at(row_class_name).at(col_class_name);
            }
        }
    }
    return 100.0 * diagonal_sum / (diagonal_sum + off_diagonal_sum);
}

void EvalTrainingData::printDataQualityEvaluation(double data_quality_index)
{
    std::cout << "\nTraining Data Quality Index: " << data_quality_index << "   (out of a possible maximum of 100)\n";
    if (data_quality_index <= 80) {
        std::cout << "\nYour training data does not possess much class "
                     "discriminatory "
                     "information.  It could be that the classes are inherently not "
                     "well "
                     "separable or that your constructor parameter choices are not "
                     "appropriate.\n";
    }
    else if (80 < data_quality_index <= 90) {
        std::cout << "\nYour training data possesses some class discriminatory "
                     "information "
                     "but it may not be sufficient for real-world applications.  "
                     "You might "
                     "try tweaking the constructor parameters to see if that "
                     "improves the "
                     "class discriminations.\n";
    }
    else if (90 < data_quality_index <= 95) {
        std::cout << "\nYour training data appears to possess good class "
                     "discriminatory "
                     "information.  Whether or not it is acceptable would depend "
                     "on your "
                     "application.\n";
    }
    else if (95 < data_quality_index < 98) {
        std::cout << "\nYour training data is of very high quality.\n";
    }
    else {
        std::cout << "\nYour training data is excellent.\n";
    }
}