#include "EvalTrainingData.hpp"

#include <algorithm>
#include <iostream>
#include <map>
#include <memory>
#include <regex>
#include <set>
#include <vector>

#include "DecisionTree.hpp" // Include necessary headers
#include "Utility.hpp"

class EvalTrainingData : public DecisionTree
{
public:
    // Constructor inheriting from the DecisionTree
    EvalTrainingData(std::map<std::string, std::string> kwargs)
        : DecisionTree(kwargs) {}

    // Method to evaluate training data
    void evaluateTrainingData()
    {
        bool evaldebug = false;

        // Check if the training data file is a CSV
        if (_trainingDatafile.substr(_trainingDatafile.find_last_of(".") + 1) !=
            "csv")
        {
            throw std::runtime_error(
                "The data evaluation function can only be used for CSV files.");
        }

        std::cout << "\nWill run a 10-fold cross-validation test on your training "
                     "data...\n";
        auto all_training_data = _trainingDataDict;
        std::vector<std::string> all_sample_names;

        // Sort samples based on some index
        for (const auto &entry : all_training_data)
        {
            all_sample_names.push_back(entry.first);
        }
        std::sort(all_sample_names.begin(), all_sample_names.end(),
                  [](const std::string &a, const std::string &b)
                  {
                      return sampleIndex(a) < sampleIndex(b);
                  });

        int fold_size = static_cast<int>(0.1 * all_training_data.size());
        std::map<std::string, std::map<std::string, int>> confusion_matrix;

        for (const auto &class_name : _classNames)
        {
            confusion_matrix[class_name] = std::map<std::string, int>();
            for (const auto &class_name2 : _classNames)
            {
                confusion_matrix[class_name][class_name2] = 0;
            }
        }

        // Perform 10-fold cross-validation
        for (int fold_index = 0; fold_index < 10; ++fold_index)
        {
            std::cout << "\nStarting fold " << fold_index
                      << " of the 10-fold cross-validation test.\n";
            std::vector<std::string> testing_samples(
                all_sample_names.begin() + fold_size * fold_index,
                all_sample_names.begin() + fold_size * (fold_index + 1));
            std::vector<std::string> training_samples(
                all_sample_names.begin(),
                all_sample_names.begin() + fold_size * fold_index);
            training_samples.insert(
                training_samples.end(),
                all_sample_names.begin() + fold_size * (fold_index + 1),
                all_sample_names.end());

            std::map<std::string, std::vector<std::string>> testing_data,
                training_data;
            for (const auto &sample : testing_samples)
            {
                testing_data[sample] = all_training_data[sample];
            }
            for (const auto &sample : training_samples)
            {
                training_data[sample] = all_training_data[sample];
            }
            std::map<std::string, std::string> kwargs = {{"evalmode", "1"}};
            DecisionTree trainingDT(kwargs);
            trainingDT.setTrainingDataDict(training_data);
            trainingDT._classNames = _classNames;
            trainingDT.setFeatureNames(_featureNames);
            trainingDT.setEntropyThreshold(_entropyThreshold);
            trainingDT.setMaxDepthDesired(_maxDepthDesired);
            trainingDT.setSymbolicToNumericCardinalityThreshold(
                _symbolicToNumericCardinalityThreshold);
            trainingDT.setSamplesClassLabelDict(_samplesClassLabelDict);

            // Populate features and values dict for training data
            trainingDT._featuresAndValuesDict.clear();
            std::regex pattern(R "((\S+)\s*=\s*(\S+))");
            for (const auto &item : trainingDT._trainingDataDict)
            {
                for (const auto &feature_and_value : item.second)
                {
                    std::smatch match;
                    if (std::regex_search(feature_and_value, match, pattern))
                    {
                        std::string feature = match[1].str();
                        std::string value = match[2].str();
                        if (value != "NA")
                        {
                            trainingDT._featuresAndValuesDict[feature].insert(convert(value));
                        }
                    }
                }
            }

            // Set unique values for features
            for (auto &pair : trainingDT._featuresAndValuesDict)
            {
                std::set<std::string> unique_values(pair.second.begin(),
                                                    pair.second.end());
                trainingDT._featuresAndUniqueValuesDict[pair.first] =
                    std::set<std::string>(unique_values.begin(), unique_values.end());
            }

            // Calculate numeric value ranges for features
            trainingDT._numericFeaturesValueRangeDict.clear();
            for (const auto &feature : _numericFeaturesValueRangeDict)
            {
                trainingDT._numericFeaturesValueRangeDict[feature.first] = {
                    std::min_element(
                        trainingDT._featuresAndUniqueValuesDict[feature.first].begin(),
                        trainingDT._featuresAndUniqueValuesDict[feature.first].end()),
                    std::max_element(
                        trainingDT._featuresAndUniqueValuesDict[feature.first].begin(),
                        trainingDT._featuresAndUniqueValuesDict[feature.first].end())};
            }

            // Optional debug output
            if (evaldebug)
            {
                printDebugInformation(trainingDT, testing_samples);
            }

            trainingDT.calculateFirstOrderProbabilities();
            trainingDT.calculateClassPriors();
            auto root_node = trainingDT.constructDecisionTreeClassifier();

            // Optional tree display
            if (evaldebug)
            {
                root_node->displayDecisionTree("     ");
            }

            // Classify test samples
            for (const auto &test_sample_name : testing_samples)
            {
                std::vector<std::string> test_sample_data =
                    all_training_data[test_sample_name];
                if (evaldebug)
                {
                    std::cout << "Original data in test sample: " << test_sample_name
                              << "\n";
                }

                // Remove 'NA' values
                test_sample_data.erase(
                    std::remove_if(test_sample_data.begin(), test_sample_data.end(),
                                   [](const std::string &val)
                                   {
                                       return val.find("=NA") != std::string::npos;
                                   }),
                    test_sample_data.end());

                if (evaldebug)
                {
                    std::cout << "Data in test sample after removing 'NA' values: "
                              << test_sample_data << "\n";
                }
                auto classification = trainingDT.classify(root_node, test_sample_data);
                std::vector<std::string> which_classes;
                for (const auto &pair : classification)
                {
                    if (pair.first != "solution_path")
                    {
                        which_classes.push_back(pair.first);
                    }
                }
                std::sort(
                    which_classes.begin(), which_classes.end(),
                    [&classification](const std::string &a, const std::string &b)
                    {
                        return classification.at(a) > classification.at(b);
                    });

                std::string most_likely_class_label = which_classes[0];
                if (evaldebug)
                {
                    printClassificationInfo(which_classes, classification,
                                            most_likely_class_label, root_node);
                }

                std::string true_class_label = _samplesClassLabelDict[test_sample_name];
                confusion_matrix[true_class_label][most_likely_class_label] += 1;
            }
        }

        // Display confusion matrix
        displayConfusionMatrix(confusion_matrix);

        // Calculate data quality index
        double data_quality_index = calculateDataQualityIndex(confusion_matrix);
        std::cout << "\nTraining Data Quality Index: " << data_quality_index
                  << " (out of a maximum of 100)\n";
        printDataQualityEvaluation(data_quality_index);

        // update the data quality index
        _dataQualityIndex = data_quality_index;
    }

    double _dataQualityIndex;

private:
    // methods to print information << NEEDS TO BE IMPLEMENTED >>
    void printDebugInformation(DecisionTree &trainingDT,
                               const std::vector<std::string> &testing_samples)
    {
        // if evaldebug:
        //         print("\n\nprinting samples in the testing set: " +
        //         str(testing_samples)) print("\n\nPrinting features and their
        //         values in the training set:\n") for item in
        //         sorted(trainingDT._features_and_values_dict.items()):
        //             print(item[0]  + "  =>  "  + str(item[1]))
        //         print("\n\nPrinting unique values for features:\n")
        //         for item in
        //         sorted(trainingDT._features_and_unique_values_dict.items()):
        //             print(item[0]  + "  =>  "  + str(item[1]))
        //         print("\n\nPrinting unique value ranges for features:\n")
        //         for item in
        //         sorted(trainingDT._numeric_features_valuerange_dict.items()):
        //             print(item[0]  + "  =>  "  + str(item[1]))

        std::cout << "\n\nPrinting samples in the testing set: " << testing_samples
                  << "\n";
        std::cout
            << "\n\nPrinting features and their values in the training set:\n";
        for (const auto &item : trainingDT._featuresAndValuesDict)
        {
            std::cout << item.first << "  =>  " << item.second << "\n";
        }
        std::cout << "\n\nPrinting unique values for features:\n";
        for (const auto &item : trainingDT._featuresAndUniqueValuesDict)
        {
            std::cout << item.first << "  =>  " << item.second << "\n";
        }
        std::cout << "\n\nPrinting unique value ranges for features:\n";
        for (const auto &item : trainingDT._numericFeaturesValueRangeDict)
        {
            std::cout << item.first << "  =>  " << item.second << "\n";
        }
    }

    void printClassificationInfo(
        const std::vector<std::string> &which_classes,
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
        for (const auto &which_class : which_classes)
        {
            if (which_class != "solution_path")
            {
                std::cout << "     " << std::setw(30) << which_class
                          << classification.at(which_class) << "\n";
            }
        }
        std::cout << "\nSolution path in the decision tree: "
                  << classification.at("solution_path") << "\n";
        std::cout << "\nNumber of nodes created: " << root_node->HowManyNodes()
                  << "\n";
    }
    void displayConfusionMatrix(
        const std::map<std::string, std::map<std::string, int>> &
            confusion_matrix)
    {
        //   print("\n\n       DISPLAYING THE CONFUSION MATRIX FOR THE 10-FOLD
        //   CROSS-VALIDATION TEST:\n")
        // matrix_header = " " * 30
        // for class_name in self._class_names:
        //     matrix_header += '{:^30}'.format(class_name)
        // print("\n" + matrix_header + "\n")
        // for row_class_name in sorted(confusion_matrix.keys()):
        //     row_display = str.rjust(row_class_name, 30)
        //     for col_class_name in
        //     sorted(confusion_matrix[row_class_name].keys()):
        //         row_display +=
        //         '{:^30}'.format(str(confusion_matrix[row_class_name][col_class_name])
        //         )
        //     print(row_display + "\n")
        // diagonal_sum, off_diagonal_sum = 0,0
        // for row_class_name in sorted(confusion_matrix.keys()):
        //     for col_class_name in
        //     sorted(confusion_matrix[row_class_name].keys()):
        //         if row_class_name == col_class_name:
        //             diagonal_sum +=
        //             confusion_matrix[row_class_name][col_class_name]
        //         else:
        //             off_diagonal_sum +=
        //             confusion_matrix[row_class_name][col_class_name]
        std::cout << "\n\n       DISPLAYING THE CONFUSION MATRIX FOR THE 10-FOLD "
                     "CROSS-VALIDATION TEST:\n";
        std::string matrix_header = std::string(30, ' ');
        for (const auto &class_name : _classNames)
        {
            matrix_header += class_name;
        }
        std::cout << "\n"
                  << matrix_header << "\n";
        for (const auto &row_class_name : _classNames)
        {
            std::string row_display = std::string(30, ' ');
            row_display += row_class_name;
            for (const auto &col_class_name : _classNames)
            {
                row_display += std::to_string(
                    confusion_matrix.at(row_class_name).at(col_class_name));
            }
            std::cout << row_display << "\n";
        }
    };
    double calculateDataQualityIndex(const std::map<std::string, std::map<std::string, int>> &confusion_matrix)
    {
        int diagonal_sum = 0, off_diagonal_sum = 0;
        for (const auto &row_class_name : _classNames)
        {
            for (const auto &col_class_name : _classNames)
            {
                if (row_class_name == col_class_name)
                {
                    diagonal_sum +=
                        confusion_matrix.at(row_class_name).at(col_class_name);
                }
                else
                {
                    off_diagonal_sum +=
                        confusion_matrix.at(row_class_name).at(col_class_name);
                }
            }
        }
        return 100.0 * diagonal_sum / (diagonal_sum + off_diagonal_sum);
    }

    void printDataQualityEvaluation(double data_quality_index)
    {
        std::cout << "\nTraining Data Quality Index: " << data_quality_index
                  << "   (out of a possible maximum of 100)\n";
        if (data_quality_index <= 80)
        {
            std::cout
                << "\nYour training data does not possess much class "
                   "discriminatory "
                   "information.  It could be that the classes are inherently not "
                   "well "
                   "separable or that your constructor parameter choices are not "
                   "appropriate.\n";
        }
        else if (80 < data_quality_index <= 90)
        {
            std::cout << "\nYour training data possesses some class discriminatory "
                         "information "
                         "but it may not be sufficient for real-world applications.  "
                         "You might "
                         "try tweaking the constructor parameters to see if that "
                         "improves the "
                         "class discriminations.\n";
        }
        else if (90 < data_quality_index <= 95)
        {
            std::cout << "\nYour training data appears to possess good class "
                         "discriminatory "
                         "information.  Whether or not it is acceptable would depend "
                         "on your "
                         "application.\n";
        }
        else if (95 < data_quality_index < 98)
        {
            std::cout << "\nYour training data is of very high quality.\n";
        }
        else
        {
            std::cout << "\nYour training data is excellent.\n";
        }
    }
};