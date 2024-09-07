#include <iostream>
#include <map>
#include <vector>
#include <set>
#include <algorithm>
#include <memory>
#include <regex>
#include "DecisionTree.hpp" // Include necessary headers
#include "Utility.hpp"

class EvalTrainingData : public DecisionTree
{
public:
    // Constructor inheriting from the DecisionTree
    EvalTrainingData(std::map<std::string, std::string> kwargs) : DecisionTree(kwargs) {}

    // Method to evaluate training data
    void evaluateTrainingData()
    {
        bool evaldebug = false;

        // Check if the training data file is a CSV
        if (_trainingDatafile.substr(_trainingDatafile.find_last_of(".") + 1) != "csv")
        {
            throw std::runtime_error("The data evaluation function can only be used for CSV files.");
        }

        std::cout << "\nWill run a 10-fold cross-validation test on your training data...\n";
        auto all_training_data = _trainingDataDict;
        std::vector<std::string> all_sample_names;

        // Sort samples based on some index
        for (const auto &entry : all_training_data)
        {
            all_sample_names.push_back(entry.first);
        }
        std::sort(all_sample_names.begin(), all_sample_names.end(), [](const std::string &a, const std::string &b)
                  { return sampleIndex(a) < sampleIndex(b); });

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
            std::cout << "\nStarting fold " << fold_index << " of the 10-fold cross-validation test.\n";
            std::vector<std::string> testing_samples(all_sample_names.begin() + fold_size * fold_index,
                                                     all_sample_names.begin() + fold_size * (fold_index + 1));
            std::vector<std::string> training_samples(all_sample_names.begin(), all_sample_names.begin() + fold_size * fold_index);
            training_samples.insert(training_samples.end(),
                                    all_sample_names.begin() + fold_size * (fold_index + 1), all_sample_names.end());

            std::map<std::string, std::vector<std::string>> testing_data, training_data;
            for (const auto &sample : testing_samples)
            {
                testing_data[sample] = all_training_data[sample];
            }
            for (const auto &sample : training_samples)
            {
                training_data[sample] = all_training_data[sample];
            }

            DecisionTree trainingDT("evalmode");
            trainingDT.setTrainingDataDict(training_data);
            trainingDT._classNames = _classNames;
            trainingDT.setFeatureNames(_featureNames);
            trainingDT.setEntropyThreshold(_entropyThreshold);
            trainingDT.setMaxDepthDesired(_maxDepthDesired);
            trainingDT.setSymbolicToNumericCardinalityThreshold(_symbolicToNumericCardinalityThreshold);
            trainingDT.setSamplesClassLabelDict(_samplesClassLabelDict);

            // Populate features and values dict for training data
            trainingDT._featuresAndValuesDict.clear();
            std::regex pattern(R"((\S+)\s*=\s*(\S+))");
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
                            trainingDT._featuresAndValuesDict[feature].push_back(convert(value));
                        }
                    }
                }
            }

            // Set unique values for features
            for (auto &pair : trainingDT._featuresAndValuesDict)
            {
                std::set<std::string> unique_values(pair.second.begin(), pair.second.end());
                trainingDT._featuresAndUniqueValuesDict[pair.first] = std::vector<std::string>(unique_values.begin(), unique_values.end());
            }

            // Calculate numeric value ranges for features
            trainingDT._numericFeaturesValueRangeDict.clear();
            for (const auto &feature : _numericFeaturesValueRangeDict)
            {
                trainingDT._numericFeaturesValueRangeDict[feature.first] = {
                    *std::min_element(trainingDT._featuresAndUniqueValuesDict[feature.first].begin(),
                                      trainingDT._featuresAndUniqueValuesDict[feature.first].end()),
                    *std::max_element(trainingDT._featuresAndUniqueValuesDict[feature.first].begin(),
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
                std::vector<std::string> test_sample_data = all_training_data[test_sample_name];
                if (evaldebug)
                {
                    std::cout << "Original data in test sample: " << test_sample_name << "\n";
                }

                // Remove 'NA' values
                test_sample_data.erase(std::remove_if(test_sample_data.begin(), test_sample_data.end(),
                                                      [](const std::string &val)
                                                      { return val.find("=NA") != std::string::npos; }),
                                       test_sample_data.end());

                auto classification = trainingDT.classify(root_node, test_sample_data);
                std::vector<std::string> which_classes;
                for (const auto &pair : classification)
                {
                    if (pair.first != "solution_path")
                    {
                        which_classes.push_back(pair.first);
                    }
                }
                std::sort(which_classes.begin(), which_classes.end(),
                          [&classification](const std::string &a, const std::string &b)
                          {
                              return classification.at(a) > classification.at(b);
                          });

                std::string most_likely_class_label = which_classes[0];
                if (evaldebug)
                {
                    printClassificationInfo(which_classes, classification, most_likely_class_label, root_node);
                }

                std::string true_class_label = _samplesClassLabelDict[test_sample_name];
                confusion_matrix[true_class_label][most_likely_class_label] += 1;
            }
        }

        // Display confusion matrix
        displayConfusionMatrix(confusion_matrix);

        // Calculate data quality index
        double data_quality_index = calculateDataQualityIndex(confusion_matrix);
        std::cout << "\nTraining Data Quality Index: " << data_quality_index << " (out of a maximum of 100)\n";
        printDataQualityEvaluation(data_quality_index);
    }

private:
    void printDebugInformation(DecisionTree &trainingDT, const std::vector<std::string> &testing_samples);
    void printClassificationInfo(const std::vector<std::string> &which_classes,
                                 const std::map<std::string, double> &classification,
                                 const std::string &most_likely_class_label,
                                 std::shared_ptr<DecisionTreeNode> root_node);
    void displayConfusionMatrix(const std::map<std::string, std::map<std::string, int>> &confusion_matrix);
    double calculateDataQualityIndex(const std::map<std::string, std::map<std::string, int>> &confusion_matrix);
    void printDataQualityEvaluation(double data_quality_index);
};
