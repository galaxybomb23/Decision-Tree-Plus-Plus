#include "EvalTrainingData.hpp"

// Constructor inheriting from the DecisionTree
EvalTrainingData::EvalTrainingData(std::map<std::string, std::string> kwargs) : DecisionTree(kwargs) {}
EvalTrainingData::~EvalTrainingData()
{
    // Destructor
}

// Method to evaluate training data
double EvalTrainingData::evaluateTrainingData()
{
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
        // std::cout << "Sample name: " << ent << "\n";
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
        std::cout << "\nStarting the iteration indexed " << foldIndex << " of the 10-fold cross-validation test\n";

        // Define testing and training samples
        auto testingSamplesStart = allSampleNames.begin() + foldSize * foldIndex;
        auto testingSamplesEnd   = allSampleNames.begin() + foldSize * (foldIndex + 1);
        std::vector<std::string> testingSamples(testingSamplesStart, testingSamplesEnd);

        std::vector<std::string> trainingSamples(allSampleNames.begin(), testingSamplesStart);
        trainingSamples.insert(trainingSamples.end(), testingSamplesEnd, allSampleNames.end());

        // Create testing and training data
        std::map<int, std::vector<std::string>> testingData, trainingData;
        for (const auto &sample : testingSamples) {
            auto samp         = std::stoi(sample);
            testingData[samp] = allTrainingData[samp];
        }
        for (const auto &sample : trainingSamples) {
            auto samp          = std::stoi(sample);
            trainingData[samp] = allTrainingData[samp];
        }

        // print training data
        std::cout << "Training data:\n";
        for (const auto &entry : trainingData) {
            std::cout << entry.first << ": ";
            for (const auto &feature : entry.second) {
                std::cout << feature << " ";
            }
            std::cout << "\n";
        }

        // Initialize DecisionTree
        map<string, string> kwargs = {
            {"training_datafile", _trainingDatafile}
        };
        shared_ptr<DecisionTree> trainingDT                = make_unique<DecisionTree>(kwargs);
        trainingDT->_trainingDataDict                      = trainingData;
        trainingDT->_classNames                            = _classNames;
        trainingDT->_featureNames                          = _featureNames;
        trainingDT->_entropyThreshold                      = _entropyThreshold;
        trainingDT->_maxDepthDesired                       = _maxDepthDesired;
        trainingDT->_symbolicToNumericCardinalityThreshold = _symbolicToNumericCardinalityThreshold;

        // Assign samples class labels
        for (const auto &sample : trainingSamples) {
            trainingDT->_samplesClassLabelDict[std::stod(sample)] = _samplesClassLabelDict.at(std::stod(sample));
        }

        // Populate feature and values dictionary
        trainingDT->_featuresAndValuesDict.clear();
        int idx = 0;
        for (const auto &item : trainingDT->_trainingDataDict) {
            for (const auto &feature_and_value : item.second) {
                std::string feature = trainingDT->_featureNames[idx % trainingDT->_featureNames.size()];
                std::string value   = feature_and_value;

                if (value != "NA") {
                    trainingDT->_featuresAndValuesDict[feature].push_back(value);
                }
                idx++;
            }
        }

        // Calculate unique values for each feature
        trainingDT->_featuresAndUniqueValuesDict.clear();
        for (const auto &pair : trainingDT->_featuresAndValuesDict) {
            std::set<std::string> unique_values(pair.second.begin(), pair.second.end());
            trainingDT->_featuresAndUniqueValuesDict[pair.first] =
                std::set<std::string>(unique_values.begin(), unique_values.end());
        }

        // Calculate numeric feature value ranges
        trainingDT->_numericFeaturesValueRangeDict.clear();
        for (const auto &feature : _numericFeaturesValueRangeDict) {
            std::set<double> numeric_values;
            for (const auto &value : feature.second) {
                try {
                    numeric_values.insert(value);
                }
                catch (const std::invalid_argument &e) {
                    continue;
                }
            }
            if (!numeric_values.empty()) {
                std::vector<double> numeric_values_vec(numeric_values.begin(), numeric_values.end());
                std::sort(numeric_values_vec.begin(), numeric_values_vec.end());
                trainingDT->_numericFeaturesValueRangeDict[feature.first] = {numeric_values_vec.front(),
                                                                             numeric_values_vec.back()};
            }
        }

        // Calculate probabilities and construct decision tree
        cout << "Class names are: ";
        for (const auto &class_name : trainingDT->_classNames) {
            cout << class_name << " ";
        }
        trainingDT->calculateFirstOrderProbabilities();
        trainingDT->calculateClassPriors();
        auto rootNode = trainingDT->constructDecisionTreeClassifier();

        std::cout << "\nResults of the 10-fold cross-validation test for run indexed " << foldIndex + 1 << ":\n";
        for (const auto &testSampleName : testingSamples) {
            auto testSampleDataUnfiltered = allTrainingData[std::stoi(testSampleName)];
            std::vector<std::string> testSampleData;

            for (size_t idx = 0; idx < testSampleDataUnfiltered.size(); ++idx) {
                const auto &data = testSampleDataUnfiltered[idx];
                if (!data.empty() && data != "NA") {
                    testSampleData.push_back(trainingDT->_featureNames[idx % trainingDT->_featureNames.size()] + "=" +
                                             data);
                }
            }

            // print test sample data
            std::cout << "\n\nTest sample data:\n";
            for (const auto &feature : testSampleData) {
                std::cout << feature << " ";
            }

            auto classification = trainingDT->classify(rootNode, testSampleData);
            auto solutionPath   = classification["solution_path"];

            // print classification info and solution path
            // print sample number
            std::cout << "\n\nSample number: " << testSampleName << "\n";
            printClassificationInfo(trainingDT->_classNames, classification, solutionPath, rootNode);

            classification.erase("solution_path");

            std::vector<std::string> whichClasses;
            for (const auto &entry : classification) {
                whichClasses.push_back(entry.first);
            }

            std::sort(whichClasses.begin(),
                      whichClasses.end(),
                      [&classification](const std::string &a, const std::string &b) {
                          return classification.at(a) > classification.at(b);
                      });

            auto mostLikelyClassLabel = whichClasses.front();
            auto trueClassLabel       = _samplesClassLabelDict.at(std::stoi(testSampleName));
            confusion_matrix[std::stoi(trueClassLabel)][mostLikelyClassLabel] += 1;
        }
    }

    // Display confusion matrix
    displayConfusionMatrix(confusion_matrix);
    auto idx = calculateDataQualityIndex(confusion_matrix);
    printDataQualityEvaluation(idx);
    return idx;
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

    std::cout << "\n\nPrinting samples in the testing set:";
    for (const auto &sample : testing_samples) {
        std::cout << sample << "\n";
    }
    std::cout << "\n\nPrinting features and their values in the training set:\n";
    for (const auto &item : _featuresAndValuesDict) {
        for (const auto &value : item.second) {
            std::cout << item.first << "  =>  " << value << "\n";
        }
    }
    std::cout << "\n\nPrinting unique values for features:\n";
    for (const auto &item : _featuresAndUniqueValuesDict) {
        for (const auto &value : item.second) {
            std::cout << item.first << "  =>  " << value << "\n";
        }
    }
    std::cout << "\n\nPrinting unique value ranges for features:\n";
    for (const auto &item : _numericFeaturesValueRangeDict) {
        std::cout << item.first << "  =>  " << item.second[0] << " - " << item.second[1] << "\n";
    }
}

void EvalTrainingData::printClassificationInfo(const std::vector<std::string> &which_classes,
                                               const std::map<std::string, std::string> &classification,
                                               const std::string &solution_path,
                                               DecisionTreeNode* root_node)
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
    std::cout << "\nClassification for sample:\n";
    std::cout << "     " << std::setw(30) << "class name" << "  probability\n";
    std::cout << "     ----------                    -----------\n";
    for (const auto &which_class : which_classes) {
        if (which_class != "solution_path") {
            std::cout << "     " << std::setw(30) << which_class << " " << classification.at(which_class) << "\n";
        }
    }
    std::cout << "\nSolution path in the decision tree: " << classification.at("solution_path") << "\n";
    std::cout << "\nNumber of nodes created: " << root_node->HowManyNodes() << "\n";
}
void EvalTrainingData::displayConfusionMatrix(const std::map<int, std::map<std::string, int>> &confusion_matrix)
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
    for (const auto &class_name : _classNames) {
        matrix_header += class_name;
    }
    std::cout << "\n" << matrix_header << "\n";
    for (const auto &row_class_name : _classNames) {
        std::string row_display = std::string(30, ' ');
        row_display += row_class_name;
        for (const auto &col_class_name : _classNames) {
            row_display += std::to_string(confusion_matrix.at(std::stoi(row_class_name)).at(col_class_name));
        }
        std::cout << row_display << "\n";
    }
}
double EvalTrainingData::calculateDataQualityIndex(const std::map<int, std::map<std::string, int>> &confusion_matrix)
{
    int diagonal_sum = 0, off_diagonal_sum = 0;
    for (const auto &row_class_name : _classNames) {
        for (const auto &col_class_name : _classNames) {
            if (row_class_name == col_class_name) {
                diagonal_sum += confusion_matrix.at(std::stoi(row_class_name)).at(col_class_name);
            }
            else {
                off_diagonal_sum += confusion_matrix.at(std::stoi(row_class_name)).at(col_class_name);
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