#include "EvalTrainingData.hpp"

// Constructor inheriting from the DecisionTree
EvalTrainingData::EvalTrainingData(std::map<std::string, std::string> kwargs) : DecisionTree(kwargs) {}
EvalTrainingData::~EvalTrainingData()
{
    // Destructor
}

// Method to evaluate training data
/**
 * @brief Evaluates the training data using 10-fold cross-validation.
 *
 * This function performs a 10-fold cross-validation on the training data to evaluate
 * the performance of a decision tree classifier. It checks if the training data file
 * is in CSV format, splits the data into training and testing sets, trains the decision
 * tree, and evaluates its performance on the testing set. The results are stored in a
 * confusion matrix, which is used to calculate and display the data quality index.
 *
 * @return double The data quality index calculated from the confusion matrix.
 *
 * @throws std::runtime_error If the training data file is not a CSV file.
 */
double EvalTrainingData::evaluateTrainingData()
{
    bool evalDebug = false;

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
        allSampleNames.push_back(ent);
    }

    // Sort the samples
    std::sort(allSampleNames.begin(), allSampleNames.end(), [](const std::string &a, const std::string &b) {
        return std::stoi(a) < std::stoi(b);
    });

    // fold size is 10% of the training data
    int foldSize = static_cast<int>(0.1 * allTrainingData.size());
    std::map<int, std::map<std::string, int>> confusion_matrix;

    // Initialize confusion matrix
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
        auto testingSamplesStart = allSampleNames.begin() + static_cast<long>(foldSize) * foldIndex;
        auto testingSamplesEnd   = allSampleNames.begin() + static_cast<long>(foldSize) * (foldIndex + 1);
        std::vector<std::string> testingSamples(testingSamplesStart, testingSamplesEnd);

        // Combine the training samples
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

        // Initialize DecisionTree and class variables
        map<string, string> kwargs = {
            {"training_datafile", _trainingDatafile}
        };
        shared_ptr<DecisionTree> trainingDT                = make_unique<DecisionTree>(kwargs);
        trainingDT->_trainingDataDict                      = _trainingDataDict;
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
                catch (const std::invalid_argument &e) { // ignore invalid values
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

        if (evalDebug) {
            printDebugInformation(*trainingDT, testingSamples);
        }

        if (evalDebug) {
            trainingDT->_debug2 = true;
        }

        // We have the training data, calculate probabilities and priors
        trainingDT->calculateFirstOrderProbabilities();
        trainingDT->calculateClassPriors();

        // Construct the decision tree classifier
        auto rootNode = trainingDT->constructDecisionTreeClassifier();
        if (evalDebug) {
            trainingDT->getRootNode()->DisplayDecisionTree("    ");
        }

        // Show the classification results
        std::cout << "\nResults of the 10-fold cross-validation test for run indexed " << foldIndex + 1 << ":\n";
        for (const auto &testSampleName : testingSamples) {
            auto testSampleDataUnfiltered = allTrainingData[std::stoi(testSampleName)];

            // Filter out empty and NA values from the test sample data
            std::vector<std::string> testSampleData;
            for (size_t idx = 0; idx < testSampleDataUnfiltered.size(); ++idx) {
                const auto &data = testSampleDataUnfiltered[idx];
                if (!data.empty() && data != "NA") {
                    testSampleData.push_back(trainingDT->_featureNames[idx % trainingDT->_featureNames.size()] + "=" +
                                             data);
                }
            }

            if (evalDebug) {
                std::cout << "Data in test sample: ";
                for (const auto &data : testSampleData) {
                    std::cout << data << " ";
                }
            }

            auto classification = trainingDT->classify(rootNode, testSampleData);
            auto solutionPath   = classification["solution_path"];

            // print classification info and solution path
            printClassificationInfo(trainingDT->_classNames, classification, solutionPath, rootNode);
            classification.erase("solution_path");

            // Get the most likely class label
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

            if (evalDebug) {
                std::cout << "\n"
                          << testSampleName << ":   true_class: " << trueClassLabel
                          << "    estimated_class: " << mostLikelyClassLabel << "\n";
            }

            // Update confusion matrix with the classification results
            confusion_matrix[std::stoi(trueClassLabel)][mostLikelyClassLabel] += 1;
        }
    }

    // Display confusion matrix
    if (_debug1) {
        displayConfusionMatrix(confusion_matrix);
    }
    auto idx = calculateDataQualityIndex(confusion_matrix);
    printDataQualityEvaluation(idx);
    return idx;
}

/**
 * @brief Prints debug information for the training and testing data.
 *
 * This function outputs various debug information to the standard output, including:
 * - The samples in the testing set.
 * - The features and their values in the training set.
 * - The unique values for each feature.
 * - The unique value ranges for numeric features.
 *
 * @param trainingDT A reference to the DecisionTree object used for training.
 * @param testing_samples A vector of strings containing the testing samples.
 */
void EvalTrainingData::printDebugInformation(DecisionTree &trainingDT, const std::vector<std::string> &testing_samples)
{
    std::cout << "\n\nPrinting samples in the testing set:\n";
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

/**
 * @brief Prints classification information for a given sample.
 *
 * This function outputs the classification information for a sample, including the class names,
 * their associated probabilities, the solution path in the decision tree, and the number of nodes created.
 *
 * @param which_classes A vector of strings representing the class names to be printed.
 * @param classification A map where the key is the class name and the value is the classification result.
 * @param solution_path A string representing the path to the solution.
 * @param root_node A pointer to the root node of the decision tree.
 */
void EvalTrainingData::printClassificationInfo(const std::vector<std::string> &which_classes,
                                               const std::map<std::string, std::string> &classification,
                                               const std::string &solution_path,
                                               DecisionTreeNode* root_node)
{
    std::cout << "\nClassification for sample:\n";
    std::cout << "     " << std::setw(30) << std::left << "Class Name" << std::setw(14) << "Probability"
              << "\n";
    std::cout << "     " << std::string(28, '-') << "  " << std::string(15, '-') << "\n";

    for (const auto &which_class : which_classes) {
        if (which_class != "solution_path") {
            std::cout << "     " << std::setw(30) << std::left << which_class << std::setw(15)
                      << classification.at(which_class) << "\n";
        }
    }

    std::cout << "\nSolution path in the decision tree: " << classification.at("solution_path") << "\n";
    std::cout << "\nNumber of nodes created: " << root_node->HowManyNodes() << "\n";
}


/**
 * @brief Displays the confusion matrix.
 *
 * This function takes a confusion matrix as input and displays it. The confusion matrix
 * is represented as a nested map where the outer map's key is an integer representing
 * the actual class, and the inner map's key is a string representing the predicted class.
 * The value in the inner map is the count of occurrences for the actual-predicted class pair.
 *
 * @param confusion_matrix A nested map representing the confusion matrix.
 *                         The outer map's key is an integer (actual class),
 *                         and the inner map's key is a string (predicted class).
 *                         The value is the count of occurrences.
 */
void EvalTrainingData::displayConfusionMatrix(const std::map<int, std::map<std::string, int>> &confusion_matrix)
{
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

/**
 * @brief Calculates the Data Quality Index (DQI) based on the provided confusion matrix.
 *
 * This function evaluates the quality of the training data by analyzing the confusion matrix,
 * which contains the counts of true positives, false positives, true negatives, and false negatives
 * for each class.
 *
 * @param confusion_matrix A nested map where the outer map's key is the class label (int) and the value
 *                         is another map. The inner map's key is a string representing the type of count
 *                         ("TP" for true positives, "FP" for false positives, "TN" for true negatives,
 *                         "FN" for false negatives) and the value is the count (int).
 * @return double The calculated Data Quality Index (DQI) as a double.
 */
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

/**
 * @brief Prints the evaluation of the training data quality based on the provided index.
 *
 * This function outputs a message to the standard output stream that describes the quality
 * of the training data based on the given data quality index. The index is expected to be
 * a value between 0 and 100.
 *
 * @param data_quality_index A double representing the quality index of the training data.
 *                           The value should be between 0 and 100.
 *
 * The function provides different messages based on the range in which the data quality
 * index falls:
 * - If the index is less than or equal to 80, it indicates poor class discriminatory information.
 * - If the index is between 80 and 90, it indicates some class discriminatory information but may not be
 * sufficient.
 * - If the index is between 90 and 95, it indicates good class discriminatory information.
 * - If the index is between 95 and 98, it indicates very high-quality training data.
 * - If the index is greater than or equal to 98, it indicates excellent training data.
 */
void EvalTrainingData::printDataQualityEvaluation(double data_quality_index)
{
    std::cout << "\nTraining Data Quality Index: " << data_quality_index << "   (out of a possible maximum of 100)\n";

    if (data_quality_index <= 80) {
        std::cout << "\nYour training data does not possess much class "
                     "discriminatory information. It could be that the classes are "
                     "inherently not well separable or that your constructor "
                     "parameter choices are not appropriate.\n";
    }
    else if (data_quality_index > 80 && data_quality_index <= 90) {
        std::cout << "\nYour training data possesses some class discriminatory "
                     "information but it may not be sufficient for real-world "
                     "applications. You might try tweaking the constructor "
                     "parameters to see if that improves the class discriminations.\n";
    }
    else if (data_quality_index > 90 && data_quality_index <= 95) {
        std::cout << "\nYour training data appears to possess good class "
                     "discriminatory information. Whether or not it is acceptable "
                     "would depend on your application.\n";
    }
    else if (data_quality_index > 95 && data_quality_index < 98) {
        std::cout << "\nYour training data is of very high quality.\n";
    }
    else {
        std::cout << "\nYour training data is excellent.\n";
    }
}
