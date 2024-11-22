// Include
#include "DecisionTree.hpp"

#include "Utility.hpp"
#include "logger.cpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <numeric>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>


// --------------- Logger --------------- //
Logger logger("../logs/decisionTree.log");

//--------------- Constructors and Destructors ----------------//
DecisionTree::DecisionTree(map<string, string> kwargs)
{
    if (kwargs.empty()) {
        throw std::invalid_argument("Missing training datafile.");
    }

    // Allowed keys for the kwargs
    vector<string> allowedKeys = {"training_datafile",
                                  "entropy_threshold",
                                  "max_depth_desired",
                                  "csv_class_column_index",
                                  "symbolic_to_numeric_cardinality_threshold",
                                  "csv_columns_for_features",
                                  "number_of_histogram_bins",
                                  "csv_cleanup_needed",
                                  "debug1",
                                  "debug2",
                                  "debug3"};

    // Set default values
    _entropyThreshold                      = 0.01;
    _symbolicToNumericCardinalityThreshold = 10;
    _csvCleanupNeeded                      = 0;
    _csvColumnsForFeatures                 = {};
    _debug1 = _debug2 = _debug3 = 0;
    _maxDepthDesired = _csvClassColumnIndex = _numberOfHistogramBins = -1;
    _rootNode                                                        = nullptr;
    _howManyTotalTrainingSamples                                     = 0;
    _probabilityCache                                                = {};
    _entropyCache                                                    = {};
    _trainingDataDict                                                = {};
    _featuresAndValuesDict                                           = {};
    _featuresAndUniqueValuesDict                                     = {};
    _samplesClassLabelDict                                           = {};
    _classNames                                                      = {};
    _classPriorsDict                                                 = {};
    _featureNames                                                    = {};
    _numericFeaturesValueRangeDict                                   = {};
    _samplingPointsForNumericFeatureDict                             = {};
    _featureValuesHowManyUniquesDict                                 = {};
    _probDistributionNumericFeaturesDict                             = {};
    _histogramDeltaDict                                              = {};
    _numOfHistogramBinsDict                                          = {};

    // Check and set keyword arguments
    for (const auto &kv : kwargs) {
        const string &key   = kv.first;
        const string &value = kv.second;

        if (key == "training_datafile") {
            _trainingDatafile = value;
        }
        else if (key == "entropy_threshold") {
            _entropyThreshold = std::stod(value);
        }
        else if (key == "max_depth_desired") {
            _maxDepthDesired = std::stoi(value);
        }
        else if (key == "csv_class_column_index") {
            _csvClassColumnIndex = std::stoi(value);
        }
        else if (key == "csv_columns_for_features") {
            for (const auto &count : value) {
                _csvColumnsForFeatures.push_back(count);
            }
        }
        else if (key == "symbolic_to_numeric_cardinality_threshold") {
            _symbolicToNumericCardinalityThreshold = std::stoi(value);
        }
        else if (key == "number_of_histogram_bins") {
            _numberOfHistogramBins = std::stoi(value);
        }
        else if (key == "csv_cleanup_needed") {
            _csvCleanupNeeded = std::stoi(value);
        }
        else if (key == "debug1") {
            _debug1 = std::stoi(value);
        }
        else if (key == "debug2") {
            _debug2 = std::stoi(value);
        }
        else if (key == "debug3") {
            _debug3 = std::stoi(value);
        }
        else {
            throw std::invalid_argument(key + ": Wrong keyword used --- check spelling");
        }
    }
}

DecisionTree::~DecisionTree() {}

//--------------- Functions ----------------//

// Get the training data from the CSV file
void DecisionTree::getTrainingData()
{
    // Check if training data file is a CSV file
    if (_trainingDatafile.find(".csv") == string::npos) { // string.find() returns string::npos if not found
        throw std::invalid_argument("Aborted. get_training_data_from_csv() is only for CSV files");
    }

    _classNames = {};

    // Open the file
    std::ifstream file(_trainingDatafile); // std::ifstream is used to read input from a file
    if (!file.is_open()) {
        throw std::invalid_argument("Could not open file: " + _trainingDatafile);
    }

    // Read the header
    string line;
    if (std::getline(file, line)) {
        std::istringstream ss(line);
        string token;
        int columnIdx = 0; // Index of the column
        while (std::getline(ss, token, ',')) {
            // strip leading/trailing whitespaces and \" from the token
            token.erase(0, token.find_first_not_of(" \""));
            token.erase(token.find_last_not_of(" \"") + 1);
            // Check if the column is a class column, if not, add it to the feature columns
            if (std::find(_csvColumnsForFeatures.begin(), _csvColumnsForFeatures.end(), columnIdx) !=
                _csvColumnsForFeatures.end()) {
                _featureNames.push_back(token); // Get the feature names
            }
            columnIdx++;
        }
    }

    // Read the data
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        string token;
        vector<string> row;
        int columnIdx = 0;
        int uniqueId;
        string className;

        while (std::getline(ss, token, ',')) {
            // strip leading/trailing whitespaces and \" from the token
            token.erase(0, token.find_first_not_of(" \""));
            token.erase(token.find_last_not_of(" \"") + 1);

            // If the column is the idx column, set the uniqueId
            if (columnIdx == 0) {
                uniqueId = std::stoi(token);
            }
            // If the column is the class column, set the className
            else if (columnIdx == _csvClassColumnIndex) {
                className = token;
            }
            // If the column is a feature column, add the token to the row
            else if (std::find(_csvColumnsForFeatures.begin(), _csvColumnsForFeatures.end(), columnIdx) !=
                     _csvColumnsForFeatures.end()) {
                row.push_back(token);
            }
            columnIdx++;
        }

        _trainingDataDict[uniqueId]      = row;
        _samplesClassLabelDict[uniqueId] = className;
        _classNames.push_back(className);
    }

    // Close the file
    file.close();

    // Get the unique class labels
    std::sort(_classNames.begin(), _classNames.end());
    _classNames.erase(std::unique(_classNames.begin(), _classNames.end()), _classNames.end());

    // Get the number of unique class labels
    int numUniqueClassLabels = _classNames.size();

    // Get the number of training samples
    _howManyTotalTrainingSamples = _trainingDataDict.size();

    // Get the features and their values
    for (int i = 0; i < _featureNames.size(); i++) {
        vector<string> allValues;      // All values for the feature
        std::set<string> uniqueValues; // Unique values for the feature
        for (const auto &kv : _trainingDataDict) {
            allValues.push_back(kv.second[i]); // Insert the value into the vector
            uniqueValues.insert(kv.second[i]); // Insert the value into the set
        }
        _featuresAndValuesDict[_featureNames[i]]       = allValues;
        _featuresAndUniqueValuesDict[_featureNames[i]] = uniqueValues;
    }

    // Get the number of unique values for each feature
    for (const auto &kv : _featuresAndUniqueValuesDict) {
        _featureValuesHowManyUniquesDict[kv.first] = kv.second.size();
    }

    // Get the _numericFeaturesValuerangeDict
    for (const auto &feature : _featureNames) {
        // Get the min and max values of the feature and store them in a vector
        vector<double> values;
        double min = std::numeric_limits<double>::max();
        double max = std::numeric_limits<double>::min();
        for (const auto &value : _featuresAndUniqueValuesDict[feature]) {
            double numericValue = convert(value);
            if (std::isnan(numericValue)) {
                continue;
            }
            if (numericValue < min) {
                min = numericValue;
            }
            if (numericValue > max) {
                max = numericValue;
            }
        }
        if (min == std::numeric_limits<double>::max() && max == std::numeric_limits<double>::min()) {
            continue;
        }
        values.push_back(min);
        values.push_back(max);
        _numericFeaturesValueRangeDict[feature] = values;
    }
}

// Calculate first order probabilities
void DecisionTree::calculateFirstOrderProbabilities()
{
    cout << "\nEstimating probabilities...\n";

    for (const auto &feature : _featureNames) {
        // Calculate probability for the feature's value
        probabilityOfFeatureValue(feature, "");

        // Debug output if debug2 is enabled
        if (_debug2) {
            // Check if the feature has a probability distribution for numeric values
            if (_probDistributionNumericFeaturesDict.find(feature) != _probDistributionNumericFeaturesDict.end()) {
                cout << "\nPresenting probability distribution for a feature "
                        "considered to be numeric:\n";
                // Output sorted sampling points and their probabilities
                for (auto it = _probDistributionNumericFeaturesDict[feature].begin();
                     it != _probDistributionNumericFeaturesDict[feature].end();
                     ++it) {
                    string samplingPoint = std::to_string(it->first);

                    double prob = probabilityOfFeatureValue(feature, samplingPoint);
                    cout << feature << "::" << samplingPoint << " = " << std::setprecision(5) << prob << "\n";
                }
            }
            else {
                // Output probabilities for symbolic feature values
                cout << "\nPresenting probabilities for the values of a feature "
                        "considered to be symbolic:\n";
                const auto &values_for_feature = _featuresAndUniqueValuesDict[feature];
                for (const auto &value : values_for_feature) {
                    double prob = probabilityOfFeatureValue(feature, value);
                    cout << feature << "::" << value << " = " << std::setprecision(5) << prob << "\n";
                }
            }
        }
    }
}

// Show training data
void DecisionTree::showTrainingData() const
{
    for (const auto &kv : _trainingDataDict) {
        cout << kv.first << ": ";
        for (const auto &v : kv.second) {
            cout << v << " ";
        }
        cout << endl;
    }
}

//--------------- Classify ----------------//

map<string, string> DecisionTree::classify(DecisionTreeNode* rootNode, const vector<string> &featuresAndValues)
{
    /*
    Classifies one test sample at a time using the decision tree constructed from
    your training file.  The data record for the test sample must be supplied as
    shown in the scripts in the `Examples' subdirectory.  See the scripts
    construct_dt_and_classify_one_sample_caseX.py in that subdirectory.
    */
    if (!checkNamesUsed(featuresAndValues)) {
        throw std::runtime_error("\n\nError in the names you have used for features and/or values. "
                                 "Try using the csv_cleanup_needed option in the constructor call.");
    }

    vector<string> newFeaturesAndValues;
    std::regex pattern(R"((\S+)\s*=\s*(\S+))");
    std::smatch match;

    for (const auto &fv : featuresAndValues) {
        if (std::regex_match(fv, match, pattern)) {
            string feature = match[1];
            string value   = match[2];
            newFeaturesAndValues.push_back(feature + "=" + value);
        }
        else {
            throw std::runtime_error("\n\nError in the format of the feature and value pairs. "
                                     "Use the format feature=value.");
        }
    }

    // Update the features and values
    for (const auto &fv : newFeaturesAndValues) {
        string feature = fv.substr(0, fv.find("="));
        string value   = fv.substr(fv.find("=") + 1);
        _featuresAndValuesDict[feature].push_back(value);
    }

    if (_debug3) {
        cout << "\nCL1 New features and values:\n";
        for (const auto &item : newFeaturesAndValues) {
            cout << item << " ";
        }
    }

    map<string, vector<double>> answer;
    for (const auto &className : _classNames) {
        answer[className] = {};
    }
    answer["solution_path"] = {};

    map<string, double> classification = recursiveDescentForClassification(rootNode, newFeaturesAndValues, answer);
    std::reverse(answer["solution_path"].begin(), answer["solution_path"].end());

    if (_debug3) {
        cout << "\nCL2 The classification:" << endl;
        for (const auto &className : _classNames) {
            cout << "    " << className << " with probability " << classification[className] << endl;
        }
    }

    map<string, string> classificationForDisplay = {};
    for (const auto &kv : classification) {
        if (std::isfinite(kv.second)) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3) << kv.second;
            classificationForDisplay[kv.first] = oss.str();
        }
        else {
            vector<string> nodes;
            for (const auto &x : kv.first) {
                nodes.push_back("NODE" + std::to_string(x));
            }
            std::ostringstream oss;
            std::copy(nodes.begin(), nodes.end(), std::ostream_iterator<string>(oss, ", "));
            classificationForDisplay[kv.first] = oss.str();
        }
    }

    return classificationForDisplay;
}

map<string, double> DecisionTree::recursiveDescentForClassification(DecisionTreeNode* node,
                                                                    const vector<string> &featureAndValues,
                                                                    map<string, vector<double>> &answer)
{
    vector<shared_ptr<DecisionTreeNode>> children = node->GetChildren();

    if (children.empty()) {
        // If leaf node, assign class probabilities
        vector<double> leafNodeClassProbabilities = node->GetClassProbabilities();
        map<string, double> classProbabilities;
        for (size_t i = 0; i < _classNames.size(); ++i) {
            classProbabilities[_classNames[i]] = leafNodeClassProbabilities[i];
        }
        answer["solution_path"].push_back(node->GetNextSerialNum());
        return classProbabilities;
    }

    string featureTestedAtNode = node->GetFeature();
    if (_debug3) {
        cout << "\nCLRD1 Feature tested at node for classifcation: " << featureTestedAtNode << endl;
    }

    string valueForFeature;
    bool pathFound = false;
    std::regex pattern(R"((\S+)\s*=\s*(\S+))");
    std::smatch match;

    // Find the value for the feature being tested
    for (const auto &featureAndValue : featureAndValues) {
        if (std::regex_search(featureAndValue, match, pattern)) {
            string feature = match[1].str();
            string value   = match[2].str();
            if (feature == featureTestedAtNode) {
                valueForFeature = convert(value);
            }
        }
    }

    // Handle missing feature values
    if (valueForFeature.empty()) {
        vector<double> leafNodeClassProbabilities = node->GetClassProbabilities();
        map<string, double> classProbabilities;
        for (size_t i = 0; i < _classNames.size(); ++i) {
            classProbabilities[_classNames[i]] = leafNodeClassProbabilities[i];
        }
        answer["solution_path"].push_back(node->GetNextSerialNum());

        return classProbabilities;
    }

    // Numeric feature case
    if (_probDistributionNumericFeaturesDict.find(featureTestedAtNode) != _probDistributionNumericFeaturesDict.end()) {
        if (_debug3) {
            cout << "\nCLRD2 In the numeric section";
        }
        for (const auto &child : children) {
            vector<string> branchFeaturesAndValues = child->GetBranchFeaturesAndValuesOrThresholds();
            string lastFeatureAndValueOnBranch     = branchFeaturesAndValues.back();
            std::regex pattern1(R"((.+)<(.+))");
            std::regex pattern2(R"((.+)>(.+))");

            if (std::regex_search(lastFeatureAndValueOnBranch, match, pattern1)) {
                string threshold = match[2].str();
                if (std::stod(valueForFeature) <= std::stod(threshold)) {
                    pathFound   = true;
                    auto result = recursiveDescentForClassification(child.get(), featureAndValues, answer);
                    answer.insert(result.begin(), result.end());
                    answer["solution_path"].push_back(node->GetNextSerialNum());
                    break;
                }
            }
            else if (std::regex_search(lastFeatureAndValueOnBranch, match, pattern2)) {
                string threshold = match[2].str();
                if (std::stod(valueForFeature) > std::stod(threshold)) {
                    pathFound   = true;
                    auto result = recursiveDescentForClassification(child.get(), featureAndValues, answer);
                    answer.insert(result.begin(), result.end());
                    answer["solution_path"].push_back(node->GetNextSerialNum());
                    break;
                }
            }
        }

        if (pathFound) {
            map<string, double> result;
            for (const auto &kv : answer) {
                if (kv.first != "solution_path") {
                    result[kv.first] = kv.second.empty() ? 0.0 : kv.second[0];
                }
            }

            return result;
        }
    }
    else { // Symbolic feature case
        string featureValueCombo = featureTestedAtNode + "=" + valueForFeature;
        if (_debug3) {
            cout << "\nCLRD3 In the symbolic section with feature_value_combo: " << featureValueCombo;
        }

        for (const auto &child : children) {
            vector<string> branch_features_and_values = child->GetBranchFeaturesAndValuesOrThresholds();
            if (_debug3) {
                cout << "\nCLRD4 branch features and values: " << branch_features_and_values.back();
            }
            string lastFeatureAndValueOnBranch = branch_features_and_values.back();

            if (lastFeatureAndValueOnBranch == featureValueCombo) {
                auto result = recursiveDescentForClassification(child.get(), featureAndValues, answer);
                answer.insert(result.begin(), result.end());
                answer["solution_path"].push_back(node->GetNextSerialNum());
                pathFound = true;
                break;
            }
        }

        if (pathFound) {
            map<string, double> result;
            for (const auto &kv : answer) {
                if (kv.first != "solution_path") {
                    result[kv.first] = kv.second.empty() ? 0.0 : kv.second[0];
                }
            }

            return result;
        }
    }

    // If no path found, assign class probabilities from the current node
    if (!pathFound) {
        vector<double> leafNodeClassProbabilities = node->GetClassProbabilities();
        for (size_t i = 0; i < _classNames.size(); ++i) {
            answer[_classNames[i]].push_back(leafNodeClassProbabilities[i]);
        }
        answer["solution_path"].push_back(node->GetNextSerialNum());
    }

    map<string, double> result;
    for (const auto &kv : answer) {
        if (kv.first != "solution_path") {
            result[kv.first] = kv.second.empty() ? 0.0 : kv.second[0];
        }
    }

    return result;
}

//--------------- Construct Tree ----------------//

DecisionTreeNode* DecisionTree::constructDecisionTreeClassifier()
{
    /*
    Construct the root node object and set its entropy value as derived from the
    priors associated with the different classes.
    */
    cout << "\nConstructing a decision tree" << endl;

    if (_debug3) {
        // TODO //
        // determineDataCondition();
        cout << endl << "Starting construction of the decision tree:" << endl;
    }

    // Calculate prior class probabilities
    vector<double> classProbabilities;
    for (const auto &className : _classNames) {
        // TODO //
        // classProbabilities.push_back(priorProbabilityForClass(className));
    }

    if (_debug3) {
        cout << endl << "Prior probabilities for the classes:" << endl;
        for (size_t i = 0; i < _classNames.size(); ++i) {
            cout << "    " << _classNames[i] << " with probability " << classProbabilities[i] << endl;
        }
    }

    double entropy = classEntropyOnPriors();
    if (_debug3) {
        cout << endl << "Entropy on priors: " << entropy << endl;
    }

    // Create the root node
    DecisionTreeNode* rootNode = new DecisionTreeNode("root", entropy, classProbabilities, {}, *this, true);
    rootNode->SetClassNames(_classNames);
    setRootNode(std::unique_ptr<DecisionTreeNode>(rootNode));

    // Start recursive descent
    recursiveDescent(rootNode);

    return rootNode;
}

void DecisionTree::recursiveDescent(DecisionTreeNode* node) {}

BestFeatureResult DecisionTree::bestFeatureCalculator(
    const vector<string>& featuresAndValuesOrThresholdsOnBranch, 
    double existingNodeEntropy
) {
    // Define regex patterns for matching
    const std::regex pattern1(R"((.+)=(.+))");
    const std::regex pattern2(R"((.+)<(.+))");
    const std::regex pattern3(R"((.+)>(.+))");

    // Collect all symbolic features
    vector<string> allSymbolicFeatures;
    for (const auto& featureName : _featureNames) {
        if (_probDistributionNumericFeaturesDict.find(featureName) == _probDistributionNumericFeaturesDict.end()) {
            allSymbolicFeatures.push_back(featureName);
        }
    }

    // Determine symbolic features already used
    vector<string> symbolicFeaturesAlreadyUsed;
    for (const auto& item : featuresAndValuesOrThresholdsOnBranch) {
        std::smatch match;
        if (std::regex_search(item, match, pattern1)) {
            symbolicFeaturesAlreadyUsed.push_back(match[1].str());
        }
    }

    vector<string> trueNumericTypes;
    vector<string> symbolicTypes;
    vector<string> trueNumericTypesFeatureNames;
    vector<string> symbolicTypesFeatureNames;

    // MARK: Not tested in first test case, come back to it
    for (const auto& item : featuresAndValuesOrThresholdsOnBranch) {
        std::smatch match;
        if (std::regex_search(item, match, pattern2)) {
            trueNumericTypes.push_back(match[1].str()); // Might not be right
            trueNumericTypesFeatureNames.push_back(match[1].str());
        }
        else if (std::regex_search(item, match, pattern3)) {
            trueNumericTypes.push_back(match[1].str()); // Might not be right
            trueNumericTypesFeatureNames.push_back(match[1].str());
        }
        else {
            symbolicTypes.push_back(match[1].str());
            symbolicTypesFeatureNames.push_back(match[1].str());
        }
    }

    trueNumericTypesFeatureNames.erase(std::unique(trueNumericTypesFeatureNames.begin(), trueNumericTypesFeatureNames.end()), trueNumericTypesFeatureNames.end());
    symbolicTypesFeatureNames.erase(std::unique(symbolicTypesFeatureNames.begin(), symbolicTypesFeatureNames.end()), symbolicTypesFeatureNames.end());
    vector<vector<string>>boundedIntervalsNumericTypes = findBoundedIntervalsForNumericFeatures(trueNumericTypesFeatureNames);

    // Upper and lower bounds for the best feature
    map<string, double> lowerBound;
    map<string, double> upperBound;

    for (const auto& item : boundedIntervalsNumericTypes) {
        lowerBound[item[0]] = std::numeric_limits<double>::max();
        upperBound[item[0]] = std::numeric_limits<double>::min();
    }

    // Fill in the lower and upper bounds
    for (const auto& item : boundedIntervalsNumericTypes) {
        if (item[1] == ">") {
            lowerBound[item[0]] = std::stod(item[2]);
        }
        else {
            upperBound[item[0]] = std::stod(item[2]);
        }
    }

    map<string, double> entropyValuesForDifferentFeatures; // Stores entropy values for features
    map<string, map<double, std::pair<double, double>>> partitioningPointChildEntropiesDict; // Child entropies for numeric thresholds
    map<string, std::optional<double>> partitioningPointThreshold; // Thresholds for numeric features
    map<string, vector<double>> entropiesForDifferentValuesOfSymbolicFeature; // Entropies for symbolic feature values

    // Initialize maps for all features
    for (const auto& feature : _featureNames) {
        partitioningPointChildEntropiesDict[feature] = {};
        partitioningPointThreshold[feature] = std::nullopt;
        entropiesForDifferentValuesOfSymbolicFeature[feature] = {};
    }

    // Loop through all features to calculate entropies
    for (const auto& featureName : _featureNames) {
        if (_debug3) {
            std::cout << "\n\nBFC1    FEATURE BEING CONSIDERED: " << featureName << std::endl;
        }

        // Skip symbolic features that are already used
        if (std::find(symbolicFeaturesAlreadyUsed.begin(), symbolicFeaturesAlreadyUsed.end(), featureName) != symbolicFeaturesAlreadyUsed.end()) {
            continue;
        }

        // Check if the feature is numeric and exceeds the symbolic-to-numeric cardinality threshold
        if (_numericFeaturesValueRangeDict.find(featureName) != _numericFeaturesValueRangeDict.end() &&
            _featureValuesHowManyUniquesDict[featureName] > _symbolicToNumericCardinalityThreshold) {

            // Get the sampling points for the numeric feature
            const auto& values = _samplingPointsForNumericFeatureDict[featureName];
            if (_debug3) {
                std::cout << "\nBFC2 values for " << featureName << " are [";
                for (const auto& val : values) std::cout << val << ", ";
                std::cout << "]\n";
            }

            std::vector<double> newValues;

            // Check if the feature is in true numeric types and filter values within bounds
            if (std::find(trueNumericTypesFeatureNames.begin(), trueNumericTypesFeatureNames.end(), featureName) != trueNumericTypesFeatureNames.end()) {
                if (upperBound[featureName] && lowerBound[featureName] && lowerBound[featureName] >= upperBound[featureName]) {
                    // Skip if bounds are invalid
                    continue;
                } 
                else if (upperBound[featureName] && lowerBound[featureName] && lowerBound[featureName] < upperBound[featureName]) {
                    // Filter values within valid bounds
                    for (const auto& value : values) {
                        if (lowerBound[featureName] < value && value <= upperBound[featureName]) {
                            newValues.push_back(value);
                        }
                    }
                } 
                else if (upperBound[featureName]) {
                    // Filter values below upper bound
                    for (const auto& value : values) {
                        if (value <= upperBound[featureName]) {
                            newValues.push_back(value);
                        }
                    }
                } 
                else if (lowerBound[featureName]) {
                    // Filter values above lower bound
                    for (const auto& value : values) {
                        if (value > lowerBound[featureName]) {
                            newValues.push_back(value);
                        }
                    }
                } 
                else {
                    throw std::runtime_error("Error in bound specifications in best feature calculator");
                }
            }
            else {
                // If not in true numeric types, use all values
                newValues = values;
            }

            if (newValues.empty()) {
                // Skip if no valid values are found
                continue;
            }
        }
        else {
            if (_debug3) {
                std::cout << "\nBFC3 Best feature calculator: Entering section reserved for symbolic features";
                cout << "\nBFC4 Feature name: " << featureName;
            }

            set<string> valuesSet = _featuresAndUniqueValuesDict[featureName];
            vector<string> values(valuesSet.begin(), valuesSet.end());
            // Sort the values
            std::sort(values.begin(), values.end());

            if (_debug3) {
                cout << "\nBFC5 Values for feature " << featureName << " are: " << values;
            }

            double entropy = 0.0;

            for (const auto& value : values) {
                string featureValueString;
                double valueAsDouble = convert(value);
                if (std::isnan(valueAsDouble)) {
                    featureValueString = featureName + "=" + value;
                }
                else {
                    featureValueString = featureName + "=" + formatDouble(valueAsDouble);
                }

                if (_debug3) {
                    cout << "\nBFC6 Feature value string: " << featureValueString;
                }

                vector<string> extendedAttributes = deepCopy(featuresAndValuesOrThresholdsOnBranch);

                if (!featuresAndValuesOrThresholdsOnBranch.empty()) {
                    extendedAttributes.push_back(featureValueString);
                }
                else {
                    extendedAttributes = {featureValueString};
                }
                
                entropy += classEntropyForAGivenSequenceOfFeaturesAndValuesOrThresholds(extendedAttributes) * probabilityOfASequenceOfFeaturesAndValuesOrThresholds(extendedAttributes);

                if (_debug3) {
                    cout << "\nBFC7 Entropy calculated for symbolic feature value choice (" << featureName << ", " << value << ") is " << entropy;
                }

                entropiesForDifferentValuesOfSymbolicFeature[featureName].push_back(entropy);
            }

            if (entropy < existingNodeEntropy) {
                entropyValuesForDifferentFeatures[featureName] = entropy;
            }
        }
    }

    double minEntropyForBestFeature = std::numeric_limits<double>::max();
    string bestFeatureName;

    // MARK: Correct up to here for symbolic


    return {"bestFeatureName", 0.0, {}, 0};
}

//--------------- Entropy Calculators ----------------//

/**
 * @brief Calculates the entropy of the class priors.
 *
 * This function computes the entropy based on the prior probabilities of the classes.
 * It first checks if the entropy for 'priors' is already cached. If so, it returns the cached value.
 * Otherwise, it calculates the entropy.
 * 
 * The function ensures that probabilities very close to 0 or 1 are handled appropriately to avoid
 * numerical issues with the logarithm function.
 * 
 * @return The entropy of the class priors.
 */
double DecisionTree::classEntropyOnPriors()
{
    // Check if the entropy for 'priors' is already cached
    if (_entropyCache.find("priors") != _entropyCache.end()) {
        return _entropyCache["priors"];
    }

    double entropy = 0.0; // Initialize entropy

    // Calculate entropy based on class priors
    for (const auto &className : _classNames) {
        double prob = priorProbabilityForClass(className);

        double logProb = 0.0;
        if (prob >= 0.0001 && prob <= 0.999) {
            logProb = std::log2(prob);
        }

        if (prob < 0.0001 || prob > 0.999) {
            logProb = 0.0;
        }

        // Calculate entropy incrementally
        entropy += -1.0 * prob * logProb;
    }

    if (std::abs(entropy) < 0.0000001) {
        entropy = 0.0;
    }

    // Cache the calculated entropy
    _entropyCache["priors"] = entropy;

    return entropy;
}

/**
 * @brief Scans and calculates the entropy for a numeric feature at various sampling points.
 *
 * This function retrieves all sampling points for the specified numeric feature and calculates
 * the entropy for values less than and greater than each sampling point. The results are then
 * printed to the standard output.
 *
 * @param feature The name of the numeric feature to scan.
 */
void DecisionTree::entropyScannerForANumericFeature(const string &feature)
{
    // Retrieve all sampling points for the feature
    vector<double> allSamplingPoints = _samplingPointsForNumericFeatureDict[feature];
    vector<double> entropiesForLessThanThresholds;
    vector<double> entropiesForGreaterThanThresholds;

    // Iterate over all sampling points and calculate entropies
    for (double point : allSamplingPoints) {
        entropiesForLessThanThresholds.push_back(classEntropyForLessThanThresholdForFeature({}, feature, point));
        entropiesForGreaterThanThresholds.push_back(classEntropyForGreaterThanThresholdForFeature({}, feature, point));
    }

    // Output the results
    std::cout << "\nSCANNER: All entropies less than thresholds for feature " << feature << " are: [";
    for (const auto &entropy : entropiesForLessThanThresholds) {
        std::cout << entropy << " ";
    }
    std::cout << "]" << std::endl;

    std::cout << "\nSCANNER: All entropies greater than thresholds for feature " << feature << " are: [";
    for (const auto &entropy : entropiesForGreaterThanThresholds) {
        std::cout << entropy << " ";
    }
    std::cout << "]" << std::endl;
}

/**
 * @brief Calculates the entropy for a given feature and threshold combination.
 *
 * This function computes the entropy for a specific feature and threshold combination
 * within a given set of features and values or thresholds. It first constructs a sequence
 * string representing the combination, checks if the entropy for this sequence is already
 * cached, and if not, calculates the entropy and caches the result.
 *
 * @param arrayOfFeaturesAndValuesOrThresholds A vector of strings representing the features
 *        and their corresponding values or thresholds.
 * @param feature The feature for which the entropy is being calculated.
 * @param threshold The threshold value for the feature.
 * @param comparison The comparison operator (e.g., "<", ">", "<=", ">=") used with the threshold.
 * @return The calculated entropy for the given feature and threshold combination.
 */
double DecisionTree::EntropyForThresholdForFeature(const vector<string> &arrayOfFeaturesAndValuesOrThresholds,
                                                   const string &feature,
                                                   const double &threshold,
                                                   const string &comparison)
{
    // build a sequence string
    string featureThresholdCombo = feature + comparison + formatDouble(threshold);
    string sequence;
    for (const auto &featureValue : arrayOfFeaturesAndValuesOrThresholds) {
        if (!sequence.empty()) {
            sequence += ":";
        }
        sequence += featureValue;
    }
    sequence += ":" + featureThresholdCombo;

    // Check if the entropy for the sequence is already cached
    if (_entropyCache.find(sequence) != _entropyCache.end()) {
        return _entropyCache[sequence];
    }

    // make a copy of the array of features and values or thresholds
    vector<string> arrayOfFeaturesAndValuesOrThresholdsCopy = arrayOfFeaturesAndValuesOrThresholds;
    arrayOfFeaturesAndValuesOrThresholdsCopy.push_back(featureThresholdCombo);

    // Calculate the entropy for the sequence
    double entropy = 0.0;

    // Calculate the entropy for each class
    for (const auto &className : _classNames) {
        double logProb = 0.0;
        double prob    = probabilityOfAClassGivenSequenceOfFeaturesAndValuesOrThresholds(
            className, arrayOfFeaturesAndValuesOrThresholdsCopy);
        if (prob >= .0001 && prob <= .999) {
            logProb = std::log2(prob);
        }
        else {
            logProb = 0.0;
        }
        entropy += -1.0 * prob * logProb;
    }

    // check floating point precision
    if (std::abs(entropy) < 0.0000001) {
        entropy = 0.0;
    }
    // cache the result
    _entropyCache[sequence] = entropy;
    return entropy;
}

/**
 * @brief Calculates the entropy of a class for a given feature when the feature's value is less than a specified threshold.
 *
 * @param arrayOfFeaturesAndValuesOrThresholds A vector containing the features and their corresponding values or thresholds.
 * @param feature The feature for which the entropy is to be calculated.
 * @param threshold The threshold value to compare the feature's value against.
 * @return The entropy of the class for the given feature when its value is less than the specified threshold.
 */
double DecisionTree::classEntropyForLessThanThresholdForFeature(
    const vector<string> &arrayOfFeaturesAndValuesOrThresholds, const string &feature, const double &threshold)
{
    return EntropyForThresholdForFeature(arrayOfFeaturesAndValuesOrThresholds, feature, threshold, "<");
}

/**
 * @brief Calculates the entropy of a class for a given feature when the feature's value is greater than a specified threshold.
 *
 * This function computes the entropy for a specific feature in the dataset when the feature's value is greater than the provided threshold.
 * It utilizes the EntropyForThresholdForFeature function with the ">" operator to determine the entropy.
 *
 * @param arrayOfFeaturesAndValuesOrThresholds A vector of strings representing the features and their corresponding values or thresholds.
 * @param feature The feature for which the entropy is to be calculated.
 * @param threshold The threshold value for the feature.
 * @return The entropy of the class for the given feature when the feature's value is greater than the threshold.
 */
double DecisionTree::classEntropyForGreaterThanThresholdForFeature(
    const vector<string> &arrayOfFeaturesAndValuesOrThresholds, const string &feature, const double &threshold)
{
    return EntropyForThresholdForFeature(arrayOfFeaturesAndValuesOrThresholds, feature, threshold, ">");
}

/**
 * @brief Calculates the entropy for a given sequence of features and values or thresholds.
 *
 * This function computes the entropy for a given sequence of features and values or thresholds.
 * It first joins the array of features and values or thresholds into a sequence string and checks
 * if the entropy for the sequence is already cached. If cached, it returns the cached value.
 * Otherwise, it calculates the entropy for each class and caches the result.
 *
 * @param arrayOfFeaturesAndValuesOrThresholds A vector of strings representing the sequence of features and values or thresholds.
 * @return The calculated entropy for the given sequence.
 */
double DecisionTree::classEntropyForAGivenSequenceOfFeaturesAndValuesOrThresholds(
    const vector<string> &arrayOfFeaturesAndValuesOrThresholds)
{
    // Join the array of features and values or thresholds into a sequence string
    string sequence;
    for (const auto &featureValue : arrayOfFeaturesAndValuesOrThresholds) {
        if (!sequence.empty()) {
            sequence += ":";
        }
        sequence += featureValue;
    }

    // Check if the entropy for the sequence is already cached
    if (_entropyCache.find(sequence) != _entropyCache.end()) {
        return _entropyCache[sequence];
    }

    double entropy = 0.0;
    double logProb = 0.0;

    // Calculate the entropy for each class
    for (const auto &className : _classNames) {
        double prob = probabilityOfAClassGivenSequenceOfFeaturesAndValuesOrThresholds(
            className, arrayOfFeaturesAndValuesOrThresholds);

        if (prob >= 0.0001 && prob <= 0.999) {
            logProb = std::log2(prob);
        }
        // If probability is too small or too large, set logProb to zero
        else {
            logProb = 0.0;
        }
        
        // Calculate entropy incrementally
        entropy += -1.0 * prob * logProb;
    }

    if (std::abs(entropy) < 0.0000001) {
        entropy = 0.0;
    }

    // Cache the result
    _entropyCache[sequence] = entropy;

    return entropy;
}

//--------------- Probability Calculators ----------------//
double DecisionTree::priorProbabilityForClass(const string &className)
{
    // Generate a cache key for prior probability of a specific class
    string classNameCacheKey = "prior::" + className;
    // logger.log(LogLevel(0), "priorProbabilityForClass:: classNameInCache: " + classNameCacheKey);

    // Check if the probability is already in the cache (memoization)
    if (_probabilityCache.find(classNameCacheKey) != _probabilityCache.end()) {
        // // logger.log(LogLevel(0),
        //            "priorProbabilityForClass:: probability found in cache: " +
        //                std::to_string(_probabilityCache[classNameCacheKey]));
        return _probabilityCache[classNameCacheKey];
    }

    // logger.log(LogLevel(0), "priorProbabilityForClass:: probability not found in cache");

    // Calculate prior probability for all classes and store in cache
    size_t totalNumSamples   = _samplesClassLabelDict.size();
    vector<string> allValues = {};
    for (const auto &kv : _samplesClassLabelDict) {
        allValues.push_back(kv.second);
    }

    // Iterate over all class names to calculate their prior probabilities
    for (const auto &className : _classNames) {
        // Get the number of samples for the class
        size_t numSamplesForClass = std::count(allValues.begin(), allValues.end(), className);
        // Calculate the prior probability for the class
        double priorProbability = static_cast<double>(numSamplesForClass) / static_cast<double>(totalNumSamples);

        // store the prior probability in the _classPriorsDict
        _classPriorsDict[className] = priorProbability;

        // Store the prior probability in the cache
        string classNamePrior             = "prior::" + className;
        _probabilityCache[classNamePrior] = priorProbability;
        // logger.log(LogLevel(0),
        //            "priorProbabilityForClass:: prior probability for " + className + ": " +
        //                std::to_string(priorProbability));
    }
    return _probabilityCache[classNameCacheKey];
}

void DecisionTree::calculateClassPriors()
{
    cout << "\nCalculating class priors...\n";

    // Return if the class priors have already been calculated
    if (_classPriorsDict.size() > 1) {
        return;
    }

    for (const auto &className : _classNames) {
        int totalNumSamples      = _samplesClassLabelDict.size();
        vector<string> allValues = {};
        for (const auto &kv : _samplesClassLabelDict) {
            allValues.push_back(kv.second);
        }
        int numSamplesForClass  = std::count(allValues.begin(), allValues.end(), className);
        double priorProbability = static_cast<double>(numSamplesForClass) / static_cast<double>(totalNumSamples);

        _classPriorsDict[className]       = priorProbability;
        string classNamePrior             = "prior::" + className;
        _probabilityCache[classNamePrior] = priorProbability;
    }

    if (_debug2) {
        cout << "\nClass priors calculated:\n" << endl;
        for (const auto &className : _classNames) {
            cout << className << " = " << _classPriorsDict[className] << endl;
        }
    }
}

double DecisionTree::probabilityOfFeatureValue(const string &feature, const string &value)
{
    // Prepare feature value and initialize variables
    string adjustedValue = value; // Create a copy of the value
    double valueAsDouble = convert(adjustedValue);
    string featureAndValue;

    // If the feature is numeric, find the closest sampling point
    if (!std::isnan(valueAsDouble) &&
        _samplingPointsForNumericFeatureDict.find(feature) != _samplingPointsForNumericFeatureDict.end()) {
        adjustedValue =
            std::to_string(ClosestSamplingPoint(_samplingPointsForNumericFeatureDict[feature], valueAsDouble));
    }

    // If the feature is numeric, format the double for storing it into the cache
    // This will remove trailing zeroes from the double as a string
    if (!std::isnan(valueAsDouble)) {
        adjustedValue = formatDouble(convert(adjustedValue));
    }

    // Create a combined feature and value string
    if (!adjustedValue.empty()) {
        featureAndValue = feature + "=" + adjustedValue;
    }

    // Check if the probability is already cached, if so, return it
    if (_probabilityCache.find(featureAndValue) != _probabilityCache.end()) {
        return _probabilityCache[featureAndValue];
    }

    // Initialize variables for histogram calculations
    double histogramDelta     = 0.0;
    double diffRange          = 0.0;
    vector<double> valueRange = {};
    int numOfHistogramBins    = 0;

    // Check if feature is numeric with sufficient unique values for histogram calculations
    if (_numericFeaturesValueRangeDict.find(feature) != _numericFeaturesValueRangeDict.end()) {
        if (_featureValuesHowManyUniquesDict[feature] > _symbolicToNumericCardinalityThreshold) {
            // Calculate histogram delta based on median difference between unique sorted values
            if (_samplingPointsForNumericFeatureDict.find(feature) == _samplingPointsForNumericFeatureDict.end()) {
                valueRange = _numericFeaturesValueRangeDict[feature];
                diffRange  = valueRange[1] - valueRange[0];

                vector<string> values = _featuresAndValuesDict[feature];
                std::set<double> uniqueValues;
                for (const auto &v : values) { // Remove NA values
                    if (v != "NA") {
                        uniqueValues.insert(convert(v));
                    }
                }

                // Get unique values
                vector<double> sortedUniqueValues(uniqueValues.begin(), uniqueValues.end());
                std::sort(sortedUniqueValues.begin(), sortedUniqueValues.end());

                // Calc diffs
                vector<double> diffs;
                for (size_t diffIdx = 1; diffIdx < sortedUniqueValues.size(); ++diffIdx) {
                    diffs.push_back(sortedUniqueValues[diffIdx] - sortedUniqueValues[diffIdx - 1]);
                }
                std::sort(diffs.begin(), diffs.end());

                double medianDiff = diffs[(diffs.size() / 2) - 1];
                histogramDelta  = medianDiff * 2.0;

                if (histogramDelta < diffRange / 500.0) {
                    if (_numberOfHistogramBins > 0) {
                        histogramDelta = diffRange / static_cast<double>(_numberOfHistogramBins);
                    }
                    else {
                        histogramDelta = diffRange / 500.0;
                    }
                }

                _histogramDeltaDict[feature]     = histogramDelta;
                numOfHistogramBins               = static_cast<int>(diffRange / histogramDelta) + 1;
                _numOfHistogramBinsDict[feature] = numOfHistogramBins;

                vector<double> samplingPointsForFeature;
                for (size_t histIdx = 0; histIdx < numOfHistogramBins; ++histIdx) {
                    samplingPointsForFeature.push_back(valueRange[0] + histogramDelta * histIdx);
                }

                _samplingPointsForNumericFeatureDict[feature] = samplingPointsForFeature;
            }
        }
    }

    if (_numericFeaturesValueRangeDict.find(feature) != _numericFeaturesValueRangeDict.end()) {
        if (_featureValuesHowManyUniquesDict[feature] > _symbolicToNumericCardinalityThreshold) {
            auto samplingPointsForFeature = _samplingPointsForNumericFeatureDict[feature];
            vector<size_t> countsAtSamplingPoints(samplingPointsForFeature.size(), 0);
            vector<string> actualValuesForFeature = _featuresAndValuesDict[feature];
            vector<double> actualValuesForFeatureAsDoubles;

            for (const auto &v : actualValuesForFeature) {
                if (v != "NA") {
                    double valueAsDouble = convert(v);
                    if (!std::isnan(valueAsDouble)) {
                        actualValuesForFeatureAsDoubles.push_back(valueAsDouble);
                    }
                }
            }

            // Count the number of values at each sampling point
            for (size_t i = 0; i < samplingPointsForFeature.size(); ++i) {
                for (size_t j = 0; j < actualValuesForFeatureAsDoubles.size(); ++j) {
                    if (abs(samplingPointsForFeature[i] - actualValuesForFeatureAsDoubles[j]) < histogramDelta) {
                        countsAtSamplingPoints[i] += 1;
                    }
                }
            }

            // Calculate the total counts
            int totalCounts = 0;
            for (const auto &count : countsAtSamplingPoints) {
                totalCounts += count;
            }

            // Calculate the probabilities
            vector<double> probabilities;
            for (const auto &count : countsAtSamplingPoints) {
                probabilities.push_back(static_cast<double>(count) / static_cast<double>(totalCounts));
            }

            // Cache the probabilities
            map<double, double> binProbDict;
            for (size_t i = 0; i < samplingPointsForFeature.size(); ++i) {
                binProbDict[samplingPointsForFeature[i]] = probabilities[i];
            }
            _probDistributionNumericFeaturesDict[feature] = binProbDict;

            vector<string> valuesForFeature;
            valuesForFeature.reserve(samplingPointsForFeature.size());
            std::transform(samplingPointsForFeature.begin(),
                           samplingPointsForFeature.end(),
                           std::back_inserter(valuesForFeature),
                           [&feature](int x) { return feature + "=" + std::to_string(x); });

            // Cache rest
            for (size_t i = 0; i < valuesForFeature.size(); ++i) {
                _probabilityCache[valuesForFeature[i]] = probabilities[i];
            }

            if (!std::isnan(valueAsDouble) && (_probabilityCache.find(featureAndValue) != _probabilityCache.end())) {
                return _probabilityCache[featureAndValue];
            }
            else {
                return 0.0;
            }
        }
        else {
            // This section if for those numeric features treated symbolically
            std::set<string> uniqueValuesForFeature =
                set(_featuresAndValuesDict[feature].begin(), _featuresAndValuesDict[feature].end());
            vector<string> valuesForFeature(uniqueValuesForFeature.begin(), uniqueValuesForFeature.end());
            valuesForFeature.erase(std::remove(valuesForFeature.begin(), valuesForFeature.end(), "NA"),
                                   valuesForFeature.end());

            // Create a feature and value string
            for (size_t i = 0; i < valuesForFeature.size(); ++i) {
                valuesForFeature[i] = feature + "=" + valuesForFeature[i];
            }

            // Calculate the counts for each value
            vector<int> valueCounts(valuesForFeature.size(), 0);
            for (const auto &sample : _trainingDataDict) {
                vector<string> featuresAndValues = sample.second;
                for (size_t i = 0; i < valuesForFeature.size(); ++i) {
                    for (size_t j = 0; j < featuresAndValues.size(); ++j) {
                        string name = _featureNames[j] + "=" + featuresAndValues[j];
                        if (valuesForFeature[i] == name) {
                            valueCounts[i]++;
                        }
                    }
                }
            }

            // Assigning counts
            int totalCounts = 0;
            for (int count : valueCounts) {
                totalCounts += count;
            }

            // Assigning probabilities
            vector<double> probabilities;
            for (int count : valueCounts) {
                probabilities.push_back((double) count / (double) totalCounts);
            }

            // Assigning probability cache
            for (size_t i = 0; i < valuesForFeature.size(); ++i) {
                _probabilityCache[valuesForFeature[i]] = probabilities[i];
            }

            // If the feature and value exists in the probability cache, return it
            if (_probabilityCache.find(featureAndValue) != _probabilityCache.end()) {
                return _probabilityCache[featureAndValue];
            }
            else {
                return 0.0;
            }
        }
    }
    // Symbolic feature case
    else {
        vector<string> valuesForFeatures = _featuresAndValuesDict[feature];
        for (size_t i = 0; i < valuesForFeatures.size(); ++i) {
            valuesForFeatures[i] = feature + "=" + valuesForFeatures[i];
        }
        vector<int> countsForValues(valuesForFeatures.size(), 0);

        for (const auto &sample : _trainingDataDict) {
            vector<string> featuresAndValues = sample.second;
            for (int i = 0; i < valuesForFeatures.size(); i++) {
                for (int j = 0; j < featuresAndValues.size(); j++) {
                    string name = _featureNames[j] + "=" + featuresAndValues[j];
                    if (valuesForFeatures[i] == name) {
                        countsForValues[i]++;
                    }
                }
            }
        }

        int totalNumSamples = _trainingDataDict.size();

        vector<double> probabilities;
        for (int count : countsForValues) {
            probabilities.push_back((double) count / (double) totalNumSamples);
        }

        for (size_t i = 0; i < valuesForFeatures.size(); ++i) {
            string name             = valuesForFeatures[i];
            _probabilityCache[name] = probabilities[i];
        }

        if (_probabilityCache.find(featureAndValue) != _probabilityCache.end()) {
            return _probabilityCache[featureAndValue];
        }
        else {
            return 0.0;
        }
    }
    return 0.0;
}

double
DecisionTree::probabilityOfFeatureValueGivenClass(const string &feature, const string &value, const string &className)
{
    // Prepare feature value class and initialize variables
    string adjustedValue = value;
    double valueAsDouble = convert(adjustedValue);
    string featureAndValueClass;

    // If the feature is numeric, find the closest sampling point
    if (!std::isnan(valueAsDouble) &&
        _samplingPointsForNumericFeatureDict.find(feature) != _samplingPointsForNumericFeatureDict.end()) {
        adjustedValue =
            std::to_string(ClosestSamplingPoint(_samplingPointsForNumericFeatureDict[feature], valueAsDouble));
    }

    

    // If the feature is numeric, format the double for storing it into the cache
    if (!std::isnan(valueAsDouble)) {
        adjustedValue = formatDouble(convert(adjustedValue));
    }

    // Create a combined feature and value string
    if (!adjustedValue.empty()) {
        featureAndValueClass = feature + "=" + adjustedValue + "::" + className;
    }

    // Check if the probability is already cached
    if (_probabilityCache.find(featureAndValueClass) != _probabilityCache.end()) {
        return _probabilityCache[featureAndValueClass];
    }

    // Initialize variables for histogram calculations
    double histogramDelta     = 0.0;
    double diffrange          = 0.0;
    vector<double> valuerange = {};
    int numOfHistogramBins    = 0;

    // If feature in numericFeaturesValueRangeDict
    if (_numericFeaturesValueRangeDict.find(feature) != _numericFeaturesValueRangeDict.end()) {
        if (_featureValuesHowManyUniquesDict[feature] > _symbolicToNumericCardinalityThreshold) {
            histogramDelta     = _histogramDeltaDict[feature];
            numOfHistogramBins = _numOfHistogramBinsDict[feature];
            valuerange         = _numericFeaturesValueRangeDict[feature];
            diffrange          = static_cast<int>(valuerange[1] - valuerange[0]);
        }
    }

    vector<int> samplesForClass = {}; // Vector to store all sample indices for the given class

    // Accumulate all samples names for the given class
    for (const auto &sampleName : _samplesClassLabelDict) {
        if (_samplesClassLabelDict[sampleName.first] == className) {
            samplesForClass.push_back(sampleName.first);
        }
    }

    // Numeric feature case
    if (_numericFeaturesValueRangeDict.find(feature) != _numericFeaturesValueRangeDict.end()) {
        if (_featureValuesHowManyUniquesDict[feature] > _symbolicToNumericCardinalityThreshold) {
            vector<double> samplingPointsForFeature = _samplingPointsForNumericFeatureDict[feature];
            vector<int> countsAtSamplingPoints(samplingPointsForFeature.size(), 0);
            vector<string> actualFeatureValuesForSamplesInClass;

            for (const auto &sample : samplesForClass) {
                int featureIndex = 0;
                for (const auto &value : _trainingDataDict[sample]) {
                    string featureName = _featureNames[featureIndex++];
                    if (featureName == feature && value != "NA") {
                        actualFeatureValuesForSamplesInClass.push_back(value);
                    }
                }
            }

            for (size_t i = 0; i < samplingPointsForFeature.size(); ++i) {
                for (size_t j = 0; j < actualFeatureValuesForSamplesInClass.size(); ++j) {
                    if (std::abs(samplingPointsForFeature[i] - stoi(actualFeatureValuesForSamplesInClass[j])) <
                        histogramDelta) {
                        countsAtSamplingPoints[i]++;
                    }
                }
            }

            // Calculate the total counts (sum the counts at each sampling point)
            size_t totalCounts = std::accumulate(countsAtSamplingPoints.begin(), countsAtSamplingPoints.end(), 0);
            
            // Check for total counts being zero
            if (totalCounts == 0) {
                throw std::runtime_error("PFVC1 Something is wrong with your training file. It contains no training "
                                         "samples for Class " +
                                         className + " and Feature " + feature);
            }

            // Probabilities
            vector<double> probabilities(countsAtSamplingPoints.size(), 0.0);
            for (size_t i = 0; i < countsAtSamplingPoints.size(); ++i) {
                probabilities[i] = static_cast<double>(countsAtSamplingPoints[i]) / static_cast<double>(totalCounts);
            }

            // Create values for feature and class (should be the same as list of a map)
            vector<string> valuesForFeatureAndClass;
            for (const auto &point : samplingPointsForFeature) {
                valuesForFeatureAndClass.push_back(feature + "=" + formatDouble(point) + "::" + className);
            }

            // Cache probabilities
            for (size_t i = 0; i < valuesForFeatureAndClass.size(); ++i) {
                _probabilityCache[valuesForFeatureAndClass[i]] = probabilities[i];
            }

            // Return the probability for the given feature-value-class pair if cached, else return 0
            if (_probabilityCache.find(featureAndValueClass) != _probabilityCache.end()) {
                return _probabilityCache[featureAndValueClass];
            }
            else {
                return 0;
            }
        }
        else {
            // Extract unique values for the feature from _featuresAndValuesDict
            std::set<string> uniqueValues(_featuresAndValuesDict[feature].begin(),
                                               _featuresAndValuesDict[feature].end());

            // Remove "NA" values
            uniqueValues.erase("NA");

            // Format values as "feature=value"
            vector<string> valuesForFeature;
            for (const auto &value : uniqueValues) {
                string formattedValue = feature + "=" + value;
                valuesForFeature.push_back(formattedValue);
            }

            // Initialize counts for each value
            vector<int> valueCounts(valuesForFeature.size(), 0);

            // Count occurrences of feature values within samples for the class
            for (const auto &sample : samplesForClass) {
                vector<string> featuresAndValues;
                for (size_t j = 0; j < _featureNames.size(); ++j) {
                    featuresAndValues.push_back(_featureNames[j] + "=" + _trainingDataDict[sample][j]);
                }

                for (size_t i = 0; i < valuesForFeature.size(); ++i) {
                    for (const auto &currentValue : featuresAndValues) {
                        // Compare full "feature=value" strings
                        if (valuesForFeature[i] == currentValue) {
                            valueCounts[i]++;
                        }
                    }
                }
            }

            // Calculate the total count
            int totalCount = std::accumulate(valueCounts.begin(), valueCounts.end(), 0);
            if (totalCount == 0) {
                throw std::runtime_error(
                    "PFVC2 Something is wrong with your training file. It contains no training samples for Class " +
                    className + " and Feature " + feature);
            }

            // Normalize and cache probabilities
            for (size_t i = 0; i < valuesForFeature.size(); ++i) {
                string featureAndValueAndClass = valuesForFeature[i] + "::" + className;
                _probabilityCache[featureAndValueAndClass] =
                    static_cast<double>(valueCounts[i]) / static_cast<double>(totalCount);
            }

            // Check for cached value
            string featureValueClass = feature + "=" + adjustedValue + "::" + className;

            for (const auto &entry : _probabilityCache) {
                if (entry.first == featureValueClass) {
                    return entry.second;
                }
            }
            return 0.0;
        }
    }
    // Purely symbolic case
    else {
        vector<string> valuesForFeature = _featuresAndValuesDict[feature];
        for (size_t i = 0; i < valuesForFeature.size(); ++i) {
            valuesForFeature[i] = feature + "=" + valuesForFeature[i];
        }

        // Removing the duplicate values
        std::sort(valuesForFeature.begin(), valuesForFeature.end());
        auto it = std::unique(valuesForFeature.begin(), valuesForFeature.end());
        valuesForFeature.erase(it, valuesForFeature.end());

        vector<int> countsForValues(valuesForFeature.size(), 0);

        for (const auto &sample : samplesForClass) {
            vector<string> featuresAndValues = _trainingDataDict[sample];
            for (int i = 0; i < valuesForFeature.size(); i++) {
                for (int j = 0; j < featuresAndValues.size(); j++) {
                    string name = _featureNames[j] + "=" + featuresAndValues[j];
                    if (valuesForFeature[i] == name) {
                        countsForValues[i]++;
                    }
                }
            }
        }

        int totalNumSamples = samplesForClass.size();
        if (totalNumSamples == 0) {
            return 0.0;
        }

        for (int i = 0; i < valuesForFeature.size(); i++) {
            string featureAndValueForClass = valuesForFeature[i] + "::" + className;
            _probabilityCache[featureAndValueForClass] =
                static_cast<double>(countsForValues[i]) / static_cast<double>(totalNumSamples);
        }

        string featureAndValueAndClass = feature + "=" + adjustedValue + "::" + className;
        if (_probabilityCache.find(featureAndValueAndClass) != _probabilityCache.end()) {
            return _probabilityCache[featureAndValueAndClass];
        }
        else {
            return 0.0;
        }
    }

    return 0.0;
}

// This is used for NUMERIC features only, threshold is a string, but should be a double
double DecisionTree::probabilityOfFeatureLessThanThreshold(const string &featureName, const string &threshold)
{
    double thresholdAsDouble     = convert(threshold);
    string featureThresholdCombo = featureName + "<" + formatDouble(thresholdAsDouble);

    // Check if the probability is already cached
    if (_probabilityCache.find(featureThresholdCombo) != _probabilityCache.end()) {
        return _probabilityCache[featureThresholdCombo];
    }

    // Get all values for the feature
    vector<string> valuesForFeature = _featuresAndValuesDict[featureName];
    vector<double> valuesForFeatureAsDoubles;
    for (const auto &v : valuesForFeature) {
        if (v != "NA") // Remove NA
        {
            double valueAsDouble = convert(v);
            if (!std::isnan(valueAsDouble)) {
                valuesForFeatureAsDoubles.push_back(valueAsDouble);
            }
        }
    }

    // Get all values less than the threshold
    vector<double> allValuesLessThanThreshold;
    for (const auto &v : valuesForFeatureAsDoubles) {
        if (v <= thresholdAsDouble) {
            allValuesLessThanThreshold.push_back(v);
        }
    }

    // Calculate the probability
    double probability =
        static_cast<double>(allValuesLessThanThreshold.size()) / static_cast<double>(valuesForFeatureAsDoubles.size());
    _probabilityCache[featureThresholdCombo] = probability;
    return probability;
}

double DecisionTree::probabilityOfFeatureLessThanThresholdGivenClass(const string &featureName,
                                                                     const string &threshold,
                                                                     const string &className)
{
    double thresholdAsDouble     = convert(threshold);
    string featureThresholdCombo = featureName + "<" + std::to_string(thresholdAsDouble) + "::" + className;

    // Check if the probability is already cached
    if (_probabilityCache.find(featureThresholdCombo) != _probabilityCache.end()) {
        return _probabilityCache[featureThresholdCombo];
    }

    // Accumulate all smaples for given class
    vector<int> dataSamplesForClass;
    for (const auto &kv : _samplesClassLabelDict) {
        if (kv.second == className) {
            dataSamplesForClass.push_back(kv.first);
        }
    }

    // Get all values for the feature
    vector<string> actualFeatureValuesForSamplesInClass;
    for (const auto &sampleIdx : dataSamplesForClass) {
        int featureIdx = 0;
        for (const auto &value : _trainingDataDict[sampleIdx]) {
            // if feature matches, add it to the samples list
            if (_featureNames[featureIdx++] == featureName && value != "NA") {
                actualFeatureValuesForSamplesInClass.push_back(value);
            }
        }
    }

    // Get all values less than or Equal to the threshold
    vector<double> actualPointsForFeatureLessThanThreshold;
    for (const auto &v : actualFeatureValuesForSamplesInClass) {
        double valueAsDouble = convert(v);
        if (!std::isnan(valueAsDouble) && valueAsDouble <= thresholdAsDouble) {
            actualPointsForFeatureLessThanThreshold.push_back(valueAsDouble);
        }
    }

    // Calculate and cache the probability
    double probability = static_cast<double>(actualPointsForFeatureLessThanThreshold.size()) /
                         static_cast<double>(actualFeatureValuesForSamplesInClass.size());
    _probabilityCache[featureThresholdCombo] = probability;
    return probability;
}

double DecisionTree::probabilityOfASequenceOfFeaturesAndValuesOrThresholds(
    const vector<string> &arrayOfFeaturesAndValuesOrThresholds)
{
    // This method requires that all truly numeric types only be expressed as '<' or '>' constructs in the array
    // of branch features and thresholds. The symbolic types should be expressed as 'feature=value' constructs.
    if (arrayOfFeaturesAndValuesOrThresholds.size() == 0) {
        return std::nan("");
    }

    // Generate sequence string
    string sequence = "";
    for (const auto &item : arrayOfFeaturesAndValuesOrThresholds) {
        if (item != arrayOfFeaturesAndValuesOrThresholds.back()) {
            sequence += item + ":";
        }
        else {
            sequence += item;
        }
    }

    // Check if the sequence is in the cache
    if (_probabilityCache.find(sequence) != _probabilityCache.end()) {
        return _probabilityCache[sequence];
    }

    // Setup the ritual table
    double probability = 0.0;
    regex pattern1(R"((.+)=(.+))"); // Symbolic feature pattern
    regex pattern2(R"((.+)<(.+))"); // Numeric feature pattern
    regex pattern3(R"((.+)>(.+))"); // Numeric feature pattern
    vector<string> trueNumericTypes;
    std::set<string> trueNumericTypesFeatureNames;
    vector<string> symbolicTypes;
    std::set<string> symbolicTypesFeatureNames; // unsure if this is needed

    // Cast draw the incantation
    for (const auto &item : arrayOfFeaturesAndValuesOrThresholds) {
        smatch match;
        string feature;
        string value;

        if (regex_search(item, match, pattern2)) {
            feature = match[1];
            value   = match[2];
            trueNumericTypes.push_back(item);
            trueNumericTypesFeatureNames.insert(feature);
        }
        else if (regex_search(item, match, pattern3)) {
            feature = match[1];
            value   = match[2];
            trueNumericTypes.push_back(item);
            trueNumericTypesFeatureNames.insert(feature);
        }
        else {
            regex_search(item, match, pattern1);
            feature = match[1];
            value   = match[2];
            symbolicTypes.push_back(item);
            symbolicTypesFeatureNames.insert(feature);
        }
    }

    // get bounded intervals for numeric features
    vector<vector<string>> boundedIntervalsNumericTypes = findBoundedIntervalsForNumericFeatures(trueNumericTypes);

    // Calculate the upper and the lower bounds to be used when searching for the best
    // threshold for each of the numeric features that are in play at the current node:

    // Populate bounds with feature names
    map<string, double> lowerBound;
    map<string, double> upperBound;
    for (const auto &feature : trueNumericTypesFeatureNames) {
        lowerBound[feature] = std::numeric_limits<double>::max();
        upperBound[feature] = std::numeric_limits<double>::min();
    }

    // Populate with values from the bounded intervals
    for (const auto &item : boundedIntervalsNumericTypes) {
        if (item[1] == ">") {
            lowerBound[item[0]] = convert(item[2]);
        }
        else {
            upperBound[item[0]] = convert(item[2]);
        }
    }

    // Numeric feature case
    for (const auto &featureName : trueNumericTypesFeatureNames) {
        if (lowerBound[featureName] != std::numeric_limits<double>::max() &&
            upperBound[featureName] != std::numeric_limits<double>::min()) { // Both bounds are set
            if (upperBound[featureName] <= lowerBound[featureName]) {
                return 0; // Return 0 if upper bound is less than or equal to lower bound
            }
            else {
                if (!probability) {
                    probability =
                        probabilityOfFeatureLessThanThreshold(featureName, std::to_string(upperBound[featureName])) -
                        probabilityOfFeatureLessThanThreshold(featureName, std::to_string(lowerBound[featureName]));
                }
                else {
                    probability *=
                        (probabilityOfFeatureLessThanThreshold(featureName, std::to_string(upperBound[featureName])) -
                         probabilityOfFeatureLessThanThreshold(featureName, std::to_string(lowerBound[featureName])));
                }
            }
        }
        // if only upper bound is set
        else if (upperBound[featureName] != std::numeric_limits<double>::min() &&
                 lowerBound[featureName] == std::numeric_limits<double>::max()) {
            if (!probability) {
                probability =
                    probabilityOfFeatureLessThanThreshold(featureName, std::to_string(upperBound[featureName]));
            }
            else {
                probability *=
                    probabilityOfFeatureLessThanThreshold(featureName, std::to_string(upperBound[featureName]));
            }
        }
        // if only lower bound is set
        else if (lowerBound[featureName] != std::numeric_limits<double>::max() &&
                 upperBound[featureName] == std::numeric_limits<double>::min()) {
            if (!probability) {
                probability =
                    1.0 - probabilityOfFeatureLessThanThreshold(featureName, std::to_string(lowerBound[featureName]));
            }
            else {
                probability *=
                    (1.0 - probabilityOfFeatureLessThanThreshold(featureName, std::to_string(lowerBound[featureName])));
            }
        }
        else {
            throw std::runtime_error("Ill formatted call to 'probability_of_sequence' method");
        }
    }

    // Symbolic feature case
    for (const auto &featureAndValue : symbolicTypes) {
        smatch match;
        if (regex_search(featureAndValue, match, pattern1)) {
            string feature = match[1];
            string value   = match[2];
            if (!probability) {
                probability = probabilityOfFeatureValue(feature, value);
            }
            else {
                probability *= probabilityOfFeatureValue(feature, value);
            }
        }
    }

    _probabilityCache[sequence] = probability;
    return probability;
}

double DecisionTree::probabilityOfASequenceOfFeaturesAndValuesOrThresholdsGivenClass(
    const vector<string> &arrayOfFeaturesAndValuesOrThresholds, const string &className)
{
    // This method requires that all truly numeric types only be expressed as '<' or '>' constructs in the array of
    // branch features and thresholds. The symbolic types should be expressed as 'feature=value' constructs.

    if (arrayOfFeaturesAndValuesOrThresholds.size() == 0) {
        return std::nan("");
    }

    // Generate sequence string
    string sequence = "";
    for (const auto &item : arrayOfFeaturesAndValuesOrThresholds) {
        // Append a colon to the sequence if it is not the last item
        if (item != arrayOfFeaturesAndValuesOrThresholds.back()) {
            sequence += item + ":";
        }
        else {
            sequence += item;
        }
    }
    string sequenceWithClass = sequence + "::" + className;

    // Setup the ritual table
    double probability = 0.0;
    regex pattern1(R"((.+)=(.+))"); // Symbolic feature pattern
    regex pattern2(R"((.+)<(.+))"); // Numeric feature pattern
    regex pattern3(R"((.+)>(.+))"); // Numeric feature pattern
    vector<string> trueNumericTypes;
    vector<string> trueNumericTypesFeatureNames;
    vector<string> symbolicTypes;
    vector<string> symbolicTypesFeatureNames;

    // Cast draw the incantation
    for (const auto &item : arrayOfFeaturesAndValuesOrThresholds) {
        smatch match;
        string feature;
        string value;
        if (regex_search(item, match, pattern2)) {
            feature = match[1];
            value   = match[2];
            trueNumericTypes.push_back(item);
            trueNumericTypesFeatureNames.push_back(feature);
        }
        else if (regex_search(item, match, pattern3)) {
            feature = match[1];
            value   = match[2];
            trueNumericTypes.push_back(item);
            trueNumericTypesFeatureNames.push_back(feature);
        }
        else {
            regex_search(item, match, pattern1);
            feature = match[1]; // group 1
            value   = match[2]; // group 2
            symbolicTypes.push_back(item);
            symbolicTypesFeatureNames.push_back(feature);
        }
    }

    // Remove duplicates from feature names in-place
    trueNumericTypesFeatureNames.erase(unique(trueNumericTypesFeatureNames.begin(), trueNumericTypesFeatureNames.end()),
                                       trueNumericTypesFeatureNames.end());
    symbolicTypesFeatureNames.erase(unique(symbolicTypesFeatureNames.begin(), symbolicTypesFeatureNames.end()),
                                    symbolicTypesFeatureNames.end());
    vector<vector<string>> boundedIntervalsNumericTypes = findBoundedIntervalsForNumericFeatures(trueNumericTypes);

    // Calculate the upper and the lower bounds to be used when searching for the best
    // threshold for each of the numeric features that are in play at the current node:
    map<string, double> lowerBound;
    map<string, double> upperBound;
    for (const auto &feature : trueNumericTypesFeatureNames) {
        lowerBound[feature] = std::numeric_limits<double>::max();
        upperBound[feature] = std::numeric_limits<double>::min();
    }

    // Populate with values from the bounded intervals
    for (const auto &item : boundedIntervalsNumericTypes) {
        if (item[1] == ">") {
            lowerBound[item[0]] = convert(item[2]);
        }
        else {
            upperBound[item[0]] = convert(item[2]);
        }
    }

    // Numeric feature case
    for (const auto &featureName : trueNumericTypesFeatureNames) {
        // If the feature has both a lower and upper bound
        if (lowerBound[featureName] != std::numeric_limits<double>::max() &&
            upperBound[featureName] != std::numeric_limits<double>::min()) {
            // If the upper bound is less than or equal to the lower bound, return 0
            if (upperBound[featureName] <= lowerBound[featureName]) {
                return 0;
            }
            else {
                if (!probability) {
                    probability = probabilityOfFeatureLessThanThresholdGivenClass(
                                      featureName, std::to_string(upperBound[featureName]), className) -
                                  probabilityOfFeatureLessThanThresholdGivenClass(
                                      featureName, std::to_string(lowerBound[featureName]), className);
                }
                else {
                    probability *= (probabilityOfFeatureLessThanThresholdGivenClass(
                                        featureName, std::to_string(upperBound[featureName]), className) -
                                    probabilityOfFeatureLessThanThresholdGivenClass(
                                        featureName, std::to_string(lowerBound[featureName]), className));
                }
            }
        }
        // If the feature has only an upper bound
        else if (upperBound[featureName] != std::numeric_limits<double>::min() &&
                 lowerBound[featureName] == std::numeric_limits<double>::max()) {

            if (!probability) {
                probability = probabilityOfFeatureLessThanThresholdGivenClass(
                    featureName, std::to_string(upperBound[featureName]), className);
            }
            else {
                probability *= probabilityOfFeatureLessThanThresholdGivenClass(
                    featureName, std::to_string(upperBound[featureName]), className);
            }
        }
        // If the feature has only a lower bound
        else if (lowerBound[featureName] != std::numeric_limits<double>::max() &&
                 upperBound[featureName] == std::numeric_limits<double>::min()) {

            if (!probability) {
                probability = 1.0 - probabilityOfFeatureLessThanThresholdGivenClass(
                                        featureName, std::to_string(lowerBound[featureName]), className);
            }
            else {
                probability *= (1.0 - probabilityOfFeatureLessThanThresholdGivenClass(
                                          featureName, std::to_string(lowerBound[featureName]), className));
            }
        }
        else {
            throw std::runtime_error("Ill formatted call to 'probability_of_sequence' method");
        }
    }

    // Symbolic feature case
    for (const auto &featureAndValue : symbolicTypes) {
        smatch match;
        if (regex_search(featureAndValue, match, pattern1)) {
            string feature = match[1];
            string value   = match[2];
            if (!probability) {
                probability = probabilityOfFeatureValueGivenClass(feature, value, className);
            }
            else {
                probability *= probabilityOfFeatureValueGivenClass(feature, value, className);
            }
        }
    }


    _probabilityCache[sequenceWithClass] = probability;
    return probability;
}

double DecisionTree::probabilityOfAClassGivenSequenceOfFeaturesAndValuesOrThresholds(
    const string &className, const vector<string> &arrayOfFeaturesAndValuesOrThresholds)
{

    string sequence = "";
    for (const auto &item : arrayOfFeaturesAndValuesOrThresholds) {
        if (item != arrayOfFeaturesAndValuesOrThresholds.back()) {
            sequence += item + ":";
        }
        else {
            sequence += item;
        }
    }
    string classAndSequence = className + "::" + sequence;

    // Check if the probability is already cached
    if (_probabilityCache.find(classAndSequence) != _probabilityCache.end()) {
        return _probabilityCache[classAndSequence];
    }

    // Calculate the probability
    vector<double> arrayOfClassProbabilities = vector<double>(_classNames.size(), 0.0);

    for (size_t i = 0; i < _classNames.size(); ++i) {
        string currentClassName = _classNames[i];
        double probability      = probabilityOfASequenceOfFeaturesAndValuesOrThresholdsGivenClass(
            arrayOfFeaturesAndValuesOrThresholds, currentClassName);
        // check if prob is ~ 0
        if (probability < .000001) {
            arrayOfClassProbabilities[i] = 0.0;
            continue;
        }
        double probOfFeatureSequence = probabilityOfASequenceOfFeaturesAndValuesOrThresholds(
            arrayOfFeaturesAndValuesOrThresholds); // could proll be moved outta loop
        double prior = _classPriorsDict[currentClassName];
        if (probOfFeatureSequence) {
            arrayOfClassProbabilities[i] = (probability * prior) / probOfFeatureSequence;
        }
        else {
            arrayOfClassProbabilities[i] = prior;
        }
    }

    // Normalize the probs
    double sumProbabilities = std::accumulate(arrayOfClassProbabilities.begin(), arrayOfClassProbabilities.end(), 0.0);
    if (sumProbabilities == 0) {
        arrayOfClassProbabilities = vector<double>(_classNames.size(), 1.0 / _classNames.size());
    }
    else {
        // normalize the probabilities
        for (size_t i = 0; i < _classNames.size(); ++i) {
            arrayOfClassProbabilities[i] /= sumProbabilities;
        }
    }

    // Cache the probabilities
    for (size_t i = 0; i < _classNames.size(); ++i) {
        string key             = _classNames[i] + "::" + sequence;
        _probabilityCache[key] = arrayOfClassProbabilities[i];
    }

    // Return the probability
    return _probabilityCache[classAndSequence];
}

//--------------- Class Based Utilities ----------------//

bool DecisionTree::checkNamesUsed(const vector<string> &featuresAndValues)
{
    for (const auto &featureAndValue : featuresAndValues) {
        std::regex pattern(R"(\S+)\s*=\s*(\S+)");
        std::smatch match;
        std::regex_search(featureAndValue, match, pattern);

        auto feature = match[1];
        auto value   = match[2];

        if (feature == "" || value == "") {
            throw std::runtime_error("Your test data has a formatting error");
        }
        if (_featuresAndValuesDict.find(feature) == _featuresAndValuesDict.end()) {
            return false;
        }
        return true;
    }
    return false;
}

DecisionTree &DecisionTree::operator=(const DecisionTree &dt)
{
    return *this;
}

vector<vector<string>> DecisionTree::findBoundedIntervalsForNumericFeatures(const vector<string> &trueNumericTypes)
{
    std::unordered_map<string, pair<double, double>> featureBounds; // Stores {featureName, {min < value, max > value}}
    std::unordered_map<string, bool> hasMinMax;                     // Tracks if a feature has a min or max bound
    // Step 1: Parse each condition and update feature bounds
    for (const string &condition : trueNumericTypes) {
        istringstream ss(condition);
        string featureName, op, valueStr;
        double value;

        // Split the condition into feature name, operator, and value
        size_t operatorPos = condition.find_first_of("<>");
        if (operatorPos != string::npos) {
            featureName = condition.substr(0, operatorPos);
            op          = condition.substr(operatorPos, 1);  // Extract the operator (either '<' or '>')
            valueStr    = condition.substr(operatorPos + 1); // Extract the value
        }
        else {
            // Handle the case where no operator is found (optional, if conditions are well-formed)
            cerr << "Invalid condition: no operator found!" << endl;
        }

        // Convert value to double for comparison
        value = convert(valueStr);

        // Update bounds
        if (op == "<") {
            if (!hasMinMax[featureName]) {
                featureBounds[featureName].first  = value;
                featureBounds[featureName].second = -numeric_limits<double>::infinity();
                hasMinMax[featureName]            = true;
            }
            else {
                featureBounds[featureName].first = std::min(featureBounds[featureName].first, value);
            }
        }
        else if (op == ">") {
            if (!hasMinMax[featureName]) {
                featureBounds[featureName].second = value;
                featureBounds[featureName].first  = numeric_limits<double>::infinity();
                hasMinMax[featureName]            = true;
            }
            else {
                featureBounds[featureName].second = std::max(featureBounds[featureName].second, value);
            }
        }
    }

    // Step 2: Prepare the result in the required format
    vector<vector<string>> result;
    for (const auto &[featureName, bounds] : featureBounds) {
        double minValue = bounds.first;
        double maxValue = bounds.second;

        if (minValue != numeric_limits<double>::infinity()) {
            result.push_back({featureName, "<", to_string(minValue)});
        }
        if (maxValue != -numeric_limits<double>::infinity()) {
            result.push_back({featureName, ">", to_string(maxValue)});
        }
    }

    return result;
}

// print the stree variables
void DecisionTree::printStats()
{
    cout << "Training Datafile:                            " << _trainingDatafile << endl;
    cout << "Entropy Threshold:                            " << _entropyThreshold << endl;
    cout << "Max Depth Desired:                            " << _maxDepthDesired << endl;
    cout << "CSV Class Column Index:                       " << _csvClassColumnIndex << endl;
    cout << "Symbolic To Numeric Cardinality Threshold:    " << _symbolicToNumericCardinalityThreshold << endl;
    cout << "Number Of Histogram Bins:                     " << _numberOfHistogramBins << endl;
    cout << "CSV Cleanup Needed:                           " << _csvCleanupNeeded << endl;
    cout << "Debug1:                                       " << _debug1 << endl;
    cout << "Debug2:                                       " << _debug2 << endl;
    cout << "Debug3:                                       " << _debug3 << endl;
    cout << "How Many Total Training Samples:              " << _howManyTotalTrainingSamples << endl;
    cout << "Feature Names: \n";
    for (const auto &feature : _featureNames) {
        cout << feature << " ";
    }
    cout << endl;
    cout << "Training Data Dict: \n";
    for (const auto &kv : _trainingDataDict) {
        cout << kv.first << ": ";
        for (const auto &v : kv.second) {
            cout << v << " ";
        }
        cout << endl;
    }
    cout << "Features And Values Dict: \n";
    for (const auto &kv : _featuresAndValuesDict) {
        cout << kv.first << ": ";
        for (const auto &v : kv.second) {
            cout << v << " ";
        }
        cout << endl;
    }
}

//--------------- Getters ----------------//
string DecisionTree::getTrainingDatafile() const
{
    return _trainingDatafile;
}

double DecisionTree::getEntropyThreshold() const
{
    return _entropyThreshold;
}

int DecisionTree::getMaxDepthDesired() const
{
    return _maxDepthDesired;
}

int DecisionTree::getCsvClassColumnIndex() const
{
    return _csvClassColumnIndex;
}

vector<int> DecisionTree::getCsvColumnsForFeatures() const
{
    return _csvColumnsForFeatures;
}

int DecisionTree::getSymbolicToNumericCardinalityThreshold() const
{
    return _symbolicToNumericCardinalityThreshold;
}

int DecisionTree::getNumberOfHistogramBins() const
{
    return _numberOfHistogramBins;
}

int DecisionTree::getCsvCleanupNeeded() const
{
    return _csvCleanupNeeded;
}

int DecisionTree::getDebug1() const
{
    return _debug1;
}

int DecisionTree::getDebug2() const
{
    return _debug2;
}

int DecisionTree::getDebug3() const
{
    return _debug3;
}

int DecisionTree::getHowManyTotalTrainingSamples() const
{
    return _howManyTotalTrainingSamples;
}

vector<string> DecisionTree::getFeatureNames() const
{
    return _featureNames;
}

map<int, vector<string>> DecisionTree::getTrainingDataDict() const
{
    return _trainingDataDict;
}

map<string, vector<string>> DecisionTree::getFeaturesAndValuesDict() const
{
    return _featuresAndValuesDict;
}

//--------------- Setters ----------------//
void DecisionTree::setTrainingDatafile(const string &trainingDatafile)
{
    _trainingDatafile = trainingDatafile;
}

void DecisionTree::setEntropyThreshold(double entropyThreshold)
{
    _entropyThreshold = entropyThreshold;
}

void DecisionTree::setMaxDepthDesired(int maxDepthDesired)
{
    _maxDepthDesired = maxDepthDesired;
}

void DecisionTree::setCsvClassColumnIndex(int csvClassColumnIndex)
{
    _csvClassColumnIndex = csvClassColumnIndex;
}

void DecisionTree::setCsvColumnsForFeatures(const vector<int> &csvColumnsForFeatures)
{
    _csvColumnsForFeatures = csvColumnsForFeatures;
}

void DecisionTree::setSymbolicToNumericCardinalityThreshold(int symbolicToNumericCardinalityThreshold)
{
    _symbolicToNumericCardinalityThreshold = symbolicToNumericCardinalityThreshold;
}

void DecisionTree::setNumberOfHistogramBins(int numberOfHistogramBins)
{
    _numberOfHistogramBins = numberOfHistogramBins;
}

void DecisionTree::setCsvCleanupNeeded(int csvCleanupNeeded)
{
    _csvCleanupNeeded = csvCleanupNeeded;
}

void DecisionTree::setDebug1(int debug1)
{
    _debug1 = debug1;
}

void DecisionTree::setDebug2(int debug2)
{
    _debug2 = debug2;
}

void DecisionTree::setDebug3(int debug3)
{
    _debug3 = debug3;
}

void DecisionTree::setRootNode(std::unique_ptr<DecisionTreeNode> rootNode)
{
    _rootNode = std::move(rootNode);
}