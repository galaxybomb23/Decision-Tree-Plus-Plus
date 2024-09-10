// Include
#include "DecisionTree.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <iterator>
#include <regex>
#include <iomanip>

//--------------- Constructors and Destructors ----------------//
DecisionTree::DecisionTree(std::map<std::string, std::string> kwargs)
{
    if (kwargs.empty()) {
        throw std::invalid_argument("Missing training datafile.");
    }

    // Allowed keys for the kwargs
    std::vector<std::string> allowedKeys = {
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
    _entropyThreshold = 0.01;
    _symbolicToNumericCardinalityThreshold = 10;
    _csvCleanupNeeded = 0;
    _csvColumnsForFeatures = {};
    _debug1 = _debug2 = _debug3 = 0;
    _maxDepthDesired = _csvClassColumnIndex = _numberOfHistogramBins = -1;
    _rootNode = nullptr;
    _howManyTotalTrainingSamples = 0;
    _probabilityCache = {};
    _entropyCache = {};
    _trainingDataDict = {};
    _featuresAndValuesDict = {};
    _featuresAndUniqueValuesDict = {};
    _samplesClassLabelDict = {};
    _classNames = {};
    _classPriorsDict = {};
    _featureNames = {};
    _numericFeaturesValueRangeDict = {};
    _samplingPointsForNumericFeatureDict = {};
    _featureValuesHowManyUniquesDict = {};
    _probDistributionNumericFeaturesDict = {};
    _histogramDeltaDict = {};
    _numOfHistogramBinsDict = {};

    // Check and set keyword arguments
    for (const auto& kv : kwargs) {
        const std::string& key = kv.first;
        const std::string& value = kv.second;

        if (key == "training_datafile") {
            _trainingDatafile = value;
        } else if (key == "entropy_threshold") {
            _entropyThreshold = std::stod(value);
        } else if (key == "max_depth_desired") {
            _maxDepthDesired = std::stoi(value);
        } else if (key == "csv_class_column_index") {
            _csvClassColumnIndex = std::stoi(value);
        } else if (key == "csv_columns_for_features") {
            for (const auto& c : value) {
                _csvColumnsForFeatures.push_back(c);
            }
        } else if (key == "symbolic_to_numeric_cardinality_threshold") {
            _symbolicToNumericCardinalityThreshold = std::stoi(value);
        } else if (key == "number_of_histogram_bins") {
            _numberOfHistogramBins = std::stoi(value);
        } else if (key == "csv_cleanup_needed") {
            _csvCleanupNeeded = std::stoi(value);
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


//--------------- Functions ----------------//

// Get training data
void DecisionTree::getTrainingData() 
{
    // Check if training data file is a CSV file
    if (_trainingDatafile.find(".csv") == std::string::npos) { // std::string.find() returns std::string::npos if not found
        throw std::invalid_argument("Aborted. get_training_data_from_csv() is only for CSV files");
    }
    
    _classNames = {};

    // Open the file
    std::ifstream file(_trainingDatafile); // std::ifstream is used to read input from a file
    if (!file.is_open()) {
        throw std::invalid_argument("Could not open file: " + _trainingDatafile);
    }


    // Read the header
    std::string line;
    if (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string token;
        while (std::getline(ss, token, ',')) {
            // strip leading/trailing whitespaces and \" from the token
            token.erase(0, token.find_first_not_of(" \""));
            token.erase(token.find_last_not_of(" \"") + 1);
            _featureNames.push_back(token); // Get the feature names
        }
    }

    // Read the data
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string token;
        std::vector<std::string> row;
        while (std::getline(ss, token, ',')) {
            // strip leading/trailing whitespaces and \" from the token
            token.erase(0, token.find_first_not_of(" \""));
            token.erase(token.find_last_not_of(" \"") + 1);
            row.push_back(token);
        }
        _trainingDataDict[row[0]] = row;
        _classNames.push_back(row[_csvClassColumnIndex]);
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

    // Get the unique values for each feature
    for (int i = 1; i < _featureNames.size(); i++) {
        std::set<std::string> uniqueValues;
        for (const auto& kv : _trainingDataDict) {
            uniqueValues.insert(kv.second[i]);
        }
        _featuresAndValuesDict[_featureNames[i]] = uniqueValues;
    }
}

// Calculate first order probabilities
void DecisionTree::calculateFirstOrderProbabilities() {
    std::cout << "\nEstimating probabilities...\n";
        for (const auto& feature : _featureNames) {
            // Calculate probability for the feature's value
            probabilityOfFeatureValue(feature, "");

            // Debug output if debug2 is enabled
            if (_debug2) {
                // Check if the feature has a probability distribution for numeric values
                if (_probDistributionNumericFeaturesDict.find(feature) != _probDistributionNumericFeaturesDict.end()) {
                    std::cout << "\nPresenting probability distribution for a feature considered to be numeric:\n";
                    // Output sorted sampling points and their probabilities
                    for (auto it = _probDistributionNumericFeaturesDict[feature].begin(); it != _probDistributionNumericFeaturesDict[feature].end(); ++it) {
                        double samplingPoint = *it;
                        double prob = probabilityOfFeatureValue(feature, samplingPoint);
                        std::cout << feature << "::" << samplingPoint << " = " 
                                  << std::setprecision(5) << prob << "\n";
                    }
                } else {
                    // Output probabilities for symbolic feature values
                    std::cout << "\nPresenting probabilities for the values of a feature considered to be symbolic:\n";
                    const auto& values_for_feature = _featuresAndUniqueValuesDict[feature];
                    for (const auto& value : values_for_feature) {
                        double prob = probabilityOfFeatureValue(feature, value);
                        std::cout << feature << "::" << value << " = " 
                                  << std::setprecision(5) << prob << "\n";
                    }
                }
            }
        }
}

// Show training data
void DecisionTree::showTrainingData() const {
    for (const auto& kv : _trainingDataDict) {
        std::cout << kv.first << ": ";
        for (const auto& v : kv.second) {
            std::cout << v << " ";
        }
        std::cout << std::endl;
    }
}


//--------------- Classify ----------------//

std::map<std::string, std::string> DecisionTree::classify(void* root_node, const std::vector<std::string>& features_and_values) {
    return {};
}


//--------------- Construct Tree ----------------//

DecisionTreeNode* DecisionTree::constructDecisionTreeClassifier() {
    return nullptr;
}


//--------------- Entropy Calculators ----------------//

double DecisionTree::classEntropyOnPriors() {
    return 0.0;
}


//--------------- Probability Calculators ----------------//

double DecisionTree::probabilityOfFeatureValue(const std::string& feature, const std::string& value) {
    return 1.0;
}

double DecisionTree::probabilityOfFeatureValue(const std::string& feature, double sampling_point) {
    return 1.0;
}


//--------------- Class Based Utilities ----------------//

// Getters
std::string DecisionTree::getTrainingDatafile() const {
    return _trainingDatafile;
}

double DecisionTree::getEntropyThreshold() const {
    return _entropyThreshold;
}

int DecisionTree::getMaxDepthDesired() const {
    return _maxDepthDesired;
}

int DecisionTree::getCsvClassColumnIndex() const {
    return _csvClassColumnIndex;
}

std::vector<int> DecisionTree::getCsvColumnsForFeatures() const {
    return _csvColumnsForFeatures;
}

int DecisionTree::getSymbolicToNumericCardinalityThreshold() const {
    return _symbolicToNumericCardinalityThreshold;
}

int DecisionTree::getNumberOfHistogramBins() const {
    return _numberOfHistogramBins;
}

int DecisionTree::getCsvCleanupNeeded() const {
    return _csvCleanupNeeded;
}

int DecisionTree::getDebug1() const {
    return _debug1;
}

int DecisionTree::getDebug2() const {
    return _debug2;
}

int DecisionTree::getDebug3() const {
    return _debug3;
}

int DecisionTree::getHowManyTotalTrainingSamples() const {
    return _howManyTotalTrainingSamples;
}

std::vector<std::string> DecisionTree::getFeatureNames() const {
    return _featureNames;
}

std::map<std::string, std::vector<std::string>> DecisionTree::getTrainingDataDict() const {
    return _trainingDataDict;
}

// Setters
void DecisionTree::setTrainingDatafile(const std::string& trainingDatafile) {
    _trainingDatafile = trainingDatafile;
}

void DecisionTree::setEntropyThreshold(double entropyThreshold) {
    _entropyThreshold = entropyThreshold;
}

void DecisionTree::setMaxDepthDesired(int maxDepthDesired) {
    _maxDepthDesired = maxDepthDesired;
}

void DecisionTree::setCsvClassColumnIndex(int csvClassColumnIndex) {
    _csvClassColumnIndex = csvClassColumnIndex;
}

void DecisionTree::setCsvColumnsForFeatures(const std::vector<int>& csvColumnsForFeatures) {
    _csvColumnsForFeatures = csvColumnsForFeatures;
}

void DecisionTree::setSymbolicToNumericCardinalityThreshold(int symbolicToNumericCardinalityThreshold) {
    _symbolicToNumericCardinalityThreshold = symbolicToNumericCardinalityThreshold;
}

void DecisionTree::setNumberOfHistogramBins(int numberOfHistogramBins) {
    _numberOfHistogramBins = numberOfHistogramBins;
}

void DecisionTree::setCsvCleanupNeeded(int csvCleanupNeeded) {
    _csvCleanupNeeded = csvCleanupNeeded;
}

void DecisionTree::setDebug1(int debug1) {
    _debug1 = debug1;
}

void DecisionTree::setDebug2(int debug2) {
    _debug2 = debug2;
}

void DecisionTree::setDebug3(int debug3) {
    _debug3 = debug3;
}