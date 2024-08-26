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

// Other functions below