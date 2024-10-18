// Include
#include "DecisionTree.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "Utility.hpp"
#include "logger.cpp"

using std::string, std::vector, std::map;

// --------------- Logger --------------- //
Logger logger("../logs/decisionTree.log");

//--------------- Constructors and Destructors ----------------//
DecisionTree::DecisionTree(map<string, string> kwargs)
{
  if (kwargs.empty())
  {
    throw std::invalid_argument("Missing training datafile.");
  }

  // Allowed keys for the kwargs
  vector<string> allowedKeys = {
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
      "debug3"};

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
  for (const auto &kv : kwargs)
  {
    const string &key = kv.first;
    const string &value = kv.second;

    if (key == "training_datafile")
    {
      _trainingDatafile = value;
    }
    else if (key == "entropy_threshold")
    {
      _entropyThreshold = std::stod(value);
    }
    else if (key == "max_depth_desired")
    {
      _maxDepthDesired = std::stoi(value);
    }
    else if (key == "csv_class_column_index")
    {
      _csvClassColumnIndex = std::stoi(value);
    }
    else if (key == "csv_columns_for_features")
    {
      for (const auto &c : value)
      {
        _csvColumnsForFeatures.push_back(c);
      }
    }
    else if (key == "symbolic_to_numeric_cardinality_threshold")
    {
      _symbolicToNumericCardinalityThreshold = std::stoi(value);
    }
    else if (key == "number_of_histogram_bins")
    {
      _numberOfHistogramBins = std::stoi(value);
    }
    else if (key == "csv_cleanup_needed")
    {
      _csvCleanupNeeded = std::stoi(value);
    }
    else if (key == "debug1")
    {
      _debug1 = std::stoi(value);
    }
    else if (key == "debug2")
    {
      _debug2 = std::stoi(value);
    }
    else if (key == "debug3")
    {
      _debug3 = std::stoi(value);
    }
    else
    {
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
  if (_trainingDatafile.find(".csv") == string::npos)
  { // string.find() returns string::npos if not found
    throw std::invalid_argument("Aborted. get_training_data_from_csv() is only for CSV files");
  }

  _classNames = {};

  // Open the file
  std::ifstream file(_trainingDatafile); // std::ifstream is used to read input from a file
  if (!file.is_open())
  {
    throw std::invalid_argument("Could not open file: " + _trainingDatafile);
  }

  // Read the header
  string line;
  if (std::getline(file, line))
  {
    std::istringstream ss(line);
    string token;
    while (std::getline(ss, token, ','))
    {
      // strip leading/trailing whitespaces and \" from the token
      token.erase(0, token.find_first_not_of(" \""));
      token.erase(token.find_last_not_of(" \"") + 1);
      _featureNames.push_back(token); // Get the feature names
    }
  }

  // Read the data
  while (std::getline(file, line))
  {
    std::istringstream ss(line);
    string token;
    vector<string> row;
    while (std::getline(ss, token, ','))
    {
      // strip leading/trailing whitespaces and \" from the token
      token.erase(0, token.find_first_not_of(" \""));
      token.erase(token.find_last_not_of(" \"") + 1);
      row.push_back(token);
    }

    // remove the first element from the row
    int uniqueId = std::stoi(row.front());
    // row.erase(row.begin());
    _trainingDataDict[uniqueId] = row;
    _samplesClassLabelDict[uniqueId] = row[_csvClassColumnIndex];
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

  // Get the features and their values
  for (int i = 1; i < _featureNames.size(); i++)
  {
    vector<string> allValues;
    std::set<string> uniqueValues;
    for (const auto &kv : _trainingDataDict)
    {
      allValues.push_back(kv.second[i]);
      uniqueValues.insert(kv.second[i]);
    }
    _featuresAndValuesDict[_featureNames[i]] = allValues;
    _featuresAndUniqueValuesDict[_featureNames[i]] = uniqueValues;
  }

  // itterate the _trainingDataDict remove the first element from the row
  for (auto &kv : _trainingDataDict)
  {
    kv.second.erase(kv.second.begin());
  }
}

// Calculate first order probabilities
void DecisionTree::calculateFirstOrderProbabilities()
{
  std::cout << "\nEstimating probabilities...\n";
  for (const auto &feature : _featureNames)
  {
    // Calculate probability for the feature's value
    probabilityOfFeatureValue(feature, "");

    // Debug output if debug2 is enabled
    if (_debug2)
    {
      // Check if the feature has a probability distribution for numeric values
      if (_probDistributionNumericFeaturesDict.find(feature) !=
          _probDistributionNumericFeaturesDict.end())
      {
        std::cout << "\nPresenting probability distribution for a feature "
                     "considered to be numeric:\n";
        // Output sorted sampling points and their probabilities
        for (auto it = _probDistributionNumericFeaturesDict[feature].begin();
             it != _probDistributionNumericFeaturesDict[feature].end(); ++it)
        {
          string samplingPoint = std::to_string(*it);
          double prob = probabilityOfFeatureValue(feature, samplingPoint);
          std::cout << feature << "::" << samplingPoint << " = "
                    << std::setprecision(5) << prob << "\n";
        }
      }
      else
      {
        // Output probabilities for symbolic feature values
        std::cout << "\nPresenting probabilities for the values of a feature "
                     "considered to be symbolic:\n";
        const auto &values_for_feature = _featuresAndUniqueValuesDict[feature];
        for (const auto &value : values_for_feature)
        {
          double prob = probabilityOfFeatureValue(feature, value);
          std::cout << feature << "::" << value << " = " << std::setprecision(5)
                    << prob << "\n";
        }
      }
    }
  }
}

// Show training data
void DecisionTree::showTrainingData() const
{
  for (const auto &kv : _trainingDataDict)
  {
    std::cout << kv.first << ": ";
    for (const auto &v : kv.second)
    {
      std::cout << v << " ";
    }
    std::cout << std::endl;
  }
}

//--------------- Classify ----------------//

map<string, string> DecisionTree::classify(
    DecisionTreeNode *rootNode,
    const vector<string> &featuresAndValues)
{
  /*
  Classifies one test sample at a time using the decision tree constructed from
  your training file.  The data record for the test sample must be supplied as
  shown in the scripts in the `Examples' subdirectory.  See the scripts
  construct_dt_and_classify_one_sample_caseX.py in that subdirectory.
  */
  if (!checkNamesUsed(featuresAndValues))
  {
    throw std::runtime_error(
        "\n\nError in the names you have used for features and/or values. "
        "Try using the csv_cleanup_needed option in the constructor call.");
  }

  vector<string> newFeaturesAndValues;
  std::regex pattern(R"((\S+)\s*=\s*(\S+))");
  std::smatch match;

  for (const auto &fv : featuresAndValues)
  {
    if (std::regex_match(fv, match, pattern))
    {
      string feature = match[1];
      string value = match[2];
      newFeaturesAndValues.push_back(feature + "=" + value);
    }
    else
    {
      throw std::runtime_error(
          "\n\nError in the format of the feature and value pairs. "
          "Use the format feature=value.");
    }
  }

  // Update the features and values
  for (const auto &fv : newFeaturesAndValues)
  {
    string feature = fv.substr(0, fv.find("="));
    string value = fv.substr(fv.find("=") + 1);
    _featuresAndValuesDict[feature].push_back(value);
  }

  if (_debug3)
  {
    std::cout << "\nCL1 New features and values:\n";
    for (const auto &item : newFeaturesAndValues)
    {
      std::cout << item << " ";
    }
  }

  map<string, vector<double>> answer;
  for (const auto &className : _classNames)
  {
    answer[className] = {};
  }
  answer["solution_path"] = {};

  map<string, double> classification =
      recursiveDescentForClassification(rootNode, newFeaturesAndValues, answer);
  std::reverse(answer["solution_path"].begin(), answer["solution_path"].end());

  if (_debug3)
  {
    std::cout << "\nCL2 The classification:" << std::endl;
    for (const auto &className : _classNames)
    {
      std::cout << "    " << className << " with probability "
                << classification[className] << std::endl;
    }
  }

  map<string, string> classificationForDisplay = {};
  for (const auto &kv : classification)
  {
    if (std::isfinite(kv.second))
    {
      std::ostringstream oss;
      oss << std::fixed << std::setprecision(3) << kv.second;
      classificationForDisplay[kv.first] = oss.str();
    }
    else
    {
      vector<string> nodes;
      for (const auto &x : kv.first)
      {
        nodes.push_back("NODE" + std::to_string(x));
      }
      std::ostringstream oss;
      std::copy(nodes.begin(), nodes.end(),
                std::ostream_iterator<string>(oss, ", "));
      classificationForDisplay[kv.first] = oss.str();
    }
  }

  return classificationForDisplay;
}

map<string, double> DecisionTree::recursiveDescentForClassification(
    DecisionTreeNode *node, const vector<string> &featureAndValues,
    map<string, vector<double>> &answer)
{
  vector<shared_ptr<DecisionTreeNode>> children = node->GetChildren();

  if (children.empty())
  {
    // If leaf node, assign class probabilities
    vector<double> leafNodeClassProbabilities =
        node->GetClassProbabilities();
    map<string, double> classProbabilities;
    for (size_t i = 0; i < _classNames.size(); ++i)
    {
      classProbabilities[_classNames[i]] = leafNodeClassProbabilities[i];
    }
    answer["solution_path"].push_back(node->GetNextSerialNum());
    return classProbabilities;
  }

  string featureTestedAtNode = node->GetFeature();
  if (_debug3)
  {
    std::cout << "\nCLRD1 Feature tested at node for classifcation: "
              << featureTestedAtNode << std::endl;
  }

  string valueForFeature;
  bool pathFound = false;
  std::regex pattern(R"((\S+)\s*=\s*(\S+))");
  std::smatch match;

  // Find the value for the feature being tested
  for (const auto &featureAndValue : featureAndValues)
  {
    if (std::regex_search(featureAndValue, match, pattern))
    {
      string feature = match[1].str();
      string value = match[2].str();
      if (feature == featureTestedAtNode)
      {
        valueForFeature = convert(value);
      }
    }
  }

  // Handle missing feature values
  if (valueForFeature.empty())
  {
    vector<double> leafNodeClassProbabilities =
        node->GetClassProbabilities();
    map<string, double> classProbabilities;
    for (size_t i = 0; i < _classNames.size(); ++i)
    {
      classProbabilities[_classNames[i]] = leafNodeClassProbabilities[i];
    }
    answer["solution_path"].push_back(node->GetNextSerialNum());

    return classProbabilities;
  }

  // Numeric feature case
  if (_probDistributionNumericFeaturesDict.find(featureTestedAtNode) !=
      _probDistributionNumericFeaturesDict.end())
  {
    if (_debug3)
      std::cout << "\nCLRD2 In the numeric section";
    for (const auto &child : children)
    {
      vector<string> branchFeaturesAndValues =
          child->GetBranchFeaturesAndValuesOrThresholds();
      string lastFeatureAndValueOnBranch = branchFeaturesAndValues.back();
      std::regex pattern1(R"((.+)<(.+))");
      std::regex pattern2(R"((.+)>(.+))");

      if (std::regex_search(lastFeatureAndValueOnBranch, match, pattern1))
      {
        string threshold = match[2].str();
        if (std::stod(valueForFeature) <= std::stod(threshold))
        {
          pathFound = true;
          auto result = recursiveDescentForClassification(
              child.get(), featureAndValues, answer);
          answer.insert(result.begin(), result.end());
          answer["solution_path"].push_back(node->GetNextSerialNum());
          break;
        }
      }
      else if (std::regex_search(lastFeatureAndValueOnBranch, match,
                                 pattern2))
      {
        string threshold = match[2].str();
        if (std::stod(valueForFeature) > std::stod(threshold))
        {
          pathFound = true;
          auto result = recursiveDescentForClassification(
              child.get(), featureAndValues, answer);
          answer.insert(result.begin(), result.end());
          answer["solution_path"].push_back(node->GetNextSerialNum());
          break;
        }
      }
    }

    if (pathFound)
    {
      map<string, double> result;
      for (const auto &kv : answer)
      {
        if (kv.first != "solution_path")
        {
          result[kv.first] = kv.second.empty() ? 0.0 : kv.second[0];
        }
      }

      return result;
    }
  }
  else
  { // Symbolic feature case
    string featureValueCombo = featureTestedAtNode + "=" + valueForFeature;
    if (_debug3)
      std::cout << "\nCLRD3 In the symbolic section with feature_value_combo: "
                << featureValueCombo;

    for (const auto &child : children)
    {
      vector<string> branch_features_and_values =
          child->GetBranchFeaturesAndValuesOrThresholds();
      if (_debug3)
        std::cout << "\nCLRD4 branch features and values: "
                  << branch_features_and_values.back();
      string lastFeatureAndValueOnBranch =
          branch_features_and_values.back();

      if (lastFeatureAndValueOnBranch == featureValueCombo)
      {
        auto result = recursiveDescentForClassification(
            child.get(), featureAndValues, answer);
        answer.insert(result.begin(), result.end());
        answer["solution_path"].push_back(node->GetNextSerialNum());
        pathFound = true;
        break;
      }
    }

    if (pathFound)
    {
      map<string, double> result;
      for (const auto &kv : answer)
      {
        if (kv.first != "solution_path")
        {
          result[kv.first] = kv.second.empty() ? 0.0 : kv.second[0];
        }
      }

      return result;
    }
  }

  // If no path found, assign class probabilities from the current node
  if (!pathFound)
  {
    vector<double> leafNodeClassProbabilities =
        node->GetClassProbabilities();
    for (size_t i = 0; i < _classNames.size(); ++i)
    {
      answer[_classNames[i]].push_back(leafNodeClassProbabilities[i]);
    }
    answer["solution_path"].push_back(node->GetNextSerialNum());
  }

  map<string, double> result;
  for (const auto &kv : answer)
  {
    if (kv.first != "solution_path")
    {
      result[kv.first] = kv.second.empty() ? 0.0 : kv.second[0];
    }
  }

  return result;
}

//--------------- Construct Tree ----------------//

DecisionTreeNode *DecisionTree::constructDecisionTreeClassifier()
{
  /*
  Construct the root node object and set its entropy value as derived from the
  priors associated with the different classes.
  */
  std::cout << "\nConstructing a decision tree" << std::endl;
  if (_debug3)
  {
    // TODO //
    // determineDataCondition();
    std::cout << std::endl
              << "Starting construction of the decision tree:" << std::endl;
  }

  // Calculate prior class probabilities
  vector<double> classProbabilities;
  for (const auto &className : _classNames)
  {
    // TODO //
    // classProbabilities.push_back(priorProbabilityForClass(className));
  }

  if (_debug3)
  {
    std::cout << std::endl
              << "Prior probabilities for the classes:" << std::endl;
    for (size_t i = 0; i < _classNames.size(); ++i)
    {
      std::cout << "    " << _classNames[i] << " with probability "
                << classProbabilities[i] << std::endl;
    }
  }

  double entropy = classEntropyOnPriors();
  if (_debug3)
  {
    std::cout << std::endl
              << "Entropy on priors: " << entropy << std::endl;
  }

  // Create the root node
  DecisionTreeNode *rootNode = new DecisionTreeNode(
      "root", entropy, classProbabilities, {}, *this, true);
  rootNode->SetClassNames(_classNames);
  setRootNode(std::unique_ptr<DecisionTreeNode>(rootNode));

  // Start recursive descent
  recursiveDescent(rootNode);

  return rootNode;
}

void DecisionTree::recursiveDescent(DecisionTreeNode *node) {}

//--------------- Entropy Calculators ----------------//

double DecisionTree::classEntropyOnPriors() { return 0.0; }

//--------------- Probability Calculators ----------------//
double DecisionTree::priorProbabilityForClass(const string &className,
                                              bool overloadCache)
{
  // make a cache key
  string classNameInCache = "prior::" + className;
  logger.log(LogLevel(0), "priorProbabilityForClass:: classNameInCache: " +
                              classNameInCache);

  // Check if the probability is already in the cache (memoization)
  if (_probabilityCache.find(classNameInCache) != _probabilityCache.end() &&
      !overloadCache)
  {
    logger.log(LogLevel(0),
               "priorProbabilityForClass:: probability found in cache: " +
                   std::to_string(_probabilityCache[classNameInCache]));
    return _probabilityCache[classNameInCache];
  }

  logger.log(LogLevel(0),
             "priorProbabilityForClass:: probability not found in cache");
  size_t totalNumSamples = _samplesClassLabelDict.size();
  // get get value from the dictionary
  vector<string> allValues = {};
  for (const auto &kv : _samplesClassLabelDict)
  {
    allValues.push_back(kv.second);
  }

  // itterate over all class names to calculate their prior probabilities
  for (const auto &className : _classNames)
  {
    // get the number of samples for the class
    size_t numSamplesForClass =
        std::count(allValues.begin(), allValues.end(), className);
    // calculate the prior probability for the class
    double priorProbability = static_cast<double>(numSamplesForClass) /
                              static_cast<double>(totalNumSamples);

    // store the prior probability in the cache
    string thisClassName = "prior::" + className;
    _probabilityCache[thisClassName] = priorProbability;
    logger.log(LogLevel(0),
               "priorProbabilityForClass:: prior probability for " + className +
                   ": " + std::to_string(priorProbability));
  }
  return _probabilityCache[classNameInCache];
}

void DecisionTree::calculateClassPriors()
{
  std::cout << "\nCalculating class priors...\n";
  if (_samplesClassLabelDict.size() > 1)
  {
    return;
  }

  //
  for (const auto &className : _classNames)
  {
    priorProbabilityForClass(className, true);
  }
  if (_debug2)
  {
    std::cout << "\nClass priors calculated:\n"
              << std::endl;
    for (const auto &className : _classNames)
    {
      std::cout << className << " = " << priorProbabilityForClass(className)
                << std::endl;
    }
  }
}

double DecisionTree::probabilityOfFeatureValue(const string &feature,
                                               const string &Value)
{
	string value = Value;
	// Convert the value to double, or NAN if it is symbolic
    double valueAsDouble = convert(value);
    string featureAndValue;

    // If the feature is numeric, find the closest sampling point
    if (!std::isnan(valueAsDouble) && _samplingPointsForNumericFeatureDict.find(feature) != _samplingPointsForNumericFeatureDict.end()) {
		value = std::to_string(ClosestSamplingPoint(_samplingPointsForNumericFeatureDict[feature], valueAsDouble));
    }

	// Create a combined feature and value string
    if (!value.empty()) {
        featureAndValue = feature + "=" + std::to_string(convert(value));
    }

	// Check if the probability is already cached
    if (_probabilityCache.find(featureAndValue) != _probabilityCache.end()) {
        return _probabilityCache[featureAndValue];
    }

	// Initialize variables for histogram calculations
    double histogramDelta = 0.0;
    double diffrange = 0.0;
    vector<double> valuerange = {};
    int numOfHistogramBins = 0;

    // Check if the feature is numeric and has a large number of unique values
    if (_numericFeaturesValueRangeDict.find(feature) != _numericFeaturesValueRangeDict.end()) {
        if (_featureValuesHowManyUniquesDict[feature] > _symbolicToNumericCardinalityThreshold) {
            if (_samplingPointsForNumericFeatureDict.find(feature) != _samplingPointsForNumericFeatureDict.end()) {
                // Calculate the histogram delta and sampling points for the feature
				valuerange = _numericFeaturesValueRangeDict[feature];
                diffrange = valuerange[1] - valuerange[0];

                vector<string> values = _featuresAndValuesDict[feature];
                std::set<string> uniqueValues;

                for (const auto &v : values) {
                    if (v != "NA") {
                        uniqueValues.insert(v);
                    }
                }

                vector<string> sortedUniqueValues(uniqueValues.begin(), uniqueValues.end());
                std::sort(sortedUniqueValues.begin(), sortedUniqueValues.end());

                vector<double> diffs;
                for (size_t i = 1; i < sortedUniqueValues.size(); ++i) {
                    diffs.push_back(convert(sortedUniqueValues[i]) - convert(sortedUniqueValues[i - 1]));
                }
                std::sort(diffs.begin(), diffs.end());

                auto medianDiff = diffs[(diffs.size() / 2) - 1];
                histogramDelta = medianDiff * 2.0;
                if (histogramDelta < diffrange / 500.0) {
                    if (_numberOfHistogramBins > 0) {
                        histogramDelta = diffrange / static_cast<double>(_numberOfHistogramBins);
                    } else {
                        histogramDelta = diffrange / 500.0;
                    }
                }

                _histogramDeltaDict[feature] = histogramDelta;
                numOfHistogramBins = static_cast<int>(diffrange / histogramDelta);
                _numOfHistogramBinsDict[feature] = numOfHistogramBins;
                
                vector<double> sampling_points_for_feature;
                for (int j = 0; j < numOfHistogramBins; ++j) {
                    sampling_points_for_feature.push_back(valuerange[0] + histogramDelta * j);
                }

                _samplingPointsForNumericFeatureDict[feature] = sampling_points_for_feature;

            }
        }
    }
    

    
    return 1.0;
}

//--------------- Class Based Utilities ----------------//

bool DecisionTree::checkNamesUsed(
    const vector<string> &featuresAndValues)
{
  return true;
}

// Getters
string DecisionTree::getTrainingDatafile() const
{
  return _trainingDatafile;
}

double DecisionTree::getEntropyThreshold() const { return _entropyThreshold; }

int DecisionTree::getMaxDepthDesired() const { return _maxDepthDesired; }

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

int DecisionTree::getCsvCleanupNeeded() const { return _csvCleanupNeeded; }

int DecisionTree::getDebug1() const { return _debug1; }

int DecisionTree::getDebug2() const { return _debug2; }

int DecisionTree::getDebug3() const { return _debug3; }

int DecisionTree::getHowManyTotalTrainingSamples() const
{
  return _howManyTotalTrainingSamples;
}

vector<string> DecisionTree::getFeatureNames() const
{
  return _featureNames;
}

map<int, vector<string>>
DecisionTree::getTrainingDataDict() const
{
  return _trainingDataDict;
}

map<string, vector<string>>
DecisionTree::getFeaturesAndValuesDict() const
{
  return _featuresAndValuesDict;
}

// Setters
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

void DecisionTree::setCsvColumnsForFeatures(
    const vector<int> &csvColumnsForFeatures)
{
  _csvColumnsForFeatures = csvColumnsForFeatures;
}

void DecisionTree::setSymbolicToNumericCardinalityThreshold(
    int symbolicToNumericCardinalityThreshold)
{
  _symbolicToNumericCardinalityThreshold =
      symbolicToNumericCardinalityThreshold;
}

void DecisionTree::setNumberOfHistogramBins(int numberOfHistogramBins)
{
  _numberOfHistogramBins = numberOfHistogramBins;
}

void DecisionTree::setCsvCleanupNeeded(int csvCleanupNeeded)
{
  _csvCleanupNeeded = csvCleanupNeeded;
}

void DecisionTree::setDebug1(int debug1) { _debug1 = debug1; }

void DecisionTree::setDebug2(int debug2) { _debug2 = debug2; }

void DecisionTree::setDebug3(int debug3) { _debug3 = debug3; }

void DecisionTree::setRootNode(std::unique_ptr<DecisionTreeNode> rootNode)
{
  _rootNode = std::move(rootNode);
}

// print the stree variables
void DecisionTree::printStats()
{
  std::cout << "Training Datafile: " << _trainingDatafile << std::endl;
  std::cout << "Entropy Threshold: " << _entropyThreshold << std::endl;
  std::cout << "Max Depth Desired: " << _maxDepthDesired << std::endl;
  std::cout << "CSV Class Column Index: " << _csvClassColumnIndex << std::endl;
  std::cout << "Symbolic To Numeric Cardinality Threshold: "
            << _symbolicToNumericCardinalityThreshold << std::endl;
  std::cout << "Number Of Histogram Bins: " << _numberOfHistogramBins
            << std::endl;
  std::cout << "CSV Cleanup Needed: " << _csvCleanupNeeded << std::endl;
  std::cout << "Debug1: " << _debug1 << std::endl;
  std::cout << "Debug2: " << _debug2 << std::endl;
  std::cout << "Debug3: " << _debug3 << std::endl;
  std::cout << "How Many Total Training Samples: "
            << _howManyTotalTrainingSamples << std::endl;
  std::cout << "Feature Names: ";
  for (const auto &feature : _featureNames)
  {
    std::cout << feature << " ";
  }
  std::cout << std::endl;
  std::cout << "Training Data Dict: ";
  for (const auto &kv : _trainingDataDict)
  {
    std::cout << kv.first << ": ";
    for (const auto &v : kv.second)
    {
      std::cout << v << " ";
    }
    std::cout << std::endl;
  }
  std::cout << "Features And Values Dict: ";
  for (const auto &kv : _featuresAndValuesDict)
  {
    std::cout << kv.first << ": ";
    for (const auto &v : kv.second)
    {
      std::cout << v << " ";
    }
    std::cout << std::endl;
  }
}
