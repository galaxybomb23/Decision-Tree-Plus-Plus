#ifndef DECISION_TREE_HPP
#define DECISION_TREE_HPP

// Include
#include "DecisionTreeNode.hpp"

#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

using std::string, std::vector, std::map;

class DecisionTreeNode;
class DecisionTree {
  public:
    //--------------- Constructors and Destructors ----------------//
    DecisionTree(map<string, string> kwargs); // constructor
    ~DecisionTree();                          // destructor

    //--------------- Functions ----------------//
    void getTrainingData();
    void calculateFirstOrderProbabilities();
    void showTrainingData() const;

    //--------------- Classify ----------------//
    map<string, string> classify(DecisionTreeNode* rootNode, const vector<string> &featuresAndValues);
    map<string, double> recursiveDescentForClassification(DecisionTreeNode* node,
                                                          const vector<string> &feature_and_values,
                                                          map<string, vector<double>> &answer);

    //--------------- Construct Tree ----------------//
    DecisionTreeNode* constructDecisionTreeClassifier();
    void recursiveDescent(DecisionTreeNode* node);

  //--------------- Entropy Calculators ----------------//
  double classEntropyOnPriors();
  void entropyScannerForANumericFeature(const std::string& feature);
  double classEntropyForLessThanThresholdForFeature(const std::vector<std::string>& attributes, const std::string& feature, double point);
  double classEntropyForGreaterThanThresholdForFeature(const std::vector<std::string>& attributes, const std::string& feature, double point);
  double classEntropyForAGivenSequenceOfFeaturesAndValuesOrThresholds(const std::vector<std::string>& arrayOfFeaturesAndValuesOrThresholds);

    //--------------- Probability Calculators ----------------//
    double priorProbabilityForClass(const string &className, bool overloadCache = false);
    void calculateClassPriors();
    double probabilityOfFeatureValue(const string &feature, const string &value);
    double probabilityOfFeatureValueGivenClass(const string &feature, const string &value, const string &className);
    double probabilityOfFeatureLessThanThreshold(const string &featureName, const string &threshold);
    double probabilityOfFeatureLessThanThresholdGivenClass(const string &featureName,
                                                           const string &threshold,
                                                           const string &className);

    double
    probabilityOfASequenceOfFeaturesAndValuesOrThresholds(const vector<string> &arrayOfFeaturesAndValuesOrThresholds);
    double probabilityOfASequenceOfFeaturesAndValuesOrThresholdsGivenClass(
        const vector<string> &arrayOfFeaturesAndValuesOrThresholds, const string &className);
    double probabilityOfAClassGivenSequenceOfFeaturesAndValuesOrThresholds(
        const string &className, const vector<string> &arrayOfFeaturesAndValuesOrThresholds);

    //--------------- Class Based Utilities ----------------//
    bool checkNamesUsed(const vector<string> &featuresAndValues);
    DecisionTree &operator=(const DecisionTree &dt);
    vector<vector<string>> findBoundedIntervalsForNumericFeatures(const vector<string> &trueNumericTypes);
    void printStats();

    int _nodesCreated;
    string _classLabel; // The class label for the training data currently unused
    vector<string> _classNames;

    // --------------- Getters ----------------//
    string getTrainingDatafile() const;
    double getEntropyThreshold() const;
    int getMaxDepthDesired() const;
    int getNumberOfHistogramBins() const;
    int getCsvClassColumnIndex() const;
    vector<int> getCsvColumnsForFeatures() const;
    int getSymbolicToNumericCardinalityThreshold() const;
    int getCsvCleanupNeeded() const;
    int getDebug1() const;
    int getDebug2() const;
    int getDebug3() const;
    int getHowManyTotalTrainingSamples() const;
    vector<string> getFeatureNames() const;
    map<string, vector<string>> getFeaturesAndValuesDict() const;
    map<int, vector<string>> getTrainingDataDict() const;

    //---------------- Setters ----------------//
    void setTrainingDatafile(const string &trainingDatafile);
    void setEntropyThreshold(double entropyThreshold);
    void setMaxDepthDesired(int maxDepthDesired);
    void setNumberOfHistogramBins(int numberOfHistogramBins);
    void setCsvClassColumnIndex(int csvClassColumnIndex);
    void setCsvColumnsForFeatures(const vector<int> &csvColumnsForFeatures);
    void setSymbolicToNumericCardinalityThreshold(int symbolicToNumericCardinalityThreshold);
    void setCsvCleanupNeeded(int csvCleanupNeeded);
    void setDebug1(int debug1);
    void setDebug2(int debug2);
    void setDebug3(int debug3);
    void setHowManyTotalTrainingSamples(int howManyTotalTrainingSamples);
    void setRootNode(std::unique_ptr<DecisionTreeNode> rootNode);

  private:
    string _trainingDatafile;
    double _entropyThreshold;
    int _maxDepthDesired;
    int _numberOfHistogramBins;
    int _csvClassColumnIndex;
    int _symbolicToNumericCardinalityThreshold;
    int _csvCleanupNeeded;
    int _debug1, _debug2, _debug3;
    int _howManyTotalTrainingSamples;

    std::unique_ptr<DecisionTreeNode> _rootNode;
    vector<int> _csvColumnsForFeatures;
    map<string, double> _probabilityCache;
    map<string, double> _entropyCache;
    map<int, vector<string>> _trainingDataDict;
    map<string, vector<string>> _featuresAndValuesDict;
    map<string, std::set<string>> _featuresAndUniqueValuesDict;
    map<int, string> _samplesClassLabelDict;
    map<string, double> _classPriorsDict;
    vector<string> _featureNames;
    map<string, vector<double>> _numericFeaturesValueRangeDict;
    map<string, vector<double>> _samplingPointsForNumericFeatureDict;
    map<string, int> _featureValuesHowManyUniquesDict;
    map<string, map<double, double>> _probDistributionNumericFeaturesDict;
    map<string, double> _histogramDeltaDict;
    map<string, int> _numOfHistogramBinsDict;
};

#endif // DECISION_TREE_HPP