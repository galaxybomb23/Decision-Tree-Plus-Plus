#ifndef DECISION_TREE_HPP
#define DECISION_TREE_HPP

// Include
#include "Common.hpp"
#include "DecisionTreeNode.hpp"
#include "Utility.hpp"

#include <iostream>
#include <memory>

/**
 * @struct ClassificationAnswer
 * @brief Represents the result of a classification operation.
 *
 * This structure holds the probabilities of each class and the path taken
 * to reach the solution in a decision tree.
 *
 */
struct ClassificationAnswer {
    map<string, double> classProbabilities;
    vector<int> solutionPath;
};


/**
 * @struct BestFeatureResult
 * @brief A structure to hold the result of the best feature selection in a decision tree algorithm.
 *
 * This structure contains information about the best feature selected during the decision tree
 * construction process, including the feature's name, its entropy, the entropies based on its values,
 * and the decision value.
 */
struct BestFeatureResult {
    string bestFeatureName;
    double bestFeatureEntropy;
    optional<pair<double, double>> valBasedEntropies;
    optional<double> decisionValue;
};


class DecisionTreeNode;


/**
 * @class DecisionTree
 * @brief A class representing a Decision Tree for classification tasks.
 *
 * This class provides functionalities to construct, train, and use a decision tree for classification.
 * It includes methods for calculating probabilities, entropies, and classifying data based on the trained model.
 *
 */
class DecisionTree : public std::enable_shared_from_this<DecisionTree> {
  public:
    int _nodesCreated;
    string _classLabel; // The class label for the training data currently unused
    vector<string> _classNames;

    shared_ptr<DecisionTree> getShared() { return shared_from_this(); }

    //--------------- Constructors and Destructors ----------------//
    DecisionTree();                           // constructor
    DecisionTree(map<string, string> kwargs); // constructor
    virtual ~DecisionTree();                  // destructor

    //--------------- Class Functions ----------------//
    void getTrainingData();
    void calculateFirstOrderProbabilities();
    void showTrainingData() const;

    //--------------- Classify ----------------//
    map<string, string> classify(DecisionTreeNode* rootNode, const vector<string> &featuresAndValues);
    void recursiveDescentForClassification(DecisionTreeNode* node,
                                           const vector<string> &featureAndValues,
                                           ClassificationAnswer &answer);
    ClassificationAnswer classifyByAskingQuestions(DecisionTreeNode* rootNode);
    void interactiveRecursiveDescentForClassification(DecisionTreeNode* node,
                                                      ClassificationAnswer &answer,
                                                      map<string, optional<double>> &scratchpadForNumerics);

    //--------------- Construct Tree ----------------//
    DecisionTreeNode* constructDecisionTreeClassifier();
    void recursiveDescent(DecisionTreeNode* node);
    BestFeatureResult bestFeatureCalculator(const vector<string> &featuresAndValuesOrThresholdsOnBranch,
                                            double existingNodeEntropy);

    //--------------- Entropy Calculators ----------------//
    double classEntropyOnPriors();
    void entropyScannerForANumericFeature(const string &feature);
    double EntropyForThresholdForFeature(const vector<string> &arrayOfFeaturesAndValuesOrThresholds,
                                         const string &feature,
                                         const double &threshold,
                                         const string &comparison);
    double classEntropyForLessThanThresholdForFeature(const vector<string> &arrayOfFeaturesAndValuesOrThresholds,
                                                      const string &feature,
                                                      const double &threshold);

    double classEntropyForGreaterThanThresholdForFeature(const vector<string> &arrayOfFeaturesAndValuesOrThresholds,
                                                         const string &feature,
                                                         const double &threshold);

    double classEntropyForAGivenSequenceOfFeaturesAndValuesOrThresholds(
        const vector<string> &arrayOfFeaturesAndValuesOrThresholds);

    //--------------- Probability Calculators ----------------//
    double priorProbabilityForClass(const string &className);
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
    void determineDataCondition();
    bool checkNamesUsed(const vector<string> &featuresAndValues);
    DecisionTree &operator=(const DecisionTree &dt);
    vector<vector<string>> findBoundedIntervalsForNumericFeatures(const vector<string> &trueNumericTypes);
    void printStats();
    void printClassificationAnswer(ClassificationAnswer answer);

    // --------------- Getters ----------------//
    string getTrainingDatafile() const;
    double getEntropyThreshold() const;
    int getMaxDepthDesired() const;
    int getNumberOfHistogramBins() const;
    vector<string> getClassNames() const;
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
    map<int, string> getSamplesClassLabelDict() const;
    map<string, std::set<string>> getFeaturesAndUniqueValuesDict() const;
    map<string, vector<double>> getNumericFeaturesValueRangeDict() const;
    map<int, vector<string>> getTrainingDataDict() const;
    DecisionTreeNode* getRootNode() const;

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
    void setRootNode(unique_ptr<DecisionTreeNode> rootNode);
    void setClassNames(const vector<string> &classNames);

  public:
    string _trainingDatafile;
    double _entropyThreshold;
    int _maxDepthDesired;
    int _numberOfHistogramBins;
    int _csvClassColumnIndex;
    int _symbolicToNumericCardinalityThreshold;
    int _csvCleanupNeeded;
    int _debug1, _debug2, _debug3;
    int _howManyTotalTrainingSamples;

    unique_ptr<DecisionTreeNode> _rootNode;
    vector<int> _csvColumnsForFeatures;
    map<string, double> _probabilityCache;
    map<string, double> _entropyCache;
    map<int, vector<string>> _trainingDataDict;
    map<string, vector<string>> _featuresAndValuesDict;
    map<string, set<string>> _featuresAndUniqueValuesDict;
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