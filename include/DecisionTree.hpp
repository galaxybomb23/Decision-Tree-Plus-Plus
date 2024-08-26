#ifndef DECISION_TREE_HPP
#define DECISION_TREE_HPP

// Include
#include "DecisionTreeNode.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <memory>

class DecisionTreeNode;
class DecisionTree
{
public:
    DecisionTree(std::map<std::string, std::string> kwargs); // constructor
    ~DecisionTree(); // destructor

    void getTrainingData();
    void calculateFirstOrderProbabilities();
    void showTrainingData() const;

    int _nodesCreated;
    std::vector<std::string> _classNames;

private:
    std::string _trainingDatafile;
    double _entropyThreshold;
    int _maxDepthDesired;
    int _numberOfHistogramBins;
    int _csvClassColumnIndex;
    int _symbolicToNumericCardinalityThreshold;
    int _csvCleanupNeeded;
    int _debug1, _debug2, _debug3;
    int _howManyTotalTrainingSamples;

    std::unique_ptr<DecisionTreeNode> _rootNode;
    std::vector<int> _csvColumnsForFeatures;
    std::map<std::string, double> _probabilityCache;
    std::map<std::string, double> _entropyCache;
    std::map<std::string, std::vector<std::string>> _trainingDataDict;
    std::map<std::string, std::set<std::string>> _featuresAndValuesDict;
    std::map<std::string, std::set<std::string>> _featuresAndUniqueValuesDict;
    std::map<std::string, std::string> _samplesClassLabelDict;
    std::map<std::string, double> _classPriorsDict;
    std::vector<std::string> _featureNames;
    std::map<std::string, std::vector<double>> _numericFeaturesValueRangeDict;
    std::map<std::string, std::vector<double>> _samplingPointsForNumericFeatureDict;
    std::map<std::string, int> _featureValuesHowManyUniquesDict;
    std::map<std::string, std::vector<double>> _probDistributionNumericFeaturesDict;
    std::map<std::string, double> _histogramDeltaDict;
    std::map<std::string, int> _numOfHistogramBinsDict;
};

#endif // DECISION_TREE_HPP