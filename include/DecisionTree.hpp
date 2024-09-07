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
    ~DecisionTree();                                         // destructor

    void getTrainingData();
    void calculateFirstOrderProbabilities();
    void showTrainingData() const;

    int _nodesCreated;
    std::vector<std::string> _classNames;

    // Getters
    std::string getTrainingDatafile() const;
    double getEntropyThreshold() const;
    int getMaxDepthDesired() const;
    int getNumberOfHistogramBins() const;
    int getCsvClassColumnIndex() const;
    std::vector<int> getCsvColumnsForFeatures() const;
    int getSymbolicToNumericCardinalityThreshold() const;
    int getCsvCleanupNeeded() const;
    int getDebug1() const;
    int getDebug2() const;
    int getDebug3() const;
    int getHowManyTotalTrainingSamples() const;
    std::vector<std::string> getFeatureNames() const;
    std::map<std::string, std::vector<std::string>> getTrainingDataDict() const;

    // Setters
    void setTrainingDatafile(const std::string &trainingDatafile);
    void setEntropyThreshold(double entropyThreshold);
    void setMaxDepthDesired(int maxDepthDesired);
    void setNumberOfHistogramBins(int numberOfHistogramBins);
    void setCsvClassColumnIndex(int csvClassColumnIndex);
    void setCsvColumnsForFeatures(const std::vector<int> &csvColumnsForFeatures);
    void setSymbolicToNumericCardinalityThreshold(int symbolicToNumericCardinalityThreshold);
    void setCsvCleanupNeeded(int csvCleanupNeeded);
    void setDebug1(int debug1);
    void setDebug2(int debug2);
    void setDebug3(int debug3);
    void setHowManyTotalTrainingSamples(int howManyTotalTrainingSamples);
    void setTrainingDataDict(const std::map<std::string, std::vector<std::string>> &trainingDataDict);
    void setFeaturesAndValuesDict(const std::map<std::string, std::set<std::string>> &featuresAndValuesDict);
    void setFeaturesAndUniqueValuesDict(const std::map<std::string, std::set<std::string>> &featuresAndUniqueValuesDict);
    void setSamplesClassLabelDict(const std::map<std::string, std::string> &samplesClassLabelDict);
    void setClassPriorsDict(const std::map<std::string, double> &classPriorsDict);
    void setFeatureNames(const std::vector<std::string> &featureNames);
    void setNumericFeaturesValueRangeDict(const std::map<std::string, std::vector<double>> &numericFeaturesValueRangeDict);
    void setSamplingPointsForNumericFeatureDict(const std::map<std::string, std::vector<double>> &samplingPointsForNumericFeatureDict);
    void setFeatureValuesHowManyUniquesDict(const std::map<std::string, int> &featureValuesHowManyUniquesDict);
    void setProbDistributionNumericFeaturesDict(const std::map<std::string, std::vector<double>> &probDistributionNumericFeaturesDict);
    void setHistogramDeltaDict(const std::map<std::string, double> &histogramDeltaDict);
    void setNumOfHistogramBinsDict(const std::map<std::string, int> &numOfHistogramBinsDict);

protected:
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