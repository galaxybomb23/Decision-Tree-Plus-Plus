#include "DecisionTreeNode.hpp"

DecisionTreeNode::DecisionTreeNode(const std::string &feature, double entropy,
                                   const std::vector<double> &class_probabilities,
                                   const std::vector<string> &branch_features_and_values_or_thresholds,
                                   DecisionTree &dt, const bool isRoot)
    : _dt(dt), _feature(feature), _nodeCreationEntropy(entropy),
      _classProbabilities(class_probabilities),
      _branchFeaturesAndValuesOrThresholds(branch_features_and_values_or_thresholds)
{
    if (isRoot)
    {
        dt.nodesCreated = -1;
        dt.classNames.clear();
    }
    _serialNumber = GetNextSerialNum();
}

DecisionTreeNode::~DecisionTreeNode()
{
}

// Other functions below
