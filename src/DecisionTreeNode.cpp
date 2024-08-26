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
        _dt._nodesCreated = -1;
        _dt._classNames.clear();
    }
    _serialNumber = GetNextSerialNum();
}

DecisionTreeNode::~DecisionTreeNode()
{
}

int DecisionTreeNode::GetNextSerialNum() const
{
    return _dt._nodesCreated + 1; // placeholder
}

// Other functions below
