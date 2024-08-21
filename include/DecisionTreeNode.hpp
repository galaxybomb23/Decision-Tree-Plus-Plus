#ifndef DECISION_TREE_NODE_HPP
#define DECISION_TREE_NODE_HPP

// import decision Tree
#include "DecisionTree.hpp"
#include <memory>
#include <vector>
#include <string>

using namespace std;

class DecisionTreeNode
{
public:
    DecisionTreeNode(const std::string &feature, double entropy,
                     const std::vector<double> &class_probabilities,
                     const std::vector<string> &branch_features_and_values_or_thresholds,
                     DecisionTree &dt, const bool isRoot);
    ~DecisionTreeNode(); // Destructor

    int HowManyNodes();

    // Getters
    vector<string> GetClassNames();
    int GetNextSerialNum() const;
    void GetFeature() const;
    float GetNodeEntropy() const;
    vector<float> GetClassProbabilities() const;
    vector<string> GetBranchFeaturesAndValuesOrThresholds() const;
    vector<shared_ptr<DecisionTreeNode>> GetChildren() const;
    int GetSerialNum() const;

    // Setters
    void SetClassNames(const vector<string> classNames);
    void SetNodeCreationEntropy(const float entropy);
    void AddChildLink(shared_ptr<DecisionTreeNode> newNode);

    void DeleteAllLinks();

    // Displays
    void DisplayNode();
    void DisplayDecisionTree();

private:
    // Private members
    DecisionTree &_dt;
    int _serialNumber;
    string _feature;
    double _nodeCreationEntropy;
    vector<double> _classProbabilities;
    vector<string> _branchFeaturesAndValuesOrThresholds;
    vector<shared_ptr<DecisionTreeNode>> _linked_to; // maybe change to weak if cyclic referencing
};

#endif // DECISION_TREE_NODE_HPP
