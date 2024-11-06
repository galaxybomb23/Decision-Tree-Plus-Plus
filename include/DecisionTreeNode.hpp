#ifndef DECISION_TREE_NODE_HPP
#define DECISION_TREE_NODE_HPP

#include "DecisionTree.hpp"

#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
// Forward declaration
class DecisionTree;

using namespace std;

class DecisionTreeNode {
  public:
    DecisionTreeNode(DecisionTree &dt); // Constructor

    DecisionTreeNode(const std::string &feature,
                     double entropy,
                     const std::vector<double> &class_probabilities,
                     const std::vector<string> &branch_features_and_values_or_thresholds,
                     DecisionTree &dt,
                     const bool isRoot);
    ~DecisionTreeNode(); // Destructor

    int HowManyNodes();

    // Getters
    vector<string> GetClassNames() const;
    int GetNextSerialNum() const;
    string GetFeature() const;
    double GetNodeEntropy() const;
    vector<double> GetClassProbabilities() const;
    vector<string> GetBranchFeaturesAndValuesOrThresholds() const;
    vector<shared_ptr<DecisionTreeNode>> GetChildren() const;
    int GetSerialNum() const;

    // Setters
    void SetClassNames(const vector<string> classNames);
    void SetNodeCreationEntropy(const double entropy);
    void AddChildLink(shared_ptr<DecisionTreeNode> newNode);

    void DeleteAllLinks();

    // Displays
    void DisplayNode(const std::string &offset) const;
    void DisplayDecisionTree(const std::string &offset) const;

  private:
    // Private members
    DecisionTree &_dt; // by reference may be a problem later
    int _serialNumber;
    string _feature;
    double _nodeCreationEntropy;
    vector<double> _classProbabilities;
    vector<string> _branchFeaturesAndValuesOrThresholds;
    vector<shared_ptr<DecisionTreeNode>> _linkedTo; // maybe change to weak if cyclic referencing
};

#endif // DECISION_TREE_NODE_HPP
