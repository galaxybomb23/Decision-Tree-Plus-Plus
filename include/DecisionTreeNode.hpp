#ifndef DECISION_TREE_NODE_HPP
#define DECISION_TREE_NODE_HPP

#include "Common.hpp"
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

/**
 * @class DecisionTreeNode
 * @brief Represents a node in a decision tree.
 *
 * This class encapsulates the properties and behaviors of a node within a decision tree.
 * It includes information about the feature used for splitting, the entropy of the node,
 * class probabilities, and links to child nodes.
 */
class DecisionTreeNode {
  public:
    DecisionTreeNode();                                     // Constructor
    explicit DecisionTreeNode(shared_ptr<DecisionTree> dt); // Constructor
    DecisionTreeNode(const string &feature,
                     double entropy,
                     const vector<double> &class_probabilities,
                     const vector<string> &branch_features_and_values_or_thresholds,
                     shared_ptr<DecisionTree> dt,
                     bool isRoot);

    ~DecisionTreeNode(); // Destructor

    // Copy constructor
    DecisionTreeNode(const DecisionTreeNode &other);
    DecisionTreeNode &operator=(const DecisionTreeNode &other);

    int HowManyNodes() const;

    // Getters
    vector<string> GetClassNames() const;
    int GetNextSerialNum() const;
    string GetFeature() const;
    double GetNodeEntropy() const;
    vector<double> GetClassProbabilities() const;
    vector<string> GetBranchFeaturesAndValuesOrThresholds() const;
    const vector<DecisionTreeNode*> GetChildren() const;
    int GetSerialNum() const;

    // Setters
    void SetClassNames(const vector<string> classNames);
    void SetFeature(const string &feature) { _feature = feature; };
    void SetNodeCreationEntropy(const double entropy);
    void AddChildLink(unique_ptr<DecisionTreeNode> newNode);

    void DeleteAllLinks();

    // Displays
    void DisplayNode(const string &offset) const;
    void DisplayDecisionTree(const string &offset) const;

  private:
    // Private members
    std::weak_ptr<DecisionTree> _dt;
    int _serialNumber;
    string _feature;
    double _nodeCreationEntropy;
    vector<double> _classProbabilities;
    vector<string> _branchFeaturesAndValuesOrThresholds;
    vector<unique_ptr<DecisionTreeNode>> _linkedTo; // maybe change to weak if cyclic referencing
};

#endif // DECISION_TREE_NODE_HPP
