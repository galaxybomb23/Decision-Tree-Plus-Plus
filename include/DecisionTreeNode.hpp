#ifndef DECISION_TREE_NODE_HPP
#define DECISION_TREE_NODE_HPP

#include "DecisionTree.hpp"
#include "Common.hpp"

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
    // MARK: figure out whats going on with Node ptrs

    explicit DecisionTreeNode(shared_ptr<DecisionTree> dt); // Constructor

    // MARK: figure out whats going on with Node ptrs
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
    const vector<unique_ptr<DecisionTreeNode>> &GetChildren() const;
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
    // MARK: figure out whats going on with Node ptrs
    std::weak_ptr<DecisionTree> _dt;
    int _serialNumber;
    string _feature;
    double _nodeCreationEntropy;
    vector<double> _classProbabilities;
    vector<string> _branchFeaturesAndValuesOrThresholds;
    vector<unique_ptr<DecisionTreeNode>> _linkedTo; // maybe change to weak if cyclic referencing
};

#endif // DECISION_TREE_NODE_HPP
