#ifndef DT_INTROSPECTION_HPP
#define DT_INTROSPECTION_HPP

// Include
#include "Common.hpp"
#include "DecisionTree.hpp"
#include "Utility.hpp"

#include <iostream>

/**
 * @struct FeatureOpValue
 * @brief Represents a feature operation value used in decision trees.
 *
 * This structure holds information about a feature, an operation, and a value
 * associated with that feature. It is typically used in decision tree algorithms
 * to represent conditions or rules.
 */
struct FeatureOpValue {
    string feature;
    string op;
    string value;
};


/**
 * @class DTIntrospection
 * @brief A class for introspecting decision trees, providing various utilities for recursive descent, display, and
 * explanation.
 *
 * This class provides methods to recursively traverse decision trees, display training samples, and explain
 * classifications. It also includes utility functions for feature value combinations and mappings between samples and
 * nodes.
 *
 * @public
 * @method getShared
 * @brief Returns a shared pointer to the current instance.
 *
 * @constructor DTIntrospection
 * @brief Constructs a DTIntrospection object with a given decision tree.
 * @param dt A shared pointer to a DecisionTree object.
 *
 * @destructor ~DTIntrospection
 * @brief Destructs the DTIntrospection object.
 *
 * @method initialize
 * @brief Initializes the DTIntrospection object.
 *
 * @method recursiveDescent
 * @brief Recursively descends through the decision tree starting from a given node.
 * @param node A pointer to the starting DecisionTreeNode.
 *
 * @method recursiveDescentForShowingSamplesAtANode
 * @brief Recursively descends through the decision tree to show samples at a given node.
 * @param node A pointer to the starting DecisionTreeNode.
 *
 * @method recursiveDescentForSampleToNodeInfluence
 * @brief Recursively descends through the decision tree to determine sample to node influence.
 * @param nodeSerialNum The serial number of the node.
 * @param nodesAlreadyAccountedFor A vector of nodes already accounted for.
 * @param offset An offset value.
 *
 * @method displayTrainingSamplesAtAllNodesDirectInfluenceOnly
 * @brief Displays training samples at all nodes with direct influence only.
 *
 * @method displayTrainingSamplesToNodesInfluencePropagation
 * @brief Displays training samples to nodes influence propagation.
 *
 * @method explainClassificationsAtMultipleNodesInteractively
 * @brief Explains classifications at multiple nodes interactively.
 *
 * @method explainClassificationAtOneNode
 * @brief Explains classification at a single node.
 * @param nodeID The ID of the node.
 *
 * @method getSamplesForFeatureValueCombo
 * @brief Gets samples for a given feature value combination.
 * @param featureValueCombo A string representing the feature value combination.
 * @return A vector of sample indices.
 *
 * @method extractFeatureOpValue
 * @brief Extracts the feature operation value from a given feature value combination.
 * @param featureValueCombo A string representing the feature value combination.
 * @return A FeatureOpValue object.
 *
 * @method getSamplesAtNodesDict
 * @brief Gets the dictionary mapping node IDs to samples.
 * @return A map of node IDs to vectors of sample strings.
 *
 * @method getBranchFeaturesToNodesDict
 * @brief Gets the dictionary mapping node IDs to branch features.
 * @return A map of node IDs to vectors of branch feature strings.
 *
 * @method getSampleToNodeMappingDirectDict
 * @brief Gets the dictionary mapping samples to nodes directly.
 * @return A map of sample strings to vectors of node IDs.
 *
 * @method getNodeSerialNumToNodeDict
 * @brief Gets the dictionary mapping node serial numbers to nodes.
 * @return A map of node serial numbers to DecisionTreeNode pointers.
 *
 * @private
 * @var _dt
 * @brief A shared pointer to the DecisionTree object.
 *
 * @var _rootNode
 * @brief A pointer to the root DecisionTreeNode.
 *
 * @var _samplesAtNodesDict
 * @brief A dictionary mapping node IDs to samples.
 *
 * @var _branchFeaturesToNodesDict
 * @brief A dictionary mapping node IDs to branch features.
 *
 * @var _sampleToNodeMappingDirectDict
 * @brief A dictionary mapping samples to nodes directly.
 *
 * @var _nodeSerialNumToNodeDict
 * @brief A dictionary mapping node serial numbers to nodes.
 *
 * @var _awarenessRaisingMessageShown
 * @brief An integer indicating if an awareness-raising message has been shown.
 *
 * @var _debug
 * @brief An integer for debugging purposes.
 */
class DTIntrospection : public std::enable_shared_from_this<DTIntrospection> {
  public:
    shared_ptr<DTIntrospection> getShared() { return shared_from_this(); }

    //--------------- Constructors and Destructors ----------------//
    DTIntrospection(shared_ptr<DecisionTree> dt);
    ~DTIntrospection();
    void initialize();

    //--------------- Recursive Descent ----------------//
    void recursiveDescent(DecisionTreeNode* node);
    void recursiveDescentForShowingSamplesAtANode(DecisionTreeNode* node);
    void recursiveDescentForSampleToNodeInfluence(int nodeSerialNum, vector<int> nodesAlreadyAccountedFor, int offset);

    //--------------- Display ----------------//
    void displayTrainingSamplesAtAllNodesDirectInfluenceOnly();
    void displayTrainingSamplesToNodesInfluencePropagation();

    //--------------- Explanation ----------------//
    void explainClassificationsAtMultipleNodesInteractively();
    void explainClassificationAtOneNode(int nodeID);

    //--------------- Class Utility ----------------//
    vector<int> getSamplesForFeatureValueCombo(string featureValueCombo);
    FeatureOpValue extractFeatureOpValue(string featureValueCombo);

    //--------------- Getters ----------------//
    map<int, vector<string>> getSamplesAtNodesDict() const { return _samplesAtNodesDict; }
    map<int, vector<string>> getBranchFeaturesToNodesDict() const { return _branchFeaturesToNodesDict; }
    map<string, vector<int>> getSampleToNodeMappingDirectDict() const { return _sampleToNodeMappingDirectDict; }
    map<int, DecisionTreeNode*> getNodeSerialNumToNodeDict() const { return _nodeSerialNumToNodeDict; }

  private:
    shared_ptr<DecisionTree> _dt;
    DecisionTreeNode* _rootNode;
    map<int, vector<string>> _samplesAtNodesDict;
    map<int, vector<string>> _branchFeaturesToNodesDict;
    map<string, vector<int>> _sampleToNodeMappingDirectDict;
    map<int, DecisionTreeNode*> _nodeSerialNumToNodeDict;
    int _awarenessRaisingMessageShown;
    int _debug;
};

#endif // DT_INTROSPECTION_HPP