#ifndef DT_INTROSPECTION_HPP
#define DT_INTROSPECTION_HPP

// Include
#include "Common.hpp"
#include "Utility.hpp"
#include "DecisionTree.hpp"

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


class DTIntrospection : public std::enable_shared_from_this<DTIntrospection> {
public:
    shared_ptr<DTIntrospection> getShared() { return shared_from_this(); }

    //--------------- Constructors and Destructors ----------------//
    DTIntrospection(shared_ptr<DecisionTree> dt);
    ~DTIntrospection();
    void initialize();

    //--------------- Recursive Descent ----------------//
    void recursiveDescent(DecisionTreeNode *node);
    void recursiveDescentForShowingSamplesAtANode(DecisionTreeNode *node);
    void recursiveDescentForSampleToNodeInfluence(int nodeSerialNum, vector<DecisionTreeNode> &nodesAlreadyAccountedFor, int offset);

    //--------------- Display ----------------//
    void displayTrainingSamplesAtAllNodesDirectInfluenceOnly();
    void displayTrainingSamplesToNodesInfluencePropagation();

    //--------------- Explanation ----------------//
    void explainClassificationsAtMultipleNodesInteractively();
    void explainClassificationAtOneNode(int nodeID);

    //--------------- Class Utility ----------------//
    vector<string> getSamplesForFeatureValueCombo(string featureValueCombo);
    FeatureOpValue extractFeatureOpValue(string featureValueCombo);

private:
    shared_ptr<DecisionTree> _dt;
    DecisionTreeNode* _rootNode;
    map<int, vector<string>> _samplesAtNodesDict;
    map<int, vector<string>> _branchFeaturesToNodesDict;
    map<int, vector<string>> _sampleToNodeMappingDirectDict;
    map<int, DecisionTreeNode> _nodeSerialNumToNodeDict;
    int awarenessRaisingMessageShown;
    int debug;
};

#endif // DT_INTROSPECTION_HPP