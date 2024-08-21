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
int DecisionTreeNode::HowManyNodes()
{
    return _dt.nodesCreated + 1;
}

vector<string> DecisionTreeNode::GetClassNames() const
{
    return _dt.classNames;
}

int DecisionTreeNode::GetNextSerialNum() const
{
    _dt.nodesCreated++;
    return _dt.nodesCreated;
}

string DecisionTreeNode::GetFeature() const
{
    return _feature;
}

double DecisionTreeNode::GetNodeEntropy() const
{
    return _nodeCreationEntropy;
}

vector<double> DecisionTreeNode::GetClassProbabilities() const
{
    return _classProbabilities;
}

vector<string> DecisionTreeNode::GetBranchFeaturesAndValuesOrThresholds() const
{
    return _branchFeaturesAndValuesOrThresholds;
}

vector<shared_ptr<DecisionTreeNode>> DecisionTreeNode::GetChildren() const
{
    return _linkedTo;
}

int DecisionTreeNode::GetSerialNum() const
{
    return _serialNumber;
}

void DecisionTreeNode::SetClassNames(const vector<string> classNames)
{
    _dt.classNames = classNames;
}

void DecisionTreeNode::SetNodeCreationEntropy(const double entropy)
{
    _nodeCreationEntropy = entropy;
}

void DecisionTreeNode::AddChildLink(shared_ptr<DecisionTreeNode> newNode)
{
    _linkedTo.emplace_back(newNode);
}

void DecisionTreeNode::DeleteAllLinks()
{
    _linkedTo.clear();
}

void DecisionTreeNode::DisplayNode()
{
    // Handle feature at node
    std::string feature_at_node = _feature.empty() ? " " : _feature;

    // Format entropy value
    std::ostringstream entropy_stream;
    entropy_stream << std::fixed << std::setprecision(3) << _nodeCreationEntropy;
    std::string print_node_creation_entropy_at_node = entropy_stream.str();

    // Format class probabilities
    std::vector<std::string> classProbsForDisplay;
    for (double prob : _classProbabilities)
    {
        std::ostringstream probStream;
        probStream << std::fixed << std::setprecision(3) << prob;
        classProbsForDisplay.push_back(probStream.str());
    }

    // Format branch features and values
    std::ostringstream branch_features_stream;
    for (size_t i = 0; i < _branchFeaturesAndValuesOrThresholds.size(); ++i)
    {
        branch_features_stream << _branchFeaturesAndValuesOrThresholds[i];
        if (i < _branchFeaturesAndValuesOrThresholds.size() - 1)
        {
            branch_features_stream << ", ";
        }
    }
    std::string branchFeaturesAndValuesStr = branch_features_stream.str();

    // Build and display the node information
    std::ostringstream nodeDisplay;
    nodeDisplay << "\n\nNODE " << _serialNumber
                << ":\n   Branch features and values to this node: "
                << branchFeaturesAndValuesStr
                << "\n   Class probabilities at current node: [";

    for (size_t i = 0; i < classProbsForDisplay.size(); ++i)
    {
        nodeDisplay << classProbsForDisplay[i];
        if (i < classProbsForDisplay.size() - 1)
        {
            nodeDisplay << ", ";
        }
    }

    nodeDisplay << "]"
                << "\n   Entropy at current node: " << print_node_creation_entropy_at_node
                << "\n   Best feature test at current node: " << feature_at_node << "\n\n";

    std::cout << nodeDisplay.str();
}

void

    // int main()
    // {
    //     // Example usage
    //     DecisionTreeNode node;
    //     node._feature = "Feature A";
    //     node._node_creation_entropy = 0.54321;
    //     node._class_probabilities = {0.1, 0.7, 0.2};
    //     node._serial_number = 1;
    //     node._branch_features_and_values_or_thresholds = {"Threshold B", "Value C"};

    //     node.display_node(); // Display the node's information
    //     return 0;
    // }