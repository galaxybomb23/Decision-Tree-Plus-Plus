#include "DecisionTreeNode.hpp"
#include "DecisionTree.hpp"

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

DecisionTreeNode::DecisionTreeNode(DecisionTree &dt)
    : _dt(dt)
{
    _serialNumber = GetNextSerialNum();
}

DecisionTreeNode::~DecisionTreeNode()
{
}

// Other functions below
int DecisionTreeNode::HowManyNodes()
{
    return _dt._nodesCreated + 1; // placeholder
}

vector<string> DecisionTreeNode::GetClassNames() const
{
    return _dt._classNames;
}

int DecisionTreeNode::GetNextSerialNum() const
{
    _dt._nodesCreated++;
    return _dt._nodesCreated;
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
    _dt._classNames = classNames;
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

void DecisionTreeNode::DisplayNode(const std::string &offset) const
{
    // Handle feature at node
    std::string featureAtNode = _feature.empty() ? " " : _feature;

    // Format entropy value
    std::ostringstream entropyStream;
    entropyStream << std::fixed << std::setprecision(3) << _nodeCreationEntropy;
    std::string printNodeCreationEntropyAtNode = entropyStream.str();

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
    nodeDisplay << offset << "NODE " << _serialNumber << ":" << endl
                << offset << "  Branch features and values to this node: " << branchFeaturesAndValuesStr << endl
                << offset << "  Class probabilities at current node: [";

    for (size_t i = 0; i < classProbsForDisplay.size(); ++i)
    {
        nodeDisplay << classProbsForDisplay[i];
        if (i < classProbsForDisplay.size() - 1)
        {
            nodeDisplay << ", ";
        }
    }

    nodeDisplay << "]" << endl
                << offset << "  Entropy at current node: " << printNodeCreationEntropyAtNode << endl
                << offset << "  Best feature test at current node: " << featureAtNode << endl
                << endl;

    std::cout << nodeDisplay.str();
}

void DecisionTreeNode::DisplayDecisionTree(const std::string &offset) const
{
    // Display the current node
    this->DisplayNode(offset);

    // Recursively display child nodes with an increased offset
    std::string newOffset = offset + "    ";
    for (const auto &child : this->GetChildren())
    {
        child->DisplayDecisionTree(newOffset);
    }
}
