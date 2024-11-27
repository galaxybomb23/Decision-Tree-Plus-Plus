#include "DecisionTreeNode.hpp"

#include "DecisionTree.hpp"

// for vector output
std::ostream &operator<<(std::ostream &os, const vector<string> &vec)
{
    for (const auto &str : vec) {
        os << str << " ";
    }
    return os;
}

DecisionTreeNode::DecisionTreeNode(const string &feature,
                                   double entropy,
                                   const vector<double> &class_probabilities,
                                   const vector<string> &branch_features_and_values_or_thresholds,
                                   shared_ptr<DecisionTree> dt,
                                   const bool isRoot)
    : _dt(dt),
      _feature(feature),
      _nodeCreationEntropy(entropy),
      _classProbabilities(class_probabilities),
      _branchFeaturesAndValuesOrThresholds(branch_features_and_values_or_thresholds)
{
    // Lock the weak pointer once and store the shared pointer
    auto tree = _dt.lock();
    if (!tree) {
        throw std::runtime_error("DecisionTree pointer is invalid");
    }

    if (isRoot) {
        tree->_nodesCreated = -1;
        tree->_classNames.clear();
    }

    _serialNumber = GetNextSerialNum();
}


DecisionTreeNode::DecisionTreeNode(shared_ptr<DecisionTree> dt) : _dt((dt))
{
    _feature                             = "";
    _nodeCreationEntropy                 = 0;
    _classProbabilities                  = {};
    _branchFeaturesAndValuesOrThresholds = {};
    _serialNumber                        = GetNextSerialNum();
}

DecisionTreeNode::DecisionTreeNode(const DecisionTreeNode &other)
    : _feature(other._feature),
      _nodeCreationEntropy(other._nodeCreationEntropy),
      _classProbabilities(other._classProbabilities),
      _branchFeaturesAndValuesOrThresholds(other._branchFeaturesAndValuesOrThresholds),
      _dt(other._dt),
      _serialNumber(other._serialNumber)
{
    // Deep copy of children
    for (const auto &child : other._linkedTo) {
        _linkedTo.push_back(make_unique<DecisionTreeNode>(*child));
    }
    auto tree = _dt.lock();
    // Copy class names
    tree->_classNames = other.GetClassNames();

    // Update the number of nodes created
    tree->_nodesCreated = other.HowManyNodes();

    // Update the serial number
    _serialNumber = GetNextSerialNum();
}

DecisionTreeNode &DecisionTreeNode::operator=(const DecisionTreeNode &other)
{
    if (this == &other) {
        return *this; // Handle self-assignment
    }

    // Copy all member variables from the other object
    _dt                                  = other._dt; // Copy the weak_ptr (safe to copy)
    _feature                             = other._feature;
    _nodeCreationEntropy                 = other._nodeCreationEntropy;
    _classProbabilities                  = other._classProbabilities;
    _branchFeaturesAndValuesOrThresholds = other._branchFeaturesAndValuesOrThresholds;
    _serialNumber                        = other._serialNumber;

    // If _linkedTo needs to be copied, ensure deep copy
    _linkedTo.clear();
    for (const auto &child : other._linkedTo) {
        if (child) {
            _linkedTo.push_back(make_unique<DecisionTreeNode>(*child));
        }
        else {
            _linkedTo.push_back(nullptr);
        }
    }

    return *this;
}


DecisionTreeNode::~DecisionTreeNode() {}

// Other functions below
int DecisionTreeNode::HowManyNodes() const
{
    return _dt.lock()->_nodesCreated + 1; // placeholder
}

vector<string> DecisionTreeNode::GetClassNames() const
{
    return _dt.lock()->_classNames;
}

int DecisionTreeNode::GetNextSerialNum() const
{
    auto tree = _dt.lock();
    tree->_nodesCreated++;
    return tree->_nodesCreated;
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

const vector<unique_ptr<DecisionTreeNode>> &DecisionTreeNode::GetChildren() const
{
    return _linkedTo;
}

int DecisionTreeNode::GetSerialNum() const
{
    return _serialNumber;
}

void DecisionTreeNode::SetClassNames(const vector<string> classNames)
{
    _dt.lock()->setClassNames(classNames);
}

void DecisionTreeNode::SetNodeCreationEntropy(const double entropy)
{
    _nodeCreationEntropy = entropy;
}

void DecisionTreeNode::AddChildLink(unique_ptr<DecisionTreeNode> newNode)
{
    _linkedTo.emplace_back(std::move(newNode));
}

void DecisionTreeNode::DeleteAllLinks()
{
    _linkedTo.clear();
}

void DecisionTreeNode::DisplayNode(const string &offset) const
{
    // Handle feature at node
    string featureAtNode = _feature.empty() ? " " : _feature;

    // Format entropy value
    std::ostringstream entropyStream;
    entropyStream << std::fixed << std::setprecision(3) << _nodeCreationEntropy;
    string printNodeCreationEntropyAtNode = entropyStream.str();

    // Format class probabilities
    vector<string> classProbsForDisplay;
    for (double prob : _classProbabilities) {
        std::ostringstream probStream;
        probStream << std::fixed << std::setprecision(3) << prob;
        classProbsForDisplay.push_back(probStream.str());
    }

    // Format branch features and values
    std::ostringstream branch_features_stream;
    for (size_t i = 0; i < _branchFeaturesAndValuesOrThresholds.size(); ++i) {
        branch_features_stream << _branchFeaturesAndValuesOrThresholds[i];
        if (i < _branchFeaturesAndValuesOrThresholds.size() - 1) {
            branch_features_stream << ", ";
        }
    }
    string branchFeaturesAndValuesStr = branch_features_stream.str();

    // Build and display the node information
    std::ostringstream nodeDisplay;
    nodeDisplay << offset << "\n\nNODE " << _serialNumber << ":" << endl
                << offset << "  Branch features and values to this node: [" << branchFeaturesAndValuesStr << "]" << endl
                << offset << "  Class probabilities at current node: [";

    for (size_t i = 0; i < classProbsForDisplay.size(); ++i) {
        nodeDisplay << _dt.lock()->_classNames[i] << ": ";
        nodeDisplay << classProbsForDisplay[i];
        if (i < classProbsForDisplay.size() - 1) {
            nodeDisplay << ", ";
        }
    }

    nodeDisplay << "]" << endl
                << offset << "  Entropy at current node: " << printNodeCreationEntropyAtNode << endl
                << offset << "  Best feature test at current node: " << featureAtNode << endl
                << endl;

    cout << nodeDisplay.str();
}

void DecisionTreeNode::DisplayDecisionTree(const string &offset) const
{
    // Display the current node
    this->DisplayNode(offset);

    // Recursively display child nodes with an increased offset
    string newOffset = offset + "    ";
    for (const auto &child : this->GetChildren()) {
        child->DisplayDecisionTree(newOffset);
    }
}
