#include "DecisionTreeNode.hpp"

#include "DecisionTree.hpp"

DecisionTreeNode::DecisionTreeNode()
{

    throw std::runtime_error("DecisionTreeNode has no Default constructor");
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
    }

    _serialNumber = GetNextSerialNum();
}


/**
 * @brief Constructs a new DecisionTreeNode object.
 *
 * @param dt A shared pointer to the DecisionTree object associated with this node.
 *
 * This constructor initializes the following member variables:
 * - _feature: An empty string representing the feature associated with this node.
 * - _nodeCreationEntropy: A double initialized to 0, representing the entropy at the time of node creation.
 * - _classProbabilities: An empty map representing the class probabilities for this node.
 * - _branchFeaturesAndValuesOrThresholds: An empty map representing the features and values or thresholds for the
 * branches of this node.
 * - _serialNumber: A unique serial number for this node, obtained by calling GetNextSerialNum().
 */
DecisionTreeNode::DecisionTreeNode(shared_ptr<DecisionTree> dt) : _dt((dt))
{
    _feature                             = "";
    _nodeCreationEntropy                 = 0;
    _classProbabilities                  = {};
    _branchFeaturesAndValuesOrThresholds = {};
    _serialNumber                        = GetNextSerialNum();
}

/**
 * @brief Copy constructor for DecisionTreeNode.
 *
 * This constructor creates a deep copy of the given DecisionTreeNode object.
 * It copies all member variables and performs a deep copy of the children nodes.
 *
 * @param other The DecisionTreeNode object to copy from.
 */
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

/**
 * @brief Assignment operator for DecisionTreeNode.
 *
 * This operator handles the assignment of one DecisionTreeNode to another.
 * It performs a deep copy of the _linkedTo member to ensure that each node
 * has its own unique children, while copying other member variables directly.
 *
 * @param other The other DecisionTreeNode to assign from.
 * @return A reference to this DecisionTreeNode.
 */
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

const vector<DecisionTreeNode*> DecisionTreeNode::GetChildren() const
{
    vector<DecisionTreeNode*> children;
    for (const auto &child : _linkedTo) {
        children.push_back(child.get());
    }
    return children;
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
    // Format feature at the node
    string featureAtNode = _feature.empty() ? " " : _feature;

    // Format branch features and values with single quotes
    cout << "NODE " << _serialNumber << ":  " << offset << "BRANCH TESTS TO "
         << (_linkedTo.empty() ? "LEAF NODE: " : "NODE: ") << "[";

    for (size_t i = 0; i < _branchFeaturesAndValuesOrThresholds.size(); ++i) {
        cout << "'" << _branchFeaturesAndValuesOrThresholds[i] << "'";
        if (i < _branchFeaturesAndValuesOrThresholds.size() - 1) {
            cout << ", ";
        }
    }
    cout << "]" << endl;

    // Offset for the second line
    string secondLineOffset = offset + string(8 + to_string(_serialNumber).length(), ' ');

    // Format class probabilities with class names and brackets
    vector<string> classProbabilitiesWithClass;
    for (size_t i = 0; i < _classProbabilities.size(); ++i) {
        string classProbability =
            "'class=" + _dt.lock()->_classNames[i] + " => " + roundDouble(_classProbabilities[i], 3) + "'";
        classProbabilitiesWithClass.push_back(classProbability);
    }

    // Print entropy and class probabilities
    cout << secondLineOffset;
    if (_linkedTo.empty()) {
        // Leaf node: Only print entropy and probabilities
        cout << "Node Creation Entropy: " << roundDouble(_nodeCreationEntropy, 3) << "   Class Probs: "
             << "[" << join(classProbabilitiesWithClass, ", ") << "]" << endl
             << endl;
    }
    else {
        // Non-leaf node: Print feature, entropy, and probabilities
        cout << "Decision Feature: " << featureAtNode
             << "   Node Creation Entropy: " << roundDouble(_nodeCreationEntropy, 3) << "   Class Probs: "
             << "[" << join(classProbabilitiesWithClass, ", ") << "]" << endl
             << endl;
    }
}

void DecisionTreeNode::DisplayDecisionTree(const string &offset) const
{
    // Display the current node
    this->DisplayNode(offset);

    // Recursively display child nodes with an increased offset
    for (const auto &child : _linkedTo) {
        if (child) {
            child->DisplayDecisionTree(offset + "   ");
        }
    }
}