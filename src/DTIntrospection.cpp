// Include
#include "DTIntrospection.hpp"

//--------------- Constructors and Destructors ----------------//

DTIntrospection::DTIntrospection(shared_ptr<DecisionTree> dt)
{
    _dt = dt;
    _rootNode = nullptr;
    _samplesAtNodesDict = {};
    _branchFeaturesToNodesDict = {};
    _sampleToNodeMappingDirectDict = {};
    _nodeSerialNumToNodeDict = {};
    _awarenessRaisingMessageShown = 0;
    _debug = 0;
}

DTIntrospection::~DTIntrospection()
{
    _dt.reset();
    _rootNode = nullptr;
    _samplesAtNodesDict.clear();
    _branchFeaturesToNodesDict.clear();
    _sampleToNodeMappingDirectDict.clear();
    _nodeSerialNumToNodeDict.clear();
}

void DTIntrospection::initialize()
{
    _rootNode = _dt->getRootNode();

    if (_rootNode == nullptr)
    {
        throw std::runtime_error("Root node is not set. You must first construct the decision tree before using introspection.");
    }

    recursiveDescent(_rootNode);
}


//--------------- Recursive Descent ----------------//

void DTIntrospection::recursiveDescent(DecisionTreeNode *node) {
    int nodeSerialNum = node->GetSerialNum();
    _nodeSerialNumToNodeDict[nodeSerialNum] = node;

    vector<string> branchFeaturesAndValuesOrThresholds = node->GetBranchFeaturesAndValuesOrThresholds();

    if (_debug) {
        cout << "\nat node " << nodeSerialNum << ": the branch features and values are: ";
        for (const auto& feature : branchFeaturesAndValuesOrThresholds){
            cout << feature << " ";
        }
        cout << endl;
    }

    _branchFeaturesToNodesDict[nodeSerialNum] = branchFeaturesAndValuesOrThresholds;

    // Determine Samples at the Node
    optional<vector<int>> samplesAtNode;

    for (const auto& item : branchFeaturesAndValuesOrThresholds) {
        vector<int> samplesForFeatureValueCombo = getSamplesForFeatureValueCombo(item);
        if (!samplesAtNode.has_value()) {
            samplesAtNode = samplesForFeatureValueCombo;
        }
        else {
            // Intersect with existing samplesAtNode
            vector<int> intersection;
            std::set_intersection(
                samplesAtNode->begin(), samplesAtNode->end(),
                samplesForFeatureValueCombo.begin(), samplesForFeatureValueCombo.end(),
                std::back_inserter(intersection)
            );

            samplesAtNode = intersection;
        }
    }

    // If samplesAtNode has samples, sort them based on sample_index
    if (samplesAtNode.has_value() && !samplesAtNode->empty()) {
        std::sort(samplesAtNode->begin(), samplesAtNode->end(),
            [&](int a, int b) -> bool {
                return a < b;
            }
        );
    }

    if (_debug) {
        cout << "Node: " << nodeSerialNum << " the samples are: ";
        if(samplesAtNode.has_value()) {
            cout << samplesAtNode.value() << endl;
        }
        else {
            cout << "None";
        }
        cout << endl;
    }

    // Convert samplesAtNode to a vector of strings
    vector<string> samplesAtNodeStr;
    if (samplesAtNode.has_value()){
        for(auto sample : samplesAtNode.value()){
            samplesAtNodeStr.push_back(std::to_string(sample));
        }
    }

    // Store Samples at the Node
    if (samplesAtNode.has_value()) {
        _samplesAtNodesDict[nodeSerialNum] = samplesAtNodeStr;

        // Map Samples to Nodes
        for (auto sample : samplesAtNodeStr) {
            auto it = _sampleToNodeMappingDirectDict.find(sample);
            
            if (it == _sampleToNodeMappingDirectDict.end()) {
                // Sample not present in the mapping, create a new entry
                _sampleToNodeMappingDirectDict[sample] = {nodeSerialNum};
            }
            else {
                // Sample already mapped to other nodes, append the current node
                it->second.push_back(nodeSerialNum);
            }
        }
    }

    // Recursively Process Child Nodes
    vector<DecisionTreeNode*> children = node->GetChildren();
    for (auto child : children) {
        recursiveDescent(child);
    }
}

void DTIntrospection::recursiveDescentForShowingSamplesAtANode(DecisionTreeNode *node) {
    int nodeSerialNum = node->GetSerialNum();
    vector<string> branchFeaturesAndValuesOrThresholds = node->GetBranchFeaturesAndValuesOrThresholds();

    // If the nodeSerialNum is in the _samplesAtNodesDict, display the samples
    if (_samplesAtNodesDict.find(nodeSerialNum) != _samplesAtNodesDict.end()) {
        if (_debug) {
            cout << "\nat node " << nodeSerialNum << ": the branch features and values are: " << branchFeaturesAndValuesOrThresholds << endl;
        }

        cout << "Node " << nodeSerialNum << ": the samples are: " << _samplesAtNodesDict[nodeSerialNum] << endl;
    }
    else {
        cout << "Node " << nodeSerialNum << ": the samples are: None" << endl;
    }
    
    // Recursively Process Child Nodes
    vector<DecisionTreeNode*> children = node->GetChildren();
    for (auto child : children) {
        recursiveDescentForShowingSamplesAtANode(child);
    }
}

void DTIntrospection::recursiveDescentForSampleToNodeInfluence(int nodeSerialNum, vector<int> nodesAlreadyAccountedFor, int offset) {
    offset += 4;
    DecisionTreeNode* node = _nodeSerialNumToNodeDict[nodeSerialNum];
    vector<DecisionTreeNode*> children = node->GetChildren();
    vector<int> childrenSerialNums;

    for (auto child : children) {
        childrenSerialNums.push_back(child->GetSerialNum());
    }

    // Determine which children are not already accounted for
    vector<int> childrenSerialNumsAffected;
    for (auto childSerialNum : childrenSerialNums) {
        if (std::find(nodesAlreadyAccountedFor.begin(), nodesAlreadyAccountedFor.end(), childSerialNum) == nodesAlreadyAccountedFor.end()) {
            childrenSerialNumsAffected.push_back(childSerialNum);
        }
    }

    // Display the influence
    if (!childrenSerialNumsAffected.empty()) {
        cout << string(offset, ' ') << nodeSerialNum << " -> ";
        for (const auto& serialNum : childrenSerialNumsAffected) {
            cout << serialNum << " ";
        }
        cout << endl;
    }

    // Recursively process child nodes
    for (auto childSerialNum : childrenSerialNumsAffected) {
        vector<int> newNodesAlreadyAccountedFor = nodesAlreadyAccountedFor;
        newNodesAlreadyAccountedFor.push_back(childSerialNum);
        recursiveDescentForSampleToNodeInfluence(childSerialNum, newNodesAlreadyAccountedFor, offset);
    }
}


//--------------- Display ----------------//

void DTIntrospection::displayTrainingSamplesAtAllNodesDirectInfluenceOnly() {
    if (_rootNode == nullptr) {
        throw std::runtime_error("Root node is not set. You must first construct the decision tree before using introspection.");
    }

    recursiveDescentForShowingSamplesAtANode(_rootNode);
}

void DTIntrospection::displayTrainingSamplesToNodesInfluencePropagation() {
    auto trainingDataDict = _dt->getTrainingDataDict();

    for (const auto& samplePair : trainingDataDict) {
        const string sample = std::to_string(samplePair.first);

        if (_sampleToNodeMappingDirectDict.find(sample) != _sampleToNodeMappingDirectDict.end()) {
            vector<int> nodesDirectlyAffected = _sampleToNodeMappingDirectDict[sample];
            cout << "\n" << sample << ":\n" << "   nodes affected directly: ";

            for (const auto& nodeNum : nodesDirectlyAffected) {
                cout << nodeNum << " ";
            }

            cout << endl;
            cout << "   nodes affected through probabilistic generalization:" << endl;

            for (const auto& nodeSerialNum : nodesDirectlyAffected) {
                vector<int> nodesAlreadyAccountedFor = nodesDirectlyAffected;
                recursiveDescentForSampleToNodeInfluence(nodeSerialNum, nodesAlreadyAccountedFor, 4);
            }
        }
    }
}


//--------------- Explanation ----------------//

void DTIntrospection::explainClassificationsAtMultipleNodesInteractively() {

}

void DTIntrospection::explainClassificationAtOneNode(int nodeID){
    
}


//--------------- Class Utility ----------------//

vector<int> DTIntrospection::getSamplesForFeatureValueCombo(string featureValueCombo) {
    FeatureOpValue featureOpValue = extractFeatureOpValue(featureValueCombo);
    double valueAsDouble = convert(featureOpValue.value);
    vector<int> samples = {};

    auto trainingDataDict = _dt->getTrainingDataDict();
    auto featureNames = _dt->getFeatureNames();

    if (featureOpValue.op == "=") {
        for (const auto& samplePair : trainingDataDict) {
            const int sample = samplePair.first;
            vector<string> featuresAndValues = samplePair.second;

            for (int i = 0; i < featuresAndValues.size(); i++) {
                featuresAndValues[i].insert(0, featureNames[i] + "=");
            }

            if (std::find(featuresAndValues.begin(), featuresAndValues.end(), featureValueCombo) != featuresAndValues.end()) {
                samples.push_back(sample);
            }
        }
    }
    else if (featureOpValue.op == "<") {
        for (const auto& samplePair : trainingDataDict) {
            const int sample = samplePair.first;
            vector<string> featuresAndValues = samplePair.second;

            for (int i = 0; i < featuresAndValues.size(); i++) {
                featuresAndValues[i].insert(0, featureNames[i] + "=");
            }
            
            for (const auto& item : featuresAndValues) {
                FeatureOpValue featureOpValue2 = extractFeatureOpValue(item);
                double value2AsDouble = convert(featureOpValue2.value);
                
                if (!std::isnan(valueAsDouble) && !std::isnan(value2AsDouble)) {
                    if (featureOpValue.feature == featureOpValue2.feature && value2AsDouble <= valueAsDouble) {
                        samples.push_back(sample);
                        break;
                    }
                }

            }
            
        }
    }
    else if (featureOpValue.op == ">") {
        for (const auto& samplePair : trainingDataDict) {
            const int sample = samplePair.first;
            vector<string> featuresAndValues = samplePair.second;

            for (int i = 0; i < featuresAndValues.size(); i++) {
                featuresAndValues[i].insert(0, featureNames[i] + "=");
            }
            
            for (const auto& item : featuresAndValues) {
                FeatureOpValue featureOpValue2 = extractFeatureOpValue(item);
                double value2AsDouble = convert(featureOpValue2.value);
                
                if (!std::isnan(valueAsDouble) && !std::isnan(value2AsDouble)) {
                    if (featureOpValue.feature == featureOpValue2.feature && value2AsDouble > valueAsDouble) {
                        samples.push_back(sample);
                        break;
                    }
                }

            }
            
        }
    }
    else {
        throw std::runtime_error("Something is wrong with the feature-value syntax");
    }
    
    return samples;
}

FeatureOpValue DTIntrospection::extractFeatureOpValue(string featureValueCombo) {
    std::regex pattern1(R"((.+)=(.+))");
    std::regex pattern2(R"((.+)<(.+))");
    std::regex pattern3(R"((.+)>(.+))");

    string feature, op, value;
    std::smatch match;


    if (std::regex_search(featureValueCombo, match, pattern1)) {
        feature = match[1];
        op = "=";
        value = match[2];
    } else if (std::regex_search(featureValueCombo, match, pattern2)) {
        feature = match[1];
        op = "<";
        value = match[2];
    } else if (std::regex_search(featureValueCombo, match, pattern3)) {
        feature = match[1];
        op = ">";
        value = match[2];
    }
    else {
        throw std::runtime_error("Invalid feature value combo: " + featureValueCombo);
    }

    FeatureOpValue featureOpValue = {feature, op, value};
    
    return featureOpValue;
}