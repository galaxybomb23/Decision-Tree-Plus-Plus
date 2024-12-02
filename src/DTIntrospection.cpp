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

// MARK: Do we need unique_ptr here? ---------------------v
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

}

void DTIntrospection::recursiveDescentForSampleToNodeInfluence(int nodeSerialNum, vector<DecisionTreeNode> &nodesAlreadyAccountedFor, int offset) {

}


//--------------- Display ----------------//

void DTIntrospection::displayTrainingSamplesAtAllNodesDirectInfluenceOnly() {

}

void DTIntrospection::displayTrainingSamplesToNodesInfluencePropagation() {
    
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

    if (std::regex_match(featureValueCombo, pattern1)) {
        std::smatch match;
        std::regex_search(featureValueCombo, match, pattern1);
        feature = match[1];
        op = "=";
        value = match[2];
    } else if (std::regex_match(featureValueCombo, pattern2)) {
        std::smatch match;
        std::regex_search(featureValueCombo, match, pattern2);
        feature = match[1];
        op = "<";
        value = match[2];
    } else if (std::regex_match(featureValueCombo, pattern3)) {
        std::smatch match;
        std::regex_search(featureValueCombo, match, pattern3);
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