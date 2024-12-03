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
    using namespace ConsoleColors;

    if (_samplesAtNodesDict.empty()) {
        throw std::runtime_error("You called explainClassificationsAtMultipleNodesInteractively() without first initializing the DTIntrospection instance in your code. Aborting.");
    }

    cout << BOLD_BLUE + "\n\nIn order for the decision tree to introspect\n" + RESET;
    string msg = BOLD + "  DO YOU ACCEPT the fact that, in general, a region of the feature space\n"
                      "  that corresponds to a node may have NON-ZERO probabilities associated\n"
                      "  with it even when there are NO training data points in that region?\n" + RESET
                      + BOLD + "\n    Enter " + BOLD_GREEN + "'y' for yes" + RESET + BOLD + " or any " + BOLD_RED + "other character for no:  " + RESET;
    cout << msg;

    string ans;
    std::getline(std::cin, ans);
    ans.erase(ans.find_last_not_of(" \n\r\t") + 1);

    if (ans != "y" && ans != "yes") {
        throw std::runtime_error(BOLD_RED + "\n\n  Since you answered 'no' to a very real theoretical possibility, no explanations possible for the classification decisions in the decision tree. Aborting." + RESET);
    }

    _awarenessRaisingMessageShown = 1;

    while (true) {
        int nodeId = -1;

        while (true) {
            cout << BOLD_BLUE + "\nEnter the integer ID of a node (1 to " + std::to_string(_nodeSerialNumToNodeDict.size() - 1) + ") or type 'exit' to stop: " + RESET;
            std::getline(std::cin, ans);
            ans.erase(ans.find_last_not_of(" \n\r\t") + 1);

            if (ans == "exit") {
                return;
            }

            try {
                nodeId = std::stoi(ans);
            } catch (const std::invalid_argument&) {
                cout << BOLD_RED + "\nWhat you entered does not look like an integer ID for a node. Aborting!" + RESET << endl;
                return;
            } catch (const std::out_of_range&) {
                cout << BOLD_RED + "\nThe number you entered is out of range. Aborting!" + RESET << endl;
                return;
            }

            if (_samplesAtNodesDict.find(nodeId) != _samplesAtNodesDict.end()) {
                break;
            }
            else if (nodeId == 0) {
                cout << BOLD_RED + "\nNode 0 is the root node. It has no samples. Try again or enter 'exit'." + RESET << endl;
            } else {
                cout << BOLD_RED + "\nYour answer must be an integer ID of a node. Try again or enter 'exit'." + RESET << endl;
            }
        }

        explainClassificationAtOneNode(nodeId);
    }
}


void DTIntrospection::explainClassificationAtOneNode(int nodeId) {
    using namespace ConsoleColors;

    if (_samplesAtNodesDict.empty()) {
        throw std::runtime_error("You called explainClassificationAtOneNode() without first initializing the DTIntrospection instance in your code. Aborting.");
    }

    if (_samplesAtNodesDict.find(nodeId) == _samplesAtNodesDict.end()) {
        cout << "Node " << nodeId << " is not a node in the tree" << endl;
        return;
    }

    if (nodeId == 0) {
        cout << "Nothing useful to be explained at the root node" << endl;
        return;
    }

    if (!_awarenessRaisingMessageShown) {
        cout << BOLD_BLUE + "\n\nIn order for the decision tree to introspect at Node " << nodeId << ": \n" + RESET;
        string msg = BOLD + "  DO YOU ACCEPT the fact that, in general, a region of the feature space\n"
                          "  that corresponds to a DT node may have NON-ZERO probabilities associated\n"
                          "  with it even when there are NO training data points in that region?\n" + RESET
                          + BOLD + "\n    Enter " + BOLD_GREEN + "'y' for yes" + RESET + BOLD + " or any " + BOLD_RED + "other character for no:  " + RESET;
        cout << msg;
        string ans;
        std::getline(std::cin, ans);
        ans.erase(ans.find_last_not_of(" \n\r\t")+1);

        if (ans != "y" && ans != "yes") {
            throw std::runtime_error(BOLD_RED + "\n\n  Since you answered 'no' to a very real theoretical possibility, no explanations possible for the classification decision at node " + std::to_string(nodeId) + RESET);
        }

        _awarenessRaisingMessageShown = 1;
    }

    vector<string> samplesAtNode = _samplesAtNodesDict[nodeId];
    vector<string> branchFeaturesToNode = _branchFeaturesToNodesDict[nodeId];
    vector<string> classNames = _rootNode->GetClassNames();

    string msg2;
    if (!samplesAtNode.empty()) {
        msg2 = "\n    Samples in the portion of the feature space assigned to Node " + std::to_string(nodeId) + ": ";
        
        for (const auto& sample : samplesAtNode) {
            msg2 += sample + " ";
        }

        msg2 += "\n";
    } else {
        msg2 = "\n\n    There are NO training data samples directly in the region of the feature space assigned to node " + std::to_string(nodeId) + ".\n";
    }

    msg2 += "\n    Features tests on the branch to node " + std::to_string(nodeId) + ": ";
    
    for (const auto& featureTest : branchFeaturesToNode) {
        msg2 += featureTest + " ";
    }

    msg2 += "\n";
    msg2 += BOLD_BLUE + "\n\nWould you like to see the probability associated with the last feature test on the branch leading to Node " + std::to_string(nodeId) + "?\n" + RESET;
    msg2 += BOLD + "\n    Enter " + BOLD_GREEN + "'y' for yes" + RESET + BOLD + " or any " + BOLD_RED + "other character for no:  " + RESET;

    string ans;
    cout << msg2;
    std::getline(std::cin, ans);
    ans.erase(ans.find_last_not_of(" \n\r\t")+1);

    if (ans == "y" || ans == "yes") {
        vector<string> sequence = { branchFeaturesToNode.back() };
        double prob = _dt->probabilityOfASequenceOfFeaturesAndValuesOrThresholds(sequence);
        cout << "\n    Probability of [";

        for (const auto& s : sequence) {
            cout << s << " ";
        }

        cout << "] is: " << prob << endl;
    }

    string msg3 = BOLD_BLUE + "\n\nUsing Bayes rule, would you like to see the class probabilities predicated on just the last feature test on the branch leading to Node " + std::to_string(nodeId) + "?\n" + RESET;
    msg3 += BOLD + "\n    Enter " + BOLD_GREEN + "'y' for yes" + RESET + BOLD + " or any " + BOLD_RED + "other character for no:  " + RESET;
    cout << msg3;
    std::getline(std::cin, ans);
    ans.erase(ans.find_last_not_of(" \n\r\t")+1);

    if (ans == "y" || ans == "yes") {
        vector<string> sequence = { branchFeaturesToNode.back() };
        
        for (const auto& cls : classNames) {
            double prob = _dt->probabilityOfAClassGivenSequenceOfFeaturesAndValuesOrThresholds(cls, sequence);
            cout << "\n    Probability of class " << cls << " given just one feature test [";
            
            for (const auto& s : sequence) {
                cout << s << " ";
            }
            
            cout << "] is: " << prob << endl;
        }
    } else {
        cout << "goodbye" << endl;
    }

    cout << BOLD_GREEN + "\nFinished supplying information on Node " << nodeId << "\n\n" + RESET;
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