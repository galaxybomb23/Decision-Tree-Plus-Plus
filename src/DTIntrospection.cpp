// Include
#include "DTIntrospection.hpp"

//--------------- Constructors and Destructors ----------------//
DTIntrospection::DTIntrospection(shared_ptr<DecisionTree> dt)
{
    _dt = dt;
    _rootNode = dt->getRootNode();
    _samplesAtNodesDict = {};
    _branchFeaturesToNodesDict = {};
    _sampleToNodeMappingDirectDict = {};
    _nodeSerialNumToNodeDict = {};
    awarenessRaisingMessageShown = 0;
    debug = 0;
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


//--------------- Recursive Descent ----------------//


//--------------- Display ----------------//


//--------------- Explanation ----------------//


//--------------- Class Utility ----------------//
