// Include
#include "DTIntrospection.hpp"

//--------------- Constructors and Destructors ----------------//
DTIntrospection::DTIntrospection(DecisionTree dt)
{
    _dt = dt.getShared();
    _rootNode = dt.getRootNode();
    _samplesAtNodesDict = {};
    _branchFeaturesToNodesDict = {};
    _sampleToNodeMappingDirectDict = {};
    _nodeSerialNumToNodeDict = {};
    awarenessRaisingMessageShown = 0;
    debug = 0;
}


//--------------- Recursive Descent ----------------//


//--------------- Display ----------------//


//--------------- Explanation ----------------//


//--------------- Class Utility ----------------//
