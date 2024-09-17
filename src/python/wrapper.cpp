#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "DecisionTreeNode.hpp"
#include "DecisionTree.hpp"
#include "EvalTrainingData.hpp"
#include "TrainingDataGeneratorNumeric.hpp"
#include "TrainingDataGeneratorSymbolic.hpp"
#include "Utility.hpp"

namespace py = pybind11;

// Define the DecisionTreeNode module
PYBIND11_MODULE(DecisionTreeNode, m)
{
    py::class_<DecisionTreeNode>(m, "MyClass")
    {
        .def(py::init<DecisionTree &>())
            .def(py::init<const std::string &, double, const std::vector<double> &, const std::vector<string> &, DecisionTree &, const bool>())
            .def("HowManyNodes", &DecisionTreeNode::HowManyNodes)
            .def("GetClassNames", &DecisionTreeNode::GetClassNames)
            .def("GetNextSerialNum", &DecisionTreeNode::GetNextSerialNum)
            .def("GetFeature", &DecisionTreeNode::GetFeature)
            .def("GetNodeEntropy", &DecisionTreeNode::GetNodeEntropy)
            .def("GetClassProbabilities", &DecisionTreeNode::GetClassProbabilities)
            .def("GetBranchFeaturesAndValuesOrThresholds", &DecisionTreeNode::GetBranchFeaturesAndValuesOrThresholds)
            .def("GetChildren", &DecisionTreeNode::GetChildren)
            .def("GetSerialNum", &DecisionTreeNode::GetSerialNum)
            .def("SetClassNames", &DecisionTreeNode::SetClassNames)
            .def("SetNodeCreationEntropy", &DecisionTreeNode::SetNodeCreationEntropy)
            .def("AddChildLink", &DecisionTreeNode::AddChildLink)
            .def("DeleteAllLinks", &DecisionTreeNode::DeleteAllLinks)
            .def("DisplayNode", &DecisionTreeNode::DisplayNode)
            .def("DisplayDecisionTree", &DecisionTreeNode::DisplayDecisionTree)
    }
}

// Define the DecisionTree module

PYBIND11_MODULE(DecisionTree, m)
{
    // Bind the DecisionTree class
    py::class_<DecisionTree>(m, "DecisionTree")
        .def(py::init<std::map<std::string, std::string>>()) // Constructor
        .def("getTrainingData", &DecisionTree::getTrainingData)
        .def("calculateFirstOrderProbabilities", &DecisionTree::calculateFirstOrderProbabilities)
        .def("showTrainingData", &DecisionTree::showTrainingData)
        .def("classify", &DecisionTree::classify, py::arg("root_node"), py::arg("features_and_values"))
        .def("constructDecisionTreeClassifier", &DecisionTree::constructDecisionTreeClassifier)
        .def("classEntropyOnPriors", &DecisionTree::classEntropyOnPriors)
        .def("probabilityOfFeatureValue", (double(DecisionTree::*)(const std::string &, const std::string &)) & DecisionTree::probabilityOfFeatureValue)
        .def("probabilityOfFeatureValue", (double(DecisionTree::*)(const std::string &, double)) & DecisionTree::probabilityOfFeatureValue)
        .def_property("_nodesCreated", &DecisionTree::_nodesCreated, &DecisionTree::_nodesCreated)
        .def_property("_classNames", &DecisionTree::_classNames, &DecisionTree::_classNames)
        .def("getTrainingDatafile", &DecisionTree::getTrainingDatafile)
        .def("getEntropyThreshold", &DecisionTree::getEntropyThreshold)
        .def("getMaxDepthDesired", &DecisionTree::getMaxDepthDesired)
        .def("getNumberOfHistogramBins", &DecisionTree::getNumberOfHistogramBins)
        .def("getCsvClassColumnIndex", &DecisionTree::getCsvClassColumnIndex)
        .def("getCsvColumnsForFeatures", &DecisionTree::getCsvColumnsForFeatures)
        .def("getSymbolicToNumericCardinalityThreshold", &DecisionTree::getSymbolicToNumericCardinalityThreshold)
        .def("getCsvCleanupNeeded", &DecisionTree::getCsvCleanupNeeded)
        .def("getDebug1", &DecisionTree::getDebug1)
        .def("getDebug2", &DecisionTree::getDebug2)
        .def("getDebug3", &DecisionTree::getDebug3)
        .def("getHowManyTotalTrainingSamples", &DecisionTree::getHowManyTotalTrainingSamples)
        .def("getFeatureNames", &DecisionTree::getFeatureNames)
        .def("getTrainingDataDict", &DecisionTree::getTrainingDataDict)
        .def("setTrainingDatafile", &DecisionTree::setTrainingDatafile)
        .def("setEntropyThreshold", &DecisionTree::setEntropyThreshold)
        .def("setMaxDepthDesired", &DecisionTree::setMaxDepthDesired)
        .def("setNumberOfHistogramBins", &DecisionTree::setNumberOfHistogramBins)
        .def("setCsvClassColumnIndex", &DecisionTree::setCsvClassColumnIndex)
        .def("setCsvColumnsForFeatures", &DecisionTree::setCsvColumnsForFeatures)
        .def("setSymbolicToNumericCardinalityThreshold", &DecisionTree::setSymbolicToNumericCardinalityThreshold)
        .def("setCsvCleanupNeeded", &DecisionTree::setCsvCleanupNeeded)
        .def("setDebug1", &DecisionTree::setDebug1)
        .def("setDebug2", &DecisionTree::setDebug2)
        .def("setDebug3", &DecisionTree::setDebug3)
        .def("setHowManyTotalTrainingSamples", &DecisionTree::setHowManyTotalTrainingSamples);
}