#include "DecisionTree.hpp"
#include "DecisionTreeNode.hpp"
#include "EvalTrainingData.hpp"
#include "TrainingDataGeneratorNumeric.hpp"
#include "TrainingDataGeneratorSymbolic.hpp"
#include "Utility.hpp"

#include <Eigen/Dense>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define PYTHON_BUILD

namespace py = pybind11;

// Define the doughnut function for demonstration

// Define the DecisionTreeNode module
PYBIND11_MODULE(DecisionTreePP, m)
{

    m.doc() = "Decision Tree Plus Plus Module"; // Optional module documentation

    py::class_<DecisionTreeNode>(m, "DecisionTreeNode")
        .def(py::init<DecisionTree &>(), py::arg("dt"), "Constructor with DecisionTree reference")
        .def(py::init<const std::string &,
                      double,
                      const std::vector<double> &,
                      const std::vector<std::string> &,
                      DecisionTree &,
                      bool>(),
             py::arg("feature"),
             py::arg("entropy"),
             py::arg("class_probabilities"),
             py::arg("branch_features_and_values_or_thresholds"),
             py::arg("dt"),
             py::arg("root_or_not"),
             "Constructor with feature, entropy, class probabilities, branch features, DecisionTree reference, and "
             "root flag")
        .def("HowManyNodes", &DecisionTreeNode::HowManyNodes, "Get number of nodes")
        .def("GetClassNames", &DecisionTreeNode::GetClassNames, "Get class names")
        .def("GetNextSerialNum", &DecisionTreeNode::GetNextSerialNum, "Get next serial number")
        .def("GetFeature", &DecisionTreeNode::GetFeature, "Get feature")
        .def("GetNodeEntropy", &DecisionTreeNode::GetNodeEntropy, "Get node entropy")
        .def("GetClassProbabilities", &DecisionTreeNode::GetClassProbabilities, "Get class probabilities")
        .def("GetBranchFeaturesAndValuesOrThresholds",
             &DecisionTreeNode::GetBranchFeaturesAndValuesOrThresholds,
             "Get branch features and values or thresholds")
        .def("GetChildren", &DecisionTreeNode::GetChildren, "Get child nodes")
        .def("GetSerialNum", &DecisionTreeNode::GetSerialNum, "Get serial number")
        .def("SetClassNames", &DecisionTreeNode::SetClassNames, py::arg("class_names_list"), "Set class names")
        .def("SetNodeCreationEntropy",
             &DecisionTreeNode::SetNodeCreationEntropy,
             py::arg("entropy"),
             "Set node creation entropy")
        .def("AddChildLink", &DecisionTreeNode::AddChildLink, py::arg("new_node"), "Add a child link")
        .def("DeleteAllLinks", &DecisionTreeNode::DeleteAllLinks, "Delete all child links")
        .def("DisplayNode", &DecisionTreeNode::DisplayNode, "Display node information")
        .def("DisplayDecisionTree",
             &DecisionTreeNode::DisplayDecisionTree,
             py::arg("offset"),
             "Display decision tree structure");

    // Bind the DecisionTree class
    py::class_<DecisionTree>(m, "DecisionTree")
        .def(py::init<std::map<std::string, std::string>>(), "Constructor with kwargs")
        .def("getTrainingData", &DecisionTree::getTrainingData, "Retrieve training data")
        .def("calculateFirstOrderProbabilities",
             &DecisionTree::calculateFirstOrderProbabilities,
             "Calculate first order probabilities")
        .def("showTrainingData", &DecisionTree::showTrainingData, "Show training data")
        .def("classify",
             &DecisionTree::classify,
             py::arg("root_node"),
             py::arg("features_and_values"),
             "Classify based on features and values")
        .def("constructDecisionTreeClassifier",
             &DecisionTree::constructDecisionTreeClassifier,
             "Construct decision tree classifier")
        .def("classEntropyOnPriors", &DecisionTree::classEntropyOnPriors, "Calculate class entropy on priors")
        .def("probabilityOfFeatureValue",
             (double(DecisionTree::*)(const std::string &, const std::string &)) &
                 DecisionTree::probabilityOfFeatureValue,
             "Calculate probability of feature value (string)")
        .def("probabilityOfFeatureValue",
             (double(DecisionTree::*)(const std::string &, double)) & DecisionTree::probabilityOfFeatureValue,
             "Calculate probability of feature value (double)")
        .def("getTrainingDatafile", &DecisionTree::getTrainingDatafile, "Get training data file")
        .def("getEntropyThreshold", &DecisionTree::getEntropyThreshold, "Get entropy threshold")
        .def("getMaxDepthDesired", &DecisionTree::getMaxDepthDesired, "Get maximum depth desired")
        .def("getNumberOfHistogramBins", &DecisionTree::getNumberOfHistogramBins, "Get number of histogram bins")
        .def("getCsvClassColumnIndex", &DecisionTree::getCsvClassColumnIndex, "Get CSV class column index")
        .def("getCsvColumnsForFeatures", &DecisionTree::getCsvColumnsForFeatures, "Get CSV columns for features")
        .def("getSymbolicToNumericCardinalityThreshold",
             &DecisionTree::getSymbolicToNumericCardinalityThreshold,
             "Get symbolic to numeric cardinality threshold")
        .def("getCsvCleanupNeeded", &DecisionTree::getCsvCleanupNeeded, "Get CSV cleanup needed flag")
        .def("getDebug1", &DecisionTree::getDebug1, "Get debug value 1")
        .def("getDebug2", &DecisionTree::getDebug2, "Get debug value 2")
        .def("getDebug3", &DecisionTree::getDebug3, "Get debug value 3")
        .def("getHowManyTotalTrainingSamples",
             &DecisionTree::getHowManyTotalTrainingSamples,
             "Get total training samples count")
        .def("getFeatureNames", &DecisionTree::getFeatureNames, "Get feature names")
        .def("getTrainingDataDict", &DecisionTree::getTrainingDataDict, "Get training data dictionary")
        .def("setTrainingDatafile", &DecisionTree::setTrainingDatafile, "Set training data file")
        .def("setEntropyThreshold", &DecisionTree::setEntropyThreshold, "Set entropy threshold")
        .def("setMaxDepthDesired", &DecisionTree::setMaxDepthDesired, "Set maximum depth desired")
        .def("setNumberOfHistogramBins", &DecisionTree::setNumberOfHistogramBins, "Set number of histogram bins")
        .def("setCsvClassColumnIndex", &DecisionTree::setCsvClassColumnIndex, "Set CSV class column index")
        .def("setCsvColumnsForFeatures", &DecisionTree::setCsvColumnsForFeatures, "Set CSV columns for features")
        .def("setSymbolicToNumericCardinalityThreshold",
             &DecisionTree::setSymbolicToNumericCardinalityThreshold,
             "Set symbolic to numeric cardinality threshold")
        .def("setCsvCleanupNeeded", &DecisionTree::setCsvCleanupNeeded, "Set CSV cleanup needed flag")
        .def("setDebug1", &DecisionTree::setDebug1, "Set debug value 1")
        .def("setDebug2", &DecisionTree::setDebug2, "Set debug value 2")
        .def("setDebug3", &DecisionTree::setDebug3, "Set debug value 3")
        .def("setHowManyTotalTrainingSamples",
             &DecisionTree::setHowManyTotalTrainingSamples,
             "Set total training samples count",
             py::arg("how_many_total_training_samples"));

    // Bind EvalTrainingData class
    py::class_<EvalTrainingData>(m, "EvalTrainingData")
        .def(py::init<>(), "Default constructor")
        .def("evaluateTrainingData", &EvalTrainingData::evaluateTrainingData, "Evaluate the training data");

    // Bind TrainingDataGeneratorNumeric class
    py::class_<TrainingDataGeneratorNumeric>(m, "TrainingDataGeneratorNumeric")
        .def(py::init<std::map<std::string, std::string>>(), "Constructor with parameters", py::arg("kwargs"))
        .def("ReadParameterFileNumeric",
             &TrainingDataGeneratorNumeric::ReadParameterFileNumeric,
             "Read the parameter file for numeric data")
        .def("GenerateTrainingDataNumeric",
             &TrainingDataGeneratorNumeric::GenerateTrainingDataNumeric,
             "Generate the training data for numeric data")
        .def("GenerateMultivariateSamples",
             &TrainingDataGeneratorNumeric::GenerateMultivariateSamples,
             "Generate multivariate samples",
             py::arg("mean"),
             py::arg("cov"),
             py::arg("numSamples"))
        .def("getOutputCsvFile", &TrainingDataGeneratorNumeric::getOutputCsvFile, "Get output CSV file")
        .def("getParameterFile", &TrainingDataGeneratorNumeric::getParameterFile, "Get parameter file")
        .def("getNumberOfSamplesPerClass",
             &TrainingDataGeneratorNumeric::getNumberOfSamplesPerClass,
             "Get number of samples per class")
        .def("getDebug", &TrainingDataGeneratorNumeric::getDebug, "Get debug flag")
        .def("getClassNames", &TrainingDataGeneratorNumeric::getClassNames, "Get class names")
        .def("getFeaturesOrdered", &TrainingDataGeneratorNumeric::getFeaturesOrdered, "Get ordered features")
        .def("getClassNamesAndPriors",
             &TrainingDataGeneratorNumeric::getClassNamesAndPriors,
             "Get class names and priors")
        .def("getFeaturesWithValueRange",
             &TrainingDataGeneratorNumeric::getFeaturesWithValueRange,
             "Get features with value range")
        .def("getClassesAndTheirParamValues",
             &TrainingDataGeneratorNumeric::getClassesAndTheirParamValues,
             "Get classes and their parameter values");

    // Bind TrainingDataGeneratorSymbolic class
    py::class_<TrainingDataGeneratorSymbolic>(m, "TrainingDataGeneratorSymbolic")
        .def(py::init<std::map<std::string, std::string>>(), "Default constructor", py::arg("kwargs"))
        .def("ReadParameterFileSymbolic",
             &TrainingDataGeneratorSymbolic::ReadParameterFileSymbolic,
             "Read the parameter file for symbolic data")
        .def("GenerateTrainingDataSymbolic",
             &TrainingDataGeneratorSymbolic::GenerateTrainingDataSymbolic,
             "Generate the training data for symbolic data")
        .def("WriteTrainingDataToFile",
             &TrainingDataGeneratorSymbolic::WriteTrainingDataToFile,
             "Write training data to file")
        .def("getClassPriors", &TrainingDataGeneratorSymbolic::getClassPriors, "Get class priors")
        .def("getClassNames", &TrainingDataGeneratorSymbolic::getClassNames, "Get class names")
        .def("getFeaturesAndValuesDict",
             &TrainingDataGeneratorSymbolic::getFeaturesAndValuesDict,
             "Get features and values dictionary")
        .def("getBiasDict", &TrainingDataGeneratorSymbolic::getBiasDict, "Get bias dictionary")
        .def("getOutputDatafile", &TrainingDataGeneratorSymbolic::getOutputDatafile, "Get output data file")
        .def("getParameterFile", &TrainingDataGeneratorSymbolic::getParameterFile, "Get parameter file")
        .def("randomDouble", &TrainingDataGeneratorSymbolic::randomDouble, "Generate a random double")
        .def("getNumberOfTrainingSamples",
             &TrainingDataGeneratorSymbolic::getNumberOfTrainingSamples,
             "Get number of training samples")
        .def("getWriteToFile", &TrainingDataGeneratorSymbolic::getWriteToFile, "Get write to file flag")
        .def("getDebug1", &TrainingDataGeneratorSymbolic::getDebug1, "Get debug flag 1")
        .def("getDebug2", &TrainingDataGeneratorSymbolic::getDebug2, "Get debug flag 2")
        .def("getTrainingSampleRecords",
             &TrainingDataGeneratorSymbolic::getTrainingSampleRecords,
             "Get training sample records");

    // Bind utility functions
    m.def("sampleIndex", &sampleIndex, "Get the index of the sample by name");
    m.def("convert", &convert, "Convert a string to a double");
    m.def("CleanupCsvString", &CleanupCsvString, "Cleanup a CSV string");

    // Bind deepCopy function
    m.def(
        "deepCopy",
        [](const std::vector<std::string> &vec) { return deepCopy(vec); },
        "Create a deep copy of a vector of strings");

    // Bind minimum function
    m.def(
        "minimum",
        [](const std::vector<double> &vec) { return minimum(vec); },
        "Find the minimum value and its index in a vector of doubles");

    // Bind ClosestSamplingPoint function
    m.def(
        "ClosestSamplingPoint",
        [](const std::vector<double> &vec, double val) { return ClosestSamplingPoint(vec, val); },
        "Find the closest sampling point in a vector to the given value");
}
