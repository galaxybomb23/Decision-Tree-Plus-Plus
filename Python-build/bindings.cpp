#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include "DecisionTree.hpp"
#include "DecisionTreeNode.hpp"
#include "EvalTrainingData.hpp"
#include "TrainingDataGeneratorNumeric.hpp"
#include "TrainingDataGeneratorSymbolic.hpp"
#include "Utility.hpp"

// doughnut function for demo purposes
void doughnut(int fps, int distance, float increment, int refreshRate, int xpos, int ypos, int numupdates)
{
    int k;
    float A = 0, B = 0;
    float z[1760];
    char b[1760];
    float counter = .01;

    std::cout << "\x1b[2J"; // Clear screen

    while (numupdates > 0)
    {

        // sleep to meet the desired FPS
        std::this_thread::sleep_for(std::chrono::milliseconds(1000 / fps));
        memset(b, 32, 1760); // Initialize buffer with spaces
        memset(z, 0, 7040);  // Initialize z-buffer with zeroes

        for (float j = 0; j < 6.28; j += 0.17)
        {
            for (float i = 0; i < 6.28; i += 0.02)
            {
                float c = std::sin(i);
                float d = std::cos(j);
                float e = std::sin(A);
                float f = std::sin(j);
                float g = std::cos(A);
                float h = d + counter;
                float D = 1 / (c * h * e + f * g + 5);
                float l = std::cos(i);
                float m = std::cos(B);
                float n = std::sin(B);
                float t = c * h * g - f * e;

                int x = xpos + 30 * D * (l * h * m - t * n);
                int y = ypos + 15 * D * (l * h * n + t * m);
                int o = x + 80 * y;
                int N = 8 * ((f * e - c * d * g) * m - c * d * e - f * g - l * d * n);

                if (y > 0 && y < 22 && x > 0 && x < 80 && D > z[o])
                {
                    z[o] = D;
                    b[o] = ".,-~:;=!*#$@"[N > 0 ? N : 0];
                }
            }
        }

        std::cout << "\x1b[H"; // Move cursor to top left
        for (k = 0; k < 1760; k++)
        {
            std::this_thread::sleep_for(std::chrono::microseconds(refreshRate));
            std::cout << (k % 80 ? b[k] : '\n');
        }

        A += 0.04;
        B += 0.02;

        // make counter oscillate between 0 and distance
        if (counter >= distance || counter <= 0)
        {
            increment *= -1;
        }
        counter += increment;
        numupdates--;
    }
}

void display_decision_treeDemo()
{
    // Create a decision tree node
    // Class members to be used in tests
    std::map<std::string, std::string> kwargs = {
        {"training_datafile", "../test/resources/stage3cancer.csv"},
        {"entropy_threshold", "0.1"},
        {"max_depth_desired", "20"},
        {"csv_class_column_index", "1"},
        {"symbolic_to_numeric_cardinality_threshold", "20"},
        {"csv_columns_for_features", {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}},
        {"number_of_histogram_bins", "10"},
        {"csv_cleanup_needed", "1"},
        {"debug1", "1"},
        {"debug2", "2"},
        {"debug3", "3"}};
    DecisionTree dt = DecisionTree(kwargs);
    DecisionTreeNode node("feature", 0.1, {0.2}, {"branch"}, dt, true);

    // Add child nodes
    std::shared_ptr<DecisionTreeNode> child1 = std::make_shared<DecisionTreeNode>(dt);
    std::shared_ptr<DecisionTreeNode> child2 = std::make_shared<DecisionTreeNode>(dt);
    node.AddChildLink(child1);
    node.AddChildLink(child2);

    // Display the decision tree
    node.DisplayDecisionTree("");
}

#define PYTHON_BUILD

namespace py = pybind11;

// Define the doughnut function for demonstration

// Define the DecisionTreeNode module
PYBIND11_MODULE(DecisionTreePP, m)
{

    m.doc() = "Decision Tree Plus Plus Module"; // Optional module documentation

    py::class_<DecisionTreeNode>(m, "DecisionTreeNode")
        .def(py::init<DecisionTree &>(), py::arg("dt"), "Constructor with DecisionTree reference")
        .def(py::init<const std::string &, double, const std::vector<double> &, const std::vector<std::string> &, DecisionTree &, bool>(),
             py::arg("feature"), py::arg("entropy"), py::arg("class_probabilities"), py::arg("branch_features_and_values_or_thresholds"), py::arg("dt"), py::arg("root_or_not"),
             "Constructor with feature, entropy, class probabilities, branch features, DecisionTree reference, and root flag")
        .def("HowManyNodes", &DecisionTreeNode::HowManyNodes, "Get number of nodes")
        .def("GetClassNames", &DecisionTreeNode::GetClassNames, "Get class names")
        .def("GetNextSerialNum", &DecisionTreeNode::GetNextSerialNum, "Get next serial number")
        .def("GetFeature", &DecisionTreeNode::GetFeature, "Get feature")
        .def("GetNodeEntropy", &DecisionTreeNode::GetNodeEntropy, "Get node entropy")
        .def("GetClassProbabilities", &DecisionTreeNode::GetClassProbabilities, "Get class probabilities")
        .def("GetBranchFeaturesAndValuesOrThresholds", &DecisionTreeNode::GetBranchFeaturesAndValuesOrThresholds, "Get branch features and values or thresholds")
        .def("GetChildren", &DecisionTreeNode::GetChildren, "Get child nodes")
        .def("GetSerialNum", &DecisionTreeNode::GetSerialNum, "Get serial number")
        .def("SetClassNames", &DecisionTreeNode::SetClassNames, py::arg("class_names_list"), "Set class names")
        .def("SetNodeCreationEntropy", &DecisionTreeNode::SetNodeCreationEntropy, py::arg("entropy"), "Set node creation entropy")
        .def("AddChildLink", &DecisionTreeNode::AddChildLink, py::arg("new_node"), "Add a child link")
        .def("DeleteAllLinks", &DecisionTreeNode::DeleteAllLinks, "Delete all child links")
        .def("DisplayNode", &DecisionTreeNode::DisplayNode, "Display node information")
        .def("DisplayDecisionTree", &DecisionTreeNode::DisplayDecisionTree, py::arg("offset"), "Display decision tree structure");

    // Bind the DecisionTree class
    py::class_<DecisionTree>(m, "DecisionTree")
        .def(py::init<std::map<std::string, std::string>>(), "Constructor with kwargs")
        .def("getTrainingData", &DecisionTree::getTrainingData, "Retrieve training data")
        .def("calculateFirstOrderProbabilities", &DecisionTree::calculateFirstOrderProbabilities, "Calculate first order probabilities")
        .def("showTrainingData", &DecisionTree::showTrainingData, "Show training data")
        .def("classify", &DecisionTree::classify, py::arg("root_node"), py::arg("features_and_values"), "Classify based on features and values")
        .def("constructDecisionTreeClassifier", &DecisionTree::constructDecisionTreeClassifier, "Construct decision tree classifier")
        .def("classEntropyOnPriors", &DecisionTree::classEntropyOnPriors, "Calculate class entropy on priors")
        .def("probabilityOfFeatureValue", (double(DecisionTree::*)(const std::string &, const std::string &)) & DecisionTree::probabilityOfFeatureValue, "Calculate probability of feature value (string)")
        .def("probabilityOfFeatureValue", (double(DecisionTree::*)(const std::string &, double)) & DecisionTree::probabilityOfFeatureValue, "Calculate probability of feature value (double)")
        .def("getTrainingDatafile", &DecisionTree::getTrainingDatafile, "Get training data file")
        .def("getEntropyThreshold", &DecisionTree::getEntropyThreshold, "Get entropy threshold")
        .def("getMaxDepthDesired", &DecisionTree::getMaxDepthDesired, "Get maximum depth desired")
        .def("getNumberOfHistogramBins", &DecisionTree::getNumberOfHistogramBins, "Get number of histogram bins")
        .def("getCsvClassColumnIndex", &DecisionTree::getCsvClassColumnIndex, "Get CSV class column index")
        .def("getCsvColumnsForFeatures", &DecisionTree::getCsvColumnsForFeatures, "Get CSV columns for features")
        .def("getSymbolicToNumericCardinalityThreshold", &DecisionTree::getSymbolicToNumericCardinalityThreshold, "Get symbolic to numeric cardinality threshold")
        .def("getCsvCleanupNeeded", &DecisionTree::getCsvCleanupNeeded, "Get CSV cleanup needed flag")
        .def("getDebug1", &DecisionTree::getDebug1, "Get debug value 1")
        .def("getDebug2", &DecisionTree::getDebug2, "Get debug value 2")
        .def("getDebug3", &DecisionTree::getDebug3, "Get debug value 3")
        .def("getHowManyTotalTrainingSamples", &DecisionTree::getHowManyTotalTrainingSamples, "Get total training samples count")
        .def("getFeatureNames", &DecisionTree::getFeatureNames, "Get feature names")
        .def("getTrainingDataDict", &DecisionTree::getTrainingDataDict, "Get training data dictionary")
        .def("setTrainingDatafile", &DecisionTree::setTrainingDatafile, "Set training data file")
        .def("setEntropyThreshold", &DecisionTree::setEntropyThreshold, "Set entropy threshold")
        .def("setMaxDepthDesired", &DecisionTree::setMaxDepthDesired, "Set maximum depth desired")
        .def("setNumberOfHistogramBins", &DecisionTree::setNumberOfHistogramBins, "Set number of histogram bins")
        .def("setCsvClassColumnIndex", &DecisionTree::setCsvClassColumnIndex, "Set CSV class column index")
        .def("setCsvColumnsForFeatures", &DecisionTree::setCsvColumnsForFeatures, "Set CSV columns for features")
        .def("setSymbolicToNumericCardinalityThreshold", &DecisionTree::setSymbolicToNumericCardinalityThreshold, "Set symbolic to numeric cardinality threshold")
        .def("setCsvCleanupNeeded", &DecisionTree::setCsvCleanupNeeded, "Set CSV cleanup needed flag")
        .def("setDebug1", &DecisionTree::setDebug1, "Set debug value 1")
        .def("setDebug2", &DecisionTree::setDebug2, "Set debug value 2")
        .def("setDebug3", &DecisionTree::setDebug3, "Set debug value 3")
        .def("setHowManyTotalTrainingSamples", &DecisionTree::setHowManyTotalTrainingSamples, "Set total training samples count");

    // Bind EvalTrainingData class
    py::class_<EvalTrainingData>(m, "EvalTrainingData")
        .def(py::init<>(), "Default constructor")
        .def("evaluateTrainingData", &EvalTrainingData::evaluateTrainingData, "Evaluate the training data");

    // Bind TrainingDataGeneratorNumeric class
    py::class_<TrainingDataGeneratorNumeric>(m, "TrainingDataGeneratorNumeric")
        .def(py::init<std::map<std::string, std::string>>(), "Constructor with parameters")
        .def("ReadParameterFileNumeric", &TrainingDataGeneratorNumeric::ReadParameterFileNumeric, "Read the parameter file for numeric data")
        .def("GenerateTrainingDataNumeric", &TrainingDataGeneratorNumeric::GenerateTrainingDataNumeric, "Generate the training data for numeric data")
        .def("GenerateMultivariateSamples", &TrainingDataGeneratorNumeric::GenerateMultivariateSamples, "Generate multivariate samples", py::arg("mean"), py::arg("cov"), py::arg("numSamples"))
        .def("getOutputCsvFile", &TrainingDataGeneratorNumeric::getOutputCsvFile, "Get output CSV file")
        .def("getParameterFile", &TrainingDataGeneratorNumeric::getParameterFile, "Get parameter file")
        .def("getNumberOfSamplesPerClass", &TrainingDataGeneratorNumeric::getNumberOfSamplesPerClass, "Get number of samples per class")
        .def("getDebug", &TrainingDataGeneratorNumeric::getDebug, "Get debug flag")
        .def("getClassNames", &TrainingDataGeneratorNumeric::getClassNames, "Get class names")
        .def("getFeaturesOrdered", &TrainingDataGeneratorNumeric::getFeaturesOrdered, "Get ordered features")
        .def("getClassNamesAndPriors", &TrainingDataGeneratorNumeric::getClassNamesAndPriors, "Get class names and priors")
        .def("getFeaturesWithValueRange", &TrainingDataGeneratorNumeric::getFeaturesWithValueRange, "Get features with value range")
        .def("getClassesAndTheirParamValues", &TrainingDataGeneratorNumeric::getClassesAndTheirParamValues, "Get classes and their parameter values");

    // Bind TrainingDataGeneratorSymbolic class
    py::class_<TrainingDataGeneratorSymbolic>(m, "TrainingDataGeneratorSymbolic")
        .def(py::init<>(), "Default constructor")
        .def("ReadParameterFileSymbolic", &TrainingDataGeneratorSymbolic::ReadParameterFileSymbolic, "Read the parameter file for symbolic data")
        .def("GenerateTrainingDataSymbolic", &TrainingDataGeneratorSymbolic::GenerateTrainingDataSymbolic, "Generate the training data for symbolic data");

    // Bind utility functions
    m.def("sampleIndex", &sampleIndex, "Get the index of the sample by name");
    m.def("convert", &convert, "Convert a string to a double");
    m.def("CleanupCsvString", &CleanupCsvString, "Cleanup a CSV string");

    // Bind deepCopy function
    m.def("deepCopy", [](const std::vector<std::string> &vec)
          { return deepCopy(vec); }, "Create a deep copy of a vector of strings");

    // Bind minimum function
    m.def("minimum", [](const std::vector<double> &vec)
          { return minimum(vec); }, "Find the minimum value and its index in a vector of doubles");

    // Bind ClosestSamplingPoint function
    m.def("ClosestSamplingPoint", [](const std::vector<double> &vec, double val)
          { return ClosestSamplingPoint(vec, val); }, "Find the closest sampling point in a vector to the given value");

    // Bind the doughnut function
    m.def("doughnut", &doughnut, "Display a doughnut on the screen", py::arg("fps"), py::arg("distance"), py::arg("increment"), py::arg("refreshRate"), py::arg("xpos"), py::arg("ypos"), py::arg("numupdates"));

    // Bind the display_decision_treeDemo function
    m.def("display_decision_treeDemo", &display_decision_treeDemo, "Display a decision tree for demonstration purposes");
}
