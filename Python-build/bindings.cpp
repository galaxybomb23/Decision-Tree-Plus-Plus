#include "DecisionTree++.hpp"

#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>


#define PYTHON_BUILD

// Demo funcs
double evalTraningDataDemo()
{
    map<string, string> kwargs = {
        {                        "training_datafile", "../test/resources/stage3cancer.csv"},
        {                   "csv_class_column_index",                                  "2"},
        {                 "csv_columns_for_features",                   {3, 4, 5, 6, 7, 8}},
        {                        "entropy_threshold",                               "0.01"},
        {                        "max_depth_desired",                                  "5"},
        {"symbolic_to_numeric_cardinality_threshold",                                 "10"},
        {                       "csv_cleanup_needed",                                  "1"}
    };

    auto evalData = make_shared<EvalTrainingData>(kwargs);

    // Initialize and run evaluation
    evalData->getTrainingData();
    double idx = evalData->evaluateTrainingData();
    return idx;
}

void constructDemo()
{
    // setup
    std::map<std::string, std::string> kwargsS = {
        // Symbolic kwargs
        {       "training_datafile", "../test/resources/training_symbolic.csv"},
        {  "csv_class_column_index",                                       "1"},
        {"csv_columns_for_features",                              {2, 3, 4, 5}},
        {       "max_depth_desired",                                       "5"},
        {       "entropy_threshold",                                     "0.1"},
        {                  "debug3",                                       "0"}
    };
    auto dtS = std::make_shared<DecisionTree>(kwargsS); // Initialize the DecisionTree
    dtS->getTrainingData();
    dtS->calculateFirstOrderProbabilities();
    dtS->calculateClassPriors();

    // Construct the DecisionTree
    dtS->constructDecisionTreeClassifier();
    auto root_Node = dtS->getRootNode();
    root_Node->DisplayDecisionTree("  ");
}

void interactiveIntrospection()
{

    // setup
    std::map<std::string, std::string> kwargsS = {
        // Symbolic kwargs
        {       "training_datafile", "../test/resources/training_symbolic.csv"},
        {  "csv_class_column_index",                                       "1"},
        {"csv_columns_for_features",                              {2, 3, 4, 5}},
        {       "max_depth_desired",                                       "5"},
        {       "entropy_threshold",                                     "0.1"},
        {                  "debug3",                                       "0"}
    };
    auto dtS = std::make_shared<DecisionTree>(kwargsS); // Initialize the DecisionTree
    dtS->getTrainingData();
    dtS->calculateFirstOrderProbabilities();
    dtS->calculateClassPriors();
    dtS->constructDecisionTreeClassifier();

    // Constructing the introspection object
    auto dtSI = make_shared<DTIntrospection>(dtS);
    dtSI->initialize();

    dtSI->explainClassificationsAtMultipleNodesInteractively(); // Interactive Introspection
}

void interactiveClassification()
{
    // setup
    std::map<std::string, std::string> kwargsS = {
        // Symbolic kwargs
        {       "training_datafile", "../test/resources/training_symbolic.csv"},
        {  "csv_class_column_index",                                       "1"},
        {"csv_columns_for_features",                              {2, 3, 4, 5}},
        {       "max_depth_desired",                                       "5"},
        {       "entropy_threshold",                                     "0.1"},
        {                  "debug3",                                       "0"}
    };
    auto dtS = std::make_shared<DecisionTree>(kwargsS); // Initialize the DecisionTree
    dtS->getTrainingData();
    dtS->calculateFirstOrderProbabilities();
    dtS->calculateClassPriors();
    dtS->constructDecisionTreeClassifier();

    // Constructing the introspection object
    auto dtSI = make_shared<DTIntrospection>(dtS);
    dtSI->initialize();

    dtS->printClassificationAnswer(dtS->classifyByAskingQuestions(dtS->getRootNode())); // Interactive Classification
}
float BenchmarkConstructSmall()
{
    auto start                                 = std::chrono::high_resolution_clock::now();
    std::map<std::string, std::string> kwargsS = {
        // Symbolic kwargs
        {       "training_datafile", "../test/resources/training_symbolic.csv"},
        {  "csv_class_column_index",                                       "1"},
        {"csv_columns_for_features",                              {2, 3, 4, 5}},
        {       "max_depth_desired",                                       "5"},
        {       "entropy_threshold",                                     "0.1"},
        {                  "debug3",                                       "0"}
    };
    auto dtS = std::make_shared<DecisionTree>(kwargsS); // Initialize the DecisionTree
    dtS->getTrainingData();
    dtS->calculateFirstOrderProbabilities();
    dtS->calculateClassPriors();
    dtS->constructDecisionTreeClassifier();
    return std::chrono::duration_cast<std::chrono::duration<float>>(std::chrono::high_resolution_clock::now() - start)
        .count();
}

float BenchmarkConstructLarge()
{
    auto startTime = std::chrono::high_resolution_clock::now();
    std::cout << "BenchmarkConstructLarge" << std::endl;
    std::map<std::string, std::string> kwargsS = {
        // Symbolic kwargs
        {       "training_datafile", "../test/resources/training_symbolic_large1.csv"},
        {  "csv_class_column_index",                                              "1"},
        {"csv_columns_for_features",                                     {2, 3, 4, 5}},
        {       "max_depth_desired",                                              "5"},
        {       "entropy_threshold",                                            "0.1"},
        {                  "debug3",                                              "0"}
    };
    auto dtS = std::make_shared<DecisionTree>(kwargsS); // Initialize the DecisionTree
    dtS->getTrainingData();
    dtS->calculateClassPriors();
    dtS->calculateFirstOrderProbabilities();
    auto root_node = dtS->constructDecisionTreeClassifier();

    return std::chrono::duration_cast<std::chrono::duration<float>>(std::chrono::high_resolution_clock::now() -
                                                                    startTime)
        .count();
}

float BenchmarkClassifySmall()
{
    std::cout << "BenchmarkClassifySmall" << std::endl;
    auto start                                 = std::chrono::high_resolution_clock::now();
    std::map<std::string, std::string> kwargsS = {
        // Symbolic kwargs
        {       "training_datafile", "../test/resources/training_symbolic.csv"},
        {  "csv_class_column_index",                                       "1"},
        {"csv_columns_for_features",                              {2, 3, 4, 5}},
        {       "max_depth_desired",                                       "5"},
        {       "entropy_threshold",                                     "0.1"},
        {                  "debug3",                                       "0"}
    };
    auto dtS = std::make_shared<DecisionTree>(kwargsS); // Initialize the DecisionTree
    dtS->getTrainingData();
    dtS->calculateClassPriors();
    dtS->calculateFirstOrderProbabilities();
    auto root_node = dtS->constructDecisionTreeClassifier();
    dtS->classify(root_node, {"exercising=never", "smoking=heavy", "fatIntake=heavy", "videoAddiction=heavy"});

    return std::chrono::duration_cast<std::chrono::duration<float>>(std::chrono::high_resolution_clock::now() - start)
        .count();
}

float BenchmarkClassifyLarge()
{

    cout << "BenchmarkClassifyLarge" << endl;
    auto start                                 = std::chrono::high_resolution_clock::now();
    std::map<std::string, std::string> kwargsS = {
        // Symbolic kwargs
        {       "training_datafile", "../test/resources/training_symbolic_large1.csv"},
        {  "csv_class_column_index",                                              "1"},
        {"csv_columns_for_features",                                     {2, 3, 4, 5}},
        {       "max_depth_desired",                                              "5"},
        {       "entropy_threshold",                                            "0.1"},
        {                  "debug3",                                              "0"}
    };

    auto dtS = std::make_shared<DecisionTree>(kwargsS); // Initialize the DecisionTree
    dtS->getTrainingData();
    dtS->calculateClassPriors();
    dtS->calculateFirstOrderProbabilities();
    auto root_node = dtS->constructDecisionTreeClassifier();
    dtS->classify(root_node, {"exercising=never", "smoking=heavy", "fatIntake=heavy", "videoAddiction=heavy"});

    return std::chrono::duration_cast<std::chrono::duration<float>>(std::chrono::high_resolution_clock::now() - start)
        .count();
}

namespace pybind11 {
namespace detail {
template <> struct type_caster<std::unique_ptr<DecisionTreeNode>> {
  public:
    PYBIND11_TYPE_CASTER(std::unique_ptr<DecisionTreeNode>, _("std::unique_ptr<DecisionTreeNode>"));

    // Load the Python object into C++ type
    bool load(pybind11::handle src, bool)
    {
        if (src.is_none()) {
            value = nullptr;
            return true;
        }

        // Convert the Python object to a C++ unique_ptr
        if (PyObject* obj = src.ptr()) {
            // Assuming src is a pointer to a C++ DecisionTreeNode object, use pybind11 casting
            value = std::make_unique<DecisionTreeNode>(src.cast<DecisionTreeNode>());
            return true;
        }
        return false;
    }

    // Convert C++ object to Python object
    static pybind11::handle
    cast(const std::unique_ptr<DecisionTreeNode> &src, pybind11::return_value_policy, pybind11::handle)
    {
        if (src) {
            return pybind11::cast(*src);
        }
        else {
            return pybind11::none().release();
        }
    }
};
} // namespace detail
} // namespace pybind11


namespace py = pybind11;

// Define the doughnut function for demonstration

// Define the DecisionTreeNode module
PYBIND11_MODULE(DecisionTreePP, m)
{

    m.doc() = "Decision Tree Plus Plus Module"; // Optional module documentation

    // ========= DecisionTreeNode Class =========
    py::class_<DecisionTreeNode, std::shared_ptr<DecisionTreeNode>>(m, "DecisionTreeNode")

        // Constructors
        .def_static(
            "create_default",
            []() { return std::make_shared<DecisionTreeNode>(std::make_shared<DecisionTree>()); },
            "Factory method to create a default DecisionTreeNode")
        .def(py::init<std::shared_ptr<DecisionTree>>(), py::arg("dt"), "Constructor with DecisionTree reference")
        .def(py::init<const std::string &,
                      double,
                      const std::vector<double> &,
                      const std::vector<std::string> &,
                      std::shared_ptr<DecisionTree>,
                      bool>(),
             py::arg("feature"),
             py::arg("entropy"),
             py::arg("class_probabilities"),
             py::arg("branch_features_and_values_or_thresholds"),
             py::arg("dt"),
             py::arg("isRoot"),
             "Constructor with feature, entropy, class probabilities, branch features, DecisionTree reference, and "
             "root flag")

        // Methods
        .def("HowManyNodes", &DecisionTreeNode::HowManyNodes, "Returns the total number of nodes in the tree")

        // Getters
        .def("GetClassNames", &DecisionTreeNode::GetClassNames, "Returns the list of class names")
        .def("GetNextSerialNum", &DecisionTreeNode::GetNextSerialNum, "Returns the next available serial number")
        .def("GetFeature", &DecisionTreeNode::GetFeature, "Returns the feature used for splitting at this node")
        .def("GetNodeEntropy", &DecisionTreeNode::GetNodeEntropy, "Returns the entropy value of this node")
        .def("GetClassProbabilities",
             &DecisionTreeNode::GetClassProbabilities,
             "Returns the probability distribution across classes")
        .def("GetBranchFeaturesAndValuesOrThresholds",
             &DecisionTreeNode::GetBranchFeaturesAndValuesOrThresholds,
             "Returns the features and their corresponding values or thresholds used for branching")
        .def("GetChildren", &DecisionTreeNode::GetChildren, "Returns a list of child nodes")
        .def("GetSerialNum", &DecisionTreeNode::GetSerialNum, "Returns this node's serial number")

        // Setters
        .def("SetClassNames",
             &DecisionTreeNode::SetClassNames,
             py::arg("classNames"),
             "Sets the class names for this node")
        .def(
            "SetFeature", &DecisionTreeNode::SetFeature, py::arg("feature"), "Sets the splitting feature for this node")
        .def("SetNodeCreationEntropy",
             &DecisionTreeNode::SetNodeCreationEntropy,
             py::arg("entropy"),
             "Sets the entropy value for this node")

        // Child management
        .def("AddChildLink", &DecisionTreeNode::AddChildLink, py::arg("newNode"), "Adds a child node to this node")
        .def("DeleteAllLinks", &DecisionTreeNode::DeleteAllLinks, "Removes all child nodes")

        // Display methods
        .def("DisplayNode",
             &DecisionTreeNode::DisplayNode,
             py::arg("offset"),
             "Displays information about this node with the given indentation")
        .def("DisplayDecisionTree",
             &DecisionTreeNode::DisplayDecisionTree,
             py::arg("offset"),
             "Displays the entire decision tree starting from this node");

    // =========== DecisionTree Class ===========
    py::class_<DecisionTree, std::shared_ptr<DecisionTree>>(m, "DecisionTree")
        //--------------- Constructors and Destructors ----------------//
        .def(py::init<std::map<std::string, std::string>>(), "Constructor with kwargs")

        //--------------- Class Functions ----------------//
        .def("getTrainingData", &DecisionTree::getTrainingData, "Retrieve training data")
        .def("calculateFirstOrderProbabilities",
             &DecisionTree::calculateFirstOrderProbabilities,
             "Calculate first order probabilities")
        .def("showTrainingData", &DecisionTree::showTrainingData, "Show training data")

        //--------------- Classify ----------------//
        .def("classify",
             &DecisionTree::classify,
             py::arg("root_node"),
             py::arg("features_and_values"),
             "Classify based on features and values")

        .def("recursiveDescentForClassification",
             &DecisionTree::recursiveDescentForClassification,
             py::arg("node"),
             py::arg("feature_and_values"),
             py::arg("answer"),
             "Recursive descent for classification")

        .def("classifyByAskingQuestions",
             &DecisionTree::classifyByAskingQuestions,
             py::arg("root_node"),
             "Classify by asking questions")
        .def("interactiveRecursiveDescentForClassification",
             &DecisionTree::interactiveRecursiveDescentForClassification,
             py::arg("node"),
             py::arg("answer"),
             py::arg("scratchpadForNumerics"),
             "Interactive recursive descent for classification")

        // -------------- Construct Tree ----------------//
        .def("constructDecisionTreeClassifier",
             &DecisionTree::constructDecisionTreeClassifier,
             "Construct decision tree classifier")
        .def("recursiveDescent", &DecisionTree::recursiveDescent, py::arg("node"), "Recursive descent")
        .def("bestFeatureCalculator",
             &DecisionTree::bestFeatureCalculator,
             py::arg("featuresAndValuesOrThresholdsOnBranch"),
             py::arg("existingNodeEntropy"),
             "Calculate the best feature for the decision tree")

        // --------- Entropy Calculators ------------//

        .def("classEntropyOnPriors", &DecisionTree::classEntropyOnPriors, "Calculate class entropy on priors")
        .def("entropyScannerForANumericFeature",
             &DecisionTree::entropyScannerForANumericFeature,
             py::arg("feature"),
             "Scan entropy for a numeric feature")
        .def("EntropyForThresholdForFeature",
             &DecisionTree::EntropyForThresholdForFeature,
             py::arg("arrayOfFeaturesAndValuesOrThresholds"),
             py::arg("feature"),
             py::arg("threshold"),
             py::arg("comparison"),
             "Calculate entropy for a threshold for a feature")
        .def("classEntropyForLessThanThresholdForFeature",
             &DecisionTree::classEntropyForLessThanThresholdForFeature,
             py::arg("arrayOfFeaturesAndValuesOrThresholds"),
             py::arg("feature"),
             py::arg("threshold"),
             "Calculate class entropy for less than threshold for a feature")
        .def("classEntropyForGreaterThanThresholdForFeature",
             &DecisionTree::classEntropyForGreaterThanThresholdForFeature,
             py::arg("arrayOfFeaturesAndValuesOrThresholds"),
             py::arg("feature"),
             py::arg("threshold"),
             "Calculate class entropy for greater than threshold for a feature")
        .def("classEntropyForAGivenSequenceOfFeaturesAndValuesOrThresholds",
             &DecisionTree::classEntropyForAGivenSequenceOfFeaturesAndValuesOrThresholds,
             py::arg("arrayOfFeaturesAndValuesOrThresholds"),
             "Calculate class entropy for a given sequence of features and values or thresholds")

        // --------------- Probability Calculators ----------------//
        .def("priorProbabilityForClass",
             &DecisionTree::priorProbabilityForClass,
             py::arg("className"),
             "Calculate prior probability for a class")
        .def("calculateClassPriors", &DecisionTree::calculateClassPriors, "Calculate class priors")
        .def("probabilityOfFeatureValue",
             (double(DecisionTree::*)(const std::string &, const std::string &)) &
                 DecisionTree::probabilityOfFeatureValue,
             py::arg("feature"),
             py::arg("value"),
             "Calculate probability of feature value")
        .def("probabilityOfFeatureValueGivenClass",
             &DecisionTree::probabilityOfFeatureValueGivenClass,
             py::arg("feature"),
             py::arg("value"),
             py::arg("className"),
             "Calculate probability of feature value given class")
        .def("probabilityOfFeatureLessThanThreshold",
             &DecisionTree::probabilityOfFeatureLessThanThreshold,
             py::arg("featureName"),
             py::arg("threshold"),
             "Calculate probability of feature less than threshold")
        .def("probabilityOfFeatureLessThanThresholdGivenClass",
             &DecisionTree::probabilityOfFeatureLessThanThresholdGivenClass,
             py::arg("featureName"),
             py::arg("threshold"),
             py::arg("className"),
             "Calculate probability of feature less than threshold given class")
        .def("probabilityOfASequenceOfFeaturesAndValuesOrThresholds",
             &DecisionTree::probabilityOfASequenceOfFeaturesAndValuesOrThresholds,
             py::arg("arrayOfFeaturesAndValuesOrThresholds"),
             "Calculate probability of a sequence of features and values or thresholds")
        .def("probabilityOfASequenceOfFeaturesAndValuesOrThresholdsGivenClass",
             &DecisionTree::probabilityOfASequenceOfFeaturesAndValuesOrThresholdsGivenClass,
             py::arg("arrayOfFeaturesAndValuesOrThresholds"),
             py::arg("className"),
             "Calculate probability of a sequence of features and values or thresholds given class")
        .def("probabilityOfAClassGivenSequenceOfFeaturesAndValuesOrThresholds",
             &DecisionTree::probabilityOfAClassGivenSequenceOfFeaturesAndValuesOrThresholds,
             py::arg("className"),
             py::arg("arrayOfFeaturesAndValuesOrThresholds"),
             "Calculate probability of a class given a sequence of features and values or thresholds")

        // --------------- Class Based Utilities ----------------//
        .def("determineDataCondition", &DecisionTree::determineDataCondition, "Determine data condition")
        .def("checkNamesUsed", &DecisionTree::checkNamesUsed, py::arg("featuresAndValues"), "Check if names are used")
        .def("operator=", &DecisionTree::operator=, py::arg("dt"), "Assignment operator")
        .def("findBoundedIntervalsForNumericFeatures",
             &DecisionTree::findBoundedIntervalsForNumericFeatures,
             py::arg("trueNumericTypes"),
             "Find bounded intervals for numeric features")
        .def("printStats", &DecisionTree::printStats, "Print statistics")
        .def("printClassificationAnswer",
             &DecisionTree::printClassificationAnswer,
             py::arg("ClassificationAnswer"),
             "Print classification answer")

        // --------------- Getters ----------------//


        .def("getTrainingDatafile", &DecisionTree::getTrainingDatafile, "Get the training data file")
        .def("getEntropyThreshold", &DecisionTree::getEntropyThreshold, "Get the entropy threshold")
        .def("getMaxDepthDesired", &DecisionTree::getMaxDepthDesired, "Get the maximum desired depth")
        .def("getNumberOfHistogramBins", &DecisionTree::getNumberOfHistogramBins, "Get the number of histogram bins")
        .def("getClassNames", &DecisionTree::getClassNames, "Get the class names")
        .def("getCsvClassColumnIndex", &DecisionTree::getCsvClassColumnIndex, "Get the CSV class column index")
        .def("getCsvColumnsForFeatures", &DecisionTree::getCsvColumnsForFeatures, "Get the CSV columns for features")
        .def("getSymbolicToNumericCardinalityThreshold",
             &DecisionTree::getSymbolicToNumericCardinalityThreshold,
             "Get the symbolic to numeric cardinality threshold")
        .def("getCsvCleanupNeeded", &DecisionTree::getCsvCleanupNeeded, "Get the CSV cleanup needed flag")
        .def("getDebug1", &DecisionTree::getDebug1, "Get the debug1 flag")
        .def("getDebug2", &DecisionTree::getDebug2, "Get the debug2 flag")
        .def("getDebug3", &DecisionTree::getDebug3, "Get the debug3 flag")
        .def("getHowManyTotalTrainingSamples",
             &DecisionTree::getHowManyTotalTrainingSamples,
             "Get the total number of training samples")
        .def("getFeatureNames", &DecisionTree::getFeatureNames, "Get the feature names")
        .def("getFeaturesAndValuesDict",
             &DecisionTree::getFeaturesAndValuesDict,
             "Get the features and values dictionary")
        .def("getSamplesClassLabelDict",
             &DecisionTree::getSamplesClassLabelDict,
             "Get the samples class label dictionary")
        .def("getFeaturesAndUniqueValuesDict",
             &DecisionTree::getFeaturesAndUniqueValuesDict,
             "Get the features and unique values dictionary")
        .def("getNumericFeaturesValueRangeDict",
             &DecisionTree::getNumericFeaturesValueRangeDict,
             "Get the numeric features value range dictionary")
        .def("getTrainingDataDict", &DecisionTree::getTrainingDataDict, "Get the training data dictionary")
        .def("getRootNode", &DecisionTree::getRootNode, "Get the root node")

        // --------------- Setters ----------------//
        .def("setTrainingDatafile",
             &DecisionTree::setTrainingDatafile,
             py::arg("trainingDatafile"),
             "Set the training data file")
        .def("setEntropyThreshold",
             &DecisionTree::setEntropyThreshold,
             py::arg("entropyThreshold"),
             "Set the entropy threshold")
        .def("setMaxDepthDesired",
             &DecisionTree::setMaxDepthDesired,
             py::arg("maxDepthDesired"),
             "Set the maximum desired depth")
        .def("setNumberOfHistogramBins",
             &DecisionTree::setNumberOfHistogramBins,
             py::arg("numberOfHistogramBins"),
             "Set the number of histogram bins")
        .def("setCsvClassColumnIndex",
             &DecisionTree::setCsvClassColumnIndex,
             py::arg("csvClassColumnIndex"),
             "Set the CSV class column index")
        .def("setCsvColumnsForFeatures",
             &DecisionTree::setCsvColumnsForFeatures,
             py::arg("csvColumnsForFeatures"),
             "Set the CSV columns for features")
        .def("setSymbolicToNumericCardinalityThreshold",
             &DecisionTree::setSymbolicToNumericCardinalityThreshold,
             py::arg("symbolicToNumericCardinalityThreshold"),
             "Set the symbolic to numeric cardinality threshold")
        .def("setCsvCleanupNeeded",
             &DecisionTree::setCsvCleanupNeeded,
             py::arg("csvCleanupNeeded"),
             "Set the CSV cleanup needed flag")
        .def("setDebug1", &DecisionTree::setDebug1, py::arg("debug1"), "Set the debug1 flag")
        .def("setDebug2", &DecisionTree::setDebug2, py::arg("debug2"), "Set the debug2 flag")
        .def("setDebug3", &DecisionTree::setDebug3, py::arg("debug3"), "Set the debug3 flag")
        .def("setHowManyTotalTrainingSamples",
             &DecisionTree::setHowManyTotalTrainingSamples,
             py::arg("howManyTotalTrainingSamples"),
             "Set the total number of training samples")
        .def("setRootNode", &DecisionTree::setRootNode, py::arg("rootNode"), "Set the root node")
        .def("setClassNames", &DecisionTree::setClassNames, py::arg("classNames"), "Set the class names");

    //========= Training Data Generator Numeric =======//
    py::class_<TrainingDataGeneratorNumeric>(m, "TrainingDataGeneratorNumeric")
        .def(py::init<std::map<std::string, std::string>>(), "Constructor with parameters")
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

    //========= Training Data Generator Symbolic =======//
    py::class_<TrainingDataGeneratorSymbolic>(m, "TrainingDataGeneratorSymbolic")
        .def(py::init<std::map<std::string, std::string>>(), "Constructor with parameters")
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
        .def("getNumberOfTrainingSamples",
             &TrainingDataGeneratorSymbolic::getNumberOfTrainingSamples,
             "Get number of training samples")
        .def("getWriteToFile", &TrainingDataGeneratorSymbolic::getWriteToFile, "Get write to file flag")
        .def("getDebug1", &TrainingDataGeneratorSymbolic::getDebug1, "Get debug1 flag")
        .def("getDebug2", &TrainingDataGeneratorSymbolic::getDebug2, "Get debug2 flag")
        .def("getTrainingSampleRecords",
             &TrainingDataGeneratorSymbolic::getTrainingSampleRecords,
             "Get training sample records");

    //========= EvalTrainingData Class =========//
    py::class_<EvalTrainingData, DecisionTree, std::shared_ptr<EvalTrainingData>>(m, "EvalTrainingData")
        .def(py::init<std::map<std::string, std::string>>(), "Constructor with parameters")
        .def("evaluateTrainingData", &EvalTrainingData::evaluateTrainingData, "Evaluate the training data")
        .def_readwrite("_dataQualityIndex", &EvalTrainingData::_dataQualityIndex)
        .def_readwrite("_csvClassColumnIndex", &EvalTrainingData::_csvClassColumnIndex);

    // ========= DecisionTree Introspection =========
    py::class_<DTIntrospection, std::shared_ptr<DTIntrospection>>(m, "DTIntrospection")
        //--------------- Constructors and Destructors ----------------//
        .def(py::init<std::shared_ptr<DecisionTree>>(), py::arg("dt"))
        .def("initialize", &DTIntrospection::initialize, "Initialize the introspection")
        //--------------- Recursive Descent ----------------//

        .def("recursiveDescent", &DTIntrospection::recursiveDescent, py::arg("node"), "Perform recursive descent")
        .def("recursiveDescentForShowingSamplesAtANode",
             &DTIntrospection::recursiveDescentForShowingSamplesAtANode,
             py::arg("node"),
             "Show samples at each node during recursive descent")
        .def("recursiveDescentForSampleToNodeInfluence",
             &DTIntrospection::recursiveDescentForSampleToNodeInfluence,
             py::arg("nodeSerialNum"),
             py::arg("nodesAlreadyAccountedFor"),
             py::arg("offset"),
             "Analyze sample influence during recursive descent")

        //--------------- Display ----------------//
        .def("displayTrainingSamplesAtAllNodesDirectInfluenceOnly",
             &DTIntrospection::displayTrainingSamplesAtAllNodesDirectInfluenceOnly,
             "Display training samples with direct influence")
        .def("displayTrainingSamplesToNodesInfluencePropagation",
             &DTIntrospection::displayTrainingSamplesToNodesInfluencePropagation,
             "Display influence propagation of training samples")

        //--------------- Explanation ----------------//
        .def("explainClassificationsAtMultipleNodesInteractively",
             &DTIntrospection::explainClassificationsAtMultipleNodesInteractively,
             "Interactive explanation of classifications")
        .def("explainClassificationAtOneNode",
             &DTIntrospection::explainClassificationAtOneNode,
             py::arg("nodeID"),
             "Explain classification at specific node")

        //--------------- Class Utility ----------------//
        .def("getSamplesForFeatureValueCombo",
             &DTIntrospection::getSamplesForFeatureValueCombo,
             py::arg("featureValueCombo"),
             "Get samples matching feature-value combination")
        .def("extractFeatureOpValue", &DTIntrospection::extractFeatureOpValue, "Extract feature, operator, and value")

        //--------------- Getters ----------------//
        .def("getSamplesAtNodesDict", &DTIntrospection::getSamplesAtNodesDict, "Get dictionary of samples at nodes")
        .def("getBranchFeaturesToNodesDict",
             &DTIntrospection::getBranchFeaturesToNodesDict,
             "Get dictionary of branch features to nodes")
        .def("getSampleToNodeMappingDirectDict",
             &DTIntrospection::getSampleToNodeMappingDirectDict,
             "Get direct mapping of samples to nodes")
        .def("getNodeSerialNumToNodeDict",
             &DTIntrospection::getNodeSerialNumToNodeDict,
             "Get mapping of serial numbers to nodes");


    //======== Structs
    py::class_<BestFeatureResult>(m, "BestFeatureResult")
        .def(py::init<>()) // Default constructor
        .def_readwrite("bestFeatureName", &BestFeatureResult::bestFeatureName)
        .def_readwrite("bestFeatureEntropy", &BestFeatureResult::bestFeatureEntropy)
        .def_readwrite("valBasedEntropies", &BestFeatureResult::valBasedEntropies)
        .def_readwrite("decisionValue", &BestFeatureResult::decisionValue);
    py::class_<ClassificationAnswer>(m, "ClassificationAnswer")
        .def(py::init<>()) // Default constructor
        .def_readwrite("classProbabilities", &ClassificationAnswer::classProbabilities)
        .def_readwrite("solutionPath", &ClassificationAnswer::solutionPath);

    //==== demo functions
    m.def("constructDemo", &constructDemo, "Construct a demo decision tree");
    m.def("interactiveIntrospectionDemo", &interactiveIntrospection, "Interactive introspection");
    m.def("interactiveClassificationDemo", &interactiveClassification, "Interactive classification");
    m.def("evalTrainingDataDemo", &evalTraningDataDemo, "Evaluate training data");

    m.def("BenchmarkConstructSmall", &BenchmarkConstructSmall, "Benchmark the construction of a small decision tree");
    m.def("BenchmarkConstructLarge", &BenchmarkConstructLarge, "Benchmark the construction of a large decision tree");
    m.def("BenchmarkClassifySmall", &BenchmarkClassifySmall, "Benchmark the classification of a small decision tree");
    m.def("BenchmarkClassifyLarge", &BenchmarkClassifyLarge, "Benchmark the classification of a large decision tree");
}
