#include "DecisionTree.hpp"
#include "DTIntrospection.hpp"

int main() {
    // Initialize a symbolic decision tree
    map<string, string> kwargsS;
    shared_ptr<DecisionTree> dtS; // Symbolic DecisionTree
    shared_ptr<DTIntrospection> dtSI; // Symbolic DecisionTree Introspection

    kwargsS = {
        // Symbolic kwargs
        {       "training_datafile", "../test/resources/training_symbolic.csv"},
        {  "csv_class_column_index",                                       "1"},
        {"csv_columns_for_features",                              {2, 3, 4, 5}},
        {       "max_depth_desired",                                       "5"},
        {       "entropy_threshold",                                     "0.1"},
        {                  "debug3",                                       "0"}
    };

    // Constructing the tree
    dtS = make_shared<DecisionTree>(kwargsS);
    dtS->getTrainingData();
    dtS->calculateFirstOrderProbabilities();
    dtS->calculateClassPriors();
    dtS->constructDecisionTreeClassifier();

    // Constructing the introspection object
    dtSI = make_shared<DTIntrospection>(dtS);
    dtSI->initialize();


    // Initialize a numeric decision tree
    map<string, string> kwargsN;
    shared_ptr<DecisionTree> dtN; // Numeric DecisionTree
    shared_ptr<DTIntrospection> dtNI; // Numeric DecisionTree Introspection

    kwargsN = {
        // Numeric kwargs
        {       "training_datafile", "../test/resources/stage3cancer.csv"},
        {  "csv_class_column_index",                                  "2"},
        {"csv_columns_for_features",                   {3, 4, 5, 6, 7, 8}},
        {       "max_depth_desired",                                  "8"},
        {       "entropy_threshold",                               "0.01"},
        {                  "debug3",                                  "0"}
    };

    // Constructing the tree
    dtN = make_shared<DecisionTree>(kwargsN);
    dtN->getTrainingData();
    dtN->calculateFirstOrderProbabilities();
    dtN->calculateClassPriors();
    dtN->constructDecisionTreeClassifier();

    // Constructing the introspection object
    dtNI = make_shared<DTIntrospection>(dtN);
    dtNI->initialize();

    // Playing in the sand :)

    // SYMBOLIC
    // dtSI->explainClassificationsAtMultipleNodesInteractively(); // Interactive Introspection
    // dtS->printClassificationAnswer(dtS->classifyByAskingQuestions(dtS->getRootNode())); // Interactive Classification

    // NUMERIC
    // dtNI->explainClassificationsAtMultipleNodesInteractively(); // Interactive Introspection
    dtN->printClassificationAnswer(dtN->classifyByAskingQuestions(dtN->getRootNode())); // Interactive Classification

    return 0;
}