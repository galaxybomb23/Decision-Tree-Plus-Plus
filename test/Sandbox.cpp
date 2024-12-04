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

    // Playing in the sand :)
    // dtSI->explainClassificationsAtMultipleNodesInteractively(); // Interactive explanation
    cout << dtS->classifyByAskingQuestions(dtS->getRootNode()) << endl; // Non-interactive explanation

    return 0;
}