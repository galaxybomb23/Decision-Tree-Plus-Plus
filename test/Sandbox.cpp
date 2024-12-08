#include "DecisionTree++.hpp"

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
int main()
{
    cout << evalTraningDataDemo();
    return 0;
}
