#include "EvalTrainingData.hpp"

#include "DecisionTree.hpp"

#include <gtest/gtest.h>

class EvalTrainingDataTest : public ::testing::Test {
  protected:
    void SetUp() override
    {
        // called before each test
    }

    void TearDown() override
    {
        // called after each test ends
    }
};

TEST_F(EvalTrainingDataTest, testEvaluateTrainingData)
{
    std::cout << "testEvaluateTrainingData" << std::endl;

    // Input configuration
    std::string filename                      = "../test/resources/stage3cancer.csv";
    std::map<std::string, std::string> kwargs = {
        {                        "training_datafile",           filename},
        {                   "csv_class_column_index",                "2"},
        {                 "csv_columns_for_features", {3, 4, 5, 6, 7, 8}},
        {                        "entropy_threshold",             "0.01"},
        {                        "max_depth_desired",                "5"},
        {"symbolic_to_numeric_cardinality_threshold",               "10"},
        {                       "csv_cleanup_needed",                "1"}
    };

    // Initialize and run evaluation
    EvalTrainingData evalData = EvalTrainingData(kwargs);
    evalData.evaluateTrainingData();

    // Expected results
    ASSERT_EQ(1, 2);
}
