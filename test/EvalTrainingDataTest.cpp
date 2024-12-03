#include "EvalTrainingData.hpp"

#include "DecisionTree.hpp"

#include <gtest/gtest.h>
#include <unordered_set>

class EvalTrainingDataTest : public ::testing::Test {
  protected:
    string filename = "../test/resources/stage3cancer.csv";

    map<string, string> kwargs;
    shared_ptr<EvalTrainingData> evalData;

    void SetUp() override
    {
        kwargs = {
            {                        "training_datafile",           filename},
            {                   "csv_class_column_index",                "2"},
            {                 "csv_columns_for_features", {3, 4, 5, 6, 7, 8}},
            {                        "entropy_threshold",             "0.01"},
            {                        "max_depth_desired",                "5"},
            {"symbolic_to_numeric_cardinality_threshold",               "10"},
            {                       "csv_cleanup_needed",                "1"}
        };

        evalData = make_shared<EvalTrainingData>(kwargs);
    }

    void TearDown() override { evalData.reset(); }
};

TEST_F(EvalTrainingDataTest, testEvaluateTrainingDataDiscrimination)
{
    std::cout << "testEvaluateTrainingData" << std::endl;

    // Initialize and run evaluation
    evalData->getTrainingData();
    double idx = evalData->evaluateTrainingData();

    // assert within ~5 points
    ASSERT_NEAR(idx, 60.71, 0.1);
}