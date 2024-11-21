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

    // Class members to be used in tests
    std::map<std::string, std::string> kwargs = {
        {                        "training_datafile", "../test/resources/stage3cancer.csv"},
        {                        "entropy_threshold",                                "0.1"},
        {                        "max_depth_desired",                                 "20"},
        {                   "csv_class_column_index",                                  "1"},
        {"symbolic_to_numeric_cardinality_threshold",                                 "20"},
        {                 "csv_columns_for_features",      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}},
        {                 "number_of_histogram_bins",                                 "10"},
        {                       "csv_cleanup_needed",                                  "1"},
        {                                   "debug1",                                  "0"},
        {                                   "debug2",                                  "0"},
        {                                   "debug3",                                  "0"}
    };

    DecisionTree dt       = DecisionTree(kwargs);
    DecisionTreeNode node = DecisionTreeNode("feature", 0.0, {0.0}, {"branch"}, dt, true);
};

TEST_F(EvalTrainingDataTest, testExist)
{
    ASSERT_NE(&dt, nullptr);
    ASSERT_NE(&node, nullptr);
}

TEST_F(EvalTrainingDataTest, testEvaluateTrainingData)
{
    std::cout << "testEvaluateTrainingData" << std::endl;
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

    EvalTrainingData evalData = EvalTrainingData(kwargs);
    evalData.evaluateTrainingData();
    double dataQualityIndex = evalData._dataQualityIndex;

    // round to 2 decimal places
    dataQualityIndex = round(dataQualityIndex * 100) / 100;
    ASSERT_EQ(dataQualityIndex, 64.29);
}