#include <gtest/gtest.h>
#include "DecisionTree.hpp"

class DecisionTreeTest : public ::testing::Test
{
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
    DecisionTreeNode node = DecisionTreeNode("feature", 0.0, {0.0}, {"branch"}, dt, true);
};

TEST_F(DecisionTreeTest, CheckdtExists)
{
    ASSERT_NE(&dt, nullptr);
}

TEST_F(DecisionTreeTest, ConstructorInitializesNode)
{
    ASSERT_NE(&node, nullptr);
}

TEST_F(DecisionTreeTest, CheckParamsDt)
{
    ASSERT_EQ(dt.getTrainingDatafile(), "../test/resources/stage3cancer.csv");
    ASSERT_EQ(dt.getEntropyThreshold(), 0.1);
    ASSERT_EQ(dt.getMaxDepthDesired(), 20);
    ASSERT_EQ(dt.getCsvClassColumnIndex(), 1);
    ASSERT_EQ(dt.getSymbolicToNumericCardinalityThreshold(), 20);
    ASSERT_EQ(dt.getCsvColumnsForFeatures().size(), 10);
    ASSERT_EQ(dt.getNumberOfHistogramBins(), 10);
    ASSERT_EQ(dt.getCsvCleanupNeeded(), 1);
    ASSERT_EQ(dt.getDebug1(), 1);
    ASSERT_EQ(dt.getDebug2(), 2);
    ASSERT_EQ(dt.getDebug3(), 3);
    ASSERT_NO_THROW(dt.getTrainingData());
    ASSERT_EQ(dt.getHowManyTotalTrainingSamples(), 146);
}

TEST_F(DecisionTreeTest, CheckGetTrainingData)
{
    ASSERT_NO_THROW(dt.getTrainingData());
    ASSERT_EQ(dt.getHowManyTotalTrainingSamples(), 146);
}