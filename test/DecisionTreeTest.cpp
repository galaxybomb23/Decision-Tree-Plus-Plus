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
        {"training_datafile", "test/resources/stage3cancer.csv"},
        {"entropy_threshold", "0.01"},
        {"max_depth_desired", "10"},
        {"csv_class_column_index", "0"},
        {"symbolic_to_numeric_cardinality_threshold", "10"},
        {"csv_columns_for_features", {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}},
        {"number_of_histogram_bins", "10"},
        {"csv_cleanup_needed", "0"},
        {"debug1", "0"},
        {"debug2", "0"},
        {"debug3", "0"}};
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
    ASSERT_EQ(dt.getTrainingDatafile(), "test/resources/stage3cancer.csv");
    ASSERT_EQ(dt.getEntropyThreshold(), 0.01);
    ASSERT_EQ(dt.getMaxDepthDesired(), 10);
    ASSERT_EQ(dt.getCsvClassColumnIndex(), 0);
    ASSERT_EQ(dt.getCsvColumnsForFeatures().size(), 10);
    ASSERT_EQ(dt.getCsvColumnsForFeatures(), std::vector<int>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));
    ASSERT_EQ(dt.getSymbolicToNumericCardinalityThreshold(), 10);
    ASSERT_EQ(dt.getNumberOfHistogramBins(), 10);
}