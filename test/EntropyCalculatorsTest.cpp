#include <gtest/gtest.h>
#include "DecisionTree.hpp"

class EntropyCalculatorsTest : public ::testing::Test
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
        {"csv_class_column_index", "8"},
        {"symbolic_to_numeric_cardinality_threshold", "20"},
        {"csv_columns_for_features", {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}},
        {"number_of_histogram_bins", "10"},
        {"csv_cleanup_needed", "1"},
        {"debug1", "1"},
        {"debug2", "2"},
        {"debug3", "0"}};
    DecisionTree dt = DecisionTree(kwargs);
    DecisionTreeNode node = DecisionTreeNode("feature", 0.0, {0.0}, {"branch"}, dt, true);
};

TEST_F(EntropyCalculatorsTest, CheckdtExists)
{
    ASSERT_NE(&dt, nullptr);
}

TEST_F(EntropyCalculatorsTest, ConstructorInitializesNode)
{
    ASSERT_NE(&node, nullptr);
}

TEST_F(EntropyCalculatorsTest, EntropyTest)
{
    double expectedEntropyOnPriors = 0.0;
    ASSERT_EQ(dt.classEntropyOnPriors(), expectedEntropyOnPriors);
}