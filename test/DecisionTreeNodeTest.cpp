#include <gtest/gtest.h>
#include "DecisionTreeNode.hpp"

class DecisionTreeNodeTest : public ::testing::Test
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
    DecisionTreeNode node = DecisionTreeNode("feature", 0.1, {0.2}, {"branch"}, dt, true);
};

TEST_F(DecisionTreeNodeTest, ConstructorInitializesNode)
{
    ASSERT_NE(&node, nullptr);
}

TEST(DecisionTreeNodeTest, TestDisplayDecisionTree)
{
    // Create a decision tree node
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
    DecisionTreeNode node(dt);

    // Add child nodes
    std::shared_ptr<DecisionTreeNode> child1 = std::make_shared<DecisionTreeNode>(dt);
    std::shared_ptr<DecisionTreeNode> child2 = std::make_shared<DecisionTreeNode>(dt);
    node.AddChildLink(child1);
    node.AddChildLink(child2);

    // Display the decision tree
    testing::internal::CaptureStdout();
    node.DisplayDecisionTree("");
    std::string output = testing::internal::GetCapturedStdout();

    // Assert the expected output
    std::string expectedOutput = "NODE 0:  BRANCH TESTS TO NODE: []\n";
    expectedOutput += "   Decision Feature:    Node Creation Entropy: 0.000   Class Probs: []\n";
    expectedOutput += "   NODE 1:  BRANCH TESTS TO LEAF NODE: []\n";
    expectedOutput += "          Node Creation Entropy: 0.000   Class Probs: []\n";
    expectedOutput += "   NODE 2:  BRANCH TESTS TO LEAF NODE: []\n";
    expectedOutput += "          Node Creation Entropy: 0.000   Class Probs: []\n";

    // print the output
    std::cout << output << std::endl;

    ASSERT_EQ(output, expectedOutput);
}