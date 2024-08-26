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
    std::map<std::string, std::string> kwargs;
    DecisionTree dt = DecisionTree(kwargs);
    DecisionTreeNode node = DecisionTreeNode("feature", 0.0, {0.0}, {"branch"}, dt, true);
};

TEST_F(DecisionTreeNodeTest, TestDecisionTreeNode)
{
    ASSERT_NE(&node, nullptr);
}

TEST(DecisionTreeNodeTest, TestDisplayDecisionTree)
{
    // Create a decision tree node
    DecisionTree dt;
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

    ASSERT_EQ(output, expectedOutput);
}