#include <gtest/gtest.h>
#include "DecisionTreeNode.hpp"

class DecisionTreeNodeTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // called before each test
        DecisionTree dt = DecisionTree();
        node = DecisionTreeNode(&dt)
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
    // Test the DecisionTreeNode constructor
    DecisionTree dt = DecisionTree();
    DecisionTreeNode node = DecisionTreeNode(&dt);
    EXPECT_EQ(node.GetSerialNum(), 0);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
