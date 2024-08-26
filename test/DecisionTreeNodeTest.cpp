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
    DecisionTreeNode node;
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
