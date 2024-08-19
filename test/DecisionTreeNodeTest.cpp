#include <gtest/gtest.h>
#include "DecisionTreeNode.hpp"

class DecisionTreeNodeTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Initialization code here
    }

    void TearDown() override
    {
        // Cleanup code here (delete DecisionTreeNode object)
    }

    // Class members to be used in tests
    DecisionTreeNode node;
};

TEST_F(DecisionTreeNodeTest, Constructor)
{
    ASSERT_EQ(&node, nullptr);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
