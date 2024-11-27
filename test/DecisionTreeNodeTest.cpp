#include "DecisionTreeNode.hpp"

#include <gtest/gtest.h>

class DecisionTreeNodeTest : public ::testing::Test 
{
protected:
    // Class members to be used in tests
    map<string, string> kwargs;
    shared_ptr<DecisionTree> dt;
    unique_ptr<DecisionTreeNode> node;

    void SetUp() override
    {
        kwargs = {
            {                        "training_datafile", "../test/resources/stage3cancer.csv"},
            {                        "entropy_threshold",                                "0.1"},
            {                        "max_depth_desired",                                 "20"},
            {                   "csv_class_column_index",                                  "1"},
            {"symbolic_to_numeric_cardinality_threshold",                                 "20"},
            {                 "csv_columns_for_features",      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}},
            {                 "number_of_histogram_bins",                                 "10"},
            {                       "csv_cleanup_needed",                                  "1"},
            {                                   "debug1",                                  "1"},
            {                                   "debug2",                                  "2"},
            {                                   "debug3",                                  "3"}
        };
        dt   = make_shared<DecisionTree>(kwargs);
        node = make_unique<DecisionTreeNode>(
            "feature", 0.1, vector<double>{0.2}, vector<string>{"branch"}, dt, true);
    }

    void TearDown() override
    {
        dt.reset();
    }
};


TEST_F(DecisionTreeNodeTest, ConstructorInitializesNode)
{
    ASSERT_NE(&node, nullptr);
}

TEST_F(DecisionTreeNodeTest, TestDisplayDecisionTree)
{
    // Add child nodes
    auto child1 = make_unique<DecisionTreeNode>(dt);
    auto child2 = make_unique<DecisionTreeNode>(dt);
    node->AddChildLink(std::move(child1));
    node->AddChildLink(std::move(child2));

    // Display the decision tree
    ::testing::internal::CaptureStdout();
    node->DisplayDecisionTree("");
    string output = ::testing::internal::GetCapturedStdout();
    cout << output << endl;
}

// test all setters and getters
TEST_F(DecisionTreeNodeTest, TestSettersAndGetters)
{
    // Setters
    node->SetClassNames({"class1", "class2"});
    node->SetNodeCreationEntropy(0.2);
    auto child = make_unique<DecisionTreeNode>(dt->getShared());
    node->AddChildLink(std::move(child));

    // Getters
    ASSERT_EQ(node->GetClassNames(), vector<string>({"class1", "class2"}));
    ASSERT_EQ(node->GetNodeEntropy(), 0.2);
    ASSERT_EQ(node->GetChildren().size(), 1);
}

// test get serial number
TEST_F(DecisionTreeNodeTest, TestGetSerialNum)
{
    ASSERT_EQ(node->GetSerialNum(), 0);
}

// test get feature at node
TEST_F(DecisionTreeNodeTest, TestGetFeatureAtNode)
{
    ASSERT_EQ(node->GetFeature(), "feature");
}

// test get branch features and values or thresholds
TEST_F(DecisionTreeNodeTest, TestGetBranchFeaturesAndValuesOrThresholds)
{
    ASSERT_EQ(node->GetBranchFeaturesAndValuesOrThresholds(), vector<string>({"branch"}));
}

// test get class probabilities
TEST_F(DecisionTreeNodeTest, TestGetClassProbabilities)
{
    ASSERT_EQ(node->GetClassProbabilities(), vector<double>({0.2}));
}

// test how many nodes
TEST_F(DecisionTreeNodeTest, TestHowManyNodes)
{
    ASSERT_EQ(node->HowManyNodes(), 1);
}

// test delete all links
TEST_F(DecisionTreeNodeTest, TestDeleteAllLinks)
{
    auto child = make_unique<DecisionTreeNode>(dt->getShared());
    node->AddChildLink(std::move(child));
    node->DeleteAllLinks();
    ASSERT_EQ(node->GetChildren().size(), 0);
}
