#include <gtest/gtest.h>
#include "DecisionTree.hpp"

class ClassifyTest : public ::testing::Test
{
protected:
    // Class members to be used in tests
    map<string, string> kwargsS;
    shared_ptr<DecisionTree> dtS; // Symbolic DecisionTree
    unique_ptr<DecisionTreeNode> nodeS;

    void SetUp() override
    {
        kwargsS = {
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
            {"debug3", "0"}
        };
        dtS = make_shared<DecisionTree>(kwargsS);
        nodeS = make_unique<DecisionTreeNode>(
            "feature", 0.1, vector<double>{0.2}, vector<string>{"branch"}, dtS, true);
    }

    void TearDown() override
    {
        dtS.reset();
    }
};

TEST_F(ClassifyTest, CheckdtExists)
{
    ASSERT_NE(dtS, nullptr);
}

TEST_F(ClassifyTest, ConstructorInitializesNode)
{
    ASSERT_NE(nodeS, nullptr);
}

// TEST_F(ClassifyTest, ClassifyFunction)
// {
    // Construct the features and values vector out of the features and values dictionary
    // vector<string> featuresAndValues;
    // for (const auto& kv : dt.getFeaturesAndValuesDict())
    // {
    //     for (const auto& value : kv.second)
    //     {
    //         featuresAndValues.push_back(kv.first + "=" + value);
    //     }
    // }

    // map<string, string> classification = dt.classify(&node, featuresAndValues);
// }