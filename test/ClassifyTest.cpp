#include <gtest/gtest.h>
#include "DecisionTree.hpp"

class ClassifyTest : public ::testing::Test
{
protected:
    // Class members to be used in tests
    map<string, string> kwargsS;
    shared_ptr<DecisionTree> dtS; // Symbolic DecisionTree

    map<string, string> kwargsN;
    shared_ptr<DecisionTree> dtN; // Numeric DecisionTree

    void SetUp() override
    {
        kwargsS = {
            // Symbolic kwargs
            {       "training_datafile", "../test/resources/training_symbolic.csv"},
            {  "csv_class_column_index",                                       "1"},
            {"csv_columns_for_features",                              {2, 3, 4, 5}},
            {       "max_depth_desired",                                       "5"},
            {       "entropy_threshold",                                     "0.1"},
            {                  "debug3",                                       "0"}
        };

        kwargsN = {
            // Numeric kwargs
            {       "training_datafile", "../test/resources/stage3cancer.csv"},
            {  "csv_class_column_index",                                  "2"},
            {"csv_columns_for_features",                   {3, 4, 5, 6, 7, 8}},
            {       "max_depth_desired",                                  "8"},
            {       "entropy_threshold",                               "0.01"},
            {                  "debug3",                                  "0"}
        };

        dtS = make_shared<DecisionTree>(kwargsS);
        dtS->getTrainingData();
        dtS->calculateFirstOrderProbabilities();
        dtS->calculateClassPriors();

        dtN = make_shared<DecisionTree>(kwargsN);
        dtN->getTrainingData();
        dtN->calculateFirstOrderProbabilities();
        dtN->calculateClassPriors();
    }

    void TearDown() override
    {
        dtS.reset();
        dtN.reset();
    }
};

TEST_F(ClassifyTest, CheckdtExists)
{
    ASSERT_NE(dtS, nullptr);
    ASSERT_NE(dtN, nullptr);
}

TEST_F(ClassifyTest, ClassifySymbolic)
{
    // Construct Tree
    DecisionTreeNode* rootS = dtS->constructDecisionTreeClassifier();
    ASSERT_NE(rootS, nullptr);
    
    vector<string> testSample;
    map<string, string> classification;
    map<string, string> expected;

    {
        testSample = {"exercising=never", "smoking=heavy", "fatIntake=heavy", "videoAddiction=heavy"};
        classification = dtS->classify(rootS, testSample);
        cout << classification << endl;
    }

    ASSERT_EQ(1, 0);
}