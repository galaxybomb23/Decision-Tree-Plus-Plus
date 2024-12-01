#include <gtest/gtest.h>

#include "DTIntrospection.hpp"

class IntrospectionTest : public ::testing::Test
{
protected:
    // Class members to be used in tests
    map<string, string> kwargsS;
    shared_ptr<DecisionTree> dtS; // Symbolic DecisionTree
    shared_ptr<DTIntrospection> dtSI; // Symbolic DecisionTree Introspection

    map<string, string> kwargsN;
    shared_ptr<DecisionTree> dtN; // Numeric DecisionTree
    shared_ptr<DTIntrospection> dtNI; // Numeric DecisionTree Introspection

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
        dtSI = make_shared<DTIntrospection>(dtS);
        // dtSI->initialize();

        dtN = make_shared<DecisionTree>(kwargsN);
        dtN->getTrainingData();
        dtN->calculateFirstOrderProbabilities();
        dtN->calculateClassPriors();
        dtNI = make_shared<DTIntrospection>(dtN);
        // dtNI->initialize();
    }

    void TearDown() override
    {
        dtS.reset();
        dtSI.reset();

        dtN.reset();
        dtNI.reset();
    }
};

TEST_F(IntrospectionTest, CheckdtExists)
{
    ASSERT_NE(dtS, nullptr);
    ASSERT_NE(dtN, nullptr);
}

TEST_F(IntrospectionTest, CheckdtIExists)
{
    ASSERT_NE(dtSI, nullptr);
    ASSERT_NE(dtNI, nullptr);
}