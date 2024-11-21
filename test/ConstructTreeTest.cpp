#include <gtest/gtest.h>
#include "DecisionTree.hpp"

class ConstructTreeTest : public ::testing::Test
{
    protected:
    std::unique_ptr<DecisionTree> dtS; // Symbolic DecisionTree
    std::unique_ptr<DecisionTree> dtN; // Numeric DecisionTree
    void SetUp() override
    {
        map<string, string> kwargsS = {
            // Symbolic kwargs
            {       "training_datafile", "../test/resources/training_symbolic.csv"},
            {  "csv_class_column_index",                                       "1"},
            {"csv_columns_for_features",                              {2, 3, 4, 5}},
            {       "max_depth_desired",                                       "5"},
            {       "entropy_threshold",                                     "0.1"}
        };

        map<string, string> kwargsN = {
            // Numeric kwargs
            {       "training_datafile", "../test/resources/stage3cancer.csv"},
            {  "csv_class_column_index",                                  "2"},
            {"csv_columns_for_features",                   {3, 4, 5, 6, 7, 8}},
            {       "max_depth_desired",                                  "8"},
            {       "entropy_threshold",                               "0.01"}
        };

        dtS = std::make_unique<DecisionTree>(kwargsS); // Initialize the DecisionTree
        dtS->getTrainingData();
        dtS->calculateFirstOrderProbabilities();
        dtS->calculateClassPriors();

        dtN = std::make_unique<DecisionTree>(kwargsN); // Initialize the DecisionTree
        dtN->getTrainingData();
        dtN->calculateFirstOrderProbabilities();
        dtN->calculateClassPriors();
    }

    void TearDown() override
    {
        dtS.reset(); // Reset the DecisionTree
    }
};

TEST_F(ConstructTreeTest, CheckdtExists)
{
    ASSERT_NE(&dtS, nullptr);
    ASSERT_NE(&dtN, nullptr);
}

TEST_F(ConstructTreeTest, bestFeatureCalculatorSymbolic)
{
    BestFeatureResult bfr = dtS->bestFeatureCalculator({}, 0.9580420222262995);
    ASSERT_EQ(bfr.bestFeatureName, "fatIntake");
    ASSERT_NEAR(bfr.bestFeatureEntropy, 0.539, 0.001);
    ASSERT_EQ(bfr.valBasedEntropies, nullopt);
    ASSERT_EQ(bfr.decisionValue, nullopt);

}