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

// TEST_F(ConstructTreeTest, bestFeatureCalculatorSymbolic)
// {
//     BestFeatureResult bfr;
//     {
//         bfr = dtS->bestFeatureCalculator({}, 0.9580420222262995);
//         ASSERT_EQ(bfr.bestFeatureName, "fatIntake");
//         ASSERT_NEAR(bfr.bestFeatureEntropy, 0.539, 0.001);
//         ASSERT_EQ(bfr.valBasedEntropies, nullopt);
//         ASSERT_EQ(bfr.decisionValue, nullopt);
//     }
//     {
//         bfr = dtS->bestFeatureCalculator({"fatIntake=heavy"}, 0.7732266742876344);
//         ASSERT_EQ(bfr.bestFeatureName, "smoking");
//         ASSERT_NEAR(bfr.bestFeatureEntropy, 0.271, 0.001);
//         ASSERT_EQ(bfr.valBasedEntropies, nullopt);
//         ASSERT_EQ(bfr.decisionValue, nullopt);
//     }
//     {
//         bfr = dtS->bestFeatureCalculator({"fatIntake=heavy", "smoking=heavy"}, 0.28788102213037137);
//         ASSERT_EQ(bfr.bestFeatureName, "videoAddiction");
//         ASSERT_NEAR(bfr.bestFeatureEntropy, 0.058, 0.001);
//         ASSERT_EQ(bfr.valBasedEntropies, nullopt);
//         ASSERT_EQ(bfr.decisionValue, nullopt);
//     }
//     {
//         bfr = dtS->bestFeatureCalculator({"fatIntake=heavy", "smoking=heavy", "videoAddiction=heavy"}, 0.16223190039782087);
//         ASSERT_EQ(bfr.bestFeatureName, "exercising");
//         ASSERT_NEAR(bfr.bestFeatureEntropy, 0.014, 0.001);
//         ASSERT_EQ(bfr.valBasedEntropies, nullopt);
//         ASSERT_EQ(bfr.decisionValue, nullopt);
//     }
//     {
//         bfr = dtS->bestFeatureCalculator({"fatIntake=low"}, 0.5032583347756457);
//         ASSERT_EQ(bfr.bestFeatureName, "smoking");
//         ASSERT_NEAR(bfr.bestFeatureEntropy, 0.133, 0.001);
//         ASSERT_EQ(bfr.valBasedEntropies, nullopt);
//         ASSERT_EQ(bfr.decisionValue, nullopt);
//     }
//     {
//         bfr = dtS->bestFeatureCalculator({"fatIntake=low", "smoking=light"}, 0.13609257142369133);
//         ASSERT_EQ(bfr.bestFeatureName, "videoAddiction");
//         ASSERT_NEAR(bfr.bestFeatureEntropy, 0.009, 0.001);
//         ASSERT_EQ(bfr.valBasedEntropies, nullopt);
//         ASSERT_EQ(bfr.decisionValue, nullopt);
//     }
//     {
//         bfr = dtS->bestFeatureCalculator({"fatIntake=low", "smoking=never"}, 0.10265923626304851);
//         ASSERT_EQ(bfr.bestFeatureName, "videoAddiction");
//         ASSERT_NEAR(bfr.bestFeatureEntropy, 0.005, 0.001);
//         ASSERT_EQ(bfr.valBasedEntropies, nullopt);
//         ASSERT_EQ(bfr.decisionValue, nullopt);
//     }
//     {
//         bfr = dtS->bestFeatureCalculator({"fatIntake=medium"}, 0.21639693245126473);
//         ASSERT_EQ(bfr.bestFeatureName, "videoAddiction");
//         ASSERT_NEAR(bfr.bestFeatureEntropy, 0.065, 0.001);
//         ASSERT_EQ(bfr.valBasedEntropies, nullopt);
//         ASSERT_EQ(bfr.decisionValue, nullopt);
//     }
// }

TEST_F(ConstructTreeTest, bestFeatureCalculatorNumeric)
{
    BestFeatureResult bfr;
    // {
    //     bfr = dtN->bestFeatureCalculator({}, 0.9505668528932196);
    //     ASSERT_EQ(bfr.bestFeatureName, "grade");
    //     ASSERT_NEAR(bfr.bestFeatureEntropy, 0.790, 0.001);
    //     ASSERT_EQ(bfr.valBasedEntropies, nullopt);
    //     ASSERT_EQ(bfr.decisionValue, nullopt);
    // }
    // {
    //     bfr = dtN->bestFeatureCalculator({"grade=2.0"}, 0.6161661934005354);
    //     ASSERT_EQ(bfr.bestFeatureName, "gleason");
    //     ASSERT_NEAR(bfr.bestFeatureEntropy, 0.232, 0.001);
    //     ASSERT_EQ(bfr.valBasedEntropies, nullopt);
    //     ASSERT_EQ(bfr.decisionValue, nullopt);
    // }
    // {
    //     bfr = dtN->bestFeatureCalculator({"grade=2.0", "gleason=4.0"}, 0.5551649772709998);
    //     ASSERT_EQ(bfr.bestFeatureName, "g2");
    //     ASSERT_NEAR(bfr.bestFeatureEntropy, 0.009, 0.001);
    //     ASSERT_NEAR(bfr.valBasedEntropies->first, 0.072, 0.001);
    //     ASSERT_NEAR(bfr.valBasedEntropies->second, 0.537, 0.001);
    //     ASSERT_NEAR(bfr.decisionValue.value(), 3.840, 0.001);
    // }
    {
        bfr = dtN->bestFeatureCalculator({"grade=2.0", "gleason=4.0", "g2<3.84"}, 0.07170446042023888);
        ASSERT_EQ(bfr.bestFeatureName, "g2");
        ASSERT_NEAR(bfr.bestFeatureEntropy, 0.000016, 0.000001);
        ASSERT_NEAR(bfr.valBasedEntropies->first, 0.029, 0.001);
        ASSERT_NEAR(bfr.valBasedEntropies->second, 0.051, 0.001);
        ASSERT_NEAR(bfr.decisionValue.value(), 2.400, 0.001);
    }
    // {
    //     bfr = dtN->bestFeatureCalculator({"grade=2.0", "gleason=4.0", "g2<3.84", "g2<2.4"}, 0.02862448586432183);
    //     ASSERT_EQ(bfr.bestFeatureName, "age");
    //     ASSERT_NEAR(bfr.bestFeatureEntropy, 0.00000334, 0.00000001);
    //     ASSERT_NEAR(bfr.valBasedEntropies->first, 0.00142, 0.00001);
    //     ASSERT_NEAR(bfr.valBasedEntropies->second, 0.0277, 0.0001);
    //     ASSERT_NEAR(bfr.decisionValue.value(), 49.0, 0.001);
    // }
    // {
    //     bfr = dtN->bestFeatureCalculator({"grade=2.0", "gleason=4.0", "g2>3.84", "age<49", "g2>13.44", "g2>17.04"}, 0.015275967002314993);
    //     ASSERT_EQ(bfr.bestFeatureName, "g2");
    //     ASSERT_NEAR(bfr.bestFeatureEntropy, 0.0000003799, 0.00000000001);
    //     ASSERT_NEAR(bfr.valBasedEntropies->first, 0.0097, 0.0001);
    //     ASSERT_NEAR(bfr.valBasedEntropies->second, 0.00482, 0.00001);
    //     ASSERT_NEAR(bfr.decisionValue.value(), 22.56, 0.01);
    // }
}