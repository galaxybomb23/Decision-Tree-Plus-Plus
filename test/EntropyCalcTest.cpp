#include "DecisionTree.hpp"

#include <gtest/gtest.h>

class EntropyCalcTest : public ::testing::Test {
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

        dtN = std::make_unique<DecisionTree>(kwargsN); // Initialize the DecisionTree
        dtN->getTrainingData();
    }

    void TearDown() override
    {
        dtS.reset(); // Reset the DecisionTree
    }
};

TEST_F(EntropyCalcTest, CheckdtExists)
{
    ASSERT_NE(&dtS, nullptr);
}

// TODO: Write these Tests
// ------ Symbolic Data Tests ------

TEST_F(EntropyCalcTest, classEntropyOnPriorsSymbolic)
{
    double classEntropy = dtS->classEntropyOnPriors();
    ASSERT_NEAR(classEntropy, 0.958, 0.001);
}

TEST_F(EntropyCalcTest, entropyScannerForANumericFeatureSymbolic) {}

TEST_F(EntropyCalcTest, classEntropyForLessThanThresholdForFeatureSymbolic) {}

TEST_F(EntropyCalcTest, classEntropyForGreaterThanThresholdForFeatureSymbolic) {}

TEST_F(EntropyCalcTest, classEntropyForAGivenSequenceOfFeaturesAndValuesOrThresholdsSymbolic) {}

// ------ Numeric Data Tests ------

TEST_F(EntropyCalcTest, classEntropyOnPriorsNumeric)
{
    double classEntropy = dtN->classEntropyOnPriors();
    ASSERT_NEAR(classEntropy, 0.951, 0.001);
}

TEST_F(EntropyCalcTest, entropyScannerForANumericFeatureNumeric) {}

TEST_F(EntropyCalcTest, classEntropyForLessThanThresholdForFeatureNumeric)
{
    double result, expected, threshold;
    vector<string> arrayOfFeaturesAndValuesOrThresholds;
    string feature;
    double Tol = 1e-4;
    dtN->calculateFirstOrderProbabilities();
    dtN->calculateClassPriors();

    {
        // Setup
        arrayOfFeaturesAndValuesOrThresholds = { "grade=2.0", "gleason=5.0" };
        feature                              = "g2";
        threshold                            = 46.56000000000365;
        expected                             = 0.17828975177544815;

        // Tests
        result =
            dtN->classEntropyForLessThanThresholdForFeature(arrayOfFeaturesAndValuesOrThresholds, feature, threshold);

        // Assert
        ASSERT_NEAR(result, expected, Tol);
    }
    {
        // Setup
        arrayOfFeaturesAndValuesOrThresholds = { "grade=2.0", "gleason=5.0", "g2<3.84" };
        feature                              = "age";
        threshold                            = 57.0;
        expected                             = 0.02443053983169013;

        // Tests
        result =
            dtN->classEntropyForLessThanThresholdForFeature(arrayOfFeaturesAndValuesOrThresholds, feature, threshold);

        // Assert
        ASSERT_NEAR(result, expected, Tol);
    }
    {
        // Setup
        arrayOfFeaturesAndValuesOrThresholds = { "grade=2.0", "gleason=5.0", "g2<3.84" };
        feature                              = "g2";
        threshold                            = 3.84;
        expected                             = 0.01747604016929538;

        // Tests
        result =
            dtN->classEntropyForLessThanThresholdForFeature(arrayOfFeaturesAndValuesOrThresholds, feature, threshold);

        // Assert
        ASSERT_NEAR(result, expected, Tol);
    }
}

TEST_F(EntropyCalcTest, classEntropyForGreaterThanThresholdForFeatureNumeric)
{
    double result, expected, threshold;
    vector<string> arrayOfFeaturesAndValuesOrThresholds;
    string feature;
    double Tol = 1e-4;
    dtN->calculateFirstOrderProbabilities();
    dtN->calculateClassPriors();
    {
        // Setup
        arrayOfFeaturesAndValuesOrThresholds = { "grade=2.0", "gleason=5.0" };
        feature                              = "g2";
        threshold                            = 46.56000;
        expected                             = 0.26824793612598297;

        // Tests
        result = dtN->classEntropyForGreaterThanThresholdForFeature(
            arrayOfFeaturesAndValuesOrThresholds, feature, threshold);

        // Assert
        ASSERT_NEAR(result, expected, Tol);
    }
    {
        // Setup
        arrayOfFeaturesAndValuesOrThresholds = { "grade=2.0", "gleason=5.0", "g2<3.84" };
        feature                              = "age";
        threshold                            = 57.0;
        expected                             = 0.01606423071045408;

        // Tests
        result = dtN->classEntropyForGreaterThanThresholdForFeature(
            arrayOfFeaturesAndValuesOrThresholds, feature, threshold);

        // Assert
        ASSERT_NEAR(result, expected, Tol);
    }
    {
        // Setup
        arrayOfFeaturesAndValuesOrThresholds = { "grade=2.0", "gleason=5.0", "g2<3.84" };
        feature                              = "g2";
        threshold                            = 3.84;
        expected                             = 1.0;

        // Tests
        result = dtN->classEntropyForGreaterThanThresholdForFeature(
            arrayOfFeaturesAndValuesOrThresholds, feature, threshold);

        // Assert
        ASSERT_NEAR(result, expected, Tol);
    }
}

TEST_F(EntropyCalcTest, classEntropyForAGivenSequenceOfFeaturesAndValuesOrThresholdsNumeric) {}