#include "DecisionTree.hpp"

#include <gtest/gtest.h>

class EntropyCalcTest : public ::testing::Test {
  protected:
    shared_ptr<DecisionTree> dtS; // Symbolic DecisionTree
    shared_ptr<DecisionTree> dtN; // Numeric DecisionTree

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

        dtS = make_shared<DecisionTree>(kwargsS); // Initialize the DecisionTree
        dtS->getTrainingData();
        dtS->calculateFirstOrderProbabilities();
        dtS->calculateClassPriors();

        dtN = make_shared<DecisionTree>(kwargsN); // Initialize the DecisionTree
        dtN->getTrainingData();
        dtN->calculateFirstOrderProbabilities();
        dtN->calculateClassPriors();
    }

    void TearDown() override
    {
        dtS.reset(); // Reset the DecisionTree
    }
};

TEST_F(EntropyCalcTest, CheckdtExists)
{
    ASSERT_NE(&dtS, nullptr);
    ASSERT_NE(&dtN, nullptr);
}

// ------ Symbolic Data Tests ------

TEST_F(EntropyCalcTest, classEntropyOnPriorsSymbolic)
{
    double classEntropy = dtS->classEntropyOnPriors();
    ASSERT_NEAR(classEntropy, 0.958, 0.001);
}


// ------ Numeric Data Tests ------

TEST_F(EntropyCalcTest, classEntropyOnPriorsNumeric)
{
    double classEntropy = dtN->classEntropyOnPriors();
    ASSERT_NEAR(classEntropy, 0.951, 0.001);
}

TEST_F(EntropyCalcTest, entropyScannerForANumericFeatureNumeric) {
    // dtN->entropyScannerForANumericFeature("age");
}

TEST_F(EntropyCalcTest, classEntropyForLessThanThresholdForFeatureNumeric)
{
    double result, expected, threshold;
    vector<string> arrayOfFeaturesAndValuesOrThresholds;
    string feature;
    double Tol = 1e-4;

    {
        // Setup
        arrayOfFeaturesAndValuesOrThresholds = {"grade=2.0", "gleason=5.0"};
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
        arrayOfFeaturesAndValuesOrThresholds = {"grade=2.0", "gleason=5.0", "g2<3.84"};
        feature                              = "age";
        threshold                            = 57.0;
        expected                             = 0.004453027563883287;

        // Tests
        result =
            dtN->classEntropyForLessThanThresholdForFeature(arrayOfFeaturesAndValuesOrThresholds, feature, threshold);

        // Assert
        ASSERT_NEAR(result, expected, Tol);
    }
    {
        // Setup
        arrayOfFeaturesAndValuesOrThresholds = {"grade=2.0", "gleason=5.0", "g2<3.84"};
        feature                              = "g2";
        threshold                            = 3.84;
        expected                             = 0.01747604016929538;

        // Tests
        result =
            dtN->classEntropyForLessThanThresholdForFeature(arrayOfFeaturesAndValuesOrThresholds, feature, threshold);

        // Assert
        ASSERT_NEAR(result, expected, Tol);
    }
    {
        // Setup
        arrayOfFeaturesAndValuesOrThresholds = {"grade=2.0", "gleason=5.0", "g2>3.84"};
        feature                              = "g2";
        threshold                            = 3.84;
        expected                             = 1.0;

        // Tests
        result =
            dtN->classEntropyForLessThanThresholdForFeature(arrayOfFeaturesAndValuesOrThresholds, feature, threshold);

        // Assert
        ASSERT_NEAR(result, expected, Tol);
    }
    {
        // Setup
        arrayOfFeaturesAndValuesOrThresholds = {"grade=2.0", "gleason<5.0", "g2>3.84"};
        feature                              = "g2";
        threshold                            = 46.56;
        expected                             = 0.21940130771341496;

        // Tests
        result =
            dtN->classEntropyForLessThanThresholdForFeature(arrayOfFeaturesAndValuesOrThresholds, feature, threshold);

        // Assert
        ASSERT_NEAR(result, expected, Tol);
    }
    {
        arrayOfFeaturesAndValuesOrThresholds = {"grade=2.0", "gleason=4.0", "g2>3.84", "age<49", "g2>13.44", "g2>17.04"};
        feature                              = "age";
        threshold                            = 47.0;
        expected                             = 0.0314;

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

    {
        // Setup
        arrayOfFeaturesAndValuesOrThresholds = {"grade=2.0", "gleason=5.0"};
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
        arrayOfFeaturesAndValuesOrThresholds = {"grade=2.0", "gleason=5.0", "g2<3.84"};
        feature                              = "age";
        threshold                            = 57.0;
        expected                             = 0.013766658849326036;

        // Tests
        result = dtN->classEntropyForGreaterThanThresholdForFeature(
            arrayOfFeaturesAndValuesOrThresholds, feature, threshold);

        // Assert
        ASSERT_NEAR(result, expected, Tol);
    }
    {
        // Setup
        arrayOfFeaturesAndValuesOrThresholds = {"grade=2.0", "gleason=5.0", "g2<3.84"};
        feature                              = "g2";
        threshold                            = 3.84;
        expected                             = 1.0;

        // Tests
        result = dtN->classEntropyForGreaterThanThresholdForFeature(
            arrayOfFeaturesAndValuesOrThresholds, feature, threshold);

        // Assert
        ASSERT_NEAR(result, expected, Tol);
    }
    {
        // Setup
        arrayOfFeaturesAndValuesOrThresholds = {"grade=2.0", "gleason=5.0", "g2>3.84"};
        feature                              = "g2";
        threshold                            = 3.84;
        expected                             = 0.1713422270378175;

        // Tests
        result = dtN->classEntropyForGreaterThanThresholdForFeature(
            arrayOfFeaturesAndValuesOrThresholds, feature, threshold);

        // Assert
        ASSERT_NEAR(result, expected, Tol);
    }
    {
        // Setup
        arrayOfFeaturesAndValuesOrThresholds = {"grade=2.0", "gleason<5.0", "g2>3.84"};
        feature                              = "g2";
        threshold                            = 46.56;
        expected                             = 0.3396464907040562;

        // Tests
        result = dtN->classEntropyForGreaterThanThresholdForFeature(
            arrayOfFeaturesAndValuesOrThresholds, feature, threshold);

        // Assert
        ASSERT_NEAR(result, expected, Tol);
    }
}

TEST_F(EntropyCalcTest, classEntropyForAGivenSequenceOfFeaturesAndValuesOrThresholdsNumeric)
{
    double result, expected;
    vector<string> arrayOfFeaturesAndValuesOrThresholds;
    double Tol = 1e-4;

    {
        // Setup Test 1
        arrayOfFeaturesAndValuesOrThresholds = {"grade=2.0", "gleason=5.0", "g2<3.84", "age>57.0"};
        expected                             = 0.013766658849326036;

        // Tests
        result =
            dtN->classEntropyForAGivenSequenceOfFeaturesAndValuesOrThresholds(arrayOfFeaturesAndValuesOrThresholds);

        // Assert
        ASSERT_NEAR(result, expected, Tol);
    }
    {
        // Setup Test 2
        arrayOfFeaturesAndValuesOrThresholds = {"grade=2.0"};
        expected                             = 0.6161661934005354;

        // Tests
        result =
            dtN->classEntropyForAGivenSequenceOfFeaturesAndValuesOrThresholds(arrayOfFeaturesAndValuesOrThresholds);

        // Assert

        ASSERT_NEAR(result, expected, Tol);
    }
    {
        // Setup Test 3
        arrayOfFeaturesAndValuesOrThresholds = {"grade=2.0", "gleason=5.0", "g2>3.84"};
        expected                             = 0.1713422270378175;

        // Tests
        result =
            dtN->classEntropyForAGivenSequenceOfFeaturesAndValuesOrThresholds(arrayOfFeaturesAndValuesOrThresholds);

        // Assert
        ASSERT_NEAR(result, expected, Tol);
    }
    {
        // Setup Test 4
        arrayOfFeaturesAndValuesOrThresholds = {"grade=2.0", "gleason=5.0", "g2>3.84", "age>47.0"};
        expected                             = 0.1689155664534898;

        // Tests
        result =
            dtN->classEntropyForAGivenSequenceOfFeaturesAndValuesOrThresholds(arrayOfFeaturesAndValuesOrThresholds);

        // Assert
        ASSERT_NEAR(result, expected, Tol);
    }
    {
        // Setup Test 5
        arrayOfFeaturesAndValuesOrThresholds = {"grade=2.0", "gleason>2", "g2>3.84", "age<49.0"};
        expected                             = 0.0546437614213809;

        // Tests
        result =
            dtN->classEntropyForAGivenSequenceOfFeaturesAndValuesOrThresholds(arrayOfFeaturesAndValuesOrThresholds);

        // Assert
        ASSERT_NEAR(result, expected, Tol);
    }
    {
        // Setup Test 6
        arrayOfFeaturesAndValuesOrThresholds = {"grade=2.0", "gleason=5.0", "g2>25.0", "age=65", "g2<28.0"};
        expected                             = 0.9840902467406283;

        // Tests
        result =
            dtN->classEntropyForAGivenSequenceOfFeaturesAndValuesOrThresholds(arrayOfFeaturesAndValuesOrThresholds);

        // Assert
        ASSERT_NEAR(result, expected, Tol);
    }
}