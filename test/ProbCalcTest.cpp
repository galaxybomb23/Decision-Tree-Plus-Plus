#include <gtest/gtest.h>

#include "DecisionTree.hpp"

class ProbCalcTest : public ::testing::Test
{
    protected:
    std::unique_ptr<DecisionTree> dtS; // Symbolic DecisionTree
    std::unique_ptr<DecisionTree> dtN; // Numeric DecisionTree
    void SetUp() override
    {
        map<string,string> kwargsS = { // Symbolic kwargs
            {"training_datafile", "../test/resources/training_symbolic.csv"},
            {"csv_class_column_index", "1"},
            {"csv_columns_for_features", {2,3,4,5}},
            {"max_depth_desired", "5"},
            {"entropy_threshold", "0.1"}
        };

        map<string,string> kwargsN = { // Numeric kwargs
            {"training_datafile", "../test/resources/stage3cancer.csv"},
            {"csv_class_column_index", "2"},
            {"csv_columns_for_features", {3,4,5,6,7,8}},
            {"max_depth_desired", "8"},
            {"entropy_threshold", "0.01"}
        };
        
        dtS = std::make_unique<DecisionTree>(kwargsS);  // Initialize the DecisionTree
        dtS->getTrainingData();

        dtN = std::make_unique<DecisionTree>(kwargsN);  // Initialize the DecisionTree
        dtN->getTrainingData();
    }

    void TearDown() override
    {
        dtS.reset();  // Reset the DecisionTree
        
    }

};

TEST_F(ProbCalcTest, CheckdtExists) { ASSERT_NE(&dtS, nullptr); }

// ------ Symbolic Data Tests ------

TEST_F(ProbCalcTest, priorProbabilityForClassSymbolic) {
    double prob = dtS->priorProbabilityForClass("benign");
    ASSERT_EQ(prob, 0.62);

    prob = dtS->priorProbabilityForClass("malignant");
    ASSERT_EQ(prob, 0.38);
}

TEST_F(ProbCalcTest, calculateClassPriorsSymbolic) {
    dtS->calculateClassPriors();
    vector<float> expected = {0.62, 0.38};
    vector<float> priors;
    for (int i = 0; i < dtS->_classNames.size(); i++) {
        priors.push_back(dtS->priorProbabilityForClass(dtS->_classNames[i]));
        ASSERT_EQ(expected[i], priors[i]);
    }
}

TEST_F(ProbCalcTest, probabilityOfFeatureValueSymbolic) {
    double prob0 = dtS->probabilityOfFeatureValue("smoking", "never");
    ASSERT_EQ(prob0, 0.16);
    double prob1 = dtS->probabilityOfFeatureValue("smoking", "light");
    ASSERT_EQ(prob1, 0.23);
    double prob2 = dtS->probabilityOfFeatureValue("smoking", "medium");
    ASSERT_EQ(prob2, 0.17);
    double prob3 = dtS->probabilityOfFeatureValue("smoking", "heavy");
    ASSERT_EQ(prob3, 0.44);
}

TEST_F(ProbCalcTest, probabilityOfFeatureValueGivenClassSymbolic) {
    double prob0 = dtS->probabilityOfFeatureValueGivenClass("smoking", "never", "benign");
    ASSERT_NEAR(prob0, 0.242, 0.001);
    double prob1 = dtS->probabilityOfFeatureValueGivenClass("smoking", "light", "benign");
    ASSERT_NEAR(prob1, 0.339, 0.001);
    double prob2 = dtS->probabilityOfFeatureValueGivenClass("smoking", "medium", "benign");
    ASSERT_NEAR(prob2, 0.258, 0.001);
    double prob3 = dtS->probabilityOfFeatureValueGivenClass("smoking", "heavy", "benign");
    ASSERT_NEAR(prob3, 0.161, 0.001);

    double prob4 = dtS->probabilityOfFeatureValueGivenClass("smoking", "never", "malignant");
    ASSERT_NEAR(prob4, 0.026, 0.001);
    double prob5 = dtS->probabilityOfFeatureValueGivenClass("smoking", "light", "malignant");
    ASSERT_NEAR(prob5, 0.053, 0.001);
    double prob6 = dtS->probabilityOfFeatureValueGivenClass("smoking", "medium", "malignant");
    ASSERT_NEAR(prob6, 0.026, 0.001);
    double prob7 = dtS->probabilityOfFeatureValueGivenClass("smoking", "heavy", "malignant");
    ASSERT_NEAR(prob7, 0.895, 0.001);
}

TEST_F(ProbCalcTest, probabilityOfASequenceOfFeaturesAndValuesOrThresholdsSymbolic) {
    double prob0 = dtS->probabilityOfASequenceOfFeaturesAndValuesOrThresholds({ "exercising=never" });
    ASSERT_NEAR(prob0, 0.43, 0.001);
    double prob1 = dtS->probabilityOfASequenceOfFeaturesAndValuesOrThresholds({ "fatIntake=heavy" });
    ASSERT_NEAR(prob1, 0.44, 0.001);
    double prob2 = dtS->probabilityOfASequenceOfFeaturesAndValuesOrThresholds({ "fatIntake=low", "smoking=heavy" });
    ASSERT_NEAR(prob2, 0.119, 0.001);
    double prob3 = dtS->probabilityOfASequenceOfFeaturesAndValuesOrThresholds({ "fatIntake=low", "smoking=never", "exercising=regularly" });
    ASSERT_NEAR(prob3, 0.011, 0.001);
    double prob4 = dtS->probabilityOfASequenceOfFeaturesAndValuesOrThresholds({ "fatIntake=medium", "exercising=occasionally" });
    ASSERT_NEAR(prob4, 0.089, 0.001);
}

// ------ Numeric Data Tests ------

// TODO: Add tests for numeric data for the following functions:
// priorProbabilityForClass
// calculateClassPriors
// probabilityOfFeatureValue
// probabilityOfFeatureValueGivenClass
// probabilityOfASequenceOfFeaturesAndValuesOrThresholds

TEST_F(ProbCalcTest, priorProbabilityForClassNumeric) {
    
}

TEST_F(ProbCalcTest, calculateClassPriorsNumeric) {
    
}

// TODO: Write more asserts
TEST_F(ProbCalcTest, probabilityOfFeatureValueNumeric) {
    double prob0 = dtN->probabilityOfFeatureValue("grade", "2.0");
    ASSERT_NEAR(prob0, 0.404, 0.001);
}

TEST_F(ProbCalcTest, probabilityOfFeatureValueGivenClassNumeric) {
    
}

TEST_F(ProbCalcTest, probabilityOfFeatureLessThanThresholdNumeric) {
    double prob0 = dtN->probabilityOfFeatureLessThanThreshold("age", "47");
    ASSERT_NEAR(prob0, 0.007, 0.001);
    double prob1 = dtN->probabilityOfFeatureLessThanThreshold("age", "50");
    ASSERT_NEAR(prob1, 0.021, 0.001);
    double prob2 = dtN->probabilityOfFeatureLessThanThreshold("age", "100");
    ASSERT_NEAR(prob2, 1.0, 0.001);
}

TEST_F(ProbCalcTest, probabilityOfFeatureLessThanThresholdGivenClassNumeric) {
    double prob0 = dtN->probabilityOfFeatureLessThanThresholdGivenClass("age", "47", "1");
    ASSERT_NEAR(prob0, 0.019, 0.001);
    double prob1 = dtN->probabilityOfFeatureLessThanThresholdGivenClass("age", "90", "1");
    ASSERT_NEAR(prob1, 1.0, 0.001);
    double prob2 = dtN->probabilityOfFeatureLessThanThresholdGivenClass("age", "68", "1");
    ASSERT_NEAR(prob2, 0.852, 0.001);
    double prob3 = dtN->probabilityOfFeatureLessThanThresholdGivenClass("age", "73", "1");
    ASSERT_NEAR(prob3, 0.981, 0.001);
    double prob4 = dtN->probabilityOfFeatureLessThanThresholdGivenClass("eet", "2", "1");
    ASSERT_NEAR(prob4, 1.0, 0.001);
    double prob5 = dtN->probabilityOfFeatureLessThanThresholdGivenClass("g2", "14.5", "1");
    ASSERT_NEAR(prob5, 0.509, 0.001);
    double prob6 = dtN->probabilityOfFeatureLessThanThresholdGivenClass("grade", "3", "1");
    ASSERT_NEAR(prob6, 0.888, 0.001);
    double prob7 = dtN->probabilityOfFeatureLessThanThresholdGivenClass("gleason", "6", "1");
    ASSERT_NEAR(prob7, 0.333, 0.001);
}

TEST_F(ProbCalcTest, probabilityOfASequenceOfFeaturesAndValuesOrThresholdsNumeric) {
    double prob0 = dtN->probabilityOfASequenceOfFeaturesAndValuesOrThresholds({ "age<47.0" });
    ASSERT_NEAR(prob0, 0.007, 0.001);
    double prob1 = dtN->probabilityOfASequenceOfFeaturesAndValuesOrThresholds({ "g2<8.640000000000052" });
    ASSERT_NEAR(prob1, 0.230, 0.001);
    double prob2 = dtN->probabilityOfASequenceOfFeaturesAndValuesOrThresholds({ "grade=2.0", "g2>49.20000000000039" });
    ASSERT_NEAR(prob2, 0.003, 0.001);
    double prob3 = dtN->probabilityOfASequenceOfFeaturesAndValuesOrThresholds({ "grade=2.0", "gleason=4.0", "g2>49.20000000000039" });
    ASSERT_NEAR(prob3, 0.000122, 0.000001);
    double prob4 = dtN->probabilityOfASequenceOfFeaturesAndValuesOrThresholds({ "grade=2.0", "gleason=5.0", "g2<3.840000000000012", "ploidy=aneuploid" });
    ASSERT_NEAR(prob4, 0.000161, 0.000001);
    double prob5 = dtN->probabilityOfASequenceOfFeaturesAndValuesOrThresholds({ "grade=2.0", "gleason=5.0", "g2<3.840000000000012", "ploidy=tetraploid" });
    ASSERT_NEAR(prob5, 0.000994, 0.000001);
}