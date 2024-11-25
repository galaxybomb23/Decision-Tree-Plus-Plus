#include "DecisionTree.hpp"

#include <gtest/gtest.h>

class ProbCalcTest : public ::testing::Test {
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
            {       "entropy_threshold",                               "0.01"},
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

TEST_F(ProbCalcTest, CheckdtExists)
{
    ASSERT_NE(&dtS, nullptr);
    ASSERT_NE(&dtN, nullptr);
}

// ------ Symbolic Data Tests ------

TEST_F(ProbCalcTest, priorProbabilityForClassSymbolic)
{
    double prob = dtS->priorProbabilityForClass("benign");
    ASSERT_EQ(prob, 0.62);

    prob = dtS->priorProbabilityForClass("malignant");
    ASSERT_EQ(prob, 0.38);
}

TEST_F(ProbCalcTest, calculateClassPriorsSymbolic)
{
    dtS->calculateClassPriors();
    vector<float> expected = {0.62, 0.38};
    vector<float> priors;
    for (int i = 0; i < dtS->_classNames.size(); i++) {
        priors.push_back(dtS->priorProbabilityForClass(dtS->_classNames[i]));
        ASSERT_EQ(expected[i], priors[i]);
    }
}

TEST_F(ProbCalcTest, probabilityOfFeatureValueSymbolic)
{
    double prob0 = dtS->probabilityOfFeatureValue("smoking", "never");
    ASSERT_EQ(prob0, 0.16);
    double prob1 = dtS->probabilityOfFeatureValue("smoking", "light");
    ASSERT_EQ(prob1, 0.23);
    double prob2 = dtS->probabilityOfFeatureValue("smoking", "medium");
    ASSERT_EQ(prob2, 0.17);
    double prob3 = dtS->probabilityOfFeatureValue("smoking", "heavy");
    ASSERT_EQ(prob3, 0.44);
}

TEST_F(ProbCalcTest, probabilityOfFeatureValueGivenClassSymbolic)
{
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

TEST_F(ProbCalcTest, probabilityOfASequenceOfFeaturesAndValuesOrThresholdsSymbolic)
{
    double prob0 = dtS->probabilityOfASequenceOfFeaturesAndValuesOrThresholds({"exercising=never"});
    ASSERT_NEAR(prob0, 0.43, 0.001);
    double prob1 = dtS->probabilityOfASequenceOfFeaturesAndValuesOrThresholds({"fatIntake=heavy"});
    ASSERT_NEAR(prob1, 0.44, 0.001);
    double prob2 = dtS->probabilityOfASequenceOfFeaturesAndValuesOrThresholds({"fatIntake=low", "smoking=heavy"});
    ASSERT_NEAR(prob2, 0.119, 0.001);
    double prob3 = dtS->probabilityOfASequenceOfFeaturesAndValuesOrThresholds(
        {"fatIntake=low", "smoking=never", "exercising=regularly"});
    ASSERT_NEAR(prob3, 0.011, 0.001);
    double prob4 =
        dtS->probabilityOfASequenceOfFeaturesAndValuesOrThresholds({"fatIntake=medium", "exercising=occasionally"});
    ASSERT_NEAR(prob4, 0.089, 0.001);
}

TEST_F(ProbCalcTest, probabilityOfASequenceOfFeaturesAndValuesOrThresholdsGivenClassSymbolic)
{
    double prob0 = dtS->probabilityOfASequenceOfFeaturesAndValuesOrThresholdsGivenClass({"exercising=never"}, "benign");
    ASSERT_NEAR(prob0, 0.161, 0.001);
    double prob1 = dtS->probabilityOfASequenceOfFeaturesAndValuesOrThresholdsGivenClass({"smoking=heavy"}, "malignant");
    ASSERT_NEAR(prob1, 0.895, 0.001);
    double prob2 = dtS->probabilityOfASequenceOfFeaturesAndValuesOrThresholdsGivenClass(
        {"fatIntake=heavy", "exercising=never"}, "benign");
    ASSERT_NEAR(prob2, 0.026, 0.001);
    double prob3 = dtS->probabilityOfASequenceOfFeaturesAndValuesOrThresholdsGivenClass(
        {"fatIntake=heavy", "videoAddiction=medium"}, "malignant");
    ASSERT_NEAR(prob3, 0.235, 0.001);
    double prob4 = dtS->probabilityOfASequenceOfFeaturesAndValuesOrThresholdsGivenClass(
        {"fatIntake=heavy", "smoking=heavy", "videoAddiction=none"}, "benign");
    ASSERT_NEAR(prob4, 0.007, 0.001);
    double prob5 = dtS->probabilityOfASequenceOfFeaturesAndValuesOrThresholdsGivenClass(
        {"fatIntake=heavy", "smoking=heavy", "videoAddiction=heavy", "exercising=regularly"}, "benign");
    ASSERT_NEAR(prob5, 0.001, 0.001);
}


// ------ Numeric Data Tests ------

TEST_F(ProbCalcTest, priorProbabilityForClassNumeric)
{
    double prob0 = dtN->priorProbabilityForClass("1");
    ASSERT_NEAR(prob0, 0.369, 0.001);
    double prob1 = dtN->priorProbabilityForClass("0");
    ASSERT_NEAR(prob1, 0.630, 0.001);
}

TEST_F(ProbCalcTest, calculateClassPriorsNumeric)
{
    // dtN->calculateClassPriors();
    vector<float> expected = {0.630, 0.369};
    vector<float> priors;
    for (int i = 0; i < dtN->_classNames.size(); i++) {
        priors.push_back(dtN->priorProbabilityForClass(dtN->_classNames[i]));
        ASSERT_NEAR(expected[i], priors[i], 0.001);
    }
}

TEST_F(ProbCalcTest, probabilityOfFeatureValueNumeric)
{
    double prob0 = dtN->probabilityOfFeatureValue("grade", "2.0");
    ASSERT_NEAR(prob0, 0.404, 0.001);
    double prob1 = dtN->probabilityOfFeatureValue("grade", "2");
    ASSERT_NEAR(prob1, 0.404, 0.001);
    double prob2 = dtN->probabilityOfFeatureValue("grade", "3.0");
    ASSERT_NEAR(prob2, 0.541, 0.001);
    double prob3 = dtN->probabilityOfFeatureValue("gleason", "8.0");
    ASSERT_NEAR(prob3, 0.147, 0.001);
    double prob4 = dtN->probabilityOfFeatureValue("ploidy", "tetraploid");
    ASSERT_NEAR(prob4, 0.466, 0.001);
    double prob5 = dtN->probabilityOfFeatureValue("age", "64");
    ASSERT_NEAR(prob5, 0.151, 0.001);
}

TEST_F(ProbCalcTest, probabilityOfFeatureLessThanThresholdNumeric)
{
    double prob0 = dtN->probabilityOfFeatureLessThanThreshold("age", "47");
    ASSERT_NEAR(prob0, 0.00684, 0.0001);
    double prob1 = dtN->probabilityOfFeatureLessThanThreshold("age", "50");
    ASSERT_NEAR(prob1, 0.0205, 0.0001);
    double prob2 = dtN->probabilityOfFeatureLessThanThreshold("age", "100");
    ASSERT_NEAR(prob2, 1.0, 0.0001);
}

TEST_F(ProbCalcTest, probabilityOfFeatureValueGivenClassNumeric)
{
    double prob1 = dtN->probabilityOfFeatureValueGivenClass("grade", "2.0", "1");
    ASSERT_NEAR(prob1, 0.16666666666666666, 0.001);

    double prob2 = dtN->probabilityOfFeatureValueGivenClass("grade", "3.0", "0");
    ASSERT_NEAR(prob2, 0.43478260869565216, 0.001);

    double prob3 = dtN->probabilityOfFeatureValueGivenClass("grade", "4.0", "1");
    ASSERT_NEAR(prob3, 0.1111111111111111, 0.001);

    double prob4 = dtN->probabilityOfFeatureValueGivenClass("grade", "2.0", "1");
    ASSERT_NEAR(prob4, 0.16666666666666666, 0.001);

    double prob5 = dtN->probabilityOfFeatureValueGivenClass("grade", "3.0", "1");
    ASSERT_NEAR(prob5, 0.7222222222222222, 0.001);

    double prob6 = dtN->probabilityOfFeatureValueGivenClass("gleason", "7", "1");
    ASSERT_NEAR(prob6, 0.3333333333333333, 0.001);

    double prob7 = dtN->probabilityOfFeatureValueGivenClass("ploidy", "diploid", "1");
    ASSERT_NEAR(prob7, 0.24074074074074073, 0.001);

    double prob8 = dtN->probabilityOfFeatureValueGivenClass("ploidy", "tetraploid", "1");
    ASSERT_NEAR(prob8, 0.6296296296296297, 0.001);

    double prob9 = dtN->probabilityOfFeatureValueGivenClass("eet", "2", "1");
    ASSERT_NEAR(prob9, 0.7307692307692307, 0.001);

    double prob10 = dtN->probabilityOfFeatureValueGivenClass("gleason", "8", "0");
    ASSERT_NEAR(prob10, 0.10112359550561797, 0.001);

    double prob11 = dtN->probabilityOfFeatureValueGivenClass("ploidy", "aneuploid", "0");
    ASSERT_NEAR(prob11, 0.043478260869565216, 0.001);

    double prob12 = dtN->probabilityOfFeatureValueGivenClass("eet", "1", "0");
    ASSERT_NEAR(prob12, 0.2391304347826087, 0.001);

    double prob13 = dtN->probabilityOfFeatureValueGivenClass("ploidy", "aneuploid", "0");
    ASSERT_NEAR(prob13, 0.043478260869565216, 0.001);

    double prob14 = dtN->probabilityOfFeatureValueGivenClass("grade", "2.0", "1");
    ASSERT_NEAR(prob14, 0.16666666666666666, 0.001);

    double prob15 = dtN->probabilityOfFeatureValueGivenClass("grade", "3.0", "0");
    ASSERT_NEAR(prob15, 0.43478260869565216, 0.001);

    double prob16 = dtN->probabilityOfFeatureValueGivenClass("grade", "4.0", "1");
    ASSERT_NEAR(prob16, 0.1111111111111111, 0.001);

    double prob17 = dtN->probabilityOfFeatureValueGivenClass("grade", "2.0", "1");
    ASSERT_NEAR(prob17, 0.16666666666666666, 0.001);

    double prob18 = dtN->probabilityOfFeatureValueGivenClass("grade", "3.0", "1");
    ASSERT_NEAR(prob18, 0.7222222222222222, 0.001);

    double prob19 = dtN->probabilityOfFeatureValueGivenClass("gleason", "7", "1");
    ASSERT_NEAR(prob19, 0.3333333333333333, 0.001);

    double prob20 = dtN->probabilityOfFeatureValueGivenClass("ploidy", "diploid", "1");
    ASSERT_NEAR(prob20, 0.24074074074074073, 0.001);

    double prob21 = dtN->probabilityOfFeatureValueGivenClass("ploidy", "tetraploid", "1");
    ASSERT_NEAR(prob21, 0.6296296296296297, 0.001);

    double prob22 = dtN->probabilityOfFeatureValueGivenClass("eet", "2", "1");
    ASSERT_NEAR(prob22, 0.7307692307692307, 0.001);

    double prob23 = dtN->probabilityOfFeatureValueGivenClass("gleason", "8", "0");
    ASSERT_NEAR(prob23, 0.10112359550561797, 0.001);

    double prob24 = dtN->probabilityOfFeatureValueGivenClass("ploidy", "aneuploid", "0");
    ASSERT_NEAR(prob24, 0.043478260869565216, 0.001);

    double prob25 = dtN->probabilityOfFeatureValueGivenClass("eet", "1", "0");
    ASSERT_NEAR(prob25, 0.2391304347826087, 0.001);

    double prob26 = dtN->probabilityOfFeatureValueGivenClass("ploidy", "aneuploid", "0");
    ASSERT_NEAR(prob26, 0.043478260869565216, 0.001);

    double prob27 = dtN->probabilityOfFeatureValueGivenClass("age", "62", "0");
    ASSERT_NEAR(prob27, 0.1, 0.001);
}

TEST_F(ProbCalcTest, probabilityOfFeatureLessThanThresholdGivenClassNumeric)
{
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

TEST_F(ProbCalcTest, probabilityOfASequenceOfFeaturesAndValuesOrThresholdsNumeric)
{
    double prob0 = dtN->probabilityOfASequenceOfFeaturesAndValuesOrThresholds({"age<47.0"});
    ASSERT_NEAR(prob0, 0.007, 0.001);
    double prob1 = dtN->probabilityOfASequenceOfFeaturesAndValuesOrThresholds({"g2<8.640000000000052"});
    ASSERT_NEAR(prob1, 0.230, 0.001);
    double prob2 = dtN->probabilityOfASequenceOfFeaturesAndValuesOrThresholds({"grade=2.0", "g2>49.20000000000039"});
    ASSERT_NEAR(prob2, 0.003, 0.001);
    double prob3 = dtN->probabilityOfASequenceOfFeaturesAndValuesOrThresholds(
        {"grade=2.0", "gleason=4.0", "g2>49.20000000000039"});
    ASSERT_NEAR(prob3, 0.000122, 0.000001);
    double prob4 = dtN->probabilityOfASequenceOfFeaturesAndValuesOrThresholds(
        {"grade=2.0", "gleason=5.0", "g2<3.840000000000012", "ploidy=aneuploid"});
    ASSERT_NEAR(prob4, 0.000161, 0.000001);
    double prob5 = dtN->probabilityOfASequenceOfFeaturesAndValuesOrThresholds(
        {"grade=2.0", "gleason=5.0", "g2<3.840000000000012", "ploidy=tetraploid"});
    ASSERT_NEAR(prob5, 0.000994, 0.000001);

    double prob6 = dtN->probabilityOfASequenceOfFeaturesAndValuesOrThresholds(
        {"grade=2.0", "gleason=4.0", "g2>3.84000", "age<49.0", "g2>13.44000", "g2>17.04000", "g2>49.20000"});
    ASSERT_NEAR(prob6, 1.671e-6, 0.01e-6);
}

TEST_F(ProbCalcTest, probabilityOfASequenceOfFeaturesAndValuesOrThresholdsGivenClassNumeric)
{
    double prob0 = dtN->probabilityOfASequenceOfFeaturesAndValuesOrThresholdsGivenClass({"age<47.0"}, "1");
    ASSERT_NEAR(prob0, 0.019, 0.001);
    double prob1 = dtN->probabilityOfASequenceOfFeaturesAndValuesOrThresholdsGivenClass(
        {"grade=2.0", "gleason=5.0", "g2<3.840000000000012", "age>51.0"}, "0");
    ASSERT_NEAR(prob1, 0.195, 0.001);
    double prob2 = dtN->probabilityOfASequenceOfFeaturesAndValuesOrThresholdsGivenClass(
        {"grade=2.0", "gleason=5.0", "g2<3.840000000000012", "ploidy=aneuploid"}, "0");
    ASSERT_NEAR(prob2, 0.008, 0.001);
    double prob3 = dtN->probabilityOfASequenceOfFeaturesAndValuesOrThresholdsGivenClass(
        {"grade=2.0", "g2>42.00000000000033"}, "0");
    ASSERT_NEAR(prob3, 0.006, 0.001);
    double prob4 = dtN->probabilityOfASequenceOfFeaturesAndValuesOrThresholdsGivenClass(
        {"grade=2.0", "gleason=5.0", "g2>25", "age=62", "g2<28.0"}, "1");
    ASSERT_NEAR(prob4, 0.001016, 0.000001);
}

TEST_F(ProbCalcTest, probabilityOfAClassGivenSequenceOfFeaturesAndValuesOrThresholds)
{
    double prob0 = dtN->probabilityOfAClassGivenSequenceOfFeaturesAndValuesOrThresholds("0", {"age>47.0"});
    ASSERT_NEAR(prob0, 0.634, 0.001);
    double prob1 = dtN->probabilityOfAClassGivenSequenceOfFeaturesAndValuesOrThresholds("1", {"age>47.0"});
    ASSERT_NEAR(prob1, 0.366, 0.001);
    double prob2 =
        dtN->probabilityOfAClassGivenSequenceOfFeaturesAndValuesOrThresholds("0", {"age>47.0", "gleason=4.0 "});
    ASSERT_NEAR(prob2, 0.678, 0.001);
    double prob3 = dtN->probabilityOfAClassGivenSequenceOfFeaturesAndValuesOrThresholds(
        "1", {"grade=2.0", "gleason=4.0", "g2>3.84000", "age<49.0", "g2>13.44000", "g2>17.04000", "g2>49.20000"});
    ASSERT_NEAR(prob3, 0.928, 0.001);
    double prob4 = dtN->probabilityOfAClassGivenSequenceOfFeaturesAndValuesOrThresholds("0", {"grade=2.0", "g2<36.96"});
    ASSERT_NEAR(prob4, 0.848, 0.001);
    double prob5 = dtN->probabilityOfAClassGivenSequenceOfFeaturesAndValuesOrThresholds(
        "0", {"grade=2.0", "gleason=5.0", "g2<46.56"});
    ASSERT_NEAR(prob5, 0.973, 0.001);
    double prob6 = dtN->probabilityOfAClassGivenSequenceOfFeaturesAndValuesOrThresholds(
        "1", {"grade=2.0", "gleason=5.0", "g2<46.56"});
    ASSERT_NEAR(prob6, 0.0268, 0.001);
    double prob7 = dtN->probabilityOfAClassGivenSequenceOfFeaturesAndValuesOrThresholds(
        "1", {"grade=2.0", "gleason=5.0", "g2>25", "age=65", "g2<28.0"});
    ASSERT_NEAR(prob7, 0.425881, 0.000001);
}