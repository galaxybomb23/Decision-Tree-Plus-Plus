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
            {"csv_columns_for_features", "{2,3,4,5}"},
            {"max_depth_desired", "5"},
            {"entropy_threshold", "0.1"}
        };

        map<string,string> kwargsN = { // Numeric kwargs
            {"training_datafile", "../test/resources/stage3cancer.csv"},
            {"csv_class_column_index", "2"},
            {"csv_columns_for_features", "{3,4,5,6,7,8}"},
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
// Pyhton setup:
// dt = DecisionTree( training_datafile = r"penis",  
//                         csv_class_column_index = 1,
//                         csv_columns_for_features = [2,3,4,5],
//                         max_depth_desired = 5,
//                         entropy_threshold = 0.1,
//                      )

TEST_F(ProbCalcTest, CheckdtExists) { ASSERT_NE(&dtS, nullptr); }

TEST_F(ProbCalcTest, priorProbabilityForClass) {
    double prob = dtS->priorProbabilityForClass("benign");
    ASSERT_EQ(prob, 0.62);

    prob = dtS->priorProbabilityForClass("malignant");
    ASSERT_EQ(prob, 0.38);
}

TEST_F(ProbCalcTest, calculateClassPriors) {
    dtS->calculateClassPriors();
    vector<float> expected = {0.62, 0.38};
    vector<float> priors;
    for (int i = 0; i < dtS->_classNames.size(); i++) {
        priors.push_back(dtS->priorProbabilityForClass(dtS->_classNames[i]));
        ASSERT_EQ(expected[i], priors[i]);
    }
}

TEST_F(ProbCalcTest, probabilityOfFeatureValue) {
    auto prob0 = dtS->probabilityOfFeatureValue("smoking", "never");
    ASSERT_EQ(prob0, 0.16);
    auto prob1 = dtS->probabilityOfFeatureValue("smoking", "light");
    ASSERT_EQ(prob1, 0.23);
    auto prob2 = dtS->probabilityOfFeatureValue("smoking", "medium");
    ASSERT_EQ(prob2, 0.17);
    auto prob3 = dtS->probabilityOfFeatureValue("smoking", "heavy");
    ASSERT_EQ(prob3, 0.44);
}

TEST_F(ProbCalcTest, probabilityOfFeatureValueGivenClass) {
    auto prob0 = dtS->probabilityOfFeatureValueGivenClass("smoking", "never", "benign");
    ASSERT_NEAR(prob0, 0.242, 0.001);
    auto prob1 = dtS->probabilityOfFeatureValueGivenClass("smoking", "light", "benign");
    ASSERT_NEAR(prob1, 0.339, 0.001);
    auto prob2 = dtS->probabilityOfFeatureValueGivenClass("smoking", "medium", "benign");
    ASSERT_NEAR(prob2, 0.258, 0.001);
    auto prob3 = dtS->probabilityOfFeatureValueGivenClass("smoking", "heavy", "benign");
    ASSERT_NEAR(prob3, 0.161, 0.001);

    auto prob4 = dtS->probabilityOfFeatureValueGivenClass("smoking", "never", "malignant");
    ASSERT_NEAR(prob4, 0.026, 0.001);
    auto prob5 = dtS->probabilityOfFeatureValueGivenClass("smoking", "light", "malignant");
    ASSERT_NEAR(prob5, 0.053, 0.001);
    auto prob6 = dtS->probabilityOfFeatureValueGivenClass("smoking", "medium", "malignant");
    ASSERT_NEAR(prob6, 0.026, 0.001);
    auto prob7 = dtS->probabilityOfFeatureValueGivenClass("smoking", "heavy", "malignant");
    ASSERT_NEAR(prob7, 0.895, 0.001);
}

TEST_F(ProbCalcTest, probabilityOfFeatureLessThanThreshold) {
    auto prob0 = dtN->probabilityOfFeatureLessThanThreshold("age", "47");
    ASSERT_NEAR(prob0, 0.007, 0.001);
    auto prob1 = dtN->probabilityOfFeatureLessThanThreshold("age", "50");
    ASSERT_NEAR(prob1, 0.021, 0.001);
    auto prob2 = dtN->probabilityOfFeatureLessThanThreshold("age", "100");
    ASSERT_NEAR(prob2, 1.0, 0.001);
}

TEST_F(ProbCalcTest, probabilityOfFeatureLessThanThresholdGivenClass) {
    auto prob0 = dtN->probabilityOfFeatureLessThanThresholdGivenClass("age", "47", "1");
    ASSERT_NEAR(prob0, 0.007, 0.001);
}