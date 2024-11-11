#include <gtest/gtest.h>

#include "DecisionTree.hpp"

class ProbCalcTest : public ::testing::Test
{
    protected:
    std::unique_ptr<DecisionTree> dt;  // Use a pointer to the DecisionTree
    void SetUp() override
    {
        // called before each test starts
        map<string,string> kwargs = {
            {"training_datafile", "../test/resources/training_symbolic.csv"},
            {"csv_class_column_index", "1"},
            {"csv_columns_for_features", "{2,3,4,5}"},
            {"max_depth_desired", "5"},
            {"entropy_threshold", "0.1"}
        };
        
        dt = std::make_unique<DecisionTree>(kwargs);  // Initialize the DecisionTree
        dt->getTrainingData();
    }

    void TearDown() override
    {
        dt.reset();  // Reset the DecisionTree
        
    }

};
// Pyhton setup:
// dt = DecisionTree( training_datafile = r"penis",  
//                         csv_class_column_index = 1,
//                         csv_columns_for_features = [2,3,4,5],
//                         max_depth_desired = 5,
//                         entropy_threshold = 0.1,
//                      )

TEST_F(ProbCalcTest, CheckdtExists) { ASSERT_NE(&dt, nullptr); }

TEST_F(ProbCalcTest, priorProbabilityForClass) {
    double prob = dt->priorProbabilityForClass("benign");
    ASSERT_EQ(prob, 0.62);

    prob = dt->priorProbabilityForClass("malignant");
    ASSERT_EQ(prob, 0.38);
}

TEST_F(ProbCalcTest, calculateClassPriors) {
    dt->calculateClassPriors();
    vector<float> expected = {0.62, 0.38};
    vector<float> priors;
    for (int i = 0; i < dt->_classNames.size(); i++) {
        priors.push_back(dt->priorProbabilityForClass(dt->_classNames[i]));
        ASSERT_EQ(expected[i], priors[i]);
    }
}

TEST_F(ProbCalcTest, probabilityOfFeatureValue) {
    auto prob0 = dt->probabilityOfFeatureValue("smoking", "never");
    ASSERT_EQ(prob0, 0.16);
    auto prob1 = dt->probabilityOfFeatureValue("smoking", "light");
    ASSERT_EQ(prob1, 0.23);
    auto prob2 = dt->probabilityOfFeatureValue("smoking", "medium");
    ASSERT_EQ(prob2, 0.17);
    auto prob3 = dt->probabilityOfFeatureValue("smoking", "heavy");
    ASSERT_EQ(prob3, 0.44);
}

TEST_F(ProbCalcTest, probabilityOfFeatureValueGivenClass) {
    auto prob0 = dt->probabilityOfFeatureValueGivenClass("smoking", "never", "benign");
    ASSERT_NEAR(prob0, 0.242, 0.001);
    auto prob1 = dt->probabilityOfFeatureValueGivenClass("smoking", "light", "benign");
    ASSERT_NEAR(prob1, 0.339, 0.001);
    auto prob2 = dt->probabilityOfFeatureValueGivenClass("smoking", "medium", "benign");
    ASSERT_NEAR(prob2, 0.258, 0.001);
    auto prob3 = dt->probabilityOfFeatureValueGivenClass("smoking", "heavy", "benign");
    ASSERT_NEAR(prob3, 0.161, 0.001);

    auto prob4 = dt->probabilityOfFeatureValueGivenClass("smoking", "never", "malignant");
    ASSERT_NEAR(prob4, 0.026, 0.001);
    auto prob5 = dt->probabilityOfFeatureValueGivenClass("smoking", "light", "malignant");
    ASSERT_NEAR(prob5, 0.053, 0.001);
    auto prob6 = dt->probabilityOfFeatureValueGivenClass("smoking", "medium", "malignant");
    ASSERT_NEAR(prob6, 0.026, 0.001);
    auto prob7 = dt->probabilityOfFeatureValueGivenClass("smoking", "heavy", "malignant");
    ASSERT_NEAR(prob7, 0.895, 0.001);
}