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
    
}

TEST_F(ProbCalcTest, probabilityOfFeatureValue) {
    auto prob = dt->probabilityOfFeatureValue("smoking", "heavy");
    // ASSERT_EQ(prob, 0.44);
    
}