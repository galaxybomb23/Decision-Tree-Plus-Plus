#include <gtest/gtest.h>
#include "DecisionTree.hpp"

class ClassifyTest : public ::testing::Test
{
protected:
    // Class members to be used in tests
    map<string, string> kwargsS;
    shared_ptr<DecisionTree> dtS; // Symbolic DecisionTree

    map<string, string> kwargsN;
    shared_ptr<DecisionTree> dtN; // Numeric DecisionTree

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

        dtN = make_shared<DecisionTree>(kwargsN);
        dtN->getTrainingData();
        dtN->calculateFirstOrderProbabilities();
        dtN->calculateClassPriors();
    }

    void TearDown() override
    {
        dtS.reset();
        dtN.reset();
    }
};

TEST_F(ClassifyTest, CheckdtExists)
{
    ASSERT_NE(dtS, nullptr);
    ASSERT_NE(dtN, nullptr);
}

TEST_F(ClassifyTest, ClassifySymbolic)
{
    // Construct Tree
    DecisionTreeNode* rootS = dtS->constructDecisionTreeClassifier();
    ASSERT_NE(rootS, nullptr);
    
    vector<string> testSample;
    map<string, string> classification;
    map<string, string> expected;

    {
        testSample = {"exercising=never", "smoking=heavy", "fatIntake=heavy", "videoAddiction=heavy"};
        classification = dtS->classify(rootS, testSample);
        expected["benign"] = "0.005";
        expected["malignant"] = "0.995";
        expected["solution_path"] = "NODE0, NODE1, NODE2, NODE3, NODE4";
        ASSERT_EQ(classification, expected);
    }
    {
        testSample = {"videoAddiction=heavy"};
        classification = dtS->classify(rootS, testSample);
        expected["benign"] = "0.620";
        expected["malignant"] = "0.380";
        expected["solution_path"] = "NODE0";
        ASSERT_EQ(classification, expected);
    }
    {
        testSample = {""};
        ASSERT_THROW(dtS->classify(rootS, testSample), std::runtime_error);
    }
    {
        testSample = {"exercising=occasionally", "smoking=never", "fatIntake=low", "videoAddiction=none"};
        classification = dtS->classify(rootS, testSample);
        expected["benign"] = "0.987";
        expected["malignant"] = "0.013";
        expected["solution_path"] = "NODE0, NODE5, NODE8";
        ASSERT_EQ(classification, expected);
    }
    {
        testSample = {"exercising=never", "smoking=never", "fatIntake=heavy", "videoAddiction=heavy"};
        classification = dtS->classify(rootS, testSample);
        expected["benign"] = "0.227";
        expected["malignant"] = "0.773";
        expected["solution_path"] = "NODE0, NODE1";
        ASSERT_EQ(classification, expected);
    }
}

TEST_F(ClassifyTest, ClassifyNumeric) 
{
    // Construct Tree
    DecisionTreeNode* rootN = dtN->constructDecisionTreeClassifier();
    ASSERT_NE(rootN, nullptr);
    
    vector<string> testSample;
    map<string, string> classification;
    map<string, string> expected;

    {
        testSample = {"age=60", "eet=2", "g2=10.22", "grade=2", "gleason=4", "ploidy=diploid"};
        classification = dtN->classify(rootN, testSample);
        expected["1"] = "0.119";
        expected["0"] = "0.881";
        expected["solution_path"] = "NODE0, NODE2, NODE5, NODE19, NODE23";
        ASSERT_EQ(classification, expected);
    }
    {
        testSample = {"age=71", "eet=1", "g2=16.92", "grade=4", "gleason=7", "ploidy=aneuploid"};
        classification = dtN->classify(rootN, testSample);
        expected["1"] = "1.000";
        expected["0"] = "0.000";
        expected["solution_path"] = "NODE0, NODE27";
        ASSERT_EQ(classification, expected);
    }
    {
        testSample = {"age=65", "eet=2", "g2=6.20", "grade=2", "gleason=5", "ploidy=tetraploid"};
        classification = dtN->classify(rootN, testSample);
        expected["1"] = "0.027";
        expected["0"] = "0.973";
        expected["solution_path"] = "NODE0, NODE2, NODE25";
        ASSERT_EQ(classification, expected);
    }
    {
        testSample = {"age=52", "grade=2", "ploidy=tetraploid"};
        classification = dtN->classify(rootN, testSample);
        expected["1"] = "0.153";
        expected["0"] = "0.847";
        expected["solution_path"] = "NODE0, NODE2";
        ASSERT_EQ(classification, expected);
    }
    {
        testSample = {"age=49", "eet=NA", "g2=NA", "grade=3", "gleason=NA", "ploidy=diploid"};
        classification = dtN->classify(rootN, testSample);
        expected["1"] = "0.370";
        expected["0"] = "0.630";
        expected["solution_path"] = "NODE0";
        ASSERT_EQ(classification, expected);
    }
}