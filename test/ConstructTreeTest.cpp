#include "DecisionTree.hpp"

#include <gtest/gtest.h>

class ConstructTreeTest : public ::testing::Test {
  protected:
    shared_ptr<DecisionTree> dtS; // Symbolic DecisionTree
    shared_ptr<DecisionTree> dtN; // Numeric DecisionTree
    map<string, string> kwargsS;
    map<string, string> kwargsN;

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
        dtN.reset(); // Reset the DecisionTree
    }
};

TEST_F(ConstructTreeTest, CheckdtExists)
{
    ASSERT_NE(&dtS, nullptr);
    ASSERT_NE(&dtN, nullptr);
}

TEST_F(ConstructTreeTest, bestFeatureCalculatorSymbolic)
{
    BestFeatureResult bfr;
    {
        bfr = dtS->bestFeatureCalculator({}, 0.9580420222262995);
        ASSERT_EQ(bfr.bestFeatureName, "fatIntake");
        ASSERT_NEAR(bfr.bestFeatureEntropy, 0.539, 0.001);
        ASSERT_EQ(bfr.valBasedEntropies, nullopt);
        ASSERT_EQ(bfr.decisionValue, nullopt);
    }
    {
        bfr = dtS->bestFeatureCalculator({"fatIntake=heavy"}, 0.7732266742876344);
        ASSERT_EQ(bfr.bestFeatureName, "smoking");
        ASSERT_NEAR(bfr.bestFeatureEntropy, 0.271, 0.001);
        ASSERT_EQ(bfr.valBasedEntropies, nullopt);
        ASSERT_EQ(bfr.decisionValue, nullopt);
    }
    {
        bfr = dtS->bestFeatureCalculator({"fatIntake=heavy", "smoking=heavy"}, 0.28788102213037137);
        ASSERT_EQ(bfr.bestFeatureName, "videoAddiction");
        ASSERT_NEAR(bfr.bestFeatureEntropy, 0.058, 0.001);
        ASSERT_EQ(bfr.valBasedEntropies, nullopt);
        ASSERT_EQ(bfr.decisionValue, nullopt);
    }
    {
        bfr = dtS->bestFeatureCalculator({"fatIntake=heavy", "smoking=heavy", "videoAddiction=heavy"},
                                         0.16223190039782087);
        ASSERT_EQ(bfr.bestFeatureName, "exercising");
        ASSERT_NEAR(bfr.bestFeatureEntropy, 0.014, 0.001);
        ASSERT_EQ(bfr.valBasedEntropies, nullopt);
        ASSERT_EQ(bfr.decisionValue, nullopt);
    }
    {
        bfr = dtS->bestFeatureCalculator({"fatIntake=low"}, 0.5032583347756457);
        ASSERT_EQ(bfr.bestFeatureName, "smoking");
        ASSERT_NEAR(bfr.bestFeatureEntropy, 0.133, 0.001);
        ASSERT_EQ(bfr.valBasedEntropies, nullopt);
        ASSERT_EQ(bfr.decisionValue, nullopt);
    }
    {
        bfr = dtS->bestFeatureCalculator({"fatIntake=low", "smoking=light"}, 0.13609257142369133);
        ASSERT_EQ(bfr.bestFeatureName, "videoAddiction");
        ASSERT_NEAR(bfr.bestFeatureEntropy, 0.009, 0.001);
        ASSERT_EQ(bfr.valBasedEntropies, nullopt);
        ASSERT_EQ(bfr.decisionValue, nullopt);
    }
    {
        bfr = dtS->bestFeatureCalculator({"fatIntake=low", "smoking=never"}, 0.10265923626304851);
        ASSERT_EQ(bfr.bestFeatureName, "videoAddiction");
        ASSERT_NEAR(bfr.bestFeatureEntropy, 0.005, 0.001);
        ASSERT_EQ(bfr.valBasedEntropies, nullopt);
        ASSERT_EQ(bfr.decisionValue, nullopt);
    }
    {
        bfr = dtS->bestFeatureCalculator({"fatIntake=medium"}, 0.21639693245126473);
        ASSERT_EQ(bfr.bestFeatureName, "videoAddiction");
        ASSERT_NEAR(bfr.bestFeatureEntropy, 0.065, 0.001);
        ASSERT_EQ(bfr.valBasedEntropies, nullopt);
        ASSERT_EQ(bfr.decisionValue, nullopt);
    }
}

TEST_F(ConstructTreeTest, bestFeatureCalculatorNumeric)
{
    BestFeatureResult bfr;
    {
        bfr = dtN->bestFeatureCalculator({}, 0.9505668528932196);
        ASSERT_EQ(bfr.bestFeatureName, "grade");
        ASSERT_NEAR(bfr.bestFeatureEntropy, 0.790, 0.001);
        ASSERT_EQ(bfr.valBasedEntropies, nullopt);
        ASSERT_EQ(bfr.decisionValue, nullopt);
    }
    {
        bfr = dtN->bestFeatureCalculator({"grade=2.0"}, 0.6161661934005354);
        ASSERT_EQ(bfr.bestFeatureName, "gleason");
        ASSERT_NEAR(bfr.bestFeatureEntropy, 0.232, 0.001);
        ASSERT_EQ(bfr.valBasedEntropies, nullopt);
        ASSERT_EQ(bfr.decisionValue, nullopt);
    }
    {
        bfr = dtN->bestFeatureCalculator({"grade=2.0", "gleason=4.0"}, 0.5551649772709998);
        ASSERT_EQ(bfr.bestFeatureName, "g2");
        ASSERT_NEAR(bfr.bestFeatureEntropy, 0.009, 0.001);
        ASSERT_NEAR(bfr.valBasedEntropies->first, 0.072, 0.001);
        ASSERT_NEAR(bfr.valBasedEntropies->second, 0.537, 0.001);
        ASSERT_NEAR(bfr.decisionValue.value(), 3.840, 0.001);
    }
    {
        bfr = dtN->bestFeatureCalculator({"grade=2.0", "gleason=4.0", "g2<3.84"}, 0.07170446042023888);
        ASSERT_EQ(bfr.bestFeatureName, "age");
        ASSERT_NEAR(bfr.bestFeatureEntropy, 0.0000148, 0.000001);
        ASSERT_NEAR(bfr.valBasedEntropies->first, 0.043, 0.001);
        ASSERT_NEAR(bfr.valBasedEntropies->second, 0.038, 0.001);
        ASSERT_NEAR(bfr.decisionValue.value(), 63.000, 0.001);
    }
    {
        bfr = dtN->bestFeatureCalculator({"grade=2.0", "gleason=4.0", "g2>3.84", "age<49.0"}, 0.04651186386689919);
        ASSERT_EQ(bfr.bestFeatureName, "age");
        ASSERT_NEAR(bfr.bestFeatureEntropy, 0.00000588, 0.00000001);
        ASSERT_NEAR(bfr.valBasedEntropies->first, 0.0259, 0.0001);
        ASSERT_NEAR(bfr.valBasedEntropies->second, 0.0259, 0.0001);
        ASSERT_NEAR(bfr.decisionValue.value(), 47.0, 0.001);
    }
    {
        bfr = dtN->bestFeatureCalculator({"grade=2.0", "gleason=4.0", "g2<3.84", "age<63", "age>55", "age<59"},
                                         0.017289523234007915);
        ASSERT_EQ(bfr.bestFeatureName, "age");
        ASSERT_NEAR(bfr.bestFeatureEntropy, 0.0000005197, 0.0000000001);
        ASSERT_NEAR(bfr.valBasedEntropies->first, 0.00829, 0.00001);
        ASSERT_NEAR(bfr.valBasedEntropies->second, 0.00829, 0.00001);
        ASSERT_NEAR(bfr.decisionValue.value(), 57, 0.01);
    }
}

TEST_F(ConstructTreeTest, constructDecisionTreeClassifierSymbolic)
{
    // Construct Tree
    DecisionTreeNode* rootS = dtS->constructDecisionTreeClassifier();
    ASSERT_NE(rootS, nullptr);

    // Capture the output of DisplayDecisionTree
    std::ostringstream outputBuffer;
    std::streambuf* oldCoutBuffer = cout.rdbuf(outputBuffer.rdbuf());
    rootS->DisplayDecisionTree(" ");
    cout.rdbuf(oldCoutBuffer); // Restore the original buffer

    // The captured output
    string actualOutput = normalizeString(outputBuffer.str());

    // Define the expected output
    string expectedOutput = R"(NODE 0:    BRANCH TESTS TO NODE: []
           Decision Feature: fatIntake   Node Creation Entropy: 0.958   Class Probs: ['class=benign => 0.620', 'class=malignant => 0.380']

NODE 1:       BRANCH TESTS TO NODE: ['fatIntake=heavy']
              Decision Feature: smoking   Node Creation Entropy: 0.773   Class Probs: ['class=benign => 0.227', 'class=malignant => 0.773']

NODE 2:          BRANCH TESTS TO NODE: ['fatIntake=heavy', 'smoking=heavy']
                 Decision Feature: videoAddiction   Node Creation Entropy: 0.288   Class Probs: ['class=benign => 0.050', 'class=malignant => 0.950']

NODE 3:             BRANCH TESTS TO NODE: ['fatIntake=heavy', 'smoking=heavy', 'videoAddiction=heavy']
                    Decision Feature: exercising   Node Creation Entropy: 0.162   Class Probs: ['class=benign => 0.024', 'class=malignant => 0.976']

NODE 4:                BRANCH TESTS TO LEAF NODE: ['fatIntake=heavy', 'smoking=heavy', 'videoAddiction=heavy', 'exercising=never']
                       Node Creation Entropy: 0.042   Class Probs: ['class=benign => 0.005', 'class=malignant => 0.995']

NODE 5:       BRANCH TESTS TO NODE: ['fatIntake=low']
              Decision Feature: smoking   Node Creation Entropy: 0.503   Class Probs: ['class=benign => 0.889', 'class=malignant => 0.111']

NODE 6:          BRANCH TESTS TO LEAF NODE: ['fatIntake=low', 'smoking=light']
                 Node Creation Entropy: 0.136   Class Probs: ['class=benign => 0.981', 'class=malignant => 0.019']

NODE 7:          BRANCH TESTS TO LEAF NODE: ['fatIntake=low', 'smoking=medium']
                 Node Creation Entropy: 0.097   Class Probs: ['class=benign => 0.987', 'class=malignant => 0.013']

NODE 8:          BRANCH TESTS TO LEAF NODE: ['fatIntake=low', 'smoking=never']
                 Node Creation Entropy: 0.103   Class Probs: ['class=benign => 0.987', 'class=malignant => 0.013']

NODE 9:       BRANCH TESTS TO LEAF NODE: ['fatIntake=medium']
              Node Creation Entropy: 0.216   Class Probs: ['class=benign => 0.966', 'class=malignant => 0.034'])";

    // Normalize the expected output
    expectedOutput = normalizeString(expectedOutput);

    // Split normalized strings into lines
    std::istringstream actualStream(actualOutput);
    std::istringstream expectedStream(expectedOutput);
    string actualLine, expectedLine;

    // Compare line by line
    while (std::getline(expectedStream, expectedLine)) {
        ASSERT_TRUE(std::getline(actualStream, actualLine)) << "Actual output is shorter than expected!";
        ASSERT_EQ(actualLine, expectedLine) << "Mismatch found:\nExpected: " << expectedLine << "\nActual: " << actualLine;
    }

    // Check for extra lines in the actual output
    while (std::getline(actualStream, actualLine)) {
        ADD_FAILURE() << "Extra line in actual output: " << actualLine;
    }
}

TEST_F(ConstructTreeTest, constructDecisionTreeClassifierNumeric)
{
    // Construct Tree
    DecisionTreeNode* rootN = dtN->constructDecisionTreeClassifier();
    ASSERT_NE(rootN, nullptr);

    // Capture the output of DisplayDecisionTree
    std::ostringstream outputBuffer;
    std::streambuf* oldCoutBuffer = cout.rdbuf(outputBuffer.rdbuf());
    rootN->DisplayDecisionTree(" ");
    cout.rdbuf(oldCoutBuffer); // Restore the original buffer

    // The captured output
    string actualOutput = normalizeString(outputBuffer.str());

    // Define the expected output
    string expectedOutput = R"(NODE 0:    BRANCH TESTS TO NODE: []
           Decision Feature: grade   Node Creation Entropy: 0.951   Class Probs: ['class=0 => 0.630', 'class=1 => 0.370']

NODE 1:       BRANCH TESTS TO LEAF NODE: ['grade=1']
              Node Creation Entropy: 0.000   Class Probs: ['class=0 => 1.000', 'class=1 => 0.000']

NODE 2:       BRANCH TESTS TO NODE: ['grade=2']
              Decision Feature: gleason   Node Creation Entropy: 0.616   Class Probs: ['class=0 => 0.847', 'class=1 => 0.153']

NODE 3:          BRANCH TESTS TO LEAF NODE: ['grade=2', 'gleason=10']
                 Node Creation Entropy: 0.000   Class Probs: ['class=0 => 0.000', 'class=1 => 1.000']

NODE 4:          BRANCH TESTS TO LEAF NODE: ['grade=2', 'gleason=3']
                 Node Creation Entropy: 0.000   Class Probs: ['class=0 => 1.000', 'class=1 => 0.000']

NODE 5:          BRANCH TESTS TO NODE: ['grade=2', 'gleason=4']
                 Decision Feature: g2   Node Creation Entropy: 0.555   Class Probs: ['class=0 => 0.871', 'class=1 => 0.129']

NODE 6:             BRANCH TESTS TO NODE: ['grade=2', 'gleason=4', 'g2<3.84']
                    Decision Feature: age   Node Creation Entropy: 0.072   Class Probs: ['class=0 => 0.991', 'class=1 => 0.009']

NODE 7:                BRANCH TESTS TO NODE: ['grade=2', 'gleason=4', 'g2<3.84', 'age<63']
                       Decision Feature: age   Node Creation Entropy: 0.043   Class Probs: ['class=0 => 0.995', 'class=1 => 0.005']

NODE 8:                   BRANCH TESTS TO NODE: ['grade=2', 'gleason=4', 'g2<3.84', 'age<63', 'age<59']
                          Decision Feature: age   Node Creation Entropy: 0.029   Class Probs: ['class=0 => 0.997', 'class=1 => 0.003']

NODE 9:                      BRANCH TESTS TO LEAF NODE: ['grade=2', 'gleason=4', 'g2<3.84', 'age<63', 'age<59', 'age<55']
                             Node Creation Entropy: 0.014   Class Probs: ['class=0 => 0.999', 'class=1 => 0.001']

NODE 10:                      BRANCH TESTS TO LEAF NODE: ['grade=2', 'gleason=4', 'g2<3.84', 'age<63', 'age<59', 'age>55']
                              Node Creation Entropy: 0.017   Class Probs: ['class=0 => 0.998', 'class=1 => 0.002']

NODE 11:                   BRANCH TESTS TO NODE: ['grade=2', 'gleason=4', 'g2<3.84', 'age<63', 'age>59']
                           Decision Feature: age   Node Creation Entropy: 0.019   Class Probs: ['class=0 => 0.998', 'class=1 => 0.002']

NODE 12:                      BRANCH TESTS TO LEAF NODE: ['grade=2', 'gleason=4', 'g2<3.84', 'age<63', 'age>59', 'age<61']
                              Node Creation Entropy: 0.008   Class Probs: ['class=0 => 0.999', 'class=1 => 0.001']

NODE 13:                BRANCH TESTS TO NODE: ['grade=2', 'gleason=4', 'g2<3.84', 'age>63']
                        Decision Feature: age   Node Creation Entropy: 0.038   Class Probs: ['class=0 => 0.996', 'class=1 => 0.004']

NODE 14:                   BRANCH TESTS TO NODE: ['grade=2', 'gleason=4', 'g2<3.84', 'age>63', 'age<67']
                           Decision Feature: age   Node Creation Entropy: 0.026   Class Probs: ['class=0 => 0.997', 'class=1 => 0.003']

NODE 15:                      BRANCH TESTS TO LEAF NODE: ['grade=2', 'gleason=4', 'g2<3.84', 'age>63', 'age<67', 'age<65']
                              Node Creation Entropy: 0.013   Class Probs: ['class=0 => 0.999', 'class=1 => 0.001']

NODE 16:                      BRANCH TESTS TO NODE: ['grade=2', 'gleason=4', 'g2<3.84', 'age>63', 'age<67', 'age>65']
                              Decision Feature: g2   Node Creation Entropy: 0.016   Class Probs: ['class=0 => 0.999', 'class=1 => 0.001']

NODE 17:                         BRANCH TESTS TO LEAF NODE: ['grade=2', 'gleason=4', 'g2<3.84', 'age>63', 'age<67', 'age>65', 'g2<2.4']
                                 Node Creation Entropy: 0.005   Class Probs: ['class=0 => 1.000', 'class=1 => 0.000']

NODE 18:                   BRANCH TESTS TO LEAF NODE: ['grade=2', 'gleason=4', 'g2<3.84', 'age>63', 'age>67']
                           Node Creation Entropy: 0.016   Class Probs: ['class=0 => 0.999', 'class=1 => 0.001']

NODE 19:             BRANCH TESTS TO NODE: ['grade=2', 'gleason=4', 'g2>3.84']
                     Decision Feature: age   Node Creation Entropy: 0.537   Class Probs: ['class=0 => 0.877', 'class=1 => 0.123']

NODE 20:                BRANCH TESTS TO NODE: ['grade=2', 'gleason=4', 'g2>3.84', 'age<49']
                        Decision Feature: age   Node Creation Entropy: 0.047   Class Probs: ['class=0 => 0.995', 'class=1 => 0.005']

NODE 21:                   BRANCH TESTS TO LEAF NODE: ['grade=2', 'gleason=4', 'g2>3.84', 'age<49', 'age<47']
                           Node Creation Entropy: 0.026   Class Probs: ['class=0 => 0.997', 'class=1 => 0.003']

NODE 22:                   BRANCH TESTS TO LEAF NODE: ['grade=2', 'gleason=4', 'g2>3.84', 'age<49', 'age>47']
                           Node Creation Entropy: 0.026   Class Probs: ['class=0 => 0.997', 'class=1 => 0.003']

NODE 23:                BRANCH TESTS TO NODE: ['grade=2', 'gleason=4', 'g2>3.84', 'age>49']
                        Decision Feature: g2   Node Creation Entropy: 0.525   Class Probs: ['class=0 => 0.881', 'class=1 => 0.119']

NODE 24:                   BRANCH TESTS TO LEAF NODE: ['grade=2', 'gleason=4', 'g2>3.84', 'age>49', 'g2<8.88']
                           Node Creation Entropy: 0.126   Class Probs: ['class=0 => 0.983', 'class=1 => 0.017']

NODE 25:          BRANCH TESTS TO NODE: ['grade=2', 'gleason=5']
                  Decision Feature: g2   Node Creation Entropy: 0.179   Class Probs: ['class=0 => 0.973', 'class=1 => 0.027']

NODE 26:             BRANCH TESTS TO LEAF NODE: ['grade=2', 'gleason=5', 'g2<3.84']
                     Node Creation Entropy: 0.017   Class Probs: ['class=0 => 0.998', 'class=1 => 0.002']

NODE 27:       BRANCH TESTS TO LEAF NODE: ['grade=4']
               Node Creation Entropy: 0.000   Class Probs: ['class=0 => 0.000', 'class=1 => 1.000'])";

    // Normalize the expected output
    expectedOutput = normalizeString(expectedOutput);

    // Split normalized strings into lines
    std::istringstream actualStream(actualOutput);
    std::istringstream expectedStream(expectedOutput);
    string actualLine, expectedLine;

    // Compare line by line
    while (std::getline(expectedStream, expectedLine)) {
        ASSERT_TRUE(std::getline(actualStream, actualLine)) << "Actual output is shorter than expected!";
        ASSERT_EQ(actualLine, expectedLine) << "Mismatch found:\nExpected: " << expectedLine << "\n  Actual: " << actualLine;
    }

    // Check for extra lines in the actual output
    while (std::getline(actualStream, actualLine)) {
        ADD_FAILURE() << "Extra line in actual output: " << actualLine;
    }
}