#include <gtest/gtest.h>

#include "DTIntrospection.hpp"

class IntrospectionTest : public ::testing::Test
{
protected:
    // Class members to be used in tests
    map<string, string> kwargsS;
    shared_ptr<DecisionTree> dtS; // Symbolic DecisionTree
    shared_ptr<DTIntrospection> dtSI; // Symbolic DecisionTree Introspection

    map<string, string> kwargsN;
    shared_ptr<DecisionTree> dtN; // Numeric DecisionTree
    shared_ptr<DTIntrospection> dtNI; // Numeric DecisionTree Introspection

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
        dtSI = make_shared<DTIntrospection>(dtS);

        dtN = make_shared<DecisionTree>(kwargsN);
        dtN->getTrainingData();
        dtN->calculateFirstOrderProbabilities();
        dtN->calculateClassPriors();
        dtNI = make_shared<DTIntrospection>(dtN);
    }

    void TearDown() override
    {
        dtS.reset();
        dtSI.reset();

        dtN.reset();
        dtNI.reset();
    }
};

TEST_F(IntrospectionTest, CheckdtExists)
{
    ASSERT_NE(dtS, nullptr);
    ASSERT_NE(dtN, nullptr);
}

TEST_F(IntrospectionTest, CheckdtIExists)
{
    ASSERT_NE(dtSI, nullptr);
    ASSERT_NE(dtNI, nullptr);
}

TEST_F(IntrospectionTest, CheckdtIInitialize)
{
    dtS->constructDecisionTreeClassifier();
    dtN->constructDecisionTreeClassifier();
    
    ASSERT_NO_THROW(dtSI->initialize());
    ASSERT_NO_THROW(dtNI->initialize());
}

TEST_F(IntrospectionTest, CheckdtIInitializeThrows)
{
    ASSERT_THROW(dtSI->initialize(), std::runtime_error);
    ASSERT_THROW(dtNI->initialize(), std::runtime_error);
}

TEST_F(IntrospectionTest, CheckdtIRecursiveDescentSymbolic)
{
    dtS->constructDecisionTreeClassifier();
    
    ASSERT_NO_THROW(dtSI->initialize());

    map<int, vector<string>> samplesAtNodesDict = dtSI->getSamplesAtNodesDict();
    map<int, vector<string>> branchFeaturesToNodesDict = dtSI->getBranchFeaturesToNodesDict();
    map<string, vector<int>> sampleToNodeMappingDirectDict = dtSI->getSampleToNodeMappingDirectDict();

    map<int, vector<string>> expectedSamplesAtNodesDict = {
        {1, {"4", "6", "9", "11", "12", "18", "19", "20", "21", "22", "23", "24", "25", "29", "30", "31", "37", "38", "40", "41", "43", "44", "50", "52", "53", "56", "62", "65", "66", "67", "68", "69", "71", "76", "83", "84", "85", "86", "88", "90", "92", "93", "94", "99"}}, 
        {2, {"4", "6", "11", "12", "22", "23", "24", "25", "29", "30", "31", "38", "40", "43", "44", "50", "53", "56", "65", "67", "68", "69", "71", "76", "84", "85", "86", "88", "90", "92", "93", "94", "99"}}, 
        {3, {"4", "6", "11", "29", "31", "65", "67", "71", "76", "88", "93"}}, 
        {4, {"4", "6", "11", "29", "31", "65", "67", "71", "76", "88"}}, 
        {5, {"0", "1", "3", "8", "10", "14", "17", "26", "28", "32", "34", "35", "42", "45", "49", "51", "55", "57", "59", "61", "64", "77", "79", "81", "82", "91", "96"}}, 
        {6, {"49", "51", "55", "59", "77", "79", "82"}}, 
        {7, {"8", "17", "28", "34", "42", "57", "96"}}, 
        {8, {"0", "10", "14", "32", "64", "81", "91"}}, 
        {9, {"2", "5", "7", "13", "15", "16", "27", "33", "36", "39", "46", "47", "48", "54", "58", "60", "63", "70", "72", "73", "74", "75", "78", "80", "87", "89", "95", "97", "98"}}
    };

    map<int, vector<string>> expectedBranchFeaturesToNodesDict = {
        {0, {}},
        {1, {"fatIntake=heavy"}}, 
        {2, {"fatIntake=heavy", "smoking=heavy"}}, 
        {3, {"fatIntake=heavy", "smoking=heavy", "videoAddiction=heavy"}}, 
        {4, {"fatIntake=heavy", "smoking=heavy", "videoAddiction=heavy", "exercising=never"}}, 
        {5, {"fatIntake=low"}}, 
        {6, {"fatIntake=low", "smoking=light"}}, 
        {7, {"fatIntake=low", "smoking=medium"}}, 
        {8, {"fatIntake=low", "smoking=never"}}, 
        {9, {"fatIntake=medium"}}
    };

    map<string, vector<int>> expectedSampleToNodeMappingDirectDict = {
        {"4", {1, 2, 3, 4}}, 
        {"6", {1, 2, 3, 4}}, 
        {"9", {1}}, 
        {"11", {1, 2, 3, 4}}, 
        {"12", {1, 2}}, 
        {"18", {1}}, 
        {"19", {1}}, 
        {"20", {1}}, 
        {"21", {1}}, 
        {"22", {1, 2}}, 
        {"23", {1, 2}}, 
        {"24", {1, 2}}, 
        {"25", {1, 2}}, 
        {"29", {1, 2, 3, 4}}, 
        {"30", {1, 2}}, 
        {"31", {1, 2, 3, 4}}, 
        {"37", {1}}, 
        {"38", {1, 2}}, 
        {"40", {1, 2}}, 
        {"41", {1}}, 
        {"43", {1, 2}}, 
        {"44", {1, 2}}, 
        {"50", {1, 2}}, 
        {"52", {1}}, 
        {"53", {1, 2}}, 
        {"56", {1, 2}}, 
        {"62", {1}}, 
        {"65", {1, 2, 3, 4}}, 
        {"66", {1}}, 
        {"67", {1, 2, 3, 4}}, 
        {"68", {1, 2}}, 
        {"69", {1, 2}}, 
        {"71", {1, 2, 3, 4}}, 
        {"76", {1, 2, 3, 4}}, 
        {"83", {1}}, 
        {"84", {1, 2}}, 
        {"85", {1, 2}}, 
        {"86", {1, 2}}, 
        {"88", {1, 2, 3, 4}}, 
        {"90", {1, 2}}, 
        {"92", {1, 2}}, 
        {"93", {1, 2, 3}}, 
        {"94", {1, 2}}, 
        {"99", {1, 2}}, 
        {"0", {5, 8}}, 
        {"1", {5}}, 
        {"3", {5}}, 
        {"8", {5, 7}}, 
        {"10", {5, 8}}, 
        {"14", {5, 8}}, 
        {"17", {5, 7}}, 
        {"26", {5}}, 
        {"28", {5, 7}}, 
        {"32", {5, 8}}, 
        {"34", {5, 7}}, 
        {"35", {5}}, 
        {"42", {5, 7}}, 
        {"45", {5}}, 
        {"49", {5, 6}}, 
        {"51", {5, 6}}, 
        {"55", {5, 6}}, 
        {"57", {5, 7}}, 
        {"59", {5, 6}}, 
        {"61", {5}}, 
        {"64", {5, 8}}, 
        {"77", {5, 6}}, 
        {"79", {5, 6}}, 
        {"81", {5, 8}}, 
        {"82", {5, 6}}, 
        {"91", {5, 8}}, 
        {"96", {5, 7}}, 
        {"2", {9}}, 
        {"5", {9}}, 
        {"7", {9}}, 
        {"13", {9}}, 
        {"15", {9}}, 
        {"16", {9}}, 
        {"27", {9}}, 
        {"33", {9}}, 
        {"36", {9}}, 
        {"39", {9}}, 
        {"46", {9}}, 
        {"47", {9}}, 
        {"48", {9}}, 
        {"54", {9}}, 
        {"58", {9}}, 
        {"60", {9}}, 
        {"63", {9}}, 
        {"70", {9}}, 
        {"72", {9}}, 
        {"73", {9}}, 
        {"74", {9}}, 
        {"75", {9}}, 
        {"78", {9}}, 
        {"80", {9}}, 
        {"87", {9}}, 
        {"89", {9}}, 
        {"95", {9}}, 
        {"97", {9}}, 
        {"98", {9}}
    };

    ASSERT_EQ(samplesAtNodesDict, expectedSamplesAtNodesDict);
    ASSERT_EQ(branchFeaturesToNodesDict, expectedBranchFeaturesToNodesDict);
    ASSERT_EQ(sampleToNodeMappingDirectDict, expectedSampleToNodeMappingDirectDict);
}

TEST_F(IntrospectionTest, CheckdtIRecursiveDescentNumeric)
{
    dtN->constructDecisionTreeClassifier();
    
    ASSERT_NO_THROW(dtNI->initialize());

    map<int, vector<string>> samplesAtNodesDict = dtNI->getSamplesAtNodesDict();
    map<int, vector<string>> branchFeaturesToNodesDict = dtNI->getBranchFeaturesToNodesDict();
    map<string, vector<int>> sampleToNodeMappingDirectDict = dtNI->getSampleToNodeMappingDirectDict();

    map<int, vector<string>> expectedSamplesAtNodesDict = {
        {1, {"46", "58"}},
        {2, {"1", "4", "7", "14", "17", "18", "21", "23", "24", "25", "29", "31", "33", "34", "36", "37", "38", "41", "43", "45", "48", "49", "50", "52", "53", "59", "62", "64", "65", "66", "67", "68", "69", "70", "71", "75", "80", "83", "87", "89", "91", "92", "93", "99", "104", "105", "106", "113", "114", "119", "127", "129", "132", "135", "136", "138", "139", "142", "144"}},
        {3, {}},
        {4, {}},
        {5, {"1", "4", "18", "31", "83", "135"}},
        {6, {"4"}},
        {7, {"4"}},
        {8, {}},
        {9, {}},
        {10, {}},
        {11, {"4"}},
        {12, {}},
        {13, {}},
        {14, {}},
        {15, {}},
        {16, {}},
        {17, {}},
        {18, {}},
        {19, {"1", "18", "31", "83", "135"}},
        {20, {}},
        {21, {}},
        {22, {}},
        {23, {"1", "18", "31", "83", "135"}},
        {24, {"83"}},
        {25, {"17", "21", "23", "25", "29", "34", "36", "37", "41", "43", "45", "49", "50", "52", "53", "59", "62", "64", "65", "66", "68", "70", "71", "89", "92", "99", "119", "127", "132", "136", "138", "139", "142", "144"}},
        {26, {}},
        {27, {"5", "12", "55", "56", "115", "143"}}
    };

    map<int, vector<string>> expectedBranchFeaturesToNodesDict = {
        {0, {}},
        {1, {"grade=1"}},
        {2, {"grade=2"}},
        {3, {"grade=2", "gleason=10"}},
        {4, {"grade=2", "gleason=3"}},
        {5, {"grade=2", "gleason=4"}},
        {6, {"grade=2", "gleason=4", "g2<3.84"}},
        {7, {"grade=2", "gleason=4", "g2<3.84", "age<63"}},
        {8, {"grade=2", "gleason=4", "g2<3.84", "age<63", "age<59"}},
        {9, {"grade=2", "gleason=4", "g2<3.84", "age<63", "age<59", "age<55"}},
        {10, {"grade=2", "gleason=4", "g2<3.84", "age<63", "age<59", "age>55"}},
        {11, {"grade=2", "gleason=4", "g2<3.84", "age<63", "age>59"}},
        {12, {"grade=2", "gleason=4", "g2<3.84", "age<63", "age>59", "age<61"}},
        {13, {"grade=2", "gleason=4", "g2<3.84", "age>63"}},
        {14, {"grade=2", "gleason=4", "g2<3.84", "age>63", "age<67"}},
        {15, {"grade=2", "gleason=4", "g2<3.84", "age>63", "age<67", "age<65"}},
        {16, {"grade=2", "gleason=4", "g2<3.84", "age>63", "age<67", "age>65"}},
        {17, {"grade=2", "gleason=4", "g2<3.84", "age>63", "age<67", "age>65", "g2<2.4"}},
        {18, {"grade=2", "gleason=4", "g2<3.84", "age>63", "age>67"}},
        {19, {"grade=2", "gleason=4", "g2>3.84"}},
        {20, {"grade=2", "gleason=4", "g2>3.84", "age<49"}},
        {21, {"grade=2", "gleason=4", "g2>3.84", "age<49", "age<47"}},
        {22, {"grade=2", "gleason=4", "g2>3.84", "age<49", "age>47"}},
        {23, {"grade=2", "gleason=4", "g2>3.84", "age>49"}},
        {24, {"grade=2", "gleason=4", "g2>3.84", "age>49", "g2<8.88"}},
        {25, {"grade=2", "gleason=5"}},
        {26, {"grade=2", "gleason=5", "g2<3.84"}},
        {27, {"grade=4"}}
    };

    map<string, vector<int>> expectedSampleToNodeMappingDirectDict = {
        {"46", {1}}, 
        {"58", {1}}, 
        {"1", {2, 5, 19, 23}}, 
        {"4", {2, 5, 6, 7, 11}}, 
        {"7", {2}}, 
        {"14", {2}}, 
        {"17", {2, 25}}, 
        {"18", {2, 5, 19, 23}}, 
        {"21", {2, 25}}, 
        {"23", {2, 25}}, 
        {"24", {2}}, 
        {"25", {2, 25}}, 
        {"29", {2, 25}}, 
        {"31", {2, 5, 19, 23}}, 
        {"33", {2}}, 
        {"34", {2, 25}}, 
        {"36", {2, 25}}, 
        {"37", {2, 25}}, 
        {"38", {2}}, 
        {"41", {2, 25}}, 
        {"43", {2, 25}}, 
        {"45", {2, 25}}, 
        {"48", {2}}, 
        {"49", {2, 25}}, 
        {"50", {2, 25}}, 
        {"52", {2, 25}}, 
        {"53", {2, 25}}, 
        {"59", {2, 25}}, 
        {"62", {2, 25}}, 
        {"64", {2, 25}}, 
        {"65", {2, 25}}, 
        {"66", {2, 25}}, 
        {"67", {2}}, 
        {"68", {2, 25}}, 
        {"69", {2}}, 
        {"70", {2, 25}}, 
        {"71", {2, 25}}, 
        {"75", {2}}, 
        {"80", {2}}, 
        {"83", {2, 5, 19, 23, 24}}, 
        {"87", {2}}, 
        {"89", {2, 25}}, 
        {"91", {2}}, 
        {"92", {2, 25}}, 
        {"93", {2}}, 
        {"99", {2, 25}}, 
        {"104", {2}}, 
        {"105", {2}}, 
        {"106", {2}}, 
        {"113", {2}}, 
        {"114", {2}}, 
        {"119", {2, 25}}, 
        {"127", {2, 25}}, 
        {"129", {2}}, 
        {"132", {2, 25}}, 
        {"135", {2, 5, 19, 23}}, 
        {"136", {2, 25}}, 
        {"138", {2, 25}}, 
        {"139", {2, 25}}, 
        {"142", {2, 25}}, 
        {"144", {2, 25}}, 
        {"5", {27}}, 
        {"12", {27}}, 
        {"55", {27}}, 
        {"56", {27}}, 
        {"115", {27}}, 
        {"143", {27}}
    };

    ASSERT_EQ(samplesAtNodesDict, expectedSamplesAtNodesDict);
    ASSERT_EQ(branchFeaturesToNodesDict, expectedBranchFeaturesToNodesDict);
    ASSERT_EQ(sampleToNodeMappingDirectDict, expectedSampleToNodeMappingDirectDict);
}
