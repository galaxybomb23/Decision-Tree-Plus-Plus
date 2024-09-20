#include <gtest/gtest.h>
#include "TrainingDataGeneratorSymbolic.hpp"

class TrainingDataGeneratorSymbolicTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // called before each test
    }

    void TearDown() override
    {
        // called after each test ends
    }

    // Class members to be used in tests
    std::map<std::string, std::string> kwargs = {
        {"output_datafile", "../test/resources/training_symbolic.csv"},
        {"parameter_file", "../test/resources/param_symbolic.txt"},
        {"number_of_training_samples", "100"},
        {"write_to_file", "1"},
        {"debug1", "1"},
        {"debug2", "1"}};
    TrainingDataGeneratorSymbolic tdgs = TrainingDataGeneratorSymbolic(kwargs);
};

TEST_F(TrainingDataGeneratorSymbolicTest, ChecktdgsExists)
{
    ASSERT_NE(&tdgs, nullptr);
}

TEST_F(TrainingDataGeneratorSymbolicTest, CheckParamsTdgs)
{
    ASSERT_EQ(tdgs.getOutputDatafile(), "../test/resources/training_symbolic.csv");
    ASSERT_EQ(tdgs.getParameterFile(), "../test/resources/param_symbolic.txt");
    ASSERT_EQ(tdgs.getNumberOfTrainingSamples(), 100);
    ASSERT_EQ(tdgs.getWriteToFile(), 1);
    ASSERT_EQ(tdgs.getDebug1(), 1);
    ASSERT_EQ(tdgs.getDebug2(), 1);
    ASSERT_NO_THROW(tdgs.ReadParameterFileSymbolic());
    ASSERT_NO_THROW(tdgs.GenerateTrainingDataSymbolic());
}

TEST_F(TrainingDataGeneratorSymbolicTest, CheckFeaturesAndValues)
{
    ASSERT_NO_THROW(tdgs.ReadParameterFileSymbolic());

    // check class priors   
    ASSERT_EQ(tdgs.getClassPriors().size(), 2);
    ASSERT_EQ(tdgs.getClassPriors()[0], 0.4);
    ASSERT_EQ(tdgs.getClassPriors()[1], 0.6);

    // check feature and values dictionary
    ASSERT_EQ(tdgs.getFeaturesAndValuesDict().size(), 4);
    std::vector<std::string> features = {"exercising", "fatIntake", "smoking", "videoAddiction"};
    std::vector<std::vector<std::string>> values = {
        {"never", "occasionally", "regularly"},
        {"low", "medium", "heavy"},
        {"heavy", "medium", "light", "never"},
        {"none", "low", "medium", "heavy"}};

    int ctr = 0;
    for (auto const &feature : tdgs.getFeaturesAndValuesDict())
    {
        ASSERT_EQ(feature.first, features[ctr]);
        ASSERT_EQ(feature.second.size(), values[ctr].size());
        for (int i = 0; i < feature.second.size(); i++)
        {
            ASSERT_EQ(feature.second[i], values[ctr][i]);
        }
        ctr++;
    }
}

TEST_F(TrainingDataGeneratorSymbolicTest, CheckBias)
{
    ASSERT_NO_THROW(tdgs.ReadParameterFileSymbolic());
    std::map<std::string, std::map<std::string, std::vector<std::string>>> biasDict = tdgs.getBiasDict();
    ASSERT_EQ(biasDict.size(), 2);

    std::vector<std::string> expectedClasses = {"benign", "malignant"};
    std::vector<std::vector<std::string>> expectedFeatures = {
        {"exercising", "fatIntake", "smoking", "videoAddiction"},
        {"exercising", "fatIntake", "smoking", "videoAddiction"}};
    std::vector<std::vector<std::vector<std::string>>> expectedValues = {
        {{"never=0.2"}, {"heavy=0.2"}, {"heavy=0.2"}, {"heavy=0.2"}},
        {{"never=0.8"}, {"heavy=0.2"}, {"heavy=0.2"}, {""}}
    };

    int classIndex = 0;
    for (auto const &bias : biasDict)
    {
        ASSERT_EQ(bias.first, expectedClasses[classIndex]);
        int featureIndex = 0;
        for (auto const &feature : bias.second)
        {
            ASSERT_EQ(feature.first, expectedFeatures[classIndex][featureIndex]);
            ASSERT_EQ(feature.second.size(), expectedValues[classIndex][featureIndex].size());
            ASSERT_EQ(feature.second.size(), 1);
            featureIndex++;
        }
        classIndex++;
    }
}