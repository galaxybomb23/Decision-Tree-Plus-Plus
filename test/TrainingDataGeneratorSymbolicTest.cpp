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
        {"debug1", "0"},
        {"debug2", "0"}};
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
    ASSERT_EQ(tdgs.getDebug1(), 0);
    ASSERT_EQ(tdgs.getDebug2(), 0);
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
    std::vector <std::string> features = {"smoking", "exercising", "fatIntake", "videoAddiction"};

    int ctr = 0;
    for (auto const &feature : tdgs.getFeaturesAndValuesDict())
    {
        ASSERT_TRUE(std::find(features.begin(), features.end(), feature.first) != features.end());
        if (feature.first == "smoking")
        {
            ASSERT_EQ(feature.second.size(), 4);
            ASSERT_EQ(feature.second[0], "heavy");
            ASSERT_EQ(feature.second[1], "medium");
            ASSERT_EQ(feature.second[2], "light");
            ASSERT_EQ(feature.second[3], "never");
        }
        if (feature.first == "exercising")
        {
            ASSERT_EQ(feature.second.size(), 3);
            ASSERT_EQ(feature.second[0], "never");
            ASSERT_EQ(feature.second[1], "occasionally");
            ASSERT_EQ(feature.second[2], "regularly");
        }
        if (feature.first == "fatIntake")
        {
            ASSERT_EQ(feature.second.size(), 3);
            ASSERT_EQ(feature.second[0], "low");
            ASSERT_EQ(feature.second[1], "medium");
            ASSERT_EQ(feature.second[2], "heavy");
        }
        if (feature.first == "videoAddiction")
        {
            ASSERT_EQ(feature.second.size(), 4);
            ASSERT_EQ(feature.second[0], "none");
            ASSERT_EQ(feature.second[1], "low");
            ASSERT_EQ(feature.second[2], "medium");
            ASSERT_EQ(feature.second[3], "heavy");
        }
        ctr++;
    }
    ASSERT_EQ(ctr, 4);
}

TEST_F(TrainingDataGeneratorSymbolicTest, CheckBias)
{
    ASSERT_NO_THROW(tdgs.ReadParameterFileSymbolic());
    std::map<std::string, std::map<std::string, std::vector<std::string>>> biasDict = tdgs.getBiasDict();
    ASSERT_EQ(biasDict.size(), 2);

    for (auto const &bias : biasDict)
{
    if (bias.first == "malignant")
    {
        ASSERT_EQ(bias.second.size(), 4);

        for (auto const &feature : bias.second)
        {
            if (feature.first == "smoking")
            {
                ASSERT_EQ(feature.second.size(), 1);
                ASSERT_EQ(feature.second[0], "heavy=0.8");
            }
            else if (feature.first == "exercising")
            {
                ASSERT_EQ(feature.second.size(), 1);
                ASSERT_EQ(feature.second[0], "never=0.8");
            }
            else if (feature.first == "fatIntake")
            {
                ASSERT_EQ(feature.second.size(), 1);
                ASSERT_EQ(feature.second[0], "heavy=0.8");
            }
            else if (feature.first == "videoAddiction")
            {
                ASSERT_EQ(feature.second.size(), 1);
                ASSERT_EQ(feature.second[0], "");
            }
        }
    }
    else if (bias.first == "benign")
    {
        ASSERT_EQ(bias.second.size(), 4);

        for (auto const &feature : bias.second)
        {
            if (feature.first == "smoking")
            {
                ASSERT_EQ(feature.second.size(), 1);
                ASSERT_EQ(feature.second[0], "heavy=0.2");
            }
            else if (feature.first == "exercising")
            {
                ASSERT_EQ(feature.second.size(), 1);
                ASSERT_EQ(feature.second[0], "never=0.2");
            }
            else if (feature.first == "fatIntake")
            {
                ASSERT_EQ(feature.second.size(), 1);
                ASSERT_EQ(feature.second[0], "heavy=0.2");
            }
            else if (feature.first == "videoAddiction")
            {
                ASSERT_EQ(feature.second.size(), 1);
                ASSERT_EQ(feature.second[0], "heavy=0.2");
            }
        }
    }
}


}