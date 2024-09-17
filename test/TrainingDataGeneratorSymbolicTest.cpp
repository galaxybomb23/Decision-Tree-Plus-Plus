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

TEST_F(TrainingDataGeneratorSymbolicTest, CheckReadParameterFileSymbolic)
{
    ASSERT_NO_THROW(tdgs.ReadParameterFileSymbolic());

    // check class priors   
    ASSERT_EQ(tdgs.getClassPriors().size(), 2);
    ASSERT_EQ(tdgs.getClassPriors()[0], 0.4);
    ASSERT_EQ(tdgs.getClassPriors()[1], 0.6);

    // check feature and values dictionary
    ASSERT_EQ(tdgs.getFeaturesAndValuesDict().size(), 4);

}