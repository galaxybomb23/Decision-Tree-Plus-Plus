#include <gtest/gtest.h>
#include "TrainingDataGeneratorNumeric.hpp"

class TrainingDataGeneratorNumericTest : public ::testing::Test
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
        {"output_csv_file", "../test/resources/param_numeric_out.txt"},
        {"parameter_file", "../test/resources/param_numeric.txt"},
        {"number_of_samples_per_class", "3000"},
        {"debug", "1"}};
    TrainingDataGeneratorNumeric tdgn = TrainingDataGeneratorNumeric(kwargs);
};

TEST_F(TrainingDataGeneratorNumericTest, ConstructorInitializesTdgn)
{
    ASSERT_NE(&tdgn, nullptr);
}

TEST_F(TrainingDataGeneratorNumericTest, CheckParamsTdgn)
{
    ASSERT_EQ(tdgn.getOutputCsvFile(), "../test/resources/param_numeric_out.txt");
    ASSERT_EQ(tdgn.getParameterFile(), "../test/resources/param_numeric.txt");
    ASSERT_EQ(tdgn.getNumberOfSamplesPerClass(), 3000);
    ASSERT_EQ(tdgn.getDebug(), 1);
}

TEST_F(TrainingDataGeneratorNumericTest, TestReadParameterFileNumeric)
{
    // Read the parameter file
    tdgn.ReadParameterFileNumeric();
}