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

// further tests of class variables after running ReadParameterFileNumeric
TEST_F(TrainingDataGeneratorNumericTest, TestReadParameterFileNumericAll)
{
    // Read the parameter file
    tdgn.ReadParameterFileNumeric();

    // Check class names
    std::vector<std::string> class_names = tdgn.getClassNames();
    ASSERT_EQ(class_names.size(), 2);
    ASSERT_EQ(class_names[0], "recession");
    ASSERT_EQ(class_names[1], "goodtimes");

    // Check class names and priors
    std::map<std::string, double> class_names_and_priors = tdgn.getClassNamesAndPriors();
    ASSERT_EQ(class_names_and_priors.size(), 2);
    ASSERT_EQ(class_names_and_priors["recession"], 0.4);
    ASSERT_EQ(class_names_and_priors["goodtimes"], 0.6);

    // Check features ordered
    std::vector<std::string> features_ordered = tdgn.getFeaturesOrdered();
    ASSERT_EQ(features_ordered.size(), 2);
    ASSERT_EQ(features_ordered[0], "gdp");
    ASSERT_EQ(features_ordered[1], "return_on_invest");

    // Check features with value range
    std::map<std::string, std::pair<double, double>> features_with_value_range = tdgn.getFeaturesWithValueRange();
    ASSERT_EQ(features_with_value_range.size(), 2);
    ASSERT_EQ(features_with_value_range["gdp"].first, 0.0);
    ASSERT_EQ(features_with_value_range["gdp"].second, 100.0);
    ASSERT_EQ(features_with_value_range["return_on_invest"].first, 0.0);
    ASSERT_EQ(features_with_value_range["return_on_invest"].second, 100.0);

    // Check classes and their parameter values
    std::map<std::string, std::map<std::string, std::vector<double>>> classes_and_their_param_values = tdgn.getClassesAndTheirParamValues();
    ASSERT_EQ(classes_and_their_param_values["recession"]["mean"].size(), 2);
    ASSERT_EQ(classes_and_their_param_values["recession"]["mean"][0], 50.0);
    ASSERT_EQ(classes_and_their_param_values["recession"]["mean"][1], 30.0);
}

