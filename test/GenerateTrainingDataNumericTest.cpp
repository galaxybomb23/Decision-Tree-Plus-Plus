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
        {"debug", "0"}};
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
    ASSERT_EQ(tdgn.getDebug(), 0);
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
    ASSERT_EQ(classes_and_their_param_values.size(), 2);
    ASSERT_EQ(classes_and_their_param_values["recession"]["mean"].size(), 2);
    ASSERT_EQ(classes_and_their_param_values["recession"]["mean"][0], 50.0);
    ASSERT_EQ(classes_and_their_param_values["recession"]["mean"][1], 30.0);
    ASSERT_EQ(classes_and_their_param_values["recession"]["covariance"].size(), 4);
    ASSERT_EQ(classes_and_their_param_values["recession"]["covariance"][0], 1.0);
    ASSERT_EQ(classes_and_their_param_values["recession"]["covariance"][1], 0.0);
    ASSERT_EQ(classes_and_their_param_values["recession"]["covariance"][2], 0.0);
    ASSERT_EQ(classes_and_their_param_values["recession"]["covariance"][3], 20.0);
    ASSERT_EQ(classes_and_their_param_values["goodtimes"]["mean"].size(), 2);
    ASSERT_EQ(classes_and_their_param_values["goodtimes"]["mean"][0], 50.0);
    ASSERT_EQ(classes_and_their_param_values["goodtimes"]["mean"][1], 60.0);
    ASSERT_EQ(classes_and_their_param_values["goodtimes"]["covariance"].size(), 4);
    ASSERT_EQ(classes_and_their_param_values["goodtimes"]["covariance"][0], 1.0);
    ASSERT_EQ(classes_and_their_param_values["goodtimes"]["covariance"][1], 0.0);
    ASSERT_EQ(classes_and_their_param_values["goodtimes"]["covariance"][2], 0.0);
    ASSERT_EQ(classes_and_their_param_values["goodtimes"]["covariance"][3], 20.0);
}

TEST_F(TrainingDataGeneratorNumericTest, TestGenerateMultivariateSamples)
{
    // Generate samples for each class
    std::vector<double> mean = {50.0, 30.0};
    MatrixXd cov_matrix(2, 2);
    cov_matrix << 0.01, 0.0, 0.0, 0.01;
    int num_samples = 3000;
    std::vector<VectorXd> samples = tdgn.GenerateMultivariateSamples(mean, cov_matrix, num_samples);

    // Check the number of samples
    ASSERT_EQ(samples.size(), 3000);

    // Check the first sample
    ASSERT_EQ(samples[0].size(), 2);
    ASSERT_NEAR(samples[0](0), 50.0, 0.5);
    ASSERT_NEAR(samples[0](1), 30.0, 0.5);
}

TEST_F(TrainingDataGeneratorNumericTest, TestGenerateTrainingDataNumeric)
{
    // Read the parameter file
    tdgn.ReadParameterFileNumeric();

    // Generate the training data
    tdgn.GenerateTrainingDataNumeric();

    // verify the output file format is correct
    std::ifstream file("../test/resources/param_numeric_out.txt");
    std::string line;
    std::getline(file, line);
    ASSERT_EQ(line, "\"\",class_name,gdp,return_on_invest");

    for (int i = 0; i < 1000; ++i)
    {
        std::getline(file, line);
        std::istringstream ss(line);
        std::string token;
        std::getline(ss, token, ',');
        ASSERT_EQ(token, std::to_string(i + 1));
        std::getline(ss, token, ',');
        ASSERT_TRUE(token == "recession" || token == "goodtimes");
        std::getline(ss, token, ',');
        double gdp = std::stod(token);
        ASSERT_GE(gdp, 0.0);
        ASSERT_LE(gdp, 100.0);
        std::getline(ss, token, ',');
        double return_on_invest = std::stod(token);
        ASSERT_GE(return_on_invest, 0.0);
        ASSERT_LE(return_on_invest, 100.0);
    }

    // see if first 1000 samples approximately follow the mean and covariance
    int recession_count = 0;
    int goodtimes_count = 0;
    int recession_sum_1 = 0;
    int recession_sum_2 = 0;
    int goodtimes_sum_1 = 0;
    int goodtimes_sum_2 = 0;
    for (int i = 0; i < 1000; ++i)
    {
        std::getline(file, line);
        std::istringstream ss(line);
        std::string token;
        std::getline(ss, token, ',');
        std::getline(ss, token, ',');
        std::string class_name = token;
        std::getline(ss, token, ',');
        double gdp = std::stod(token);
        std::getline(ss, token, ',');
        double return_on_invest = std::stod(token);
        if (class_name == "recession")
        {
            recession_count++;
            recession_sum_1 += gdp;
            recession_sum_2 += return_on_invest;
        }
        else
        {
            goodtimes_count++;
            goodtimes_sum_1 += gdp;
            goodtimes_sum_2 += return_on_invest;
        }
    }

    // Check the mean of the samples
    ASSERT_NEAR(recession_sum_1 / recession_count, 50.0, 5);
    ASSERT_NEAR(recession_sum_2 / recession_count, 30.0, 5);
    ASSERT_NEAR(goodtimes_sum_1 / goodtimes_count, 50.0, 5);
    ASSERT_NEAR(goodtimes_sum_2 / goodtimes_count, 60.0, 5);

    file.close();
}

