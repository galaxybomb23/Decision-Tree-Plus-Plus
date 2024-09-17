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
    std::vector<std::string> classNames = tdgn.getClassNames();
    ASSERT_EQ(classNames.size(), 2);
    ASSERT_EQ(classNames[0], "recession");
    ASSERT_EQ(classNames[1], "goodtimes");

    // Check class names and priors
    std::map<std::string, double> classNamesAndPriors = tdgn.getClassNamesAndPriors();
    ASSERT_EQ(classNamesAndPriors.size(), 2);
    ASSERT_EQ(classNamesAndPriors["recession"], 0.4);
    ASSERT_EQ(classNamesAndPriors["goodtimes"], 0.6);

    // Check features ordered
    std::vector<std::string> featuresOrdered = tdgn.getFeaturesOrdered();
    ASSERT_EQ(featuresOrdered.size(), 2);
    ASSERT_EQ(featuresOrdered[0], "gdp");
    ASSERT_EQ(featuresOrdered[1], "return_on_invest");

    // Check features with value range
    std::map<std::string, std::pair<double, double>> featuresWithValueRange = tdgn.getFeaturesWithValueRange();
    ASSERT_EQ(featuresWithValueRange.size(), 2);
    ASSERT_EQ(featuresWithValueRange["gdp"].first, 0.0);
    ASSERT_EQ(featuresWithValueRange["gdp"].second, 100.0);
    ASSERT_EQ(featuresWithValueRange["return_on_invest"].first, 0.0);
    ASSERT_EQ(featuresWithValueRange["return_on_invest"].second, 100.0);

    // Check classes and their parameter values
    std::map<std::string, std::map<std::string, std::vector<double>>> classesAndTheirParamValues = tdgn.getClassesAndTheirParamValues();
    ASSERT_EQ(classesAndTheirParamValues.size(), 2);
    ASSERT_EQ(classesAndTheirParamValues["recession"]["mean"].size(), 2);
    ASSERT_EQ(classesAndTheirParamValues["recession"]["mean"][0], 50.0);
    ASSERT_EQ(classesAndTheirParamValues["recession"]["mean"][1], 30.0);
    ASSERT_EQ(classesAndTheirParamValues["recession"]["covariance"].size(), 4);
    ASSERT_EQ(classesAndTheirParamValues["recession"]["covariance"][0], 1.0);
    ASSERT_EQ(classesAndTheirParamValues["recession"]["covariance"][1], 0.0);
    ASSERT_EQ(classesAndTheirParamValues["recession"]["covariance"][2], 0.0);
    ASSERT_EQ(classesAndTheirParamValues["recession"]["covariance"][3], 20.0);
    ASSERT_EQ(classesAndTheirParamValues["goodtimes"]["mean"].size(), 2);
    ASSERT_EQ(classesAndTheirParamValues["goodtimes"]["mean"][0], 50.0);
    ASSERT_EQ(classesAndTheirParamValues["goodtimes"]["mean"][1], 60.0);
    ASSERT_EQ(classesAndTheirParamValues["goodtimes"]["covariance"].size(), 4);
    ASSERT_EQ(classesAndTheirParamValues["goodtimes"]["covariance"][0], 1.0);
    ASSERT_EQ(classesAndTheirParamValues["goodtimes"]["covariance"][1], 0.0);
    ASSERT_EQ(classesAndTheirParamValues["goodtimes"]["covariance"][2], 0.0);
    ASSERT_EQ(classesAndTheirParamValues["goodtimes"]["covariance"][3], 20.0);
}

TEST_F(TrainingDataGeneratorNumericTest, TestGenerateMultivariateSamples)
{
    // Generate samples for each class
    std::vector<double> mean = {50.0, 30.0};
    MatrixXd covMatrix(2, 2);
    covMatrix << 0.01, 0.0, 0.0, 0.01;
    int numSamples = 3000;
    std::vector<VectorXd> samples = tdgn.GenerateMultivariateSamples(mean, covMatrix, numSamples);

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
        double returnOnInvest = std::stod(token);
        ASSERT_GE(returnOnInvest, 0.0);
        ASSERT_LE(returnOnInvest, 100.0);
    }

    // see if first 1000 samples approximately follow the mean and covariance
    int recessionCount = 0;
    int goodtimes_count = 0;
    int recessionSum1 = 0;
    int recessionSum2 = 0;
    int goodtimesSum1 = 0;
    int goodtimesSum2 = 0;
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
        double returnOnInvest = std::stod(token);
        if (class_name == "recession")
        {
            recessionCount++;
            recessionSum1 += gdp;
            recessionSum2 += returnOnInvest;
        }
        else
        {
            goodtimes_count++;
            goodtimesSum1 += gdp;
            goodtimesSum2 += returnOnInvest;
        }
    }

    // Check the mean of the samples
    ASSERT_NEAR(recessionSum1 / recessionCount, 50.0, 5);
    ASSERT_NEAR(recessionSum2 / recessionCount, 30.0, 5);
    ASSERT_NEAR(goodtimesSum1 / goodtimes_count, 50.0, 5);
    ASSERT_NEAR(goodtimesSum2 / goodtimes_count, 60.0, 5);

    file.close();
}

