#include "DecisionTree.hpp"

#include <gtest/gtest.h>
bool areVectorsAlmostEqual(const vector<vector<string>> &a, const vector<vector<string>> &b);
const double TOLERANCE = 1e-9;

// Helper function for Google Test to compare vectors of vectors with floating-point values
void assertVectorsAlmostEqual(const vector<vector<string>> &actual, const vector<vector<string>> &expected, const string &message = "")
{
	double tolerance = TOLERANCE;
	ASSERT_EQ(actual.size(), expected.size()) << message << " Sizes of actual and expected vectors are different.";
	for (size_t i = 0; i < actual.size(); ++i) {
		ASSERT_EQ(actual[i][0], expected[i][0]) << message << " Feature names differ at index " << i;
		ASSERT_EQ(actual[i][1], expected[i][1]) << message << " Operators differ at index " << i;

		// Convert string values to double for approximate comparison
		double actualValue	 = stod(actual[i][2]);
		double expectedValue = stod(expected[i][2]);
		ASSERT_NEAR(actualValue, expectedValue, tolerance)
			<< message << " Values differ at index " << i << ": Expected " << expectedValue << " but got " << actualValue;
	}
}


class DecisionTreeTest : public ::testing::Test {
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
	map<string, string> kwargs = {
		{		 "training_datafile", "../test/resources/training_symbolic.csv"},
		{	 "csv_class_column_index",									   "1"},
		{"csv_columns_for_features",								 {2, 3, 4, 5}},
		{		 "max_depth_desired",									   "5"},
		{		 "entropy_threshold",									 "0.1"}
		//   ,{"symbolic_to_numeric_cardinality_threshold", "20"},
		//   {"number_of_histogram_bins", "10"},
		//   {"csv_cleanup_needed", "1"},
		//   {"debug1", "1"},
		//   {"debug2", "2"},
		//   {"debug3", "3"}
	};
	DecisionTree dt		  = DecisionTree(kwargs);
	DecisionTreeNode node = DecisionTreeNode("feature", 0.0, {0.0}, {"branch"}, dt, true);
};

TEST_F(DecisionTreeTest, CheckdtExists)
{
	ASSERT_NE(&dt, nullptr);
}

TEST_F(DecisionTreeTest, ConstructorInitializesNode)
{
	ASSERT_NE(&node, nullptr);
}

TEST_F(DecisionTreeTest, CheckParamsDt)
{
	map<string, string> kargs = {
		{						"training_datafile", "../test/resources/stage3cancer.csv"},
		{				   "csv_class_column_index",								   "2"},
		{				 "csv_columns_for_features",					 {3, 4, 5, 6, 7, 8}},
		{						"max_depth_desired",								  "8"},
		{						"entropy_threshold",							   "0.01"},
		{"symbolic_to_numeric_cardinality_threshold",								  "20"},
		{				 "number_of_histogram_bins",								 "10"},
		{					   "csv_cleanup_needed",								  "1"},
		{								   "debug1",								  "1"},
		{								   "debug2",								  "2"},
		{								   "debug3",								  "3"}
	   };

	DecisionTree dt = DecisionTree(kargs);

	ASSERT_EQ(dt.getTrainingDatafile(), "../test/resources/stage3cancer.csv");
	ASSERT_EQ(dt.getEntropyThreshold(), 0.01);
	ASSERT_EQ(dt.getMaxDepthDesired(), 8);
	ASSERT_EQ(dt.getCsvClassColumnIndex(), 2);
	ASSERT_EQ(dt.getSymbolicToNumericCardinalityThreshold(), 20);
	ASSERT_EQ(dt.getCsvColumnsForFeatures().size(), 6);
	ASSERT_EQ(dt.getNumberOfHistogramBins(), 10);
	ASSERT_EQ(dt.getCsvCleanupNeeded(), 1);
	ASSERT_EQ(dt.getDebug1(), 1);
	ASSERT_EQ(dt.getDebug2(), 2);
	ASSERT_EQ(dt.getDebug3(), 3);
	ASSERT_NO_THROW(dt.getTrainingData());
	ASSERT_EQ(dt.getHowManyTotalTrainingSamples(), 146);
}

TEST_F(DecisionTreeTest, CheckGetTrainingData)
{
	map<string, string> kargs = {
		{						"training_datafile", "../test/resources/stage3cancer.csv"},
		{				   "csv_class_column_index",								   "2"},
		{				 "csv_columns_for_features",					 {3, 4, 5, 6, 7, 8}},
		{						"max_depth_desired",								  "8"},
		{						"entropy_threshold",							   "0.01"},
		{"symbolic_to_numeric_cardinality_threshold",								  "20"},
		{				 "number_of_histogram_bins",								 "10"},
		{					   "csv_cleanup_needed",								  "1"},
		{								   "debug1",								  "1"},
		{								   "debug2",								  "2"},
		{								   "debug3",								  "3"}
	   };

	DecisionTree dt = DecisionTree(kargs);

	ASSERT_NO_THROW(dt.getTrainingData());
	ASSERT_EQ(dt.getHowManyTotalTrainingSamples(), 146);

	vector<string> expectedFeatureNames = {"age", "eet", "g2", "grade", "gleason", "ploidy"};
	ASSERT_EQ(dt.getFeatureNames(), expectedFeatureNames);

	// check if getTrainingDataDict() contains the following randomly selected
	// data
	map<int, vector<string>> expectedTrainingDataDict = {
		{	 1, {"64", "2", "10.26", "2", "4", "diploid"}},
		{146,  {"56", "2", "9.01", "3", "7", "diploid"}},
		{ 28, {"57", "2", "12.13", "3", "6", "diploid"}},
		{ 55,	 {"61", "1", "2.4", "4", "10", "diploid"}}
	  };
	map<int, vector<string>> trainingDataDict = dt.getTrainingDataDict();
	for (auto data : expectedTrainingDataDict) {
		ASSERT_EQ(trainingDataDict[data.first], data.second);
	}

	// check if getFeaturesAndValuesDict() contains the following randomly
	// selected data
	map<string, vector<string>> expectedFeaturesAndValuesDict = {
		{	 "age",					 {"64", "57", "61"}},
		{	 "eet",					   {"2", "1", "NA"}},
		{	 "g2",	   {"10.26", "9.01", "12.13", "2.4"}},
		{	 "grade",						{"2", "3", "4"}},
		{"gleason",				{"4", "7", "6", "8", "10"}},
		{ "ploidy", {"aneuploid", "diploid", "tetraploid"}}
	  };
	map<string, vector<string>> featuresAndValuesDict = dt.getFeaturesAndValuesDict();

	for (const auto &data : expectedFeaturesAndValuesDict) {
		for (const auto &value : data.second) {
			ASSERT_TRUE(std::find(featuresAndValuesDict[data.first].begin(), featuresAndValuesDict[data.first].end(), value) !=
						featuresAndValuesDict[data.first].end());
		}
	}

	// check if _classNames is set correctly
	vector<string> expectedClassNames = {"0", "1"};
	ASSERT_EQ(dt._classNames, expectedClassNames);
}


TEST_F(DecisionTreeTest, findBoundedIntervalsForNumericFeatures)
{
	// Test case 1: Single feature with ">" condition only
	{
		vector<string> input			= {"g2>51.360000000000404"};
		vector<vector<string>> expected = {
			{"g2", ">", "51.360000000000404"}
		 };
		auto output = dt.findBoundedIntervalsForNumericFeatures(input);
		assertVectorsAlmostEqual(output, expected, "Test case 1 ");
	}

	// Test case 2: Single feature with multiple "<" conditions; should return the minimum value for "<"
	{
		vector<string> input			= {"g2<3.840000000000012", "g2<2.4"};
		vector<vector<string>> expected = {
			{"g2", "<", "2.4"}
		  };
		auto output = dt.findBoundedIntervalsForNumericFeatures(input);
		assertVectorsAlmostEqual(output, expected, "Test case 2 ");
	}

	// Test case 3: Multiple features with mixed "<" and ">" conditions
	{
		vector<string> input			= {"g2<3.840000000000012", "age<63.0", "g2>1.5", "age>18.0"};
		vector<vector<string>> expected = {
			{"age", "<",				 "63.0"},
			  {"age", ">",			   "18.0"},
			{ "g2", "<", "3.840000000000012"},
			  { "g2", ">",				  "1.5"}
		};
		auto output = dt.findBoundedIntervalsForNumericFeatures(input);
		assertVectorsAlmostEqual(output, expected, "Test case 3 ");
	}

	// Test case 4: Mixed conditions with "<" and ">" for the same feature
	{
		vector<string> input			= {"height<200.0", "height>150.0", "height>140.0", "height<180.0"};
		vector<vector<string>> expected = {
			{"height", "<", "180.0"},
			  {"height", ">", "150.0"}
		  };
		auto output = dt.findBoundedIntervalsForNumericFeatures(input);
		assertVectorsAlmostEqual(output, expected, "Test case 4 ");
	}

	// Test case 5: No conditions
	{
		vector<string> input			= {};
		vector<vector<string>> expected = {};
		auto output						= dt.findBoundedIntervalsForNumericFeatures(input);
		assertVectorsAlmostEqual(output, expected, "Test case 5 ");
	}

	// Test case 6: Single feature with only one bound ("<" or ">")
	{
		vector<string> input			= {"age>18.0"};
		vector<vector<string>> expected = {
			{"age", ">", "18.0"}
		};
		auto output = dt.findBoundedIntervalsForNumericFeatures(input);
		assertVectorsAlmostEqual(output, expected, "Test case 6a ");

		input	 = {"age<63.0"};
		expected = {
			{"age", "<", "63.0"}
		};
		output = dt.findBoundedIntervalsForNumericFeatures(input);
		assertVectorsAlmostEqual(output, expected, "Test case 6b ");
	}

	// Test case 7: Multiple "<" and ">" conditions for the same feature, where bounds overlap
	{
		vector<string> input			= {"weight>50.0", "weight>60.0", "weight<100.0", "weight<90.0"};
		vector<vector<string>> expected = {
			{"weight", "<", "90.0"},
			 {"weight", ">", "60.0"}
		};
		auto output = dt.findBoundedIntervalsForNumericFeatures(input);
		assertVectorsAlmostEqual(output, expected, "Test case 7 ");
	}

	// Test case 8: Large numbers and high precision
	{
		vector<string> input			= {"g2>51.360000000000404", "g2<3.840000000000012", "g2<2.4000000000001"};
		vector<vector<string>> expected = {
			{"g2", "<",	"2.4000000000001"},
			{"g2", ">", "51.360000000000404"}
		 };
		auto output = dt.findBoundedIntervalsForNumericFeatures(input);
		assertVectorsAlmostEqual(output, expected, "Test case 8 ");
	}
}
