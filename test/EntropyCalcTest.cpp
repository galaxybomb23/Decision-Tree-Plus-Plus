#include "DecisionTree.hpp"

#include <gtest/gtest.h>

class EntropyCalcTest : public ::testing::Test {
  protected:
	std::unique_ptr<DecisionTree> dtS; // Symbolic DecisionTree
	std::unique_ptr<DecisionTree> dtN; // Numeric DecisionTree
	void SetUp() override
	{
		map<string, string> kwargsS = {
			// Symbolic kwargs
			{		 "training_datafile", "../test/resources/training_symbolic.csv"},
			{	 "csv_class_column_index",									   "1"},
			{"csv_columns_for_features",								 {2, 3, 4, 5}},
			{		 "max_depth_desired",									   "5"},
			{		 "entropy_threshold",									 "0.1"}
		};

		map<string, string> kwargsN = {
			// Numeric kwargs
			{		 "training_datafile", "../test/resources/stage3cancer.csv"},
			{	 "csv_class_column_index",								   "2"},
			{"csv_columns_for_features",					 {3, 4, 5, 6, 7, 8}},
			{		 "max_depth_desired",								  "8"},
			{		 "entropy_threshold",								  "0.01"}
		 };

		dtS = std::make_unique<DecisionTree>(kwargsS); // Initialize the DecisionTree
		dtS->getTrainingData();

		dtN = std::make_unique<DecisionTree>(kwargsN); // Initialize the DecisionTree
		dtN->getTrainingData();
	}

	void TearDown() override
	{
		dtS.reset(); // Reset the DecisionTree
	}
};

TEST_F(EntropyCalcTest, CheckdtExists)
{
	ASSERT_NE(&dtS, nullptr);
}

// TODO: Write these tests
// ------ Symbolic Data Tests ------

TEST_F(EntropyCalcTest, classEntropyOnPriorsSymbolic)
{

}

TEST_F(EntropyCalcTest, entropyScannerForANumericFeatureSymbolic)
{

}

TEST_F(EntropyCalcTest, classEntropyForLessThanThresholdForFeatureSymbolic)
{

}

TEST_F(EntropyCalcTest, classEntropyForGreaterThanThresholdForFeatureSymbolic)
{

}

TEST_F(EntropyCalcTest, classEntropyForAGivenSequenceOfFeaturesAndValuesOrThresholdsSymbolic)
{

}

// ------ Numeric Data Tests ------

TEST_F(EntropyCalcTest, classEntropyOnPriorsNumeric)
{

}

TEST_F(EntropyCalcTest, entropyScannerForANumericFeatureNumeric)
{

}

TEST_F(EntropyCalcTest, classEntropyForLessThanThresholdForFeatureNumeric)
{

}

TEST_F(EntropyCalcTest, classEntropyForGreaterThanThresholdForFeatureNumeric)
{

}

TEST_F(EntropyCalcTest, classEntropyForAGivenSequenceOfFeaturesAndValuesOrThresholdsNumeric)
{

}