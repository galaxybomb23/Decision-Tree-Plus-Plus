#include <gtest/gtest.h>

#include "DecisionTree.hpp"

class DecisionTreeTest : public ::testing::Test
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
      {"training_datafile", "../test/resources/training_symbolic.csv"},
      {"csv_class_column_index", "1"},
      {"csv_columns_for_features", {2, 3, 4, 5}},
      {"max_depth_desired", "5"},
      {"entropy_threshold", "0.1"}
      //   ,{"symbolic_to_numeric_cardinality_threshold", "20"},
      //   {"number_of_histogram_bins", "10"},
      //   {"csv_cleanup_needed", "1"},
      //   {"debug1", "1"},
      //   {"debug2", "2"},
      //   {"debug3", "3"}
  };
  DecisionTree dt = DecisionTree(kwargs);
  DecisionTreeNode node =
      DecisionTreeNode("feature", 0.0, {0.0}, {"branch"}, dt, true);
};

TEST_F(DecisionTreeTest, CheckdtExists) { ASSERT_NE(&dt, nullptr); }

TEST_F(DecisionTreeTest, ConstructorInitializesNode)
{
  ASSERT_NE(&node, nullptr);
}

TEST_F(DecisionTreeTest, CheckParamsDt)
{
  std::map<std::string, std::string> kargs = {
      {"training_datafile", "../test/resources/stage3cancer.csv"},
      {"csv_class_column_index", "2"},
      {"csv_columns_for_features", {3,4,5,6,7,8}},
      {"max_depth_desired", "8"},
      {"entropy_threshold", "0.01"},
      {"symbolic_to_numeric_cardinality_threshold", "20"},
      {"number_of_histogram_bins", "10"},
      {"csv_cleanup_needed", "1"},
      {"debug1", "1"},
      {"debug2", "2"},
      {"debug3", "3"}};

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
  std::map<std::string, std::string> kargs = {
      {"training_datafile", "../test/resources/stage3cancer.csv"},
      {"csv_class_column_index", "2"},
      {"csv_columns_for_features", {3,4,5,6,7,8}},
      {"max_depth_desired", "8"},
      {"entropy_threshold", "0.01"},
      {"symbolic_to_numeric_cardinality_threshold", "20"},
      {"number_of_histogram_bins", "10"},
      {"csv_cleanup_needed", "1"},
      {"debug1", "1"},
      {"debug2", "2"},
      {"debug3", "3"}};

  DecisionTree dt = DecisionTree(kargs);

  ASSERT_NO_THROW(dt.getTrainingData());
  ASSERT_EQ(dt.getHowManyTotalTrainingSamples(), 146);

  std::vector<std::string> expectedFeatureNames = {
	"age", "eet", "g2", "grade", "gleason", "ploidy"};
  ASSERT_EQ(dt.getFeatureNames(), expectedFeatureNames);

  // check if getTrainingDataDict() contains the following randomly selected
  // data
  std::map<int, std::vector<std::string>> expectedTrainingDataDict = {
      {1, {"64", "2", "10.26", "2", "4", "diploid"}},
      {146, {"56", "2", "9.01", "3", "7", "diploid"}},
      {28, {"57", "2", "12.13", "3", "6", "diploid"}},
      {55, {"61", "1", "2.4", "4", "10", "diploid"}}};
  std::map<int, std::vector<std::string>> trainingDataDict =
      dt.getTrainingDataDict();
  for (auto data : expectedTrainingDataDict)
  {
    ASSERT_EQ(trainingDataDict[data.first], data.second);
  }

  // check if getFeaturesAndValuesDict() contains the following randomly
  // selected data
  std::map<std::string, std::vector<std::string>> expectedFeaturesAndValuesDict = {
      {"age", {"64", "57", "61"}},
      {"eet", {"2", "1", "NA"}},
      {"g2", {"10.26", "9.01", "12.13", "2.4"}},
      {"grade", {"2", "3", "4"}},
      {"gleason", {"4", "7", "6", "8", "10"}},
      {"ploidy", {"aneuploid", "diploid", "tetraploid"}}};
  std::map<std::string, std::vector<std::string>> featuresAndValuesDict =
      dt.getFeaturesAndValuesDict();
	
  for (const auto &data : expectedFeaturesAndValuesDict)
  {
    for (const auto &value : data.second)
    {
      ASSERT_TRUE(std::find(featuresAndValuesDict[data.first].begin(),
                            featuresAndValuesDict[data.first].end(),
                            value) != featuresAndValuesDict[data.first].end());
    }
  }

  // check if _classNames is set correctly
  std::vector<std::string> expectedClassNames = {"0", "1"};
  ASSERT_EQ(dt._classNames, expectedClassNames);
}
