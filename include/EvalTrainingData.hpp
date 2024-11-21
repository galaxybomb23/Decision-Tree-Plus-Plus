#ifndef EVAL_TRAINING_DATA_HPP
#define EVAL_TRAINING_DATA_HPP

#include "DecisionTree.hpp" // Include DecisionTree.hpp
#include "Utility.hpp"      // Include Utility.hpp

#include <iostream>
#include <map>
#include <string>

class EvalTrainingData : public DecisionTree {
  public:
    EvalTrainingData(std::map<std::string, std::string> kwargs); // Constructor
    ~EvalTrainingData();                                         // Destructor

    void evaluateTrainingData(); // Evaluate the training data
    double _dataQualityIndex;
    int _csvClassColumnIndex;

    void printDebugInformation(DecisionTree &trainingDT, const std::vector<std::string> &testing_samples);
    void printClassificationInfo(const std::vector<std::string> &which_classes,
                                 const std::map<std::string, double> &classification,
                                 const std::string &most_likely_class_label,
                                 std::shared_ptr<DecisionTreeNode> root_node);
    void displayConfusionMatrix(const std::map<int, std::map<std::string, int>> &confusion_matrix);
    double calculateDataQualityIndex(const std::map<std::string, std::map<std::string, int>> &confusion_matrix);
    void printDataQualityEvaluation(double data_quality_index);

  private:
    // Private members
};
;

#endif // EVAL_TRAINING_DATA_HPP