#ifndef EVAL_TRAINING_DATA_HPP
#define EVAL_TRAINING_DATA_HPP

#include "DecisionTree.hpp" // Include DecisionTree.hpp
#include "Utility.hpp"      // Include Utility.hpp

#include <iostream>
#include <map>
#include <string>

/**
 * @class EvalTrainingData
 * @brief A class that evaluates training data for a decision tree.
 *
 * This class inherits from the DecisionTree class and provides methods to evaluate
 * the quality of training data, print debug information, and display evaluation results.
 */
class EvalTrainingData : public DecisionTree {
  public:
    EvalTrainingData(std::map<std::string, std::string> kwargs); // Constructor
    ~EvalTrainingData();                                         // Destructor

    double evaluateTrainingData(); // Evaluate the training data
    void evaluationResults(std::vector<std::string> testing_samples,
                           std::map<int, std::vector<std::string>> allTrainingData,
                           DecisionTreeNode* root_node,
                           std::map<int, std::map<std::string, int>> confusion_matrix,
                           bool evalDebug);
    double _dataQualityIndex;
    int _csvClassColumnIndex;

    void printDebugInformation(DecisionTree &trainingDT, const std::vector<std::string> &testing_samples);
    void printClassificationInfo(const std::vector<std::string> &which_classes,
                                 const std::map<std::string, std::string> &classification,
                                 const std::string &most_likely_class_label,
                                 DecisionTreeNode* root_node);
    void displayConfusionMatrix(const std::map<int, std::map<std::string, int>> &confusion_matrix);
    double calculateDataQualityIndex(const std::map<int, std::map<std::string, int>> &confusion_matrix);
    void printDataQualityEvaluation(double data_quality_index);

  private:
    // Private members
};
;

#endif // EVAL_TRAINING_DATA_HPP