#ifndef EVAL_TRAINING_DATA_HPP
#define EVAL_TRAINING_DATA_HPP

#include <iostream>
#include <string>
#include <map>
#include "DecisionTree.hpp" // Include DecisionTree.hpp
#include "Utility.hpp"      // Include Utility.hpp

class EvalTrainingData : public DecisionTree
{
public:
    EvalTrainingData(std::map<std::string, std::string> kwargs); // Constructor
    ~EvalTrainingData();                                         // Destructor

    void evaluateTrainingData(); // Evaluate the training data
    double _dataQualityIndex;

private:
    // Private members
};
;

#endif // EVAL_TRAINING_DATA_HPP