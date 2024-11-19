#ifndef EVAL_TRAINING_DATA_HPP
#define EVAL_TRAINING_DATA_HPP

class EvalTrainingData {
  public:
    EvalTrainingData();  // Constructor
    ~EvalTrainingData(); // Destructor

    void evaluateTrainingData(); // Evaluate the training data
    double _dataQualityIndex;

  private:
    // Private members
};
;

#endif // EVAL_TRAINING_DATA_HPP