#ifndef TRAINING_DATA_GENERATOR_NUMERIC_HPP
#define TRAINING_DATA_GENERATOR_NUMERIC_HPP

class TrainingDataGeneratorNumeric
{
private:
    /* data */
public:
    TrainingDataGeneratorNumeric(/* args */);
    ~TrainingDataGeneratorNumeric();

    void ReadParameterFileNumeric();    // Read the parameter file for numeric data
    void GenerateTrainingDataNumeric(); // Generate the training data for numeric data
};

#endif // TRAINING_DATA_GENERATOR_NUMERIC_HPP
