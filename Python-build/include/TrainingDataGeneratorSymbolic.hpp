#ifndef TRAINING_DATA_GENERATOR_SYMBOLIC_HPP
#define TRAINING_DATA_GENERATOR_SYMBOLIC_HPP

class TrainingDataGeneratorSymbolic
{
private:
    /* data */
public:
    TrainingDataGeneratorSymbolic(/* args */);
    ~TrainingDataGeneratorSymbolic();

    void ReadParameterFileSymbolic();    // Read the parameter file for symbolic data
    void GenerateTrainingDataSymbolic(); // Generate the training data for symbolic data
};

#endif // TRAINING_DATA_GENERATOR_SYMBOLIC_HPP