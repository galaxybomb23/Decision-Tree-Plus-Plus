#ifndef TRAINING_DATA_GENERATOR_SYMBOLIC_HPP
#define TRAINING_DATA_GENERATOR_SYMBOLIC_HPP

#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <regex>

class TrainingDataGeneratorSymbolic
{
private:
    std::string _outputDatafile;
    std::string _parameterFile;
    int _numberOfTrainingSamples;
    int _writeToFile;
    int _debug1;
    int _debug2;
    std::map<std::string, std::vector<double>> _trainingSampleRecords;
    std::map<std::string, std::vector<double>> _featuresAndValuesDict;
    std::map<std::string, double> _biasDict;
    std::vector<std::string> _classNames;
    std::vector<double> _classPriors;

public:
    TrainingDataGeneratorSymbolic(std::map<std::string, std::string> kwargs);
    ~TrainingDataGeneratorSymbolic();

    void ReadParameterFileSymbolic();    // Read the parameter file for symbolic data
    void GenerateTrainingDataSymbolic(); // Generate the training data for symbolic data

    // helpers
    std::vector<std::string> filterAndClean(const std::vector<std::string> &input, const std::regex &filterPattern);
};

#endif // TRAINING_DATA_GENERATOR_SYMBOLIC_HPP