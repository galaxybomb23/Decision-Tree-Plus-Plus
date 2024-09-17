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
#include <sstream>


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
    std::map<std::string, std::map<std::string, std::vector<double>>> _biasDict;
    std::vector<std::string> _classNames;
    std::vector<double> _classPriors;

    // vecToString for string and double
    template <typename T>
    std::string vecToString(const std::vector<T> &vec)
    {
        std::stringstream ss;
        for (const auto &i : vec)
        {
            ss << i << ' ';
        }
        return ss.str();
    }

public:
    TrainingDataGeneratorSymbolic(std::map<std::string, std::string> kwargs);
    ~TrainingDataGeneratorSymbolic();

    void ReadParameterFileSymbolic();    // Read the parameter file for symbolic data
    void GenerateTrainingDataSymbolic(); // Generate the training data for symbolic data

    // helpers
    std::vector<std::string> filterAndClean(const std::string &pattern, const std::vector<std::string> &input);
    std::vector<std::string> splitByRegex(const std::string &input, const std::string &pattern);

        // getters
        std::vector<double> getClassPriors()
    {
        return _classPriors;
    }
    std::vector<std::string> getClassNames() { return _classNames; }
    std::map<std::string, std::vector<double>> getFeaturesAndValuesDict() { return _featuresAndValuesDict; }
    std::map<std::string, std::map<std::string, std::vector<double>>> getBiasDict() { return _biasDict; }
    std::string getOutputDatafile() { return _outputDatafile; }
    std::string getParameterFile() { return _parameterFile; }
    int getNumberOfTrainingSamples() { return _numberOfTrainingSamples; }
    int getWriteToFile() { return _writeToFile; }
    int getDebug1() { return _debug1; }
    int getDebug2() { return _debug2; }
};

#endif // TRAINING_DATA_GENERATOR_SYMBOLIC_HPP