#ifndef TRAINING_DATA_GENERATOR_SYMBOLIC_HPP
#define TRAINING_DATA_GENERATOR_SYMBOLIC_HPP

// Include
#include "Common.hpp"

#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>

class TrainingDataGeneratorSymbolic {
  private:
    string _outputDatafile;
    string _parameterFile;
    int _numberOfTrainingSamples;
    int _writeToFile;
    int _debug1;
    int _debug2;
    map<int, vector<string>> _trainingSampleRecords;
    map<string, vector<string>> _featuresAndValuesDict;
    map<string, map<string, vector<string>>> _biasDict;
    vector<string> _classNames;
    vector<double> _classPriors;

    // vecToString for string and double
    template <typename T> string vecToString(const vector<T> &vec)
    {
        std::stringstream ss;
        for (const auto &i : vec) {
            ss << i << ' ';
        }
        return ss.str();
    }

  public:
    TrainingDataGeneratorSymbolic(map<string, string> kwargs);
    ~TrainingDataGeneratorSymbolic();

    void ReadParameterFileSymbolic();    // Read the parameter file for symbolic data
    void GenerateTrainingDataSymbolic(); // Generate the training data for symbolic data
    void WriteTrainingDataToFile();

    // helpers
    string sampleIndex(const string &sampleName);
    vector<string> filterAndClean(const string &pattern, const vector<string> &input);
    vector<string> splitByRegex(const string &input, const string &pattern);

    // getters
    vector<double> getClassPriors() { return _classPriors; }
    vector<string> getClassNames() { return _classNames; }
    map<string, vector<string>> getFeaturesAndValuesDict() { return _featuresAndValuesDict; }
    map<string, map<string, vector<string>>> getBiasDict() { return _biasDict; }
    string getOutputDatafile() { return _outputDatafile; }
    string getParameterFile() { return _parameterFile; }
    double randomDouble(double upper, double lower);
    int getNumberOfTrainingSamples() { return _numberOfTrainingSamples; }
    int getWriteToFile() { return _writeToFile; }
    int getDebug1() { return _debug1; }
    int getDebug2() { return _debug2; }
    map<int, vector<string>> getTrainingSampleRecords() { return _trainingSampleRecords; }
};

#endif // TRAINING_DATA_GENERATOR_SYMBOLIC_HPP