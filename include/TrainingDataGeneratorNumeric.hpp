#ifndef TRAINING_DATA_GENERATOR_NUMERIC_HPP
#define TRAINING_DATA_GENERATOR_NUMERIC_HPP

// Include
#include "Common.hpp"

#include <Eigen/Dense> // For multivariate normal generation
#include <ctime> // For seeding random shuffle
#include <fstream>
#include <random>
#include <regex>

using Eigen::MatrixXd;
using Eigen::VectorXd;

class TrainingDataGeneratorNumeric {
  private:
    // Attributes
    string _outputCsvFile;
    string _parameterFile;
    int _numberOfSamplesPerClass;
    int _debug;

    // Other attributes initialized in the constructor
    vector<string> _classNames;
    vector<string> _featuresOrdered;
    map<string, double> _classNamesAndPriors;
    map<string, pair<double, double>> _featuresWithValueRange;
    map<string, map<string, vector<double>>> _classesAndTheirParamValues;

  public:
    TrainingDataGeneratorNumeric(map<string, string> kwargs);
    ~TrainingDataGeneratorNumeric();

    void ReadParameterFileNumeric();    // Read the parameter file for numeric data
    void GenerateTrainingDataNumeric(); // Generate the training data for numeric data

    // Helpers
    vector<Eigen::VectorXd>
    GenerateMultivariateSamples(const vector<double> &mean, const Eigen::MatrixXd &cov, int numSamples);

    // Getters
    string getOutputCsvFile() const;
    string getParameterFile() const;
    int getNumberOfSamplesPerClass() const;
    int getDebug() const;
    vector<string> getClassNames() const;
    vector<string> getFeaturesOrdered() const;
    map<string, double> getClassNamesAndPriors() const;
    map<string, pair<double, double>> getFeaturesWithValueRange() const;
    map<string, map<string, vector<double>>> getClassesAndTheirParamValues() const;
};

#endif // TRAINING_DATA_GENERATOR_NUMERIC_HPP
