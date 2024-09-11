#ifndef TRAINING_DATA_GENERATOR_NUMERIC_HPP
#define TRAINING_DATA_GENERATOR_NUMERIC_HPP

#include <iostream>
#include <algorithm> // for std::shuffle
#include <map>
#include <Eigen/Dense> // For multivariate normal generation
#include <ctime>       // For seeding random shuffle#include <string>
#include <vector>
#include <algorithm> 
#include <random>
#include <regex>
#include <fstream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

class TrainingDataGeneratorNumeric
{
private:
    // Attributes
    std::string _output_csv_file;
    std::string _parameter_file;
    int _number_of_samples_per_class;
    int _debug;

    // Other attributes initialized in the constructor
    std::vector<std::string> _class_names;
    std::vector<std::string> _features_ordered;
    std::map<std::string, double> _class_names_and_priors;
    std::map<std::string, std::pair<double, double>> _features_with_value_range;
    std::map<std::string, std::map<std::string, std::vector<double>>> _classes_and_their_param_values;
public:
    TrainingDataGeneratorNumeric(std::map<std::string, std::string> kwargs);
    ~TrainingDataGeneratorNumeric();

    void ReadParameterFileNumeric();    // Read the parameter file for numeric data
    void GenerateTrainingDataNumeric(); // Generate the training data for numeric data

    // Helpers
    std::vector<Eigen::VectorXd> GenerateMultivariateSamples(const std::vector<double> &mean, const Eigen::MatrixXd &cov, int num_samples);

    // Getters
    std::string getOutputCsvFile() const;
    std::string getParameterFile() const;
    int getNumberOfSamplesPerClass() const;
    int getDebug() const;
    std::vector<std::string> getClassNames() const;
    std::vector<std::string> getFeaturesOrdered() const;
    std::map<std::string, double> getClassNamesAndPriors() const;
    std::map<std::string, std::pair<double, double>> getFeaturesWithValueRange() const;
    std::map<std::string, std::map<std::string, std::vector<double>>> getClassesAndTheirParamValues() const;
};

#endif // TRAINING_DATA_GENERATOR_NUMERIC_HPP
