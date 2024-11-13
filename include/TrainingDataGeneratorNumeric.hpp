#ifndef TRAINING_DATA_GENERATOR_NUMERIC_HPP
#define TRAINING_DATA_GENERATOR_NUMERIC_HPP

#include <Eigen/Dense> // For multivariate normal generation
#include <algorithm>   // for std::shuffle
#include <algorithm>
#include <ctime> // For seeding random shuffle
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <regex>
#include <string>
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;

class TrainingDataGeneratorNumeric {
  private:
	// Attributes
	std::string _outputCsvFile;
	std::string _parameterFile;
	int _numberOfSamplesPerClass;
	int _debug;

	// Other attributes initialized in the constructor
	std::vector<std::string> _classNames;
	std::vector<std::string> _featuresOrdered;
	std::map<std::string, double> _classNamesAndPriors;
	std::map<std::string, std::pair<double, double>> _featuresWithValueRange;
	std::map<std::string, std::map<std::string, std::vector<double>>> _classesAndTheirParamValues;

  public:
	TrainingDataGeneratorNumeric(std::map<std::string, std::string> kwargs);
	~TrainingDataGeneratorNumeric();

	void ReadParameterFileNumeric();	// Read the parameter file for numeric data
	void GenerateTrainingDataNumeric(); // Generate the training data for numeric data

	// Helpers
	std::vector<Eigen::VectorXd>
	GenerateMultivariateSamples(const std::vector<double> &mean, const Eigen::MatrixXd &cov, int numSamples);

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
