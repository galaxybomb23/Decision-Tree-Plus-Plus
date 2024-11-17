#ifndef TRAINING_DATA_GENERATOR_SYMBOLIC_HPP
#define TRAINING_DATA_GENERATOR_SYMBOLIC_HPP

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>


class TrainingDataGeneratorSymbolic {
  private:
	std::string _outputDatafile;
	std::string _parameterFile;
	int _numberOfTrainingSamples;
	int _writeToFile;
	int _debug1;
	int _debug2;
	std::map<int, std::vector<std::string>> _trainingSampleRecords;
	std::map<std::string, std::vector<std::string>> _featuresAndValuesDict;
	std::map<std::string, std::map<std::string, std::vector<std::string>>> _biasDict;
	std::vector<std::string> _classNames;
	std::vector<double> _classPriors;

	// vecToString for string and double
	template <typename T> std::string vecToString(const std::vector<T> &vec)
	{
		std::stringstream ss;
		for (const auto &i : vec) {
			ss << i << ' ';
		}
		return ss.str();
	}

  public:
	TrainingDataGeneratorSymbolic(std::map<std::string, std::string> kwargs);
	~TrainingDataGeneratorSymbolic();

	void ReadParameterFileSymbolic();	 // Read the parameter file for symbolic data
	void GenerateTrainingDataSymbolic(); // Generate the training data for symbolic data
	void WriteTrainingDataToFile();

	// helpers
	std::string sampleIndex(const std::string &sampleName);
	std::vector<std::string> filterAndClean(const std::string &pattern, const std::vector<std::string> &input);
	std::vector<std::string> splitByRegex(const std::string &input, const std::string &pattern);

	// getters
	std::vector<double> getClassPriors() { return _classPriors; }
	std::vector<std::string> getClassNames() { return _classNames; }
	std::map<std::string, std::vector<std::string>> getFeaturesAndValuesDict() { return _featuresAndValuesDict; }
	std::map<std::string, std::map<std::string, std::vector<std::string>>> getBiasDict() { return _biasDict; }
	std::string getOutputDatafile() { return _outputDatafile; }
	std::string getParameterFile() { return _parameterFile; }
	double randomDouble(double upper, double lower);
	int getNumberOfTrainingSamples() { return _numberOfTrainingSamples; }
	int getWriteToFile() { return _writeToFile; }
	int getDebug1() { return _debug1; }
	int getDebug2() { return _debug2; }
	std::map<int, std::vector<std::string>> getTrainingSampleRecords() { return _trainingSampleRecords; }
};

#endif // TRAINING_DATA_GENERATOR_SYMBOLIC_HPP