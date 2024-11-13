#include "TrainingDataGeneratorNumeric.hpp"

TrainingDataGeneratorNumeric::TrainingDataGeneratorNumeric(std::map<std::string, std::string> kwargs)
{
	std::vector<std::string> allowedKeys = {
		"output_csv_file", "parameter_file", "number_of_samples_per_class", "debug"};

	if (kwargs.empty()) {
		throw std::invalid_argument("Missing parameters.");
	}

	// Checking passed keyword arguments
	for (const auto &kv : kwargs) {
		// see if the key is in the allowed keys
		if (std::find(allowedKeys.begin(), allowedKeys.end(), kv.first) == allowedKeys.end()) {
			throw std::invalid_argument(kv.first + ": Wrong keyword used --- check spelling");
		}
	}

	// Set default values
	_debug = 0;

	// go through the passed keyword arguments
	for (const auto &kv : kwargs) {
		const std::string &key	 = kv.first;
		const std::string &value = kv.second;

		if (key == "output_csv_file") {
			_outputCsvFile = value;
		}
		else if (key == "parameter_file") {
			_parameterFile = value;
		}
		else if (key == "number_of_samples_per_class") {
			_numberOfSamplesPerClass = std::stoi(value);
		}
		else if (key == "debug") {
			_debug = std::stoi(value);
		}
	}
}

TrainingDataGeneratorNumeric::~TrainingDataGeneratorNumeric() {}

void TrainingDataGeneratorNumeric::ReadParameterFileNumeric()
{
	std::vector<std::string> classNames;
	std::map<std::string, double> classNamesAndPriors;
	std::map<std::string, std::pair<double, double>> featuresWithValueRange;
	std::map<std::string, std::map<std::string, std::vector<double>>> classesAndTheirParamValues;
	std::vector<std::string> featuresOrdered;

	std::regex fpOrSciNotation("[+-]?\\ *\\d+(\\.\\d*)?|\\.\\d+([eE][+-]?\\d+)?");

	// Read the parameter file for numeric data
	std::ifstream file(_parameterFile);
	std::string params((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
	if (params.empty()) {
		throw std::invalid_argument("Empty file.");
	}

	/*
	 *  Regex search for class names and priors
	 */
	std::regex classRegex("class names: ([\\w\\s+]+)\\W*class priors: ([\\d.\\s+]+)", std::regex::icase);
	std::smatch classMatches;

	if (std::regex_search(params, classMatches, classRegex)) {
		// Extract class names and priors, make them a vector and a map respectively
		std::string classNamesStr  = classMatches[1].str();
		std::string classPriorsStr = classMatches[2].str();
		std::istringstream classNamesStream(classNamesStr);
		std::istringstream classPriorsStream(classPriorsStr);

		std::string name, prior;
		std::vector<double> classPriors;

		// Split classNamesStream by ' ' delimiter and add to classNames
		while (classNamesStream >> name) {
			if (!name.empty()) {
				classNames.push_back(name);
			}
		}

		// Split classPriorsStream by ' ' delimiter and add to classPriors
		while (classPriorsStream >> prior) {
			if (!prior.empty()) {
				classPriors.push_back(std::stod(prior)); // Convert the string to double
			}
		}

		// Combine class names and priors into the map between class names and priors
		for (size_t i = 0; i < classNames.size(); ++i) {
			classNamesAndPriors[classNames[i]] = classPriors[i];
		}
	}
	else {
		throw std::runtime_error("Could not find 'class names' and 'class priors' in the parameter file.");
	}

	if (_debug) {
		std::cout << "Class names and priors: " << std::endl;
		for (const auto &kv : classNamesAndPriors) {
			std::cout << kv.first << " : " << kv.second << std::endl;
		}
	}

	/*
	 *  Regex search for feature names and value ranges
	 */
	std::regex featureRegex("feature name: (\\w+)\\W*value range:\\s*([\\d. -]+)", std::regex::icase);
	std::smatch featureMatches;
	auto featureBegin = std::sregex_iterator(params.begin(), params.end(), featureRegex);
	auto featureEnd	  = std::sregex_iterator();

	for (std::sregex_iterator i = featureBegin; i != featureEnd; ++i) {
		// Same as above, extract feature names and value ranges
		std::smatch match		  = *i;
		std::string featureName	  = match[1].str(); // feature name is the first match group
		std::string valueRangeStr = match[2].str(); // value range is the second match group

		std::istringstream valueRangeStream(valueRangeStr);
		std::vector<double> valueRange;
		std::string value;

		// Split valueRangeStream by '-' delimiter and add to featuresWithValueRange
		while (std::getline(valueRangeStream, value, '-')) {
			// std::cout << value << std::endl;
			if (!value.empty()) {
				valueRange.push_back(std::stod(value));
			}
		}

		// Ensure valueRange has exactly two values and add to featuresWithValueRange
		if (valueRange.size() == 2) {
			featuresWithValueRange[featureName] = std::make_pair(valueRange[0], valueRange[1]);
		}

		// Add feature name to the list of features
		featuresOrdered.push_back(featureName);
	}

	if (_debug) {
		std::cout << "Features and their value ranges: " << std::endl;
		for (const auto &kv : featuresWithValueRange) {
			// std::cout << kv.first << " : [" << kv.second.first << ", " << kv.second.second << "]" << std::endl;
		}
	}

	// Add class names and their parameter values to classesAndTheirParamValues
	for (int i = 0; i < classNames.size(); i++) {
		// std::cout << "Adding [" << classNames[i] << "]" << std::endl;
		classesAndTheirParamValues[classNames[i]] = {};
	}

	/*
	 *  Regex search for class names and their parameter values
	 */
	std::regex classParamsRegex(R"(params for class:\s+(\w+)\W*mean:\s*([\d\s.-]+)\W*covariance:\s*([\d\s.-]+))",
								std::regex_constants::icase);
	std::smatch match;

	std::string::const_iterator searchStart(params.cbegin());

	// Search for class names and their parameter values, go through all the matches
	while (std::regex_search(searchStart, params.cend(), match, classParamsRegex)) {
		std::string className		 = match[1].str();
		std::string meanString		 = match[2].str();
		std::string covarianceString = match[3].str();

		std::vector<double> classMean;
		std::istringstream meanStream(meanString);
		double val;

		// Split meanString by ' ' delimiter and add to classMean
		while (meanStream >> val) {
			classMean.push_back(val);
		}

		std::istringstream covarianceStream(covarianceString);
		std::string line;
		std::vector<std::vector<double>> covarianceMatrix;

		// Split covarianceString by '\n' delimiter and add to covarianceMatrix
		while (std::getline(covarianceStream, line)) {
			std::istringstream rowStream(line);
			std::vector<double> row;
			double value;
			while (rowStream >> value) {
				// First add value to the row
				row.push_back(value);
			}
			if (!row.empty()) {
				// Then add the row to the covariance matrix once full
				covarianceMatrix.push_back(row);
			}
		}

		// Add class name, mean and covariance to classesAndTheirParamValues
		classesAndTheirParamValues[className]["mean"]		= classMean;
		classesAndTheirParamValues[className]["covariance"] = std::vector<double>();

		// Flatten the covariance matrix into a single vector
		for (const auto &row : covarianceMatrix) {
			classesAndTheirParamValues[className]["covariance"].insert(
				classesAndTheirParamValues[className]["covariance"].end(), row.begin(), row.end());
		}

		// Update searchStart to the end of the current match
		searchStart = match.suffix().first;
	}

	if (_debug) {
		std::cout << "Classes and their parameter values: " << std::endl;
		for (const auto &kv : classesAndTheirParamValues) {
			std::cout << kv.first << " : " << std::endl;
			for (const auto &kv2 : kv.second) {
				std::cout << kv2.first << " : ";
				for (const auto &val : kv2.second) {
					std::cout << val << " ";
				}
				std::cout << std::endl;
			}
		}
	}

	// Update the class attributes
	_classNames					= classNames;
	_classNamesAndPriors		= classNamesAndPriors;
	_featuresWithValueRange		= featuresWithValueRange;
	_classesAndTheirParamValues = classesAndTheirParamValues;
	_featuresOrdered			= featuresOrdered;
}

// Function to generate multivariate normal samples, since Eigen does not have a built-in function for this
// Original Python implementation uses numpy.random.multivariate_normal, but this is not available in cpp
// We will use the Cholesky decomposition method to generate multivariate normal samples
std::vector<VectorXd> TrainingDataGeneratorNumeric::GenerateMultivariateSamples(const std::vector<double> &mean,
																				const MatrixXd &cov,
																				int numSamples)
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::normal_distribution<> dist(0, 1);

	// Generate multivariate normal samples
	std::vector<VectorXd> samples;
	Eigen::LLT<MatrixXd> llt(cov);
	MatrixXd L = llt.matrixL(); // Cholesky decomposition

	// Add a sample from the multivariate normal distribution to the samples vector
	// The sample is generated by multiplying the Cholesky decomposition (L) with a vector of random samples
	for (int i = 0; i < numSamples; ++i) {
		VectorXd z(mean.size());
		for (size_t j = 0; j < mean.size(); ++j) {
			z(j) = dist(gen);
		}
		VectorXd sample = L * z + VectorXd::Map(mean.data(), mean.size());
		samples.push_back(sample);
	}

	return samples;
}

void TrainingDataGeneratorNumeric::GenerateTrainingDataNumeric()
{
	std::map<std::string, std::vector<VectorXd>> samplesForClass;

	// Generate samples for each class
	for (const auto &classEntry : _classesAndTheirParamValues) {
		// Get class name, mean and covariance
		std::string className		= classEntry.first;
		std::vector<double> mean	= classEntry.second.at("mean");
		std::vector<double> covFlat = classEntry.second.at("covariance");

		// Convert flat covariance back to matrix
		int dim = mean.size();
		MatrixXd covMatrix(dim, dim);
		for (int i = 0; i < dim; ++i) {
			for (int j = 0; j < dim; ++j) {
				covMatrix(i, j) = covFlat[i * dim + j];
			}
		}

		// Generate multivariate normal samples
		samplesForClass[className] = GenerateMultivariateSamples(mean, covMatrix, _numberOfSamplesPerClass);
	}

	// Store data records to be written to the CSV file
	// For each class, for each sample, create a data record
	std::vector<std::string> dataRecords;
	for (const auto &classEntry : samplesForClass) {
		// For each sample in the class, create a data record
		const std::string &className = classEntry.first;
		for (int sampleIndex = 0; sampleIndex < _numberOfSamplesPerClass; ++sampleIndex) {
			std::string dataRecord = className + ",";
			// For each feature in the sample, add to the data record
			for (int featureIndex = 0; featureIndex < classEntry.second[sampleIndex].size(); ++featureIndex) {
				dataRecord += std::to_string(classEntry.second[sampleIndex](featureIndex));

				// Add comma if not the last feature
				if (featureIndex < classEntry.second[sampleIndex].size() - 1) {
					dataRecord += ",";
				}
			}
			if (_debug) {
				std::cout << "Data record: " << dataRecord << std::endl;
			}
			dataRecords.push_back(dataRecord);
		}
	}

	// Shuffle the data records randomly
	std::srand(unsigned(std::time(0)));
	std::shuffle(dataRecords.begin(), dataRecords.end(), std::mt19937{std::random_device{}()});

	// Prepare the CSV output
	std::ofstream file(_outputCsvFile);
	file << "\"\",class_name,"
		 << std::accumulate(_featuresOrdered.begin(),
							_featuresOrdered.end(),
							std::string(),
							[](const std::string &acc, const std::string &feature) {
								return acc + (acc.empty() ? "" : ",") + feature;
							})
		 << "\n";

	// Write the data records to the CSV file
	for (size_t i = 0; i < dataRecords.size(); ++i) {
		file << (i + 1) << "," << dataRecords[i] << "\n";
	}
	file.close();
}

/*
 * Getters
 */
std::string TrainingDataGeneratorNumeric::getOutputCsvFile() const
{
	return _outputCsvFile;
}

std::string TrainingDataGeneratorNumeric::getParameterFile() const
{
	return _parameterFile;
}

int TrainingDataGeneratorNumeric::getNumberOfSamplesPerClass() const
{
	return _numberOfSamplesPerClass;
}

int TrainingDataGeneratorNumeric::getDebug() const
{
	return _debug;
}

std::vector<std::string> TrainingDataGeneratorNumeric::getClassNames() const
{
	return _classNames;
}

std::vector<std::string> TrainingDataGeneratorNumeric::getFeaturesOrdered() const
{
	return _featuresOrdered;
}

std::map<std::string, double> TrainingDataGeneratorNumeric::getClassNamesAndPriors() const
{
	return _classNamesAndPriors;
}

std::map<std::string, std::pair<double, double>> TrainingDataGeneratorNumeric::getFeaturesWithValueRange() const
{
	return _featuresWithValueRange;
}

std::map<std::string, std::map<std::string, std::vector<double>>>
TrainingDataGeneratorNumeric::getClassesAndTheirParamValues() const
{
	return _classesAndTheirParamValues;
}