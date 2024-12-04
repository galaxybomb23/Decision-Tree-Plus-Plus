#include "TrainingDataGeneratorNumeric.hpp"

/**
 * @brief Constructor for TrainingDataGeneratorNumeric class.
 *
 * This constructor initializes the TrainingDataGeneratorNumeric object with the provided keyword arguments.
 * It validates the keys in the provided map and sets the corresponding member variables.
 *
 * @param kwargs A map containing the keyword arguments for initialization.
 *               Allowed keys are:
 *               - "output_csv_file": Path to the output CSV file.
 *               - "parameter_file": Path to the parameter file.
 *               - "number_of_samples_per_class": Number of samples per class (as a string, will be converted to an
 * integer).
 *               - "debug": Debug flag (as a string, will be converted to an integer).
 *
 * @throws std::invalid_argument if the kwargs map is empty or contains invalid keys.
 */
TrainingDataGeneratorNumeric::TrainingDataGeneratorNumeric(map<string, string> kwargs)
{
    vector<string> allowedKeys = {"output_csv_file", "parameter_file", "number_of_samples_per_class", "debug"};

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
        const string &key   = kv.first;
        const string &value = kv.second;

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

/**
 * @brief Reads and parses a parameter file for numeric data.
 *
 * This function reads a parameter file specified by `_parameterFile` and extracts
 * class names, class priors, feature names, feature value ranges, and class parameter values
 * (mean and covariance). The extracted data is stored in the corresponding class attributes.
 *
 * @throws std::invalid_argument if the parameter file is empty.
 * @throws std::runtime_error if the required information (class names and priors) is not found in the parameter file.
 *
 * The extracted data is stored in the following class attributes:
 * - `_classNames`: A vector of class names.
 * - `_classNamesAndPriors`: A map of class names to their corresponding priors.
 * - `_featuresWithValueRange`: A map of feature names to their value ranges (min and max values).
 * - `_classesAndTheirParamValues`: A map of class names to their parameter values (mean and covariance).
 * - `_featuresOrdered`: A vector of feature names in the order they appear in the parameter file.
 */
void TrainingDataGeneratorNumeric::ReadParameterFileNumeric()
{
    vector<string> classNames;
    map<string, double> classNamesAndPriors;
    map<string, pair<double, double>> featuresWithValueRange;
    map<string, map<string, vector<double>>> classesAndTheirParamValues;
    vector<string> featuresOrdered;

    std::regex fpOrSciNotation("[+-]?\\ *\\d+(\\.\\d*)?|\\.\\d+([eE][+-]?\\d+)?");

    // Read the parameter file for numeric data
    std::ifstream file(_parameterFile);
    string params((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
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
        string classNamesStr  = classMatches[1].str();
        string classPriorsStr = classMatches[2].str();
        std::istringstream classNamesStream(classNamesStr);
        std::istringstream classPriorsStream(classPriorsStr);

        string name, prior;
        vector<double> classPriors;

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
        cout << "Class names and priors: " << endl;
        for (const auto &kv : classNamesAndPriors) {
            cout << kv.first << " : " << kv.second << endl;
        }
    }

    /*
     *  Regex search for feature names and value ranges
     */
    std::regex featureRegex("feature name: (\\w+)\\W*value range:\\s*([\\d. -]+)", std::regex::icase);
    std::smatch featureMatches;
    auto featureBegin = std::sregex_iterator(params.begin(), params.end(), featureRegex);
    auto featureEnd   = std::sregex_iterator();

    for (std::sregex_iterator i = featureBegin; i != featureEnd; ++i) {
        // Same as above, extract feature names and value ranges
        std::smatch match    = *i;
        string featureName   = match[1].str(); // feature name is the first match group
        string valueRangeStr = match[2].str(); // value range is the second match group

        std::istringstream valueRangeStream(valueRangeStr);
        vector<double> valueRange;
        string value;

        // Split valueRangeStream by '-' delimiter and add to featuresWithValueRange
        while (std::getline(valueRangeStream, value, '-')) {
            // cout << value << endl;
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
        cout << "Features and their value ranges: " << endl;
        for (const auto &kv : featuresWithValueRange) {
            // cout << kv.first << " : [" << kv.second.first << ", " << kv.second.second << "]" << endl;
        }
    }

    // Add class names and their parameter values to classesAndTheirParamValues
    for (int i = 0; i < classNames.size(); i++) {
        // cout << "Adding [" << classNames[i] << "]" << endl;
        classesAndTheirParamValues[classNames[i]] = {};
    }

    /*
     *  Regex search for class names and their parameter values
     */
    std::regex classParamsRegex(R"(params for class:\s+(\w+)\W*mean:\s*([\d\s.-]+)\W*covariance:\s*([\d\s.-]+))",
                                std::regex_constants::icase);
    std::smatch match;

    string::const_iterator searchStart(params.cbegin());

    // Search for class names and their parameter values, go through all the matches
    while (std::regex_search(searchStart, params.cend(), match, classParamsRegex)) {
        string className        = match[1].str();
        string meanString       = match[2].str();
        string covarianceString = match[3].str();

        vector<double> classMean;
        std::istringstream meanStream(meanString);
        double val;

        // Split meanString by ' ' delimiter and add to classMean
        while (meanStream >> val) {
            classMean.push_back(val);
        }

        std::istringstream covarianceStream(covarianceString);
        string line;
        vector<vector<double>> covarianceMatrix;

        // Split covarianceString by '\n' delimiter and add to covarianceMatrix
        while (std::getline(covarianceStream, line)) {
            std::istringstream rowStream(line);
            vector<double> row;
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
        classesAndTheirParamValues[className]["mean"]       = classMean;
        classesAndTheirParamValues[className]["covariance"] = vector<double>();

        // Flatten the covariance matrix into a single vector
        for (const auto &row : covarianceMatrix) {
            classesAndTheirParamValues[className]["covariance"].insert(
                classesAndTheirParamValues[className]["covariance"].end(), row.begin(), row.end());
        }

        // Update searchStart to the end of the current match
        searchStart = match.suffix().first;
    }

    if (_debug) {
        cout << "Classes and their parameter values: " << endl;
        for (const auto &kv : classesAndTheirParamValues) {
            cout << kv.first << " : " << endl;
            for (const auto &kv2 : kv.second) {
                cout << kv2.first << " : ";
                for (const auto &val : kv2.second) {
                    cout << val << " ";
                }
                cout << endl;
            }
        }
    }

    // Update the class attributes
    _classNames                 = classNames;
    _classNamesAndPriors        = classNamesAndPriors;
    _featuresWithValueRange     = featuresWithValueRange;
    _classesAndTheirParamValues = classesAndTheirParamValues;
    _featuresOrdered            = featuresOrdered;
}

/**
 * @brief Generates multivariate normal samples.
 *
 * This function generates a specified number of samples from a multivariate normal distribution
 * with a given mean vector and covariance matrix.
 *
 * @param mean A vector of doubles representing the mean of the multivariate normal distribution.
 * @param cov A MatrixXd representing the covariance matrix of the multivariate normal distribution.
 * @param numSamples An integer specifying the number of samples to generate.
 * @return A vector of VectorXd, where each VectorXd is a sample from the multivariate normal distribution.
 */
vector<VectorXd> TrainingDataGeneratorNumeric::GenerateMultivariateSamples(const vector<double> &mean,
                                                                           const MatrixXd &cov,
                                                                           int numSamples)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist(0, 1);

    // Generate multivariate normal samples
    vector<VectorXd> samples;
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

/**
 * @brief Generates training data for numeric features and writes it to a CSV file.
 *
 * This function generates multivariate normal samples for each class based on the provided
 * mean and covariance values. It then creates data records for each sample, shuffles them,
 * and writes them to a CSV file.
 *
 * The CSV file will contain a header row with feature names and subsequent rows with
 * the generated data records.
 *
 * The function performs the following steps:
 * 1. Generates samples for each class using multivariate normal distribution.
 * 2. Creates data records for each sample.
 * 3. Shuffles the data records randomly.
 * 4. Writes the data records to a CSV file with a header row.
 *
 * @note The function assumes that the class parameters (_classesAndTheirParamValues) contain
 *       mean and covariance values for each class.
 *
 * @param None
 * @return void
 */
void TrainingDataGeneratorNumeric::GenerateTrainingDataNumeric()
{
    map<string, vector<VectorXd>> samplesForClass;

    // Generate samples for each class
    for (const auto &classEntry : _classesAndTheirParamValues) {
        // Get class name, mean and covariance
        string className       = classEntry.first;
        vector<double> mean    = classEntry.second.at("mean");
        vector<double> covFlat = classEntry.second.at("covariance");

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
    vector<string> dataRecords;
    for (const auto &classEntry : samplesForClass) {
        // For each sample in the class, create a data record
        const string &className = classEntry.first;
        for (int sampleIndex = 0; sampleIndex < _numberOfSamplesPerClass; ++sampleIndex) {
            string dataRecord = className + ",";
            // For each feature in the sample, add to the data record
            for (int featureIndex = 0; featureIndex < classEntry.second[sampleIndex].size(); ++featureIndex) {
                dataRecord += std::to_string(classEntry.second[sampleIndex](featureIndex));

                // Add comma if not the last feature
                if (featureIndex < classEntry.second[sampleIndex].size() - 1) {
                    dataRecord += ",";
                }
            }
            if (_debug) {
                cout << "Data record: " << dataRecord << endl;
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
         << std::accumulate(
                _featuresOrdered.begin(),
                _featuresOrdered.end(),
                string(),
                [](const string &acc, const string &feature) { return acc + (acc.empty() ? "" : ",") + feature; })
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
string TrainingDataGeneratorNumeric::getOutputCsvFile() const
{
    return _outputCsvFile;
}

string TrainingDataGeneratorNumeric::getParameterFile() const
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

vector<string> TrainingDataGeneratorNumeric::getClassNames() const
{
    return _classNames;
}

vector<string> TrainingDataGeneratorNumeric::getFeaturesOrdered() const
{
    return _featuresOrdered;
}

map<string, double> TrainingDataGeneratorNumeric::getClassNamesAndPriors() const
{
    return _classNamesAndPriors;
}

map<string, pair<double, double>> TrainingDataGeneratorNumeric::getFeaturesWithValueRange() const
{
    return _featuresWithValueRange;
}

map<string, map<string, vector<double>>> TrainingDataGeneratorNumeric::getClassesAndTheirParamValues() const
{
    return _classesAndTheirParamValues;
}