#include "TrainingDataGeneratorSymbolic.hpp"

TrainingDataGeneratorSymbolic::TrainingDataGeneratorSymbolic(map<string, string> kwargs)
{
    vector<string> allowedKeys = {
        "output_datafile", "parameter_file", "number_of_training_samples", "write_to_file", "debug1", "debug2"};

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

    // Assign default values
    _debug1 = 0;
    _debug2 = 0;

    // go through the passed keyword arguments
    for (const auto &kv : kwargs) {
        if (kv.first == "output_datafile") {
            _outputDatafile = kv.second;
        }
        else if (kv.first == "parameter_file") {
            _parameterFile = kv.second;
        }
        else if (kv.first == "number_of_training_samples") {
            _numberOfTrainingSamples = std::stoi(kv.second);
        }
        else if (kv.first == "write_to_file") {
            _writeToFile = std::stoi(kv.second);
        }
        else if (kv.first == "debug1") {
            _debug1 = std::stoi(kv.second);
        }
        else if (kv.first == "debug2") {
            _debug2 = std::stoi(kv.second);
        }
    }
}

TrainingDataGeneratorSymbolic::~TrainingDataGeneratorSymbolic() {}

vector<string> TrainingDataGeneratorSymbolic::filterAndClean(const string &pattern,
                                                                       const vector<string> &input)
{
    vector<string> cleaned;

    if (pattern.empty()) {
        // Remove empty strings
        std::copy_if(
            input.begin(), input.end(), std::back_inserter(cleaned), [](const string &s) { return !s.empty(); });
    }
    else {
        std::regex regexPattern(pattern);
        for (const string &item : input) {
            std::sregex_iterator iter(item.begin(), item.end(), regexPattern);
            std::sregex_iterator end;

            while (iter != end) {
                string token = iter->str();
                if (!token.empty()) {
                    cleaned.push_back(token);
                }
                ++iter;
            }
        }
    }

    return cleaned;
}

vector<string> TrainingDataGeneratorSymbolic::splitByRegex(const string &input,
                                                                     const string &pattern)
{
    std::regex regexPattern(pattern);
    std::sregex_token_iterator iter(input.begin(), input.end(), regexPattern, -1);
    std::sregex_token_iterator end;
    vector<string> result;

    while (iter != end) {
        string token = *iter++;
        if (!token.empty()) {
            result.push_back(token);
        }
    }

    return result;
}

void TrainingDataGeneratorSymbolic::ReadParameterFileSymbolic()
{
    int debug1                     = _debug1;
    int debug2                     = _debug2;
    int writeToFile                = _writeToFile;
    int numberOfTrainingSamples    = _numberOfTrainingSamples;
    string inputParameterFile = _parameterFile;

    // Read the parameter file for symbolic data
    std::ifstream file(inputParameterFile);
    string allParamsStr((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    string paramString = "";

    if (allParamsStr.empty()) {
        throw std::invalid_argument("Empty file.");
    }

    // Split by newline
    vector<string> allParams;
    std::stringstream ss(allParamsStr);
    while (std::getline(ss, paramString, '\n')) {
        allParams.push_back(paramString);
    }

    // filter allParams to get only the ones that match the pattern and are not None/FALSE
    allParams = filterAndClean("^(?![ ]*#)(.*)", allParams);

    // Make back into a string
    for (const auto &param : allParams) {
        paramString += param + "\n";
    }

    // string used in matching
    string restParams;

    // Match class names and class priors
    // Regex to match and capture class names
    std::regex classPattern("class names:\\s*(.*?)\\s*class priors:\\s*(.*?)\\s*(feature: [\\s\\S]*)");
    std::smatch m;

    if (std::regex_search(paramString, m, classPattern)) {
        string classNames  = m[1].str();
        string classPriors = m[2].str();
        restParams              = m[3].str();

        // Split class names and class priors
        vector<string> classNamesList  = filterAndClean("", splitByRegex(classNames, "\\s+"));
        vector<string> classPriorsList = filterAndClean("", splitByRegex(classPriors, "\\s+"));

        // Assign to class names and class priors
        _classNames = classNamesList;
        vector<double> classPriorsDouble;
        for (const auto &item : classPriorsList) {
            classPriorsDouble.push_back(std::stod(item));
        }
        _classPriors = classPriorsDouble;
    }
    else {
        throw std::invalid_argument("Class names and class priors not found.");
    }

    // Now match Feature and bias
    std::regex featureAndBiasPattern("(feature:[\\s\\S]*?)(?=\\s*bias:)((?=bias:)[\\s\\S]*)");
    std::smatch mFeatureBias;
    string featureString;
    string biasString;
    map<string, vector<string>> featuresAndValuesDict;

    if (std::regex_search(restParams, mFeatureBias, featureAndBiasPattern)) {
        featureString                     = mFeatureBias[1].str();
        biasString                        = mFeatureBias[2].str();
        vector<string> features = filterAndClean("", splitByRegex(featureString, "(feature[:])"));

        // for each feature
        for (const auto &feature : features) {
            if (feature.substr(0, 7) == "feature") {
                continue;
            }

            vector<string> splits = filterAndClean("", splitByRegex(feature, " "));

            // for each split
            for (int i = 0; i < splits.size(); i++) {
                // if first item, then create a new key in the dictionary
                if (i == 0) {
                    // remove anything after newline
                    std::regex newlineRegex("(.*)\\n");
                    std::smatch newlineMatch;
                    if (std::regex_search(splits[0], newlineMatch, newlineRegex)) {
                        splits[0] = newlineMatch[1].str();
                    }
                    featuresAndValuesDict[splits[0]] = {};
                }
                else {
                    // otherwise, add the value to the dictionary
                    if (splits[i].substr(0, 6) == "values") {
                        continue;
                    }
                    // remove newline
                    std::regex newlineRegex("(.*)\\n");
                    std::smatch newlineMatch;
                    if (std::regex_search(splits[i], newlineMatch, newlineRegex)) {
                        splits[i] = newlineMatch[1].str();
                    }

                    featuresAndValuesDict[splits[0]].push_back(splits[i]);
                }
            }
        }
    }
    else {
        throw std::invalid_argument("Feature and bias not found.");
    }
    _featuresAndValuesDict = featuresAndValuesDict;

    // Now onto the bias
    map<string, map<string, vector<string>>> biasDict;
    vector<string> biases = filterAndClean("", splitByRegex(biasString, "(bias[:]\\s*class[:])"));
    for (const auto &bias : biases) {
        if (std::regex_match(bias, std::regex("bias")))
            continue;

        // Split bias string by spaces and filter out empty results
        vector<string> splits = filterAndClean("", splitByRegex(bias, "\\s+"));
        string featureName;

        for (size_t i = 0; i < splits.size(); ++i) {
            if (i == 0) {
                // if first item, then create a new key in the dictionary
                biasDict[splits[0]] = {};
            }
            else if (std::regex_search(splits[i], std::regex("(^.+)[:]$"))) {
                std::smatch m;
                std::regex_search(splits[i], m, std::regex("(^.+)[:]$"));
                featureName                      = m[1].str(); // Get the matched group without the ':'
                biasDict[splits[0]][featureName] = {};         // Initialize as an empty vector
            }
            else {
                if (featureName.empty())
                    continue;                                          // If featureName is not set, skip
                biasDict[splits[0]][featureName].push_back(splits[i]); // Add to the vector
            }
        }
    }
    _biasDict = biasDict; // Assuming _biasDict is defined appropriately

    if (_debug1) {
        cout << "\n\n";
        cout << "Class names: " << vecToString(_classNames) << "\n";

        size_t num_of_classes = _classNames.size();
        cout << "Number of classes: " << num_of_classes << "\n\n";

        cout << "Class priors: " << vecToString(_classPriors) << "\n\n";

        cout << "Here are the features and their possible values:\n\n";
        for (const auto &item : _featuresAndValuesDict) {
            cout << item.first << " ===> " << vecToString(item.second) << "\n";
        }

        cout << "\nHere is the biasing for each class:\n\n";
        for (const auto &item : _biasDict) {
            cout << "\n" << item.first << "\n";
            for (const auto &bias : item.second) {
                if (bias.second.size() == 2) {
                    cout << bias.first << " ===> (two)" << bias.second[0] << " and " << bias.second[1] << "\n";
                }
                cout << bias.first << " ===> [" << vecToString(bias.second) << "]\n";
            }
        }
    }
}

void TrainingDataGeneratorSymbolic::GenerateTrainingDataSymbolic()
{
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    vector<string> classNames                                             = this->_classNames;
    vector<double> classPriors                                                 = this->_classPriors;
    map<string, vector<string>> featuresAndValuesDict           = this->_featuresAndValuesDict;
    map<string, map<string, vector<string>>> biasDict = this->_biasDict;
    int howManyTrainingSamples                                                      = this->_numberOfTrainingSamples;

    map<string, pair<double, double>> classPriorsToUnitIntervalMap;
    double accumulatedInterval = 0.0;

    // Map class priors to unit interval
    for (size_t i = 0; i < classNames.size(); ++i) {
        classPriorsToUnitIntervalMap[classNames[i]] =
            std::make_pair(accumulatedInterval, accumulatedInterval + classPriors[i]);
        accumulatedInterval += classPriors[i];
    }

    // Debugging output
    if (this->_debug1) {
        cout << "Mapping of class priors to unit interval:" << endl;
        for (const auto &item : classPriorsToUnitIntervalMap) {
            cout << item.first << " ===> (" << item.second.first << ", " << item.second.second << ")" << endl;
        }
    }

    map<string, map<string, map<string, pair<double, double>>>>
        classAndFeatureBasedValuePriorsToUnitIntervalMap;

    // Initialize maps for each class and feature
    for (const auto &className : classNames) {
        classAndFeatureBasedValuePriorsToUnitIntervalMap[className] = {};
        for (const auto &feature : featuresAndValuesDict) {
            classAndFeatureBasedValuePriorsToUnitIntervalMap[className][feature.first] = {};
        }
    }

    // Process bias for each class and feature
    for (const auto &className : classNames) {
        for (const auto &feature : featuresAndValuesDict) {
            const vector<string> &values = featuresAndValuesDict[feature.first];
            string biasString;

            if (!biasDict[className][feature.first].empty()) {
                biasString = biasDict[className][feature.first][0];
            }
            else {
                double noBias = 1.0 / values.size();
                biasString    = values[0] + "=" + std::to_string(noBias);
            }

            map<string, pair<double, double>> valuePriorsToUnitIntervalMap;
            vector<string> splits = splitByRegex(biasString, "=");
            string chosenForBiasValue  = splits[0];
            double chosenBias               = std::stod(splits[1]);
            double remainingBias            = 1.0 - chosenBias;
            double remainingPortionBias     = remainingBias / (values.size() - 1);
            double accumulated              = 0.0;

            // Assign intervals for each value
            for (size_t i = 0; i < values.size(); ++i) {
                if (values[i] == chosenForBiasValue) {
                    valuePriorsToUnitIntervalMap[values[i]] = {accumulated, accumulated + chosenBias};
                    accumulated += chosenBias;
                }
                else {
                    valuePriorsToUnitIntervalMap[values[i]] = {accumulated, accumulated + remainingPortionBias};
                    accumulated += remainingPortionBias;
                }
            }

            classAndFeatureBasedValuePriorsToUnitIntervalMap[className][feature.first] = valuePriorsToUnitIntervalMap;

            // Debugging output
            if (this->_debug2) {
                cout << "For class " << className << ": Mapping feature value priors for feature '"
                          << feature.first << "' to unit interval: " << endl;
                for (const auto &item : valuePriorsToUnitIntervalMap) {
                    cout << "    " << item.first << " ===> (" << item.second.first << ", " << item.second.second
                              << ")" << endl;
                }
            }
        }
    }

    map<int, vector<string>> trainingSampleRecords;
    int eleIndex = 0;

    // Generate training samples
    while (eleIndex < howManyTrainingSamples) {
        int sampleName                    = eleIndex;
        trainingSampleRecords[sampleName] = {};

        // Generate class label for the sample
        double roll_the_dice = randomDouble(0.0, 1.0);
        string classLabel;
        for (const auto &classEntry : classPriorsToUnitIntervalMap) {
            const pair<double, double> &interval = classEntry.second;
            if (roll_the_dice >= interval.first && roll_the_dice <= interval.second) {
                trainingSampleRecords[sampleName].push_back(classEntry.first);
                classLabel = classEntry.first;
                break;
            }
        }

        // Generate feature values for the sample
        for (const auto &feature : featuresAndValuesDict) {
            roll_the_dice = randomDouble(0.0, 1.0);
            const auto &valuePriorsToUnitIntervalMap =
                classAndFeatureBasedValuePriorsToUnitIntervalMap[classLabel][feature.first];
            for (const auto &valueEntry : valuePriorsToUnitIntervalMap) {
                const pair<double, double> &interval = valueEntry.second;
                if (roll_the_dice >= interval.first && roll_the_dice <= interval.second) {
                    trainingSampleRecords[sampleName].push_back(valueEntry.first);
                    break;
                }
            }
        }

        eleIndex++;
    }

    this->_trainingSampleRecords = trainingSampleRecords;

    // Debugging output for the generated records
    if (this->_debug2) {
        cout << "\n\nTERMINAL DISPLAY OF TRAINING RECORDS:\n\n";
        for (const auto &sampleEntry : trainingSampleRecords) {
            cout << sampleEntry.first << " = ";
            for (const auto &record : sampleEntry.second) {
                cout << record << ", ";
            }
            cout << endl;
        }
    }
}

double TrainingDataGeneratorSymbolic::randomDouble(double lower, double upper)
{
    return lower + static_cast<double>(rand()) / (static_cast<double>(RAND_MAX / (upper - lower)));
}

void TrainingDataGeneratorSymbolic::WriteTrainingDataToFile()
{
    if (!_writeToFile) {
        cout << "Write to file option is disabled. Skipping file writing." << endl;
        return;
    }

    std::ofstream outFile(_outputDatafile);
    if (!outFile.is_open()) {
        throw std::runtime_error("Unable to open output file: " + _outputDatafile);
    }

    // Write header
    outFile << ",class";
    for (const auto &feature : _featuresAndValuesDict) {
        outFile << ',' << feature.first;
    }
    outFile << endl;

    // print the sample records
    for (const auto &record : _trainingSampleRecords) {
        outFile << record.first;
        for (const auto &feature : record.second) {
            outFile << ',' << feature;
        }
        outFile << endl;
    }

    outFile.close();
    cout << "Training data written to " << _outputDatafile << endl;
}
