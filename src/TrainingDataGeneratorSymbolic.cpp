#include "TrainingDataGeneratorSymbolic.hpp"

TrainingDataGeneratorSymbolic::TrainingDataGeneratorSymbolic(std::map<std::string, std::string> kwargs)
{
    std::vector<std::string> allowedKeys = {"output_datafile", "parameter_file", "number_of_training_samples", "write_to_file", "debug1", "debug2"};

    if (kwargs.empty())
    {
        throw std::invalid_argument("Missing parameters.");
    }

    // Checking passed keyword arguments
    for (const auto &kv : kwargs)
    {
        // see if the key is in the allowed keys
        if (std::find(allowedKeys.begin(), allowedKeys.end(), kv.first) == allowedKeys.end())
        {
            throw std::invalid_argument(kv.first + ": Wrong keyword used --- check spelling");
        }
    }

    // Assign default values
    _debug1 = 0;
    _debug2 = 0;

    // go through the passed keyword arguments
    for (const auto &kv : kwargs)
    {
        if (kv.first == "output_datafile")
        {
            _outputDatafile = kv.second;
        }
        else if (kv.first == "parameter_file")
        {
            _parameterFile = kv.second;
        }
        else if (kv.first == "number_of_training_samples")
        {
            _numberOfTrainingSamples = std::stoi(kv.second);
        }
        else if (kv.first == "write_to_file")
        {
            _writeToFile = std::stoi(kv.second);
        }
        else if (kv.first == "debug1")
        {
            _debug1 = std::stoi(kv.second);
        }
        else if (kv.first == "debug2")
        {
            _debug2 = std::stoi(kv.second);
        }
    }
}

TrainingDataGeneratorSymbolic::~TrainingDataGeneratorSymbolic()
{
}

std::vector<std::string> TrainingDataGeneratorSymbolic::filterAndClean(const std::string &pattern, const std::vector<std::string> &input)
{
    std::vector<std::string> cleaned;

    if (pattern.empty())
    {
        // Remove empty strings
        std::copy_if(input.begin(), input.end(), std::back_inserter(cleaned),
                     [](const std::string &s)
                     { return !s.empty(); });
    }
    else
    {
        std::regex regexPattern(pattern);
        for (const std::string &item : input)
        {
            std::sregex_iterator iter(item.begin(), item.end(), regexPattern);
            std::sregex_iterator end;

            while (iter != end)
            {
                std::string token = iter->str();
                if (!token.empty())
                {
                    std::cout << "  [!] Pushing back token: " << token << "\n";
                    cleaned.push_back(token);
                }
                ++iter;
            }
        }
    }

    return cleaned;
}

std::vector<std::string> TrainingDataGeneratorSymbolic::splitByRegex(const std::string &input, const std::string &pattern)
{
    std::regex regexPattern(pattern);
    std::sregex_token_iterator iter(input.begin(), input.end(), regexPattern, -1);
    std::sregex_token_iterator end;
    std::vector<std::string> result;

    while (iter != end)
    {
        std::string token = *iter++;
        if (!token.empty())
        {
            result.push_back(token);
        }
    }

    return result;
}

void TrainingDataGeneratorSymbolic::ReadParameterFileSymbolic()
{
    int debug1 = _debug1;
    int debug2 = _debug2;
    int writeToFile = _writeToFile;
    int numberOfTrainingSamples = _numberOfTrainingSamples;
    std::string inputParameterFile = _parameterFile;

    // Read the parameter file for symbolic data
    std::ifstream file(inputParameterFile);
    std::string allParamsStr((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    std::string paramString = "";

    if (allParamsStr.empty())
    {
        throw std::invalid_argument("Empty file.");
    }

    // Split by newline
    std::vector<std::string> allParams;
    std::stringstream ss(allParamsStr);
    while (std::getline(ss, paramString, '\n'))
    {
        allParams.push_back(paramString);
    }

    // filter allParams to get only the ones that match the pattern and are not None/FALSE
    allParams = filterAndClean("^(?![ ]*#)(.*)", allParams);

    // Make back into a string
    for (const auto &param : allParams)
    {
        std::cout << "paramSTRING: " << param << "\n";
        paramString += param + "\n";
    }

    std::cout << "paramString: " << paramString;

    // string used in matching
    std::string restParams;

    // Match class names and class priors
    // Regex to match and capture class names
    std::regex classPattern("class names:\\s*(.*?)\\s*class priors:\\s*(.*?)\\s*(feature: [\\s\\S]*)");
    std::smatch m;

    if (std::regex_search(paramString, m, classPattern))
    {
        // print matched groups
        for (size_t i = 0; i < m.size(); ++i)
        {
            std::cout << "m[" << i << "]: " << m[i].str() << "\n\n";
        }

        std::string classNames = m[1].str();
        std::string classPriors = m[2].str();
        restParams = m[3].str();

        // Split class names and class priors
        std::vector<std::string> classNamesList = filterAndClean("", splitByRegex(classNames, "\\s+"));
        std::vector<std::string> classPriorsList = filterAndClean("", splitByRegex(classPriors, "\\s+"));

        std::cout << "Class names: " << vecToString(classNamesList) << "\n";
        std::cout << "Class priors: " << vecToString(classPriorsList) << "\n";

        // Assign to class names and class priors
        _classNames = classNamesList;
        std::vector<double> classPriorsDouble;
        for (const auto &item : classPriorsList)
        {
            classPriorsDouble.push_back(std::stod(item));
        }
        _classPriors = classPriorsDouble;
    }
    else
    {
        throw std::invalid_argument("Class names and class priors not found.");
    }

    std::cout << "Rest of the parameters: " << restParams << "\n";

    // Now match Feature and bias
    std::regex featureAndBiasPattern("(feature:[\\s\\S]*?)(?=\\s*bias:)((?=bias:)[\\s\\S]*)"); 
    std::smatch mFeatureBias;
    std::string featureString;
    std::string biasString;
    std::map<std::string, std::vector<std::string>> featuresAndValuesDict;

    if (std::regex_search(restParams, mFeatureBias, featureAndBiasPattern))
    {
        std::cout << "mFeatureBias[1]: " << mFeatureBias[1].str() << "\n";
        std::cout << "mFeatureBias[2]: " << mFeatureBias[2].str() << "\n";

        std::cout << "Feature and bias found.\n";


        featureString = mFeatureBias[1].str();
        biasString = mFeatureBias[2].str();
        std::vector<std::string> features = filterAndClean("", splitByRegex(featureString, "(feature[:])"));

        // for each feature
        for (const auto &feature : features)
        {
            if (feature.substr(0, 7) == "feature") { continue; }

            std::vector<std::string> splits = filterAndClean("", splitByRegex(feature, " "));

            // for each split
            for (int i = 0; i < splits.size(); i++)
            {
                std::cout << " SPLIT " << i << ": " << splits[i] << "\n";
                // if first item, then create a new key in the dictionary
                if (i == 0) {
                    // remove anything after newline
                    std::regex newlineRegex("(.*)\\n");
                    std::smatch newlineMatch;
                    if (std::regex_search(splits[0], newlineMatch, newlineRegex))
                    {
                        splits[0] = newlineMatch[1].str();
                    }
                    featuresAndValuesDict[splits[0]] = {};
                }
                else {
                    // otherwise, add the value to the dictionary
                    if (splits[i].substr(0, 6) == "values") { continue; }
                    // remove newline
                    std::regex newlineRegex("(.*)\\n");
                    std::smatch newlineMatch;
                    if (std::regex_search(splits[i], newlineMatch, newlineRegex))
                    {
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

    // print the features and values
    std::cout << "Here are the features and their possible values:\n\n";
    for (const auto &item : _featuresAndValuesDict)
    {
        std::cout << item.first << " ===> " << vecToString(item.second) << "\n";
    }

    // Now onto the bias
    std::map<std::string, std::map<std::string, std::vector<std::string>>> biasDict;
    std::vector<std::string> biases = filterAndClean("", splitByRegex(biasString, "(bias[:]\\s*class[:])"));
    for (const auto &bias : biases)
    {
        if (bias.substr(0, 4) == "bias") { continue; }
        std::vector<std::string> splits = filterAndClean("", splitByRegex(bias, " "));
        std::string featureName = "";

        // Process the splits
        for (size_t i = 0; i < splits.size(); ++i)
        {
            if (i == 0) 
            { 
                splits[0] = filterAndClean("", splitByRegex(splits[0], "\\n"))[0];
                biasDict[splits[0]] = {}; 
            }
            else
            {
                // Check if the current split ends with a colon
                std::regex featureRegex("(^.+)[:]$");
                std::smatch match;

                std::cout << "For class " << splits[0] << " and split " << i << ": " << splits[i] << "\n";

                if (std::regex_search(splits[i], match, featureRegex))
                {
                    // Extract feature name
                    featureName = match[1].str();
                    biasDict[splits[0]][featureName] = {};
                }
                else
                {
                    // Add to the feature's list only if featureName exists
                    if (featureName.empty()) { continue; }
                    std::regex newlineRegex("(.*)\\n");
                    std::smatch newlineMatch;
                    if (std::regex_search(splits[i], newlineMatch, newlineRegex))
                    {
                        splits[i] = newlineMatch[1].str();
                    }

                    // only add if featurename does NOT exist yet
                    if (biasDict[splits[0]][featureName].empty()) {
                        std::cout << "CLass name: " << splits[0] << " Feature name: " << featureName << " Value: " << splits[i] << "\n";
                        biasDict[splits[0]][featureName].push_back(splits[i]);
                    }
                }
            }
        }
    }
    _biasDict = biasDict;

    if (_debug1)
    {
        std::cout << "\n\n";
        std::cout << "Class names: " << vecToString(_classNames) << "\n";

        size_t num_of_classes = _classNames.size();
        std::cout << "Number of classes: " << num_of_classes << "\n\n";

        std::cout << "Class priors: " << vecToString(_classPriors) << "\n\n";

        std::cout << "Here are the features and their possible values:\n\n";
        for (const auto &item : _featuresAndValuesDict)
        {
            std::cout << item.first << " ===> " << vecToString(item.second) << "\n";
        }

        std::cout << "\nHere is the biasing for each class:\n\n";
        for (const auto &item : _biasDict)
        {
            std::cout << "\n"
                      << item.first << "\n";
            for (const auto &bias : item.second)
            {
                if (bias.second.size() == 2)
                {
                    std::cout << bias.first << " ===> (two)" << bias.second[0] << " and " << bias.second[1] << "\n";
                }
                std::cout << bias.first << " ===> " << vecToString(bias.second) << "\n";
            }
        }
    }
}

void TrainingDataGeneratorSymbolic::GenerateTrainingDataSymbolic()
{
    std::vector<std::string> classNames = _classNames;
    std::vector<double> classPriors = _classPriors;
    int howManyTrainingSamples = _numberOfTrainingSamples;
    std::map<std::string, std::vector<std::string>> featuresAndValuesDict = _featuresAndValuesDict;
    std::map<std::string, std::map<std::string, std::vector<std::string>>> biasDict = _biasDict;

    std::map<std::string, std::vector<std::string>> trainingSampleRecords;
    std::map<std::string, std::pair<double, double>> classPriorsToUnitIntervalMap;
    double accumulatedInterval = 0.0;

    // Create a map of class names to their corresponding unit intervals
    for (size_t i = 0; i < classNames.size(); ++i)
    {
        accumulatedInterval += classPriors[i];
        classPriorsToUnitIntervalMap[classNames[i]] = std::make_pair(accumulatedInterval, accumulatedInterval + classPriors[i]);
    }

    if (_debug1)
    {
        std::cout << "Mapping of class priors to unit interval:\n";
        for (const auto &item : classPriorsToUnitIntervalMap)
        {
            std::cout << item.first << " ===> " << item.second.first << " to " << item.second.second << "\n";
        }
    }

    std::map<std::string, std::map<std::string, std::map<std::string, std::pair<double, double>>>> classAndFeatureBasedValuePriorsToUnitIntervalMap;
    for (const auto &className : classNames)
    {
        // Add entry for each class
        classAndFeatureBasedValuePriorsToUnitIntervalMap[className] = {};
        // for each feature in the featuresAndValuesDict
        for (const auto &feature : featuresAndValuesDict)
        {
            // Add entry for each feature
            classAndFeatureBasedValuePriorsToUnitIntervalMap[className][feature.first] = {};
        }

        // for each class
        for (const auto &className : classNames)
        {
            // for each feature in the featuresAndValuesDict
            for (const auto &feature : featuresAndValuesDict)
            {
                auto values = featuresAndValuesDict[feature.first];
                std::string biasString;
                if (!biasDict[className][feature.first].empty())
                {
                    biasString = biasDict[className][feature.first][0];
                }
                else
                {
                    double noBias = 1.0 / values.size();
                    biasString = values[0] + "=" + std::to_string(noBias);

                }

                std::map<std::string, std::pair<double, double>> valuePriorsToUnitIntervalMap;
                auto splits = splitByRegex(biasString, "'\\s*=\\s*'");
                std::string chosenForBiasValue = splits[0];
                double chosenBias = std::stod(splits[1]);
                double remainingBias = 1.0 - chosenBias;
                double remainingPortionBias = remainingBias / (values.size() - 1);
                double accumulated = 0.0;

                for (int i = 0; i < values.size(); ++i)
                {
                    if (values[i] == chosenForBiasValue)
                    {
                        valuePriorsToUnitIntervalMap[values[i]] = {accumulated, accumulated + chosenBias};
                        accumulated += chosenBias;
                    }
                    else
                    {
                        valuePriorsToUnitIntervalMap[values[i]] = {accumulated, accumulated + remainingPortionBias};
                        accumulated += remainingPortionBias;
                    }
                }
                classAndFeatureBasedValuePriorsToUnitIntervalMap[className][feature.first] = valuePriorsToUnitIntervalMap;

                if (_debug2)
                {
                    std::cout << "\nFor class " << className << ": Mapping feature value priors for feature '" << feature.first << "' to unit interval:\n";
                    for (const auto &item : valuePriorsToUnitIntervalMap)
                    {
                        std::cout << "    " << item.first << " ===> [" << item.second.first << ", " << item.second.second << "]\n";
                    }
                }
            }
        }
        
        int eleIndex = 0;
    }
}

// Other functions below