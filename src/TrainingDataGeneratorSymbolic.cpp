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
        paramString += param + "\n";
    }

    std::cout << "paramString: " << paramString << "\n";

    // string used in matching
    std::string restParams;

    // Match class names and class priors
    // Regex to match and capture class names
    std::regex classPattern("^\\s*class names:(.*?)\\s*class priors:(.*?\\s*)(feature: .*)");
    std::smatch m;

    if (std::regex_search(paramString, m, classPattern))
    {
        std::vector<std::string> classPriorsStr;
        restParams = m[3];
        _classNames    = filterAndClean("", splitByRegex(m[1].str(), "\\s+")); // split by space
        classPriorsStr = filterAndClean("", splitByRegex(m[2].str(), "\\s+")); // split by space
        
        for (const auto &cp : classPriorsStr)
        {
            _classPriors.push_back(std::stod(cp));
        }
    }
    else
    {
        throw std::invalid_argument("Class names and class priors not found.");
    }

    // Now match Feature and bias
    std::regex featureAndBiasPattern("(feature:.*?) (bias:.*)"); // this does not match
    std::smatch mFeatureBias;
    std::string featureString;
    std::string biasString;
    std::map<std::string, std::vector<double>> featuresAndValuesDict;

    std::cout << "restParams: " << restParams << "\n";

    if (std::regex_search(restParams, mFeatureBias, featureAndBiasPattern))
    {
        std::cout << "mFeatureBias[1]: " << mFeatureBias[1].str() << "\n";
        featureString = mFeatureBias[1].str();
        biasString = mFeatureBias[2].str();
        std::vector<std::string> features = filterAndClean("", splitByRegex(featureString, "(feature[:])"));

        // for each feature
        for (const auto &feature : features)
        {
            // if item starts with "feature" then continue
            if (feature.substr(0, 7) == "feature") { continue; }
            std::vector<std::string> splits = filterAndClean("", splitByRegex(feature, " "));

            // for each split
            for (int i = 0; i < splits.size(); i++)
            {
                // if first item, then create a new key in the dictionary
                if (i == 0) { featuresAndValuesDict[splits[i]] = {}; }
                else {
                    // otherwise, add the value to the dictionary
                    if (splits[i].substr(0, 6) == "values") { continue; }
                    featuresAndValuesDict[splits[0]].push_back(std::stod(splits[i]));
                }
            }
        }
    }
    else {
        throw std::invalid_argument("Feature and bias not found.");
    }
    _featuresAndValuesDict = featuresAndValuesDict;

    // Now onto the bias
    std::map<std::string, std::map<std::string, std::vector<double>>> biasDict;
    std::vector<std::string> biases = filterAndClean("", splitByRegex(biasString, "(bias[:]\\s*class[:])"));
    for (const auto &bias : biases)
    {
        if (bias.substr(0, 4) == "bias") { continue; }
        std::vector<std::string> splits = filterAndClean("", splitByRegex(bias, " "));
        std::string featureName = "";

        // Process the splits
        for (size_t i = 0; i < splits.size(); ++i)
        {
            if (i == 0) { biasDict[splits[0]] = {}; }
            else
            {
                // Check if the current split ends with a colon
                std::regex featureRegex("(^.+)[:]$");
                std::smatch match;

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
                    biasDict[splits[0]][featureName].push_back(std::stod(splits[i]));
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
                std::cout << bias.first << " ===> " << vecToString(bias.second) << "\n";
            }
        }
    }
}

void TrainingDataGeneratorSymbolic::GenerateTrainingDataSymbolic()
{
    // Generate the training data for symbolic data
}

// Other functions below