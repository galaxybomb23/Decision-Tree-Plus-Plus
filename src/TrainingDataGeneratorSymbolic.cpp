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

std::vector<std::string> filterAndClean(const std::vector<std::string> &input, const std::regex &filterPattern)
{
    std::vector<std::string> result;
    std::smatch match;

    for (const auto &str : input)
    {
        if (std::regex_search(str, match, filterPattern))
        {
            // Strip whitespace and newline characters from the end
            std::string strippedStr = std::regex_replace(str, std::regex("[ \t\n]+$"), "");
            if (!strippedStr.empty())
            {
                result.push_back(strippedStr);
            }
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

    // Regex to get params
    std::regex pattern("^(?![ ]*#)");
    std::smatch match;

    // filter allParams to get only the ones that match the pattern and are not None/FALSE
    allParams = filterAndClean(allParams, pattern);

    // Make back into a string
    for (const auto &param : allParams)
    {
        paramString += param + "\n";
    }

    // Match class names and class priors
    // Regex to match and capture class names
    std::regex classPattern("^\\s*class names:(.*?)\\s*class priors:(.*?)(feature: .*)");
    std::smatch m;
    if (std::regex_search(paramString, m, classPattern))
    {
        std::vector<std::string> classPriorsStr;
        std::string restParams = m[3];
        _classNames = filterAndClean({m[1].str()}, std::regex("'\\s+"));
        classPriorsStr = filterAndClean({m[2].str()}, std::regex("'\\s+"));
        for (const auto &cp : classPriorsStr)
        {
            _classPriors.push_back(std::stod(cp));
        }
    }

    // Now match Feature and bias
    std::regex featureAndBiasPattern("(feature:.*?) (bias:.*)");
}

void TrainingDataGeneratorSymbolic::GenerateTrainingDataSymbolic()
{
    // Generate the training data for symbolic data
}

// Other functions below