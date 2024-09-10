#include "TrainingDataGeneratorNumeric.hpp"

TrainingDataGeneratorNumeric::TrainingDataGeneratorNumeric(std::map<std::string, std::string> kwargs)
{
    std::vector<std::string> allowedKeys = {"output_csv_file", "parameter_file", "number_of_samples_per_class", "debug"};

    if (kwargs.empty()) {
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

    // Set default values
    _debug = 0;

    // go through the passed keyword arguments
    for (const auto &kv : kwargs)
    {
        const std::string &key = kv.first;
        const std::string &value = kv.second;

        if (key == "output_csv_file")
        {
            _output_csv_file = value;
        }
        else if (key == "parameter_file")
        {
            _parameter_file = value;
        }
        else if (key == "number_of_samples_per_class")
        {
            _number_of_samples_per_class = std::stoi(value);
        }
        else if (key == "debug")
        {
            _debug = std::stoi(value);
        }
    }
}

TrainingDataGeneratorNumeric::~TrainingDataGeneratorNumeric()
{
}

void TrainingDataGeneratorNumeric::ReadParameterFileNumeric()
{
    std::vector<std::string> class_names;
    std::map<std::string, double> class_names_and_priors;
    std::map<std::string, std::pair<double, double>> features_with_value_range;
    std::map<std::string, std::map<std::string, std::vector<double>>> classes_and_their_param_values;
    std::vector<std::string> features_ordered;

    std::regex fp_or_sci_notation("[+-]?\\ *\\d+(\\.\\d*)?|\\.\\d+([eE][+-]?\\d+)?");

    // Read the parameter file for numeric data
    std::ifstream FILE(_parameter_file);
    std::string params((std::istreambuf_iterator<char>(FILE)), std::istreambuf_iterator<char>());
    if (params.empty())
    {
        throw std::invalid_argument("Empty file.");
    }

    /*
    *  Regex search for class names and priors
    */
    std::regex class_regex("class names: ([\\w\\s+]+)\\W*class priors: ([\\d.\\s+]+)", std::regex::icase);
    std::smatch class_matches;

    if (std::regex_search(params, class_matches, class_regex))
    {
        std::string class_names_str = class_matches[1].str();
        std::string class_priors_str = class_matches[2].str();
        std::istringstream class_names_stream(class_names_str);
        std::istringstream class_priors_stream(class_priors_str);

        std::string name, prior;
        std::vector<double> class_priors;

        // Read class names into the vector
        while (class_names_stream >> name)
        {
            if (!name.empty())
            {
                class_names.push_back(name);
            }
        }

        // Read class priors into the vector
        while (class_priors_stream >> prior)
        {
            if (!prior.empty())
            {
                class_priors.push_back(std::stod(prior)); // Convert the string to double
            }
        }

        // Combine class names and priors into the map between class names and priors
        for (size_t i = 0; i < class_names.size(); ++i)
        {
            class_names_and_priors[class_names[i]] = class_priors[i];
        }
    }
    else
    {
        throw std::runtime_error("Could not find 'class names' and 'class priors' in the parameter file.");
    }

    if (_debug)
    {
        std::cout << "Class names and priors: " << std::endl;
        for (const auto &kv : class_names_and_priors)
        {
            std::cout << kv.first << " : " << kv.second << std::endl;
        }
    }

    /*
    *  Regex search for feature names and value ranges
    */
    std::regex feature_regex("feature name: (\\w+)\\W*value range:\\s*([\\d. -]+)", std::regex::icase);
    std::smatch feature_matches;
    auto feature_begin = std::sregex_iterator(params.begin(), params.end(), feature_regex);
    auto feature_end = std::sregex_iterator();

    for (std::sregex_iterator i = feature_begin; i != feature_end; ++i)
    {
        std::smatch match = *i;
        std::string feature_name = match[1].str(); // feature name is the first match group
        std::string value_range_str = match[2].str(); // value range is the second match group

        std::istringstream value_range_stream(value_range_str); 
        std::vector<double> value_range;
        std::string value;

        // Split value_range_stream by '-' delimiter and add to features_with_value_range
        while (std::getline(value_range_stream, value, '-'))
        {
            std::cout << value << std::endl;
            if (!value.empty())
            {
                value_range.push_back(std::stod(value));
            }
        }

        // Ensure value_range has exactly two values
        if (value_range.size() == 2)
        {
            features_with_value_range[feature_name] = std::make_pair(value_range[0], value_range[1]);
        }

        // Add feature name to the list of features
        features_ordered.push_back(feature_name);
    }

    if (_debug)
    {
        std::cout << "Features and their value ranges: " << std::endl;
        for (const auto &kv : features_with_value_range)
        {
            std::cout << kv.first << " : [" << kv.second.first << ", " << kv.second.second << "]" << std::endl;
        }
    }

    // add class names and their parameter values to classes_and_their_param_values
    for (int i = 0; i < class_names.size(); i++)
    {
        classes_and_their_param_values[class_names[i]] = {};
    }

    /*
    *  Regex search for class names and their parameter values
    */
    std::regex regex("params for class:\\s+\\w*?\\W+?mean:[\\d\\.\\s+]+\\W*?covariance:\\W+?(?:[\\s+\\d.]+\\W+?)+");
    std::smatch feature_param_matches;
    auto feature_param_begin = std::sregex_iterator(params.begin(), params.end(), regex);
    auto feature_param_end = std::sregex_iterator();

    for (std::sregex_iterator i = feature_param_begin; i != feature_param_end; ++i)
    {
        std::smatch match = *i;
        std::string class_name = match[1].str();
        std::string mean_str = match[2].str();
        std::string covariance_str = match[3].str();

        // Parse mean values
        std::istringstream mean_stream(mean_str);
        std::vector<double> class_mean;
        std::string value;

        // Split mean_str by spaces and convert to list of doubles
        while (mean_stream >> value)
        {
            if (!value.empty())
            {
                class_mean.push_back(std::stod(value));
            }
        }

        // Parse covariance matrix
        std::istringstream covariance_stream(covariance_str);
        std::vector<std::vector<double>> covar_matrix;
        std::string line;

        while (std::getline(covariance_stream, line))
        {
            std::istringstream line_stream(line);
            std::vector<double> row;
            while (line_stream >> value)
            {
                if (!value.empty())
                {
                    row.push_back(std::stod(value));
                }
            }
            if (!row.empty())
            {
                covar_matrix.push_back(row);
            }
        }

        // Store the mean and covariance in the map
        classes_and_their_param_values[class_name]["mean"] = class_mean;
        // flatten the covariance matrix (required to fit into a vector of doubles)
        std::vector<double> covar_matrix_flattened;
        for (const auto &row : covar_matrix)
        {
            covar_matrix_flattened.insert(covar_matrix_flattened.end(), row.begin(), row.end());
        }
        classes_and_their_param_values[class_name]["covariance"] = covar_matrix_flattened;
    }

    if (_debug)
    {
        std::cout << "Classes and their parameter values: " << std::endl;
        for (const auto &kv : classes_and_their_param_values)
        {
            std::cout << kv.first << " : " << std::endl;
            for (const auto &kv2 : kv.second)
            {
                std::cout << kv2.first << " : ";
                for (const auto &val : kv2.second)
                {
                    std::cout << val << " ";
                }
                std::cout << std::endl;
            }
        }
    }

    _class_names = class_names;
    _class_names_and_priors = class_names_and_priors;
    _features_with_value_range = features_with_value_range;
    _classes_and_their_param_values = classes_and_their_param_values;
    _features_ordered = features_ordered;

    std::cout << "end of ReadParameterFileNumeric" << std::endl;
    for (const auto &kv : _classes_and_their_param_values)
    {
        std::cout << "one item is " << kv.first << std::endl;
    }
}

void TrainingDataGeneratorNumeric::GenerateTrainingDataNumeric()
{
    // Generate the training data for numeric data
}


// Getters
std::string TrainingDataGeneratorNumeric::getOutputCsvFile() const
{
    return _output_csv_file;
}

std::string TrainingDataGeneratorNumeric::getParameterFile() const
{
    return _parameter_file;
}

int TrainingDataGeneratorNumeric::getNumberOfSamplesPerClass() const
{
    return _number_of_samples_per_class;
}

int TrainingDataGeneratorNumeric::getDebug() const
{
    return _debug;
}

std::vector<std::string> TrainingDataGeneratorNumeric::getClassNames() const
{
    return _class_names;
}

std::vector<std::string> TrainingDataGeneratorNumeric::getFeaturesOrdered() const
{
    return _features_ordered;
}

std::map<std::string, double> TrainingDataGeneratorNumeric::getClassNamesAndPriors() const
{
    return _class_names_and_priors;
}

std::map<std::string, std::pair<double, double>> TrainingDataGeneratorNumeric::getFeaturesWithValueRange() const
{
    return _features_with_value_range;
}

std::map<std::string, std::map<std::string, std::vector<double>>> TrainingDataGeneratorNumeric::getClassesAndTheirParamValues() const
{
    return _classes_and_their_param_values;
}