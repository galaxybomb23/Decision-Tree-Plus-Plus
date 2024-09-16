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
        // Extract class names and priors, make them a vector and a map respectively
        std::string class_names_str = class_matches[1].str();
        std::string class_priors_str = class_matches[2].str();
        std::istringstream class_names_stream(class_names_str);
        std::istringstream class_priors_stream(class_priors_str);

        std::string name, prior;
        std::vector<double> class_priors;

        // Split class_names_stream by ' ' delimiter and add to class_names
        while (class_names_stream >> name)
        {
            if (!name.empty())
            {
                class_names.push_back(name);
            }
        }

        // Split class_priors_stream by ' ' delimiter and add to class_priors
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
        // Same as above, extract feature names and value ranges
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

        // Ensure value_range has exactly two values and add to features_with_value_range
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
        std::cout << "Adding [" << class_names[i] << "]" << std::endl;
        classes_and_their_param_values[class_names[i]] = {};
    }

    /*
    *  Regex search for class names and their parameter values
    */
    std::regex class_params_regex(R"(params for class:\s+(\w+)\W*mean:\s*([\d\s.-]+)\W*covariance:\s*([\d\s.-]+))", std::regex_constants::icase);
    std::smatch match;

    std::string::const_iterator search_start(params.cbegin());

    // Search for class names and their parameter values, go through all the matches
    while (std::regex_search(search_start, params.cend(), match, class_params_regex))
    {
        std::string class_name = match[1].str();
        std::string mean_string = match[2].str();
        std::string covariance_string = match[3].str();

        std::vector<double> class_mean;
        std::istringstream mean_stream(mean_string);
        double val;

        // Split mean_string by ' ' delimiter and add to class_mean
        while (mean_stream >> val)
        {
            class_mean.push_back(val);
        }

        std::istringstream covariance_stream(covariance_string);
        std::string line;
        std::vector<std::vector<double>> covariance_matrix;

        // Split covariance_string by '\n' delimiter and add to covariance_matrix
        while (std::getline(covariance_stream, line))
        {
            std::istringstream row_stream(line);
            std::vector<double> row;
            double value;
            while (row_stream >> value)
            {
                // First add value to the row
                row.push_back(value);
            }
            if (!row.empty())
            {
                // Then add the row to the covariance matrix once full
                covariance_matrix.push_back(row);
            }
        }

        // Add class name, mean and covariance to classes_and_their_param_values
        classes_and_their_param_values[class_name]["mean"] = class_mean;
        classes_and_their_param_values[class_name]["covariance"] = std::vector<double>();

        // Because classes_and_their_param_values is a map of string to map of string to vector of double we need to flatten the covariance matrix into a single vector
        for (const auto &row : covariance_matrix)
        {
            classes_and_their_param_values[class_name]["covariance"].insert(
                classes_and_their_param_values[class_name]["covariance"].end(),
                row.begin(), row.end());
        }

        // Update search_start to the end of the current match
        search_start = match.suffix().first;
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

    // Update the class attributes
    _class_names = class_names;
    _class_names_and_priors = class_names_and_priors;
    _features_with_value_range = features_with_value_range;
    _classes_and_their_param_values = classes_and_their_param_values;
    _features_ordered = features_ordered;
}

// Function to generate multivariate normal samples, since Eigen does not have a built-in function for this
// Original Python implementation uses numpy.random.multivariate_normal, but this is not available in cpp
// We will use the Cholesky decomposition method to generate multivariate normal samples
std::vector<VectorXd> TrainingDataGeneratorNumeric::GenerateMultivariateSamples(const std::vector<double> &mean, const MatrixXd &cov, int num_samples)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist(0, 1);

    // Generate multivariate normal samples
    std::vector<VectorXd> samples;
    Eigen::LLT<MatrixXd> llt(cov);
    MatrixXd L = llt.matrixL(); // Cholesky decomposition

    // Add the a sample from the multivariate normal distribution to the samples vector
    // The sample is generated by multiplying the Cholesky decomposition (L) with a vector of random samples
    for (int i = 0; i < num_samples; ++i)
    {
        VectorXd z(mean.size());
        for (size_t j = 0; j < mean.size(); ++j)
        {
            z(j) = dist(gen);
        }
        VectorXd sample = L * z + VectorXd::Map(mean.data(), mean.size());
        samples.push_back(sample);
    }

    return samples;
}

void TrainingDataGeneratorNumeric::GenerateTrainingDataNumeric()
{
    std::map<std::string, std::vector<VectorXd>> samples_for_class;

    // Generate samples for each class
    for (const auto &class_entry : _classes_and_their_param_values)
    {
        // get class name, mean and covariance
        std::string class_name = class_entry.first;
        std::vector<double> mean = class_entry.second.at("mean");
        std::vector<double> cov_flat = class_entry.second.at("covariance");

        // Convert flat covariance back to matrix
        int dim = mean.size();
        MatrixXd cov_matrix(dim, dim);
        for (int i = 0; i < dim; ++i)
        {
            for (int j = 0; j < dim; ++j)
            {
                cov_matrix(i, j) = cov_flat[i * dim + j];
            }
        }

        // Generate multivariate normal samples
        samples_for_class[class_name] = GenerateMultivariateSamples(mean, cov_matrix, _number_of_samples_per_class);
    }

    // Store data records to be written to the CSV file
    // for each class, for each sample, create a data record
    std::vector<std::string> data_records;
    for (const auto &class_entry : samples_for_class)
    {
        // For each sample in the class, create a data record
        const std::string &class_name = class_entry.first;
        for (int sample_index = 0; sample_index < _number_of_samples_per_class; ++sample_index)
        {
            std::string data_record = class_name + ",";
            // For each feature in the sample, add to the data record
            for (int feature_index = 0; feature_index < class_entry.second[sample_index].size(); ++feature_index)
            {
                data_record += std::to_string(class_entry.second[sample_index](feature_index));

                // Add comma if not the last feature
                if (feature_index < class_entry.second[sample_index].size() - 1)
                {
                    data_record += ",";
                }
            }
            if (_debug)
            {
                std::cout << "Data record: " << data_record << std::endl;
            }
            data_records.push_back(data_record);
        }
    }

    // Shuffle the data records randomly
    std::srand(unsigned(std::time(0)));
    std::shuffle(data_records.begin(), data_records.end(), std::mt19937{std::random_device{}()});

    // Prepare the CSV output
    std::ofstream file(_output_csv_file);
    file << "\"\",class_name," << std::accumulate(_features_ordered.begin(), _features_ordered.end(), std::string(), [](const std::string &acc, const std::string &feature)
                                                  { return acc + (acc.empty() ? "" : ",") + feature; })
         << "\n";

    // Write the data records to the CSV file
    for (size_t i = 0; i < data_records.size(); ++i)
    {
        file << (i + 1) << "," << data_records[i] << "\n";
    }
    file.close();
}

/*
* Getters
*/
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