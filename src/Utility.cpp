#include "Utility.hpp"
#include <regex>
#include <iostream>

int sampleIndex(std::string sample_name)
{
    // The purpose of this function is to return the identifying integer associated with a data record.return -1;
    std::regex regex("_(.+)$");
    std::smatch match;
    std::regex_search(sample_name, match, regex);
    return std::stoi(match[1]);
};

template <typename T>
std::vector<T> deepCopy(std::vector<T> const &vec)
{
    // The purpose of this function is to create a deep copy of a vector.
    std::vector<T> copy;
    for (T const &elem : vec)
    {
        copy.push_back(elem);
    }
    return copy;
}

template <typename T>
std::pair<T, size_t> minimum(std::vector<T> const &vec)
{
    T min = vec[0];
    size_t index = 0;
    for (size_t i = 1; i < vec.size(); ++i)
    {
        if (vec[i] < min)
        {
            min = vec[i];
            index = i;
        }
    }
    return std::make_pair(min, index);
};

double convert(std::string const &str)
{
    // The purpose of this function is to convert a string to a double.
    return std::stod(str);
};

template <typename T>
std::optional<T> ClosestSamplingPoint(std::vector<T> const &vec, T const &val)
{
    // try to cast the val to a floating point number
    double value = static_cast<double>(val);

    // find the closest sampling point
    double min_diff = std::abs(value - vec[0]);
    size_t index = 0;
    for (size_t i = 1; i < vec.size(); ++i)
    {
        double diff = std::abs(value - vec[i]);
        if (diff < min_diff)
        {
            min_diff = diff;
            index = i;
        }
    }

    // return the closest sampling point
    return vec[index];
};

std::string CleanupCsvString(std::string line)
{
    // The purpose of this function is to clean up a CSV string.
    // Replace special characters with spaces
    std::regex special_chars(":?/()[]{}'");
    line = std::regex_replace(line, special_chars, "          ");

    // Replace double-quoted strings with cleaned versions
    std::regex double_quoted("\"[^\"]+\"");
    std::smatch match;
    while (std::regex_search(line, match, double_quoted))
    {
        std::string item = match.str();
        std::string clean = std::regex_replace(item.substr(1, item.length() - 2), std::regex(","), "");
        std::regex whitespace("\\s+");
        clean = std::regex_replace(clean, whitespace, "_");
        clean = std::regex_replace(clean, std::regex("^_|_$"), "");
        line = std::regex_replace(line, double_quoted, clean);
    }

    // Replace white-spaced items between commas with cleaned versions
    std::regex white_spaced(",\\s*[^,]+(?=,|$)");
    while (std::regex_search(line, match, white_spaced))
    {
        std::string item = match.str();
        std::string litem = item;
        std::regex whitespace("\\s+");
        litem = std::regex_replace(litem, whitespace, "_");
        litem = std::regex_replace(litem, std::regex("^\\s*_|_\\s*$"), "");
        line = line.substr(0, line.find(item)) + "," + litem + line.substr(line.find(item) + item.length());
    }

    // Split the line into fields separated by commas
    std::regex fields(",");
    std::vector<std::string> newfields;
    std::sregex_token_iterator iter(line.begin(), line.end(), fields, -1);
    std::sregex_token_iterator end;
    for (; iter != end; ++iter)
    {
        std::string field = iter->str();
        std::string newfield = field;
        if (newfield.empty())
        {
            newfields.push_back("NA");
        }
        else
        {
            newfields.push_back(newfield);
        }
    }

    // Replace fields with commas
    line = std::regex_replace(line, fields, ",");

    // Return the cleaned up line
    return line;
}