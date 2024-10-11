#include "Utility.hpp"
#include <regex>
#include <iostream>
#include <sstream>

int sampleIndex(std::string sample_name)
{
    // The purpose of this function is to return the identifying integer associated with a data record.return -1;
    std::regex regex("_(.+)$");
    std::smatch match;
    std::regex_search(sample_name, match, regex);
    return std::stoi(match[1]);
};

double convert(std::string const &str)
{
    // The purpose of this function is to convert a string to a double.
    return std::stod(str);
}

std::string CleanupCsvString(const std::string &line)
{
    // Translate unwanted characters ":?/()[]{}'" to spaces
    std::string cleaned = std::regex_replace(line, std::regex("[:?/()\\[\\]{}']"), " ");

    // Handle double-quoted text
    std::regex doubleQuotedPattern(R"("[^"]+")");
    auto words_begin = std::sregex_iterator(cleaned.begin(), cleaned.end(), doubleQuotedPattern);
    auto words_end = std::sregex_iterator();
    for (std::sregex_iterator i = words_begin; i != words_end; ++i)
    {
        std::string match = (*i).str();
        std::string cleanMatch = std::regex_replace(match.substr(1, match.size() - 2), std::regex(","), "");
        cleanMatch = std::regex_replace(cleanMatch, std::regex("\\s+"), "_");
        cleaned = std::regex_replace(cleaned, std::regex(std::regex_replace(match, std::regex(R"([\{\}])"), "\\$&")), cleanMatch);
    }

    // Handle whitespace between commas
    std::regex whitespacePattern(R"(,(\s*[^,]+)(?=,|$)$)");
    words_begin = std::sregex_iterator(cleaned.begin(), cleaned.end(), whitespacePattern);
    for (std::sregex_iterator i = words_begin; i != words_end; ++i)
    {
        std::string match = (*i).str();
        std::string cleanMatch = std::regex_replace(match, std::regex("\\s+"), "_");
        cleanMatch = std::regex_replace(cleanMatch, std::regex("^\\s*_|_\\s*$"), "");
        cleaned = std::regex_replace(cleaned, std::regex(std::regex_replace(match, std::regex(R"([\{\}])"), "\\$&")), " " + cleanMatch);
    }

    // Split by comma, clean up fields
    std::vector<std::string> fields;
    std::string field;
    std::stringstream ss(cleaned);
    while (std::getline(ss, field, ',')) {
        field = std::regex_replace(field, std::regex("^(\\s|_)+|(\\s|_)+$"), ""); // Trim whitespace
        if (field == "") {
            fields.push_back("NA");
        } else {
            fields.push_back(field);
        }
    }

    // If the string ends with an empty field, add "NA" to the end
    if (cleaned.back() == ',') {
        fields.push_back("NA");
    }

    // Join the fields back together with commas
    std::string result;
    for (size_t i = 0; i < fields.size(); ++i)
    {
        result += fields[i];
        if (i < fields.size() - 1)
        {
            result += ",";
        }
    }

    return result;
}