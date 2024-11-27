#include "Utility.hpp"

#include <cmath>
#include <iostream>
#include <regex>
#include <sstream>

int sampleIndex(string sample_name)
{
    // The purpose of this function is to return the identifying integer associated with a data record.return -1;
    std::regex regex("_(.+)$");
    std::smatch match;
    std::regex_search(sample_name, match, regex);
    return std::stoi(match[1]);
};

double convert(const string &str)
{
    // The purpose of this function is to convert a string to a double.
    try {
        return std::stod(str);
    }
    catch (const std::exception &e) {
        return std::nan("");
    }
}

string CleanupCsvString(const string &line)
{
    // cout << "\nOriginal: " << line << endl;
    // Translate unwanted characters ":?/()[]{}'" to spaces
    string cleaned = std::regex_replace(line, std::regex("[:?/()\\[\\]{}']"), " ");
    // cout << "Special-Chars: " << cleaned << "|" << endl;

    // Handle double-quoted text
    std::regex doubleQuotedPattern(R"("[^"]+")");
    auto words_begin = std::sregex_iterator(cleaned.begin(), cleaned.end(), doubleQuotedPattern);
    auto words_end   = std::sregex_iterator();
    for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
        string match      = (*i).str();
        string cleanMatch = std::regex_replace(match.substr(1, match.size() - 2), std::regex(","), "");
        cleanMatch             = std::regex_replace(cleanMatch, std::regex("\\s+"), "_");
        cleaned                = std::regex_replace(
            cleaned, std::regex(std::regex_replace(match, std::regex(R"([\{\}])"), "\\$&")), cleanMatch);
    }
    // cout << "Double-quoted: " << cleaned << "|" << endl;

    // Handle whitespace between commas
    std::regex whitespacePattern(R"(,(\s*[^,]+)(?=,|$)$)");
    words_begin = std::sregex_iterator(cleaned.begin(), cleaned.end(), whitespacePattern);
    for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
        string match = (*i).str();
        // cout << "Match: " << match << endl;
        string cleanMatch = std::regex_replace(match, std::regex("\\s+"), "_");
        // cout << "Clean-Match: " << cleanMatch << endl;
        cleanMatch = std::regex_replace(cleanMatch, std::regex("^\\s*_|_\\s*$"), "");
        // cout << "Clean-Match: " << cleanMatch << endl;
        cleaned = std::regex_replace(
            cleaned, std::regex(std::regex_replace(match, std::regex(R"([\{\}])"), "\\$&")), " " + cleanMatch);
    }
    // cout << "Whitespace: " << cleaned << "|" << endl;

    // Split by comma, clean up fields
    vector<string> fields;
    string field;
    std::stringstream ss(cleaned);
    while (std::getline(ss, field, ',')) {
        field = std::regex_replace(field, std::regex("^(\\s|_)+|(\\s|_)+$"), ""); // Trim whitespace
        if (field == "") {
            fields.push_back("NA");
        }
        else {
            fields.push_back(field);
        }
    }

    // If the string ends with an empty field, add "NA" to the end
    if (cleaned.back() == ',') {
        fields.push_back("NA");
    }

    // Join the fields back together with commas
    string result;
    for (size_t i = 0; i < fields.size(); ++i) {
        result += fields[i];
        if (i < fields.size() - 1) {
            result += ",";
        }
    }

    return result;
}

string removeTrailingZeros(const string &str)
{
    // Remove trailing zeros after the decimal point
    string result = str;
    result.erase(result.find_last_not_of('0') + 1, string::npos);

    // If the last character is a decimal point, remove it
    if (result.back() == '.') {
        result.pop_back();
    }
    return result;
}

string formatDouble(double value)
{
    std::stringstream ss;
    ss << std::fixed << value;            // Convert double to string with fixed-point notation
    return removeTrailingZeros(ss.str()); // Remove any unnecessary trailing zeros
}