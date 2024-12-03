#include "Utility.hpp"

#include <cmath>
#include <iomanip>
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

/**
 * @brief Cleans up a CSV string by removing unwanted characters, handling double-quoted text,
 *        and normalizing whitespace.
 *
 * This function performs the following operations on the input CSV string:
 * 1. Translates unwanted characters (":?/()[]{}'") to spaces.
 * 2. Handles double-quoted text by removing commas within quotes and replacing whitespace with underscores.
 * 3. Normalizes whitespace between commas by replacing it with underscores.
 * 4. Splits the string by commas and trims leading/trailing whitespace or underscores from each field.
 * 5. Replaces empty fields with "NA".
 * 6. Joins the cleaned fields back together with commas.
 *
 * @param line The input CSV string to be cleaned.
 * @return A cleaned CSV string with unwanted characters removed, double-quoted text handled,
 *         and whitespace normalized.
 */
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
        cleanMatch        = std::regex_replace(cleanMatch, std::regex("\\s+"), "_");
        cleaned           = std::regex_replace(
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

/**
 * @brief Rounds a double value to a specified precision and returns it as a string.
 *
 * This function takes a double value and rounds it to the specified number of decimal places.
 * The result is then converted to a string and returned.
 *
 * @param value The double value to be rounded.
 * @param precision The number of decimal places to round to. Default is 3.
 * @return A string representation of the rounded double value.
 */
std::string roundDouble(double value, int precision)
{
    std::ostringstream out;
    out << std::fixed << std::setprecision(precision) << value;
    return out.str();
}

/**
 * @brief Joins a vector of strings into a single string with a specified delimiter.
 *
 * This function takes a vector of strings and concatenates them into a single
 * string, with each element separated by the specified delimiter.
 *
 * @param elements The vector of strings to join.
 * @param delimiter The string to use as a delimiter between elements.
 * @return A single string containing all elements of the input vector, separated by the delimiter.
 */
std::string join(const std::vector<std::string> &elements, const std::string &delimiter)
{
    std::ostringstream joined;
    for (size_t i = 0; i < elements.size(); ++i) {
        joined << elements[i];
        if (i < elements.size() - 1) {
            joined << delimiter;
        }
    }
    return joined.str();
}

/**
 * @brief Normalizes the given input string by trimming leading and trailing spaces,
 *        removing empty lines, and collapsing multiple spaces into a single space.
 *
 * @param input The input string to be normalized.
 * @return A normalized string with trimmed spaces, no empty lines, and single spaces.
 */
std::string normalizeString(const std::string &input)
{
    std::ostringstream normalized;
    std::istringstream inputStream(input);
    std::string line;

    while (std::getline(inputStream, line)) {
        // Trim leading and trailing spaces
        line.erase(0, line.find_first_not_of(" \t"));
        line.erase(line.find_last_not_of(" \t") + 1);

        // Trim empty lines found
        if (line.empty()) {
            continue;
        }

        // Replace multiple spaces with a single space
        std::string collapsed;
        bool inSpace = false;
        for (char ch : line) {
            if (ch == ' ' || ch == '\t') {
                if (!inSpace) {
                    collapsed += ' '; // Add a single space
                    inSpace = true;
                }
            }
            else {
                collapsed += ch;
                inSpace = false;
            }
        }

        // Append the cleaned-up line
        normalized << collapsed << "\n";
    }
    return normalized.str();
}

/**
 * @brief Trims leading and trailing whitespace from a given string.
 *
 * This function removes any leading and trailing spaces or tab characters
 * from the input string. If the string is empty or contains only whitespace,
 * an empty string is returned.
 *
 * @param str The input string to be trimmed.
 * @return A new string with leading and trailing whitespace removed.
 */
std::string trim(const std::string &str)
{
    size_t first = str.find_first_not_of(" \t");
    size_t last  = str.find_last_not_of(" \t");
    if (first == std::string::npos || last == std::string::npos) {
        return ""; // Empty or all whitespace
    }
    return str.substr(first, (last - first + 1));
}