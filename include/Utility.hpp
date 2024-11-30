#ifndef UTILITY_HPP
#define UTILITY_HPP

// Include
#include "Common.hpp"

#include <cmath>
#include <numeric>
#include <regex>
#include <utility>

int sampleIndex(string sample_name);

template <typename T> vector<T> deepCopy(const vector<T> &vec)
{
    // The purpose of this function is to create a deep copy of a vector.
    vector<T> copy;
    for (const T &elem : vec) {
        copy.push_back(elem);
    }
    return copy;
} // may be unnecessary

template <typename T> pair<T, size_t> minimum(const vector<T> &vec)
{
    T min        = vec[0];
    size_t index = 0;
    for (size_t i = 1; i < vec.size(); ++i) {
        if (vec[i] < min) {
            min   = vec[i];
            index = i;
        }
    }
    return std::make_pair(min, index);
}

double convert(const string &str);

inline double ClosestSamplingPoint(const vector<double> &vec, const double &val)
{
    // Check if val is NAN
    if (std::isnan(val)) {
        return val;
    }

    // find the closest sampling point
    double min_diff = std::abs(val - vec[0]);
    size_t index    = 0;
    for (size_t i = 1; i < vec.size(); ++i) {
        double diff = std::abs(val - vec[i]);
        if (diff < min_diff) {
            min_diff = diff;
            index    = i;
        }
    }

    // return the closest sampling point
    return vec[index];
};

string CleanupCsvString(const string &str);

string removeTrailingZeros(const string &str);
string formatDouble(double value);

/**
 * @brief Overloaded stream insertion operator for printing vectors.
 *
 * This template function allows for printing the contents of a vector to an
 * output stream in a formatted manner. The elements of the vector are enclosed
 * in square brackets and separated by commas.
 *
 * @tparam T The type of elements contained in the vector.
 * @param os The output stream to which the vector will be printed.
 * @param v The vector to be printed.
 * @return A reference to the output stream.
 */
template <typename T> std::ostream &operator<<(std::ostream &os, const vector<T> &v)
{
    os << "[";
    for (int i = 0; i < v.size(); ++i) {
        os << v[i];
        if (i != v.size() - 1)
            os << ", ";
    }
    os << "]";
    return os;
}

// Function to print out a map
template <typename K, typename V>
std::ostream& operator<<(std::ostream& os, const std::map<K, V>& m) {
    os << "{ ";
    bool first = true;
    for (const auto& [key, value] : m) {
        if (!first) {
            os << ", ";
        }
        first = false;
        os << key << ": " << value;
    }
    os << " }" << std::endl;
    return os;
}

// All helper functions for ConstructTreeTests
std::string roundDouble(double value, int precision = 3);
std::string join(const std::vector<std::string> &elements, const std::string &delimiter);
std::string normalizeString(const std::string &input);

// MARK: Comment
std::string trim(const std::string& str);

#endif // UTILITY_HPP
