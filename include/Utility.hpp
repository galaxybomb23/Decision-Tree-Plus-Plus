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
    std::vector<T> copy;
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

// Overload the << operator for vectors
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

#endif // UTILITY_HPP