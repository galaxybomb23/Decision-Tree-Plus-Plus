#ifndef UTILITY_HPP
#define UTILITY_HPP

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <optional>
#include <regex>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

int sampleIndex(std::string sample_name);

template <typename T> std::vector<T> deepCopy(const std::vector<T> &vec)
{
    // The purpose of this function is to create a deep copy of a vector.
    std::vector<T> copy;
    for (const T &elem : vec) {
        copy.push_back(elem);
    }
    return copy;
} // may be unnecessary

template <typename T> std::pair<T, size_t> minimum(const std::vector<T> &vec)
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

double convert(const std::string &str);

template <typename T>
double ClosestSamplingPoint(const std::vector<T> &vec,
                            const T &val) inline double ClosestSamplingPoint(const std::vector<double> &vec,
                                                                             const double &val)
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

std::string CleanupCsvString(const std::string &str);

std::string removeTrailingZeros(const std::string &str);
std::string formatDouble(double value);

// for vector output
template <typename T> std::ostream &operator<<(std::ostream &os, const std::vector<T> &vec)
{
    for (const auto &elem : vec) {
        os << elem << " ";
    }
    return os;
}

template <typename T> std::ostream &operator<<(std::ostream &os, const std::set<T> &set)
{
    for (const auto &elem : set) {
        os << elem << " ";
    }
    return os;
}

#endif // UTILITY_HPP
