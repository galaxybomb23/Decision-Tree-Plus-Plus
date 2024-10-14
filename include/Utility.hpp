#ifndef UTILITY_HPP
#define UTILITY_HPP

#include <string>
#include <vector>
#include <optional>
#include <algorithm>
#include <utility>
#include <regex>
#include <iostream>
#include <tuple>
#include <numeric>

int sampleIndex(std::string sample_name);

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
} // may be unnecessary

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
}

double convert(std::string const &str);

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

std::string CleanupCsvString(std::string const &str);

#endif // UTILITY_HPP
