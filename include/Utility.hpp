#ifndef UTILITY_HPP
#define UTILITY_HPP

#include <string>
#include <vector>
#include <optional>

int sampleIndex(std::string sample_name);

template <typename T>
std::vector<T> deepCopy(std::vector<T> const &vec); // may be unnecessary

template <typename T>
std::pair<T, size_t> minimum(std::vector<T> const &vec);

double convert(std::string const &str);

template <typename T>
std::optional<T> ClosestSamplingPoint(std::vector<T> const &vec, T const &val);

std::string CleanupCsvString(std::string const &str);

#endif // UTILITY_HPP