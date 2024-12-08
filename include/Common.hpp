#ifndef COMMON_HPP
#define COMMON_HPP

#include <algorithm>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <vector>

// Stream
using std::cout;
using std::endl;

// Data structures
using std::map;
using std::pair;
using std::set;
using std::string;
using std::tuple;
using std::vector;

// Pointers
using std::make_shared;
using std::make_unique;
using std::optional;
using std::shared_ptr;
using std::unique_ptr;

#endif // COMMON_HPP


#ifndef CONSOLE_COLORS_HPP
#define CONSOLE_COLORS_HPP

namespace ConsoleColors {
const string RESET     = "\033[0m";
const string BOLD      = "\033[1m";
const string UNDERLINE = "\033[4m";

const string BLACK   = "\033[30m";
const string RED     = "\033[31m";
const string GREEN   = "\033[32m";
const string YELLOW  = "\033[33m";
const string BLUE    = "\033[34m";
const string MAGENTA = "\033[35m";
const string CYAN    = "\033[36m";
const string WHITE   = "\033[37m";

const string BOLD_RED     = "\033[1m\033[31m";
const string BOLD_GREEN   = "\033[1m\033[32m";
const string BOLD_YELLOW  = "\033[1m\033[33m";
const string BOLD_BLUE    = "\033[1m\033[34m";
const string BOLD_MAGENTA = "\033[1m\033[35m";
const string BOLD_CYAN    = "\033[1m\033[36m";
const string BOLD_WHITE   = "\033[1m\033[37m";
} // namespace ConsoleColors

#endif // CONSOLE_COLORS_HPP