# User Manual
## Getting Started
These instructions will help you set up, build, and run the Decision Tree++ project, a C++ and Python-compatible decision tree library. With bindings provided via Pybind11, you can seamlessly use the library in both C++ and Python environments.

## Requirements

- **[CMake](https://cmake.org/download/)**: Version 3.10 or higher
- **[Python](https://www.python.org/downloads/)**: Python 3 with  `setuptools` and `pybind11`  for Python code compilation
- **[GTest](https://github.com/google/googletest)**: For unit testing (optional)

## Cloning
To clone this project run the following command:
```bash
git clone https://github.com/galaxybomb23/Decision-Tree-Plus-Plus.git
```

## Usage

Run the script with one of the following options:
```bash
./run.sh {build|clean|test|install|build-python|install-python|demo}
``` 

### Options
-   **build**: Compiles the project by creating a  `build`  directory (if not already present), configuring with CMake, and running  `make`.
```bash
./run.sh build
``` 
    
-   **clean**: Removes the  `build`  directory, cleaning up any previously compiled files.
```bash
./run.sh clean
``` 
    
-   **test**: Builds and runs all tests, outputting results and any errors encountered.
```bash
./run.sh test
``` 
    
-   **install**: Cleans the  `build`  directory, builds the project in  `Release`  mode, and installs the project locally within the  `build`  directory. For a system-wide installation, run  `sudo cmake --install .`  within the  `build`directory.
```bash
./run.sh install
``` 

-   **build-python**: Only builds the Python code using the setup script in  `Python-build/setup.py` into and shared object file `Python-Build/DecisionTreePP.cpython-310-x86_64-linux-gnu.so`.
```bash
./run.sh build-python
```

-   **install-python**: Compiles Python code using the setup script found in  `Python-build/setup.py`.
```bash
./run.sh install-python
``` 
    
-   **demo**: Compiles the Python code and then runs a demo script located in  `Python-build/demo.py`.
```bash
./run.sh demo
``` 
    
## Manual Installation
Navigate to the project's root directory and run the following commands.
```bash
mkdir build
cd build
```
Run CMAKE to configure the project.
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release
```

Build the library and test suite.
```bash
cmake --build .
```

Run test suite to validate install.
```bash
ctest --output-on-failure
```

## Using the Library
The Decision Tree++ library can be called directly in C++ or imported into Python.
To use it in C++ you have to include the library in your `.cpp` file like the following:
```c++
#include "DecisionTree.h"
```
And then link it to your project during compilation.
```bash
g++ your_program.cpp -L/path/to/DecisionTree/build -lDecisionTree -o your_program
```

To use it in Python, you’ll need to install it with the Python bindings and then import it as follows:
```python
import DecisionTreePP
```

## Examples
Training data is supplied via text files, which are read into the `DecisionTree` constructur:

```cpp
std::map<std::string, std::string> kwargs = {
      {"training_datafile", "../test/resources/training_symbolic.csv"},
      {"csv_class_column_index", "1"},
      {"csv_columns_for_features", {2, 3, 4, 5}},
      {"max_depth_desired", "5"},
      {"entropy_threshold", "0.1"}
      // further arguments
  };
DecisionTree dt = DecisionTree(kwargs);
DecisionTreeNode node =
      DecisionTreeNode("feature", 0.0, {0.0}, {"branch"}, dt, true);
}; // creating a node
```

A user may then read the training data and set up the tree for classification:

```cpp
dt.getTrainingData()
dt.calculateFirstOrderProbabilities()
dt.calculateClassPriors()

// create a root node
auto root = dt.constructDecisionTreeClassifier()

// define sample
const std::vector<std::string> featuresAndValues = {
        "g2 = 4.2",
        "grade = 2.3",
        "gleason = 4",
        "eet = 1.7",
        "age = 55.0",
        "ploidy = diploid"
    };

// classify
auto classification = dt.classify(root_node, test_sample)
```

This will return reference to a hash map where the keys represent the class names and the values indicate the corresponding classification probabilities. Additionally, this hash map contains an extra key-value pair detailing the solution path from the root node to the leaf node where the final classification was determined.

For more examples on usage and details on library functionality, check the  `demo.py`  script in the  `Python-build`  directory. You can also check the test cases located in the `test` directory for more examples on how to use the code.
