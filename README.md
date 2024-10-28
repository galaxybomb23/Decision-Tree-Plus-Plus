# Decision Tree++

## Requirements

-   **CMake**: Version 3.10 or higher
-   **Python**: Python 3 with  `setuptools`  for Python code compilation
-   **GTest**: For unit testing (optional)

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

-   **build-python**: Only builds the Python code using the setup script in  `Python-build/setup.py`.
```bash
./run.sh python
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