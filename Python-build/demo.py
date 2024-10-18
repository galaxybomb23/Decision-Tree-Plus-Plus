import DecisionTree as dt
import DecisionTreePP as dtp

import matplotlib.pyplot as plt
import pandas as pd
import time
# change cwd to the directory of the file
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def CSVParsingDemo():

    kwargs = {
        "training_datafile": "../test/resources/training_symbolic.csv",
        "csv_class_column_index": "1",
        "csv_columns_for_features": {2,3,4,5},
        "max_depth_desired": "5",
        "entropy_threshold": "0.1"
    }


    #generate pyhton object from the kwargs
    start_time = time.time()
    Pdt = dt.DecisionTree(training_datafile=kwargs["training_datafile"],
                         csv_class_column_index=int(kwargs["csv_class_column_index"]),
                         csv_columns_for_features=list(kwargs["csv_columns_for_features"]),
                         max_depth_desired=int(kwargs["max_depth_desired"]),
                         entropy_threshold=float(kwargs["entropy_threshold"]))
    Pdt.get_training_data()
    Pdt.show_training_data()
    end_time = time.time()
    python_time = end_time - start_time

    #prompt user to continue
    input("Press Enter to continue...")

    kwargs["csv_columns_for_features"] = "[2,3,4,5]"

    #generate C++ object from the kwargs
    start_time = time.time()
    Cdt = dtp.DecisionTree(kwargs)
    Cdt.getTrainingData()
    Cdt.showTrainingData()
    end_time = time.time()
    cpp_time = end_time - start_time

    #print the time it took to run the python and c++ code
    print(f"Speed: Python: {python_time:.3f}s C++: {cpp_time:.3f}s | Speedup: {python_time/cpp_time:.2f}x")

    #prompt user to continue
    input("Press Enter to continue...")

    #clear terminal
    os.system("clear")
    return

def donutDemo():
    # fps = 120
    # distance = 2
    # increment = 0
    # refreshRate = 0
    # xpos = 40
    # ypos = 10
    # numupdates = 320
    # dtp.doughnut(fps, distance, increment, refreshRate, xpos, ypos, numupdates)

    # #clear terminal
    # os.system("clear")
    return
    
def dataGenerationDemo():

    #C++ data generation demo
    kwargs = {
        "output_csv_file": "../test/resources/param_numeric_out_C++.csv",
        "parameter_file": "../test/resources/param_numeric.txt",
        "number_of_samples_per_class": "300000",
        "debug": "0"
    }
    print("Running C++ data generation...")
    start_time = time.time()
    Cdg = dtp.TrainingDataGeneratorNumeric(kwargs);
    Cdg.ReadParameterFileNumeric()
    Cdg.GenerateTrainingDataNumeric()
    end_time = time.time()
    cpp_time = end_time - start_time

    #Python data generation demo
    print("Running Python data generation...")
    kwargs["output_csv_file"] = "../test/resources/param_numeric_out_Python.txt"
    start_time = time.time()
    Pdg = dt.TrainingDataGeneratorNumeric(output_csv_file=kwargs["output_csv_file"],
                                         parameter_file=kwargs["parameter_file"],
                                         number_of_samples_per_class=int(kwargs["number_of_samples_per_class"]),
                                         debug=int(kwargs["debug"]));
    Pdg.read_parameter_file_numeric()
    Pdg.gen_numeric_training_data_and_write_to_csv()
    end_time = time.time()
    python_time = end_time - start_time
    #plot both data sets


    # Read the generated data from the CSV files
    data_Python = pd.read_csv('../test/resources/param_numeric_out_Python.txt')

    #sort data
    recession_python = data_Python[data_Python['class_name'] == 'recession']
    goodtimes_python = data_Python[data_Python['class_name'] == 'goodtimes']

    recession_pyhton_gdp_mean = recession_python['gdp'].mean()
    recession_pyhton_return_on_invest_mean = recession_python['return_on_invest'].mean()

    goodtimes_pyhton_gdp_mean = goodtimes_python['gdp'].mean()
    goodtimes_pyhton_return_on_invest_mean = goodtimes_python['return_on_invest'].mean()


    # Create a figure with 4 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Plot histograms for Gdp and Return on invest for goodtimes
    ax1.hist(goodtimes_python['gdp'], bins=30, alpha=0.5, label='gdp')
    ax1.hist(goodtimes_python['return_on_invest'], bins=30, alpha=0.5, label='return_on_invest')
    ax1.set_title('Python Goodtimes Data Distribution')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Frequency')
    ax1.legend()

    # Plot histograms for Gdp and Return on invest for recession
    ax2.hist(recession_python['gdp'], bins=30, alpha=0.5, label='gdp')
    ax2.hist(recession_python['return_on_invest'], bins=30, alpha=0.5, label='return_on_invest')
    ax2.set_title('Python Recession Data Distribution')
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Frequency')
    ax2.legend()

    #do all that for C++
    data_Cpp = pd.read_csv('../test/resources/param_numeric_out_C++.csv')

    #sort data
    recession_cpp = data_Cpp[data_Cpp['class_name'] == 'recession']
    goodtimes_cpp = data_Cpp[data_Cpp['class_name'] == 'goodtimes']

    recession_cpp_gdp_mean = recession_cpp['gdp'].mean()
    recession_cpp_return_on_invest_mean = recession_cpp['return_on_invest'].mean()
    
    goodtimes_cpp_gdp_mean = goodtimes_cpp['gdp'].mean()
    goodtimes_cpp_return_on_invest_mean = goodtimes_cpp['return_on_invest'].mean()


    # Plot histograms for Gdp and Return on invest for goodtimes
    ax3.hist(goodtimes_cpp['gdp'], bins=30, alpha=0.5, label='gdp')
    ax3.hist(goodtimes_cpp['return_on_invest'], bins=30, alpha=0.5, label='return_on_invest')
    ax3.set_title('C++ Goodtimes Data Distribution')
    ax3.set_xlabel('Value')
    ax3.set_ylabel('Frequency')
    ax3.legend()

    # Plot histograms for Gdp and Return on invest for recession
    ax4.hist(recession_cpp['gdp'], bins=30, alpha=0.5, label='gdp')
    ax4.hist(recession_cpp['return_on_invest'], bins=30, alpha=0.5, label='return_on_invest')
    ax4.set_title('C++ Recession Data Distribution')
    ax4.set_xlabel('Value')
    ax4.set_ylabel('Frequency')
    ax4.legend()

    # Adjust layout and save plot   
    plt.tight_layout()
    plt.savefig('../test/resources/param_numeric_out_combined.png')
    plt.close()

    os.system("clear")
    print("Metrics                          Pyhton/C++ : Actual")
    print(f"Goodtimes Gdp Mean               {goodtimes_pyhton_gdp_mean:.2f} / {goodtimes_cpp_gdp_mean:.2f} : 50")
    print(f"Goodtimes Return on invest Mean  {goodtimes_pyhton_return_on_invest_mean:.2f} / {goodtimes_cpp_return_on_invest_mean:.2f} : 60")
    print(f"Recession Gdp Mean               {recession_pyhton_gdp_mean:.2f} / {recession_cpp_gdp_mean:.2f} : 50")
    print(f"Recession Return on invest Mean  {recession_pyhton_return_on_invest_mean:.2f} / {recession_cpp_return_on_invest_mean:.2f} : 30")
    
    print(f"Speed: Python: {python_time:.2f}s C++: {cpp_time:.2f}s | Speedup: {python_time/cpp_time:.2f}x")

    input("Press Enter to continue...")
    #remove the generated files
    # os.remove('../test/resources/param_numeric_out_C++.txt')
    os.remove('../test/resources/param_numeric_out_Python.txt')
    os.remove('../test/resources/param_numeric_out_combined.png')
    #clear terminal
    os.system("clear")
    return

def DecisionTreeNodeDemo():
    # kwargs = {
    #     "training_datafile": "../test/resources/stage3cancer.csv",
    #     "entropy_threshold": "0.1",
    #     "max_depth_desired": "20",
    #     "csv_class_column_index": "1",
    #     "symbolic_to_numeric_cardinality_threshold": "20",
    #     "csv_columns_for_features": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    #     "number_of_histogram_bins": "10",
    #     "csv_cleanup_needed": "1",
    #     "debug1": "1",
    #     "debug2": "2",
    #     "debug3": "3"
    # }
    # dtp.display_decision_treeDemo()

    # # Prompt user to continue
    # input("Press Enter to continue...")
    # os.system("clear")
    return

def CSVParsingLargeDemo():
    kwargs = {
        "training_datafile": "../test/resources/param_numeric_out_C++.csv",
        "csv_class_column_index": "1",
        "csv_columns_for_features": {2,3},
        "max_depth_desired": "5",
        "entropy_threshold": "0.1"
    }
    start_time = time.time()
    Pdt = dt.DecisionTree(training_datafile=kwargs["training_datafile"],
                         csv_class_column_index=int(kwargs["csv_class_column_index"]),
                         csv_columns_for_features=list(kwargs["csv_columns_for_features"]),
                         max_depth_desired=int(kwargs["max_depth_desired"]),
                         entropy_threshold=float(kwargs["entropy_threshold"]))
    Pdt.get_training_data()
    end_time = time.time()
    python_time = end_time - start_time

    kwargs["csv_columns_for_features"] = "[2,3]"
    start_time = time.time()
    Cdt = dtp.DecisionTree(kwargs)
    Cdt.getTrainingData()
    end_time = time.time()
    cpp_time = end_time - start_time    

    print(f"Speed: Python: {python_time:.2f}s C++: {cpp_time:.2f}s | Speedup: {python_time/cpp_time:.2f}x")

    input("Press Enter to continue...")

    #clear terminal
    os.system("clear")
    return


#interactive demo
def display_menu():
    os.system("clear")
    print("Decision Tree Demo Menu")
    print("1. Donut Demo")
    print("2. Decision Tree Node Demo")
    print("3. Data Generation Demo")
    print("4. CSV Parsing Demo")
    print("5. CSV Parsing Large Demo")
def main():
    while True:
        display_menu()
        choice = input("Enter your choice (1-5): ")
        if choice == '1':
            donutDemo()
            display_menu()
        elif choice == '2':
            DecisionTreeNodeDemo()
        elif choice == '3':
            dataGenerationDemo()
        elif choice == '4':
            CSVParsingDemo()
        elif choice == '5':
            CSVParsingLargeDemo()
        else:
            os.system("clear")
        print("\n")

if __name__ == "__main__":
    main()
