import DecisionTree as dt
import DecisionTreePP as dtp

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
    Pdt = dt.DecisionTree(training_datafile=kwargs["training_datafile"],
                         csv_class_column_index=int(kwargs["csv_class_column_index"]),
                         csv_columns_for_features=list(kwargs["csv_columns_for_features"]),
                         max_depth_desired=int(kwargs["max_depth_desired"]),
                         entropy_threshold=float(kwargs["entropy_threshold"]))
    Pdt.get_training_data()
    Pdt.show_training_data()

    #prompt user to continue
    input("Press Enter to continue...")

    kwargs["csv_columns_for_features"] = "[2,3,4,5]"

    #generate C++ object from the kwargs
    Cdt = dtp.DecisionTree(kwargs)
    Cdt.getTrainingData()
    Cdt.showTrainingData()

    #prompt user to continue
    input("Press Enter to continue...")

    #clear terminal
    os.system("clear")
    return

CSVParsingDemo()
    
    