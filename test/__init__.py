#!/usr/bin/env python

import sys

if sys.version_info[0] == 3:
    from DecisionTree.DecisionTree import __version__
    from DecisionTree.DecisionTree import __author__
    from DecisionTree.DecisionTree import __date__
    from DecisionTree.DecisionTree import __url__
    from DecisionTree.DecisionTree import __copyright__

    from DecisionTree.DecisionTree import DecisionTree
    from DecisionTree.DecisionTree import EvalTrainingData
    from DecisionTree.DecisionTree import DTIntrospection
    from DecisionTree.DecisionTree import TrainingDataGeneratorNumeric
    from DecisionTree.DecisionTree import TrainingDataGeneratorSymbolic

    from  DecisionTree.DecisionTree import DTNode

else:
    from DecisionTree import __version__
    from DecisionTree import __author__
    from DecisionTree import __date__
    from DecisionTree import __url__
    from DecisionTree import __copyright__

    from DecisionTree import DecisionTree
    from DecisionTree import EvalTrainingData
    from DecisionTree import DTIntrospection
    from DecisionTree import TrainingDataGeneratorNumeric
    from DecisionTree import TrainingDataGeneratorSymbolic
    
    from DecisionTree import DTNode
