__version__ = '3.4.3'
__author__  = "Avinash Kak (kak@purdue.edu)"
__date__    = '2016-May-14'
__url__     = 'https://engineering.purdue.edu/kak/distDT/DecisionTree-3.4.3.html'
__copyright__ = "(C) 2016 Avinash Kak. Python Software Foundation."


__doc__ = '''

DecisionTree.py

Version: ''' + __version__ + '''
   
Author: Avinash Kak (kak@purdue.edu)

Date: ''' + __date__ + '''


@tag_changes
CHANGES:

  Version 3.4.3

    The 'csv_cleanup_needed' parameter in the module constructor is now
    optional in response to several requests received from the user
    community.  The main point made by the users was that setting
    'csv_cleanup_needed' when there was no need for CSV clean-up extracted
    a performance penalty when ingesting large database files with tens of
    thousands of line records.  In addition to making 'csv_cleanup_needed'
    optional, I have also tweaked up the code in the 'cleanup_csv()' method
    in order to extract data from a larger range of messy CSV files.

  Version 3.4.2

    As with the previous version, all the changes made are confined to the
    part of the code that deals with the calculation of the regression
    coefficients.  Besides general code clean up, I have fixed a couple of
    bugs in this part of the code.  With regard to the new constructor
    parameter 'jacobian_choice' introduced in the previous version, setting
    it to 0 is the best choice for most cases, which causes the regression
    coefficients to be estimated through linear least-squares minimization.

  Version 3.4.1

    All the changes made in this version relate to the construction of
    regression trees.  I have improved the code that estimates the
    regression coefficients using a combination of linear least-squares and
    gradient descent.  The RegressionTree class now has a new constructor
    parameter called 'jacobian_choice'.

  Version 3.4.0

    In addition to constructing decision trees, this version of the module
    also allows you to construct regression trees. The regression tree
    capability has been packed into a separate subclass, named
    RegressionTree, of the main DecisionTree class.  The subdirectory
    ExamplesRegression in the main installation directory illustrates how
    you can use this new functionality of the module.

  Version 3.3.2:

    This version fixes the problem encountered by people when using pickle
    to save a decision tree to a disk file.  The problem was solved by
    moving the nested class DTNode to the module level.

  Version 3.3.1:

    This version removes certain inefficiencies that had crept into file
    I/O operations in Version 3.3.0.

  Version 3.3.0

    This version incorporates four very significant upgrades/changes to the
    DecisionTree module: (1) The CSV cleanup is now the default. So you do
    not have to set any special parameters in the constructor calls to
    initiate CSV cleanup. (2) In the form of a new Python class named
    RandomizedTreesForBigData, this module provides you with an easy-to-use
    programming interface for attempting needle-in-a-haystack solutions for
    the case when your training data is overwhelmingly dominated by a
    single class.  You need to set the constructor parameter
    'looking_for_needles_in_haystack' to invoke the logic that constructs
    multiple decision trees, each using the minority class samples along
    with samples drawn randomly from the majority class.  The final
    classification is made through a majority vote from all the decision
    trees.  (3) Assuming you are faced with a big-data problem --- in the
    sense that you have been given a training database with a very large
    number of training records --- the class RandomizedTreesForBigData will
    also let you construct multiple decision trees by pulling training data
    randomly from your training database (without paying attention to the
    relative populations of the classes).  The final classification
    decision for a test sample is based on a majority vote from all the
    decision trees thus constructed.  See the ExamplesRandomizedTrees
    directory for how to use these new features of the module. And,
    finally, (4) Support for the old-style '.dat' training files has been
    dropped in this version.

  Version 3.2.4

    This version makes it easier to use a CSV training file that violates
    the assumption that a comma be used only to separate the different
    field values in a line record.  Some large econometrics databases use
    double-quoted values for fields, and these values may contain commas
    (presumably for better readability).  This version also allows you to
    specify the leftmost entry in the first CSV record that names all the
    fields. Previously, this entry was required to be an empty
    double-quoted string.  I have also made some minor changes to the
    'get_training_data_from_csv()' method to make it more user friendly for
    large training files that may contain tens of thousands of records.
    When pulling training data from such files, this method prints out a
    dot on the terminal screen for every 10000 records it has processed.
    See the script 'dt_example_involving_csv_cleanup.py' in the Examples
    directory for how to invoke the added functionality for CSV files.

  Version 3.2.3

    Cleanup of the HTML page for the module.  The implementation code
    and the API remain unchanged.

  Version 3.2.2

    Incorporates a fix for a minor bug related to exception handling for
    the case when the training data file named in the constructor has an
    incorrect suffix.  Also incorporated in this version is a general
    cleanup of the exception handling code.

  Version 3.2.1

    Contains a bugfix for a probability calculator function. The bug was
    triggered when a feature did not take on any values whatsoever in all
    of the training data samples for a class --- a condition likely to be
    encountered only rarely, but nonetheless important.

  Version 3.2.0

    This version brings the boosting capability to the DecisionTree module.

  Version 3.0.1

    This is a minor revision that smooths out the documentation at a couple
    of important places.  I have also fixed the typos that I discovered
    after the previous version was released.

  Version 3.0

    Version 3.0 adds bagging to the DecisionTree module. If your training
    dataset is large enough, you can ask the module to construct multiple
    decision trees using data bags extracted from your dataset.  The module
    can show you the results returned by the individual decision trees and
    also the results obtained by taking a majority vote of the
    classification decisions made by the trees.  You can specify any
    arbitrary extent of overlap between the data bags.

  Version 2.3.4

    There was an error in the packaging of version 2.3.3 of this module.
    The error related to how the `packages' keyword was specified in
    setup.py.  This version fixes that error.

  Version 2.3.3:

    The purpose of this version is merely to mention that you do NOT need
    to double-quote the entries in your CSV training files. The older
    versions of this module required the rows of a CSV file to be in the
    following sort of a format:

  Version 2.3.2:

    The introspection capability in this version packs more of a punch.
    For each training data sample, you can now figure out not only the
    decision-tree nodes that are affected directly by that sample, but also
    those nodes that are affected indirectly through the generalization
    achieved by the probabilistic modeling of the data.  The Examples
    directory of this version includes additional scripts that illustrate
    these enhancements to the introspection capability.  See the section
    "The Introspection API" for a declaration of the introspection related
    methods, old and new.

  Version 2.3.1:

    This version has a bug fix for the decision-tree introspection
    capability that was added to the module in Version 2.3. Also changed in
    this version are the names of the two scripts in the Examples directory
    that illustrate introspection.  The new names are "introspection_at_one
    _node.py" and "introspection_in_a_loop.py".

  Version 2.3:

    In response to requests from several users, this version includes a new
    capability: You can now ask the module to introspect about the
    classification decisions returned by the decision tree.  Toward that
    end, the module includes a new class named DTIntrospection.  Perhaps
    the most important bit of information you are likely to seek through DT
    introspection is the list of the training samples that fall directly in
    the portion of the feature space that is assigned to a node.  CAVEAT:
    When training samples are non-uniformly distributed in the underlying
    feature space, IT IS POSSIBLE FOR A NODE TO EXIST EVEN 
    WHEN NO TRAINING SAMPLES FALL IN THE PORTION OF THE 
    FEATURE SPACE ASSIGNED TO THE NODE.  (This is the entire 
    point of the generalization achieved by probabilistic modeling of the 
    training data.)  For additional information related to DT introspection, 
    see the section titled "DECISION TREE INTROSPECTION" in this 
    documentation page.

  Version 2.2.6:

    Removed a serious bug in the method get_training_data_from_dat() that
    is used to read the training data from `.dat' files.  The class naming
    convention internally in the code is that it is a concatenation of the
    header of the column that contains the class labels and the actual
    class name as mentioned in each row of the training data. There was a
    problem with this concatenation. Another very important bug fix was in
    the method probability_of_feature_value(). The probability calculation
    in the previous versions was erroneous for those features that acquired
    zero values.  The users of this module may not have noticed this error
    in the past if the zero values for the features occurred relatively
    infrequently.  This error has now been fixed.

  Version 2.2.5:

    If you are not using the synthetic data generation feature of this
    module, the changes made in this version do not affect you. The code
    that was changed is all in the class TrainingDataGeneratorNumeric.  The
    changes to this class remove an important bug related to the ordering
    of the feature names that are read from a user-supplied parameter
    file. The basic Decision Tree construction and classification code
    remains unchanged.

  Version 2.2.4:

    This version should prove more robust when the probability distribution
    for the values of a feature is expected to be heavy-tailed; that is,
    when the supposedly rare observations can occur with significant
    probabilities.  A new option in the DecisionTree constructor lets the
    user specify the precision with which the probability distributions
    are estimated for such features.

  Version 2.2.3:

    This version fixes a bug that was caused by the explicitly set zero
    values for numerical features being misconstrued as "false" in the
    conditional statements in some of the method definitions.

  Version 2.2.2:

    In response to requests from users, this version includes scripts in
    the Examples directory that demonstrate how to carry out bulk
    classification of all your test data records placed in a CSV file in
    one fell swoop.  Also included are scripts that demonstrate the same
    for the data records placed in the old-style `.dat' files.  The main
    module code remains unchanged.

  Version 2.2.1:

    The changes made are all in the part of the module that is used for
    evaluating the quality of training data through a 10-fold cross
    validation test.  The previous version used the default values for the
    constructor parameters when constructing the decision trees in each
    iteration of the test. The new version correctly uses the user-supplied
    values.

  Version 2.2:

    This version fixes a bug discovered in the best feature calculator
    function. This bug was triggered by certain conditions related to the
    distribution of values for the features in a training data file.
    Additionally, and VERY IMPORTANTLY, Version 2.2 allows you 
    to test the quality of your training data by running a 10-fold
    cross-validation test on the data.  This test divides all of the
    training data into ten parts, with nine parts used for training a
    decision tree and one part used for testing its ability to classify
    correctly. This selection of nine parts for training and one part for
    testing is carried out in all of the ten different possible ways.  This
    testing functionality in Version 2.2 can also be used to find the best
    values to use for the constructor parameters entropy_threshold,
    max_depth_desired, and symbolic_to_numeric_cardinality_threshold.

  Version 2.1:

    This is a cleaned up version of v. 2.0 of the module. Should run more
    efficiently for large training data files that contain both numeric and
    symbolic features.

  Version 2.0:

    This was a major rewrite of the DecisionTree module.  This revision was
    prompted by a number of users wanting to see numeric features
    incorporated in the construction of decision trees.  So here it is!
    This version allows you to use either purely symbolic features, or
    purely numeric features, or a mixture of the two. (A feature is numeric
    if it can take any floating-point value over an interval.)

  Version 1.7.1:

    This version includes a fix for a bug that was triggered by certain
    comment words in a training data file.  This version also includes
    additional safety checks that are useful for catching errors and
    inconsistencies in large training data files that do not lend
    themselves to manual checking for correctness.  As an example, the new
    version makes sure that the number of values you declare in each sample
    record matches the number of features declared at the beginning of the
    training data file.

  Version 1.7:

    This version includes safety checks on the consistency of the data you
    place in your training data file.  When a training data file contains
    thousands of records, it is difficult to manually check that you used
    the same class names in your sample records that you declared at the
    top of your training file or that the values you have for your features
    are legal vis-a-vis the earlier declarations regarding such values in
    the training file.  Another safety feature incorporated in this version
    is the non-consideration of classes that are declared at the top of the
    training file but that have no sample records in the file.

  Version 1.6.1:

    Fixed a bug in the method that generates synthetic test data.

  Version 1.6:

    This version includes several upgrades: The module now includes code
    for generating synthetic training and test data for experimenting with
    the DecisionTree classifier.  Another upgrade in the new version is
    that, after training, a decision tree can now be used in an interactive
    mode in which the user is asked to supply answers for the feature tests
    at the nodes as the classification process descends down the tree.

  Version 1.5:

    This is a Python 3.x compliant version of the DecisionTree module.
    This version should work with both Python 2.x and Python 3.x.

  Version 1.0:

    This is a Python implementation of the author's Perl module
    Algorithm::DecisionTree, Version 1.41.  The Python version should work
    faster for large decision trees since it uses probability and entropy
    caching much more extensively than Version 1.41 of the Perl module.
    (Note: I expect my next release of the Perl module to catch up with
    this Python version in terms of performance.)


@tag_usage
USAGE:

    If your training data includes numeric features (a feature is numeric
    if it can take any floating point value over an interval), you are
    expected to supply your training data through a CSV file and your call
    for constructing an instance of the DecisionTree class will look like:

@begincode
        training_datafile = "stage3cancer.csv"

        dt = DecisionTree.DecisionTree( 
                        training_datafile = training_datafile,
                        csv_class_column_index = 2,
                        csv_columns_for_features = [3,4,5,6,7,8],
                        entropy_threshold = 0.01,
                        max_depth_desired = 8,
                        symbolic_to_numeric_cardinality_threshold = 10,
                        csv_cleanup_needed = 1,
             )
@endcode

    The constructor option `csv_class_column_index' informs the module as
    to which column of your CSV file contains the class label.  THE 
    COLUMN INDEXING IS ZERO BASED.  The constructor option
    `csv_columns_for_features' specifies which columns are to be used for
    feature values.  The first row of the CSV file must specify the names
    of the features.  See examples of CSV files in the `Examples'
    subdirectory.

    The option `symbolic_to_numeric_cardinality_threshold' is also
    important.  For the example shown above, if an ostensibly numeric
    feature takes on only 10 or fewer different values in your training
    datafile, it will be treated like a symbolic feature.  The option
    `entropy_threshold' determines the granularity with which the entropies
    are sampled for the purpose of calculating entropy gain with a
    particular choice of decision threshold for a numeric feature or a
    feature value for a symbolic feature.  

    The option `csv_cleanup_needed' is by default set to 0.  If you set it
    to 1, that would cause all line records in your CSV file to be
    "sanitized" before they are used for constructing a decision tree.  You
    need this option if your CSV file uses double-quoted field names and
    field values in the line records and if such double-quoted strings are
    allowed to include commas for, presumably, better readability.

    After you have constructed an instance of the DecisionTree class as
    shown above, you read in the training data file and initialize the
    probability cache by calling:

@begincode
        dt.get_training_data()
        dt.calculate_first_order_probabilities()
        dt.calculate_class_priors()
@endcode

    Next you construct a decision tree for your training data by calling:

@begincode
        root_node = dt.construct_decision_tree_classifier()
@endcode

    where root_node is an instance of the DTNode class that is also defined
    in the module file.  Now you are ready to classify a new data record.
    Let's say that your data record looks like:

@begincode
        test_sample  = ['g2 = 4.2',
                        'grade = 2.3',
                        'gleason = 4',
                        'eet = 1.7',
                        'age = 55.0',
                        'ploidy = diploid']
@endcode

    You can classify it by calling:

@begincode
        classification = dt.classify(root_node, test_sample)
@endcode

    The call to `classify()' returns a reference to a hash whose keys are
    the class names and the values the associated classification
    probabilities.  This hash also includes another key-value pair for the
    solution path from the root node to the leaf node at which the final
    classification was carried out.

    A decision tree can quickly become much too large (and much too slow to
    construct and to yield classification results) if the total number of
    features is large and/or if the number of different possible values for
    the symbolic features is large.  You can control the size of the tree
    through the constructor options `entropy_threshold' and
    `max_depth_desired'. The latter option sets the maximum depth of your
    decision tree to max_depth_desired value.  The parameter
    `entropy_threshold' sets the granularity with which the entropies are
    sampled.  Its default value is 0.001.  The larger the value you choose
    for entropy_threshold, the smaller the tree.


@tag_intro
INTRODUCTION:

    DecisionTree is a Python module for constructing a decision tree from a
    training data file containing multidimensional data in the form of a
    table. In one form or another, decision trees have been around for over
    fifty years. From a statistical perspective, they are closely related
    to classification and regression by recursive partitioning of
    multidimensional data. Early work that demonstrated the usefulness of
    such partitioning of data for classification and regression can be
    traced to the work of Terry Therneau in the early 1980's in the
    statistics community, and to the work of Ross Quinlan in the mid 1990's
    in the machine learning community,

    For those not familiar with decision tree ideas, the traditional way to
    classify multidimensional data is to start with a feature space whose
    dimensionality is the same as that of the data.  Each feature measures
    a specific attribute of an entity.  You use the training data to carve
    up the feature space into different regions, each corresponding to a
    different class.  Subsequently, when you try to classify a new data
    sample, you locate it in the feature space and find the class label of
    the region to which it belongs.  One can also give the data point the
    same class label as that of the nearest training sample. This is
    referred to as the nearest neighbor classification. There exist
    hundreds of variations of varying power on these two basic approaches
    to the classification of multidimensional data.

    A decision tree classifier works differently.  When you construct a
    decision tree, you select for the root node a feature test that
    partitions the training data in a way that causes maximal
    disambiguation of the class labels associated with the data.  In terms
    of information content as measured by entropy, such a feature test
    would cause maximum reduction in class entropy in going from all of the
    training data taken together to the data as partitioned by the feature
    test.  You then drop from the root node a set of child nodes, one for
    each partition of the training data created by the feature test at the
    root node. When your features are purely symbolic, you'll have one
    child node for each value of the feature chosen for the feature test at
    the root.  When the test at the root involves a numeric feature, you
    find the decision threshold for the feature that best bipartitions the
    data and you drop from the root node two child nodes, one for each
    partition.  Now at each child node you pose the same question that you
    posed when you found the best feature to use at the root: Which feature
    at the child node in question would maximally disambiguate the class
    labels associated with the training data corresponding to that child
    node?

    As the reader would expect, the two key steps in any approach to
    decision-tree based classification are the construction of the decision
    tree itself from a file containing the training data, and then using
    the decision tree thus obtained for classifying new data.

    What is cool about decision tree classification is that it gives you
    soft classification, meaning it may associate more than one class label
    with a given data record.  When this happens, it may mean that your
    classes are indeed overlapping in the underlying feature space.  It
    could also mean that you simply have not supplied sufficient training
    data to the decision tree classifier.  For a tutorial introduction to
    how a decision tree is constructed and used, see

    https://engineering.purdue.edu/kak/Tutorials/DecisionTreeClassifiers.pdf


@tag_whatpractical
WHAT PRACTICAL PROBLEM IS SOLVED BY THIS MODULE?

    If you are new to the concept of a decision tree, their practical
    utility is best understood with an example that only involves symbolic
    features.  However, as mentioned earlier, versions 2.0 and higher of
    this module handle both symbolic and numeric features.

    Consider the following scenario: Let's say you are running a small
    investment company that employs a team of stockbrokers who make
    buy/sell decisions for the customers of your company.  Assume that your
    company has asked the traders to make each investment decision on the
    basis of the following five criteria:

@begincode
            price_to_earnings_ratio   (P_to_E)

            price_to_sales_ratio      (P_to_S)

            return_on_equity          (R_on_E)

            market_share              (M_S)

            sentiment                 (S)
@endcode

    Since you are the boss, you keep track of the buy/sell decisions made
    by the individual traders.  But one unfortunate day, all of your
    traders decide to quit because you did not pay them enough.  So what
    are you to do?  If you had a module like the one here, you could still
    run your company and do so in such a way that your company would, on
    the average, perform better than any of the individual traders who
    worked for you previously.  This is what you would need to do: You
    would pool together the individual trader buy/sell decisions you
    accumulated during the last one year.  This pooled information is
    likely to look like:

@begincode
      example      buy/sell     P_to_E     P_to_S     R_on_E     M_S     S
      ====================================================================

      example_1     buy          high       low        medium    low    high
      example_2     buy          medium     medium     low       low    medium
      example_3     sell         low        medium     low       high   low
      ....
      ....
@endcode

    This would constitute your training data. Assuming CSV formatting for
    the data in a file called 'training.csv', you would need to feed this
    file into the module by calling:

@begincode
        dt = DecisionTree( training_datafile = "training.csv", .... )

        dt.get_training_data()

        dt.calculate_first_order_probabilities()

        dt.calculate_class_priors()
@endcode

    Subsequently, you would construct a decision tree by calling:

@begincode
        root_node = dt.construct_decision_tree_classifier()
@endcode

    Now you and your company (with practically no employees) are ready to
    service the customers again. Suppose your computer needs to make a
    buy/sell decision about an investment prospect that is best described
    by:

@begincode
        price_to_earnings_ratio   =  low
        price_to_sales_ratio      =  very_low
        return_on_equity          =  none
        market_share              =  medium    
        sentiment                 =  low
@endcode

    All that your computer would need to do would be to construct a data
    record like

@begincode
        test_case = [ 'P_to_E=low', 
                      'P_to_S=very_low', 
                      'R_on_E=none',
                      'M_S=medium',
                      'S=low'  ]
@endcode

    and call the decision tree classifier you just constructed by

@begincode
        classification = dt.classify(root_node, test_case)

        print "Classification: ", classification
@endcode

    The answer returned will be 'buy' and 'sell', along with the associated
    probabilities.  So if the probability of 'buy' is considerably greater
    than the probability of 'sell', that's what you should instruct your
    computer to do.

    The chances are that, on the average, this approach would beat the
    performance of any of your individual traders who worked for you
    previously since the buy/sell decisions made by the computer would be
    based on the collective wisdom of all your previous traders.
    DISCLAIMER: There is obviously a lot more to good investing than what
    is captured by the silly little example here. However, it does convey
    the sense in which the current module can be used.


@tag_symvsnum
SYMBOLIC FEATURES VERSUS NUMERIC FEATURES:

    A feature is symbolic when its values are compared using string
    comparison operators.  By the same token, a feature is numeric when its
    values are compared using numeric comparison operators.  Having said
    that, features that take only a small number of numeric values in the
    training data can be treated symbolically provided you are careful
    about handling their values in the test data.  At the least, you have
    to set the test data value for such a feature to its closest value in
    the training data.  For those numeric features that the module is
    allowed to treat symbolically, this snapping of the values of the
    features in the test data to the small set of values in the training
    data is carried out automatically by the module.  That is, after a user
    has told the module which numeric features to treat symbolically, the
    user need not worry about how the feature values appear in the test
    data.

    The constructor parameter symbolic_to_numeric_cardinality_threshold
    lets you tell the module when to consider an otherwise numeric feature
    symbolically. Suppose you set this parameter to 10, that means that all
    numeric looking features that take 10 or fewer different values in the
    training datafile will be considered to be symbolic features.
    
    See the tutorial at

    https://engineering.purdue.edu/kak/Tutorials/DecisionTreeClassifiers.pdf

    for further information on the implementation issues related to the
    symbolic and numeric features.


@tag_notsonice
FEATURES WITH NOT SO "NICE" STATISTICAL PROPERTIES:

    For the purpose of estimating the probabilities, it is necessary to
    sample the range of values taken on by a numerical feature. For
    features with "nice" statistical properties, this sampling interval is
    set to the median of the differences between the successive feature
    values in the training data.  (Obviously, as you would expect, you
    first sort all the values for a feature before computing the successive
    differences.)  This logic will not work for the sort of a feature
    described below.

    Consider a feature whose values are heavy-tailed, and, at the same
    time, the values span a million to one range.  What I mean by
    heavy-tailed is that rare values can occur with significant
    probabilities.  It could happen that most of the values for such a
    feature are clustered at one of the two ends of the range. At the same
    time, there may exist a significant number of values near the end of
    the range that is less populated.  (Typically, features related to
    human economic activities --- such as wealth, incomes, etc. --- are of
    this type.)  With the logic described in the previous paragraph, you
    could end up with a sampling interval that is much too small, which
    could result in millions of sampling points for the feature if you are
    not careful.

    Beginning with Version 2.2.4, you have two options in dealing with such
    features.  You can choose to go with the default behavior of the
    module, which is to sample the value range for such a feature over a
    maximum of 500 points.  Or, you can supply an additional option to the
    constructor that sets a user-defined value for the number of points to
    use.  The name of the option is "number_of_histogram_bins".  The
    following script

@begincode
          construct_dt_for_heavytailed.py
@endcode

    in the Examples directory shows an example of how to call the
    constructor of the module with the "number_of_histogram_bins" option.


@tag_quality
TESTING THE QUALITY OF YOUR TRAINING DATA:

    Starting with version 2.2, the module includes a new class named
    EvalTrainingData, derived from the main class DecisionTree, that runs a
    10-fold cross-validation test on your training data to test its ability
    to discriminate between the classes mentioned in the training file.

    The 10-fold cross-validation test divides all of the training data into
    ten parts, with nine parts used for training a decision tree and one
    part used for testing its ability to classify correctly. This selection
    of nine parts for training and one part for testing is carried out in
    all of the ten different possible ways.  

    The following code fragment illustrates how you invoke the testing
    function of the EvalTrainingData class:

@begincode
        training_datafile = "training3.csv"
        eval_data = DecisionTree.EvalTrainingData(
                        training_datafile = training_datafile,
                        csv_class_column_index = 1,
                        csv_columns_for_features = [2,3],
                        entropy_threshold = 0.01,
                        max_depth_desired = 3,
                        symbolic_to_numeric_cardinality_threshold = 10,
                        csv_cleanup_needed = 1,
                    )

        eval_data.get_training_data()
        eval_data.evaluate_training_data()
@endcode

    The last statement above prints out a Confusion Matrix and the value of
    Training Data Quality Index on a scale of 100, with 100 designating
    perfect training data.  The Confusion Matrix shows how the different
    classes were misidentified in the 10-fold cross-validation test.

    This testing functionality can also be used to find the best values to
    use for the constructor parameters entropy_threshold,
    max_depth_desired, and symbolic_to_numeric_cardinality_threshold.

    The following two scripts in the Examples directory illustrate the use
    of the EvalTrainingData class for testing the quality of your data:

@begincode
        evaluate_training_data1.py

        evaluate_training_data2.py
@endcode

    IMPORTANT: These data evaluation scripts produce reliable indicators of
    the quality of your data only when there is a rough parity between the
    number of training samples available for the different data classes.


@tag_howto
HOW TO MAKE THE BEST CHOICES FOR THE CONSTRUCTOR PARAMETERS:

    Assuming your training data is good, the quality of the results you get
    from a decision tree would depend on the choices you make for the
    constructor parameters entropy_threshold, max_depth_desired, and
    symbolic_to_numeric_cardinality_threshold.  You can optimize your
    choices for these parameters by running the 10-fold cross-validation
    test that is made available in Versions 2.2 and higher through the new
    class EvalTrainingData that is included in the module file.  A
    description of how to run this test is in the section titled "TESTING
    THE QUALITY OF YOUR TRAINING DATA" of this document.


@tag_randomized
SOLVING NEEDLE-IN-A-HAYSTACK AND BIG DATA PROBLEMS:

    Machine learning algorithms, in general, run into difficulties when
    there is gross imbalance in how many training samples are available for
    the different data classes --- especially when the different data
    classes are not linearly separable in the underlying feature
    space. Starting with Version 3.3.0 of this module, you can try to use
    the new Python class RandomizedTreesForBigData for solving such
    problems provided you set the following two constructor parameters:

@begincode
        looking_for_needles_in_haystack

        how_many_trees
@endcode

    At the moment, RandomizedTreesForBigData is programmed for just binary
    classification. It first figures out the population imbalance between
    the majority data class and the minority data class in your dataset.
    Subsequently, by mixing the minority training samples with randomly
    drawn samples from the majority data class, it constructs a collection
    of training datasets for 'how_many_trees' number of decision trees.  A
    new data sample to be classified is fed to each of these decision trees
    and the final classification based on majority voting by all the trees.
    You must set 'looking_for_needles_in_haystack = 1' in the constructor
    call for this logic to work.  You'd also want to experiment with with
    the option 'how_many_trees' to figure out the best value for this
    parameter.  Searching for a needle in a haystack is obviously a good
    metaphor for these types of data classification problems.

    The RandomizedTreesForBigData class can also be used for solving big
    data problems if you have access to a very large training database.
    You can solve such problems by constructing multiple decision trees,
    each based on a training dataset drawn randomly from the large training
    database (without paying attention to population imbalances).
    Subsequently, the final classification for a new data sample can be
    based on majority voting by all the trees.  In order to use this
    functionality, you need to set the following two constructor parameters
    of this class:

@begincode
        how_many_training_samples_per_tree

        how_many_trees
@endcode


@tag_introspect
DECISION TREE INTROSPECTION:

    Starting with Version 2.3, you can ask the DTIntrospection class of the
    module to explain the classification decisions made at the different
    nodes of the decision tree.

    Perhaps the most important bit of information you are likely to seek
    through DT introspection is the list of the training samples that fall
    directly in the portion of the feature space that is assigned to a
    node.

    However, note that, when training samples are non-uniformly distributed
    in the underlying feature space, it is possible for a node to exist
    even when there are no training samples in the portion of the feature
    space assigned to the node.  That is because the decision tree is
    constructed from the probability densities estimated from the training
    data.  When the training samples are non-uniformly distributed, it is
    entirely possible for the estimated probability densities to be
    non-zero in a small region around a point even when there are no
    training samples specifically in that region.  (After you have created
    a statistical model for, say, the height distribution of people in a
    community, the model may return a non-zero probability for the height
    values in a small interval even if the community does not include a
    single individual whose height falls in that interval.)

    That a decision-tree node can exist even where there are no training samples in
    the portion of the feature space that belongs to that node is an important
    indication of the generalization ability of a decision-tree based classifier.

    In light of the explanation provided above, before the DTIntrospection
    class supplies any answers at all, it asks you to accept the fact that
    features can take on non-zero probabilities at a point in the feature
    space even though there are zero training samples at that point (or in
    a small region around that point).  If you do not accept this
    rudimentary fact, the introspection class will not yield any answers
    (since you are not going to believe the answers anyway).

    The point made above implies that the path leading to a node in the
    decision tree may test a feature for a certain value or threshold
    despite the fact that the portion of the feature space assigned to that
    node is devoid of any training data.

    See the following three scripts in the Examples directory for how to
    carry out DT introspection:

@begincode
        introspection_in_a_loop_interactive.py

        introspection_show_training_samples_at_all_nodes_direct_influence.py

        introspection_show_training_samples_to_nodes_influence_propagation.py
@endcode

    The first script places you in an interactive session in which you will
    first be asked for the node number you are interested in.
    Subsequently, you will be asked for whether or not you are interested
    in specific questions that the introspection can provide answers
    for. The second script descends down the decision tree and shows for
    each node the training samples that fall directly in the portion of the
    feature space assigned to that node.  The third script shows for each
    training sample how it affects the decision-tree nodes either directly
    or indirectly through the generalization achieved by the probabilistic
    modeling of the data.

    The output of the script introspection_show_training_samples_at_all_
    nodes_direct_influence.py looks like:

@begincode
        Node 0: the samples are: None
        Node 1: the samples are: ['sample_46', 'sample_58']
        Node 2: the samples are: ['sample_1', 'sample_4', 'sample_7', .....]
        Node 3: the samples are: []
        Node 4: the samples are: []
        ...
        ...            
@endcode
    
    The nodes for which no samples are listed come into existence through
    the generalization achieved by the probabilistic modeling of the data.

    The output produced by the script introspection_show_training_samples_
    to_nodes_influence_propagation.py looks like

@begincode
        sample_1:                                                                 
           nodes affected directly: [2, 5, 19, 23]                                
           nodes affected through probabilistic generalization:                   
                2=> [3, 4, 25]                                                    
                    25=> [26]                                                     
                5=> [6]                                                           
                    6=> [7, 13]                                                   
                        7=> [8, 11]                                               
                            8=> [9, 10]                                           
                            11=> [12]                                             
                        13=> [14, 18]                                             
                            14=> [15, 16]                                         
                                16=> [17]                                         
                19=> [20]                                                         
                    20=> [21, 22]                                                 
                23=> [24]                                                         
                                 
        sample_4:                                                                 
           nodes affected directly: [2, 5, 6, 7, 11]                              
           nodes affected through probabilistic generalization:                   
                2=> [3, 4, 25]                                                    
                    25=> [26]                                                     
                5=> [19]                                                          
                    19=> [20, 23]                                                 
                        20=> [21, 22]                                             
                        23=> [24]                                                 
                6=> [13]                                                          
                    13=> [14, 18]                                                 
                        14=> [15, 16]                                             
                            16=> [17]                                             
                7=> [8]                                                           
                    8=> [9, 10]                                                   
                11=> [12]                                                         
                                                                                  
        ...                                                                       
        ...  
        ...
@endcode
    
    For each training sample, the display shown above first presents the
    list of nodes that are directly affected by the sample.  A node is
    affected directly by a sample if the latter falls in the portion of the
    feature space that belongs to the former.  Subsequently, for each
    training sample, the display shows a subtree of the nodes that are
    affected indirectly by the sample through the generalization achieved
    by the probabilistic modeling of the data.  In general, a node is
    affected indirectly by a sample if it is a descendant of another node
    that is affected directly.

    Also see the section titled "The Introspection API" regarding how to
    invoke the introspection capabilities of the module in your own code.


@tag_methods
METHODS:

    The module provides the following methods for constructing a decision
    tree from training data in a disk file, and for data classification with
    the decision tree.


@tag2
    Constructing a decision tree:

@begincode    
            dt = DecisionTree( 
                       training_datafile = training_datafile,
                       csv_class_column_index = 2,
                       csv_columns_for_features = [3,4,5,6,7,8],
                       entropy_threshold = 0.01,
                       max_depth_desired = 8,
                       symbolic_to_numeric_cardinality_threshold = 10,
                       csv_cleanup_needed = 1,
                 )
@endcode
    
        This yields a new instance of the DecisionTree class.  For this
        call to make sense, the training data in the training datafile must
        conform to the traditional CSV format.  For example, the first
        record must list the features to be used for classification.  The
        first column for all rows must carry a unique integer identifier
        for each data record.
    
        The constructor option csv_class_column_index supplies to the
        module zero-based index of the column that contains the class label
        for the training data records. In the example shown above, the
        class labels are in the third column.  The option
        csv_columns_for_features tells the module which of the features are
        supposed to be used for decision tree construction.  The
        constructor option max_depth_desired sets the maximum depth of the
        decision tree. The parameter entropy_threshold sets the granularity
        with which the entropies are sampled.  The parameter
        symbolic_to_numeric_cardinality_threshold allows the module to
        treat an otherwise numeric feature symbolically if it only takes a
        small number of different values in the training data file.  For
        the constructor call shown above, if a feature takes on only 10 or
        fewer different values in the training data file, it will be
        treated like a symbolic feature.
    

@tag2
    The constructor parameters:

        training_datafile:
    
            This parameter supplies the name of the file that contains the
            training data.  This must be a CSV file if your training data
            includes both numeric and symbolic features. 
    
        csv_class_column_index:
    
            When using a CSV file for your training data, this parameter
            supplies the zero-based column index for the column that contains
            the class label for each data record in the training file.

        csv_cleanup_needed:

            You need to set this parameter to 1 if your CSV file has double
            quoted strings (which may include commas) as values for the
            fields and if such values are allowed to include commas for,
            presumably, better readability.
    
        csv_columns_for_features:
    
            When using a CSV file for your training data, this parameter
            supplies a list of columns corresponding to the features you wish
            to use for decision tree construction.  Each column is specified by
            its zero-based index.
    
        entropy_threshold:
    
            This parameter sets the granularity with which the entropies are
            sampled by the module.  For example, a feature test at a node in
            the decision tree is acceptable if the entropy gain achieved by the
            test exceeds this threshold.  The larger the value you choose for
            this parameter, the smaller the tree.  Its default value is 0.001.
    
        max_depth_desired:
    
            This parameter sets the maximum depth of the decision tree.  For
            obvious reasons, the smaller the value you choose for this
            parameter, the smaller the tree.
    
        symbolic_to_numeric_cardinality_threshold:
    
            This parameter allows the module to treat an otherwise numeric
            feature symbolically if the number of different values the feature
            takes in the training data file does not exceed the value of this
            parameter.
    
        number_of_histogram_bins:
    
            This parameter gives the user the option to set the number of
            points at which the value range for a feature should be sampled
            for estimating the probabilities.  This parameter is especially
            effective for those features that occupy a large value range
            and whose probability distributions are heavy tailed.  THIS
            PARAMETER IS ALSO IMPORTANT WHEN YOU HAVE A 
            VERY LARGE TRAINING DATASET: In general, the larger 
            the dataset, the smaller the smallest difference between any
            two values for a numeric feature in relation to the overall
            range of values for that feature. In such cases, the module may
            use too large a number of bins for estimating the probabilities
            and that may slow down the calculation of the decision tree.
            You can get around this difficulty by explicitly giving a value
            to the 'number_of_histogram_bins' parameter.

        You can choose the best values to use for the constructor parameters by
        running a 10-fold cross-validation test on your training data through
        the embedded class EvalTrainingData that comes with Versions 2.2 and
        higher of this module.  See the section "TESTING THE QUALITY OF YOUR
        TRAINING DATA" of this document page.
    

@tag2
    Reading in the training data:

        After you have constructed a new instance of the DecisionTree class,
        you must now read in the training data that is contained in the file
        named above.  This you do by:
    
            dt.get_training_data()
    
        IMPORTANT: The training data file must be in a format that makes sense
        to the decision tree constructor.  If you use numeric features, you
        must use a CSV file for supplying the training data.  The first row of
        such a file must name the features and it must begin with the empty
        string `""' as shown in the `stage3cancer.csv' file in the Examples
        subdirectory.  The first column for all subsequent rows must carry a
        unique integer identifier for each training record.
    

@tag2
    Initializing the probability cache:

        After a call to the constructor and the get_training_data() method, you
        must call the following methods for initializing the probabilities:

@begincode    
            dt.calculate_first_order_probabilities()
            dt.calculate_class_priors()
@endcode
    

@tag2
    Displaying the training data:

        If you wish to see the training data that was just digested by the
        module, call

@begincode    
            dt.show_training_data() 
@endcode


@tag2
    Constructing a decision-tree classifier:

        After the training data is ingested, it is time to construct a decision
        tree classifier.  This you do by

@begincode    
            root_node = dt.construct_decision_tree_classifier()
@endcode
    
        This call returns an instance of type DTNode.  The DTNode class is
        defined within the main package file, at its end.  So, don't forget,
        that root_node in the above example call will be instantiated to an
        instance of type DTNode.
    

@tag2
    Displaying the decision tree:

        You display a decision tree by calling

@begincode    
            root_node.display_decision_tree("   ")
@endcode
    
        This displays the decision tree in your terminal window by using a
        recursively determined offset for each node as the display routine
        descends down the tree.
    
        I have intentionally left the syntax fragment root_node in the above
        call to remind the reader that display_decision_tree() is NOT called on
        the instance of the DecisionTree we constructed earlier, but on the
        Node instance returned by the call to
        construct_decision_tree_classifier().
    

@tag2
    Classifying new data:

        You classify new data by first constructing a new data record:

@begincode    
            test_sample  = ['g2 = 4.2',
                            'grade = 2.3',
                            'gleason = 4',
                            'eet = 1.7',
                            'age = 55.0',
                            'ploidy = diploid']
@endcode
    
        and calling the classify() method as follows:

@begincode     
            classification = dt.classify(root_node, test_sample)
@endcode
    
        where, again, root_node is an instance of type Node that was returned
        by calling construct_decision_tree_classifier().  The variable
        classification is a dictionary whose keys are the class labels and
        whose values the associated probabilities.  You can print it out by

@begincode    
            print "Classification: ", classification
@endcode
    

@tag2
    Displaying the number of nodes created:

        You can print out the number of nodes in a decision tree by calling

@begincode    
            root_node.how_many_nodes()
@endcode
    

@tag2
    Using the decision tree interactively:

        Starting with Version 1.6 of the module, you can use the DecisionTree
        classifier in an interactive mode.  In this mode, after you have
        constructed the decision tree, the user is prompted for answers to the
        questions regarding the feature tests at the nodes of the tree.
        Depending on the answer supplied by the user at a node, the classifier
        takes a path corresponding to the answer to descend down the tree to
        the next node, and so on.  The following method makes this mode
        possible.  Obviously, you can call this method only after you have
        constructed the decision tree.

@begincode    
            dt.classify_by_asking_questions(root_node)
@endcode    

    
@tag_introspectapi
THE INTROSPECTION API:

    To construct an instance of DTIntrospection, you call

@begincode
        introspector = DecisionTree.DTIntrospection(dt)
@endcode

    where you supply the instance of the DecisionTree class you used for
    constructing the decision tree through the parameter dt.  After you
    have constructed an instance of the introspection class, you must
    initialize it by

@begincode
        introspector.initialize()
@endcode

    on the introspector instance. Subsequently, you can invoke either of
    the following methods:

@begincode
        introspector.explain_classification_at_one_node(node)

        introspector.explain_classifications_at_multiple_nodes_interactively()
@endcode

    depending on whether you want introspection at a single specified node
    or inside an infinite loop for an arbitrary number of nodes.

    If you want to output a tabular display that shows for each node in the
    decision tree all the training samples that fall in the portion of the
    feature space that belongs to that node, call

@begincode
        introspector.display_training_samples_at_all_nodes_direct_influence_only()
@endcode

    If you want to output a tabular display that shows for each training
    sample a list of all the nodes that are affected directly AND
    indirectly by that sample, call

@begincode
        introspector.display_training_training_samples_to_nodes_influence_propagation()
@endcode

    A training sample affects a node directly if the sample falls in the
    portion of the features space assigned to that node. On the other hand,
    a training sample is considered to affect a node indirectly if the node
    is a descendant of a node that is affected directly by the sample.


@tag_bulk
BULK CLASSIFICATION OF DATA RECORDS:

    For large test datasets, you would obviously want to process an entire
    file of test data at a time. The following scripts in the Examples
    directory illustrate how you can do that:

@begincode
        classify_test_data_in_a_file_numeric.py 
        classify_test_data_in_a_file_symbolic.py
@endcode

    These scripts require three command-line arguments, the first argument
    names the training datafile, the second the test datafile, and the
    third the file in which the classification results are to be deposited.
    The first script above is for the case of numeric/symbolic features and
    the second for the purely symbolic features.  An important point to
    remember when using these scripts for bulk classification is that the
    test file must have a column for class labels.  In real-life
    situations, obviously, the entries in that column in the test file will
    be just the empty string "".

    The other examples directories, ExamplesBagging, ExamplesBoosting, and
    ExamplesRandomizedTrees, also contain scripts that illustrate how to
    carry out bulk classification of data records when you wish to take
    advantage of bagging, boosting, or tree randomization.  In their
    respective directories, these scripts are named:
                                                                                                                    
@begincode
        bagging_for_bulk_classification.pl                                                                              
        boosting_for_bulk_classification.pl                                                                             
        classify_database_records.pl 
@endcode


@tag_howdisplayed
HOW THE CLASSIFICATION RESULTS ARE DISPLAYED:

    It depends on whether you apply the classifier at once to all the data
    records in a file, or whether you feed one data record at a time into
    the classifier.

    In general, the classifier returns soft classification for a test data
    record.  What that means is that, in general, the classifier will list
    all the classes to which a given data record could belong and the
    probability of each such class label for the data record. Run the
    examples scripts in the Examples directory to see how the output of
    classification can be displayed.
    
    With regard to the soft classifications returned by this classifier, if
    the probability distributions for the different classes overlap in the
    underlying feature space, you would want the classifier to return all
    of the applicable class labels for a test data record along with the
    corresponding class probabilities.  (However, keep in mind the fact
    that the decision tree classifier may associate significant
    probabilities with multiple class labels for a given test data record
    if the training file contains an inadequate number of training samples
    for one or more classes.)  The good thing is that the classifier would
    not lie to you (unlike, say, a hard classification rule that would
    return a single class label corresponding to the partitioning of the
    underlying feature space).  The decision tree classifier will give you
    the best classification that can be made given the training data you
    feed into it.


@tag_bagging
USING BAGGING:

    Starting with Version 3.0, you can use the class
    DecisionTreeWithBagging that comes with the module to incorporate
    bagging in your decision tree based classification.  Bagging means
    constructing multiple decision trees for different (possibly
    overlapping) segments of the data extracted from your training dataset
    and then aggregating the decisions made by the individual decision
    trees for the final classification.  The aggregation of the
    classification decisions can average out the noise and bias that may
    otherwise affect the classification decision obtained from just one
    tree.

    Whether or not you get any benefits from bagging depends on: (1) If
    your original training dataset is large enough and sufficiently varied
    to capture all of the real-world statistical variations within and
    between the classes; and (2) no single feature is too dominant in
    establishing inter-class discriminations.

@tag2
    Calling the bagging constructor:

        A typical call to the constructor for the DecisionTreeWithBagging
        class looks like:
    
@begincode
            import DecisionTreeWithBagging
    
            dtbag = DecisionTreeWithBagging.DecisionTreeWithBagging( 
                            training_datafile = training_datafile,
                            csv_class_column_index = 2,
                            csv_columns_for_features = [3,4,5,6,7,8],
                            entropy_threshold = 0.01,
                            max_depth_desired = 8,
                            symbolic_to_numeric_cardinality_threshold = 10,
                            how_many_bags = 4,
                            bag_overlap_fraction = 0.20,
                            csv_cleanup_needed = 1,
                    )
@endcode
    
        Note in particular the following two constructor parameters:

@begincode    
            how_many_bags
    
            bag_overlap_fraction
@endcode
    
        where, as the name implies, the parameter how_many_bags controls
        how many bags (and, therefore, how many decision trees) will be
        constructed from your training dataset; and where the parameter
        bag_overlap_fraction controls the degree of overlap between the
        bags.  To understand what exactly is achieved by setting the
        parameter bag_overlap_fraction to 0.2 in the above example, let's
        say that the non-overlapping partitioning of the training data
        between the bags results in 100 training samples per bag. With
        bag_overlap_fraction set to 0.2, additional 20 samples drawn
        randomly from the other bags will be added to the data in each bag.
    
        Here are the methods defined for the DecisionTreeWithBagging class;
    
@tag2
    get_training_data_for_bagging():

        This method reads your training datafile, randomizes it, and then
        partitions it into the specified number of bags.  Subsequently, if
        the constructor parameter bag_overlap_fraction is non-zero, it adds
        to each bag additional samples drawn at random from the other bags.
        The number of these additional samples added to each bag is
        controlled by the constructor parameter bag_overlap_fraction.  If
        this parameter is set to, say, 0.2, the size of each bag will grow
        by 20% with the samples drawn from the other bags.
    
@tag2
    show_training_data_in_bags()

        Shows for each bag the names of the training data samples in that
        bag.

@tag2
    calculate_first_order_probabilities()

        Calls on the appropriate methods of the main DecisionTree class to
        estimate the first-order probabilities from the samples in each
        bag.

@tag2
    calculate_class_priors()

        Calls on the appropriate method of the main DecisionTree class to
        estimate the class priors for the training data samples in each
        bag.

@tag2
    construct_decision_trees_for_bags()

        Calls on the appropriate method of the main DecisionTree class to
        construct a decision tree from the training data samples in each
        bag.

@tag2
    display_decision_trees_for_bags()

        Displays separately the decision tree for each bag.

@tag2
    classify_with_bagging( test_sample )

        Calls on the appropriate methods of the main DecisionTree class to
        classify the argument test sample.

@tag2
    display_classification_results_for_each_bag()

        Displays separately the classification decision made by the
        decision tree constructed for each bag.

@tag2
    get_majority_vote_classification()

        Using majority voting, this method aggregates the classification
        decisions made by the individual decision trees into a single
        decision.

    See the example scripts in the ExamplesBagging directory for how to
    call the methods listed above for classifying individual data samples
    and for bulk classification when you place all your test samples in a
    single file.


@tag_boosting
USING BOOSTING:

    Starting with Version 3.2.0, you can use the class BoostedDecisionTree
    for constructing a boosted decision-tree classifier.  Boosting results
    in a cascade of decision trees in which each decision tree is
    constructed with samples that are mostly those that are misclassified
    by the previous decision tree.  To be precise, you create a probability
    distribution over the training samples for the selection of samples for
    training each decision tree in the cascade.  To start out, the
    distribution is uniform over all of the samples. Subsequently, this
    probability distribution changes according to the misclassifications by
    each tree in the cascade: if a sample is misclassified by a given tree
    in the cascade, the probability of its being selected for training the
    next tree is increased significantly.  You also associate a trust
    factor with each decision tree depending on its power to classify
    correctly all of the training data samples.  After a cascade of
    decision trees is constructed in this manner, you construct a final
    classifier that calculates the class label for a test data sample by
    taking into account the classification decisions made by each
    individual tree in the cascade, the decisions being weighted by the
    trust factors associated with the individual classifiers.  These
    boosting notions --- generally referred to as the AdaBoost algorithm
    --- are based on a now celebrated paper "A Decision-Theoretic
    Generalization of On-Line Learning and an Application to Boosting" by
    Yoav Freund and Robert Schapire that appeared in 1995 in the
    Proceedings of the 2nd European Conf. on Computational Learning Theory.
    For a tutorial introduction to AdaBoost, see

    https://engineering.purdue.edu/kak/Tutorials/AdaBoost.pdf

    Keep in mind the fact that, ordinarily, the theoretical guarantees
    provided by boosting apply only to the case of binary classification.
    Additionally, your training dataset must capture all of the significant
    statistical variations in the classes represented therein.


@tag2
    Calling the BoostedDecisionTree constructor:

        If you'd like to experiment with boosting, a typical call to the
        constructor for the BoostedDecisionTree class looks like:

@begincode    
            import BoostedDecisionTree
            training_datafile = "training6.csv"

            boosted = BoostedDecisionTree.BoostedDecisionTree(
                            training_datafile = training_datafile,
                            csv_class_column_index = 1,
                            csv_columns_for_features = [2,3],
                            entropy_threshold = 0.01,
                            max_depth_desired = 8,
                            symbolic_to_numeric_cardinality_threshold = 10,
                            how_many_stages = 4,
                            csv_cleanup_needed = 1,
                      )
@endcode
    
        Note in particular the constructor parameter:

@begincode    
            how_many_stages
@endcode
    
        As its name implies, this parameter controls how many stages will
        be used in the boosted decision tree classifier.  As mentioned
        above, a separate decision tree is constructed for each stage of
        boosting using a set of training samples that are drawn through a
        probability distribution maintained over the entire training
        dataset.

@tag2
    get_training_data_for_base_tree()

        This method reads your training datafile, creates the data structures
        from the data ingested for constructing the base decision tree.

@tag2
    show_training_data_for_base_tree()

        Writes to the standard output the training data samples and also
        some relevant properties of the features used in the training
        dataset.

@tag2
    calculate_first_order_probabilities_and_class_priors()

        Calls on the appropriate methods of the main DecisionTree class to
        estimate the first-order probabilities and the class priors.

@tag2
    construct_base_decision_tree()

        Calls on the appropriate method of the main DecisionTree class to
        construct the base decision tree.

@tag2
    display_base_decision_tree()

        Displays the base decision tree in your terminal window. (The
        textual form of the decision tree is written out to the standard
        output.)

@tag2
    construct_cascade_of_trees()

        Uses the AdaBoost algorithm to construct a cascade of decision
        trees.  As mentioned earlier, the training samples for each tree in
        the cascade are drawn using a probability distribution over the
        entire training dataset. This probability distribution for any
        given tree in the cascade is heavily influenced by which training
        samples are misclassified by the previous tree.
        
@tag2
    display_decision_trees_for_different_stages()

        Displays separately in your terminal window the decision tree
        constructed for each stage of the cascade. (The textual form of the
        trees is written out to the standard output.)

@tag2
    classify_with_boosting( test_sample )

        Calls on each decision tree in the cascade to classify the argument
        test sample.

@tag2
    display_classification_results_for_each_stage()

        You can call this method to display in your terminal window the
        classification decision made by each decision tree in the cascade.
        The method also prints out the trust factor associated with each
        decision tree.  It is important to look simultaneously at the
        classification decision and the trust factor for each tree ---
        since a classification decision made by a specific tree may appear
        bizarre for a given test sample.  This method is useful primarily
        for debugging purposes.

@tag2
    show_class_labels_for_misclassified_samples_in_stage( stage_index )

        As with the previous method, this method is useful mostly for
        debugging. It returns class labels for the samples misclassified by
        the stage whose integer index is supplied as an argument to the
        method.  Say you have 10 stages in your cascade.  The value of the
        argument stage_index would go from 0 to 9, with 0 corresponding to
        the base tree.

@tag2
    trust_weighted_majority_vote_classifier()

        Uses the "final classifier" formula of the AdaBoost algorithm to
        pool together the classification decisions made by the individual
        trees while taking into account the trust factors associated with
        the trees.  As mentioned earlier, we associate with each tree of
        the cascade a trust factor that depends on the overall
        misclassification rate associated with that tree.


    See the example scripts in the ExamplesBoosting subdirectory for how to
    call the methods listed above for classifying individual data samples
    with boosting and for bulk classification when you place all your test
    samples in a single file.


@tag_usingrandomized
USING RANDOMIZED DECISION TREES:

    As mentioned earlier, the new RandomizedTreesForBigData class allows
    you to solve the following two problems: (1) Data classification using
    the needle-in-a-haystack metaphor, that is, when a vast majority of
    your training samples belong to just one class.  And (2) You have
    access to a very large database of training samples and you wish to
    construct an ensemble of decision trees for classification.

    Here is how you'd call the RandomizedTreesForBigData constructor for
    needle-in-a-haystack classification:

@begincode    
            import RandomizedTreesForBigData
            training_datafile = "MyLargeDatabase.csv"
            rt = RandomizedTreesForBigData.RandomizedTreesForBigData(
                            training_datafile = training_datafile,
                            csv_class_column_index = 48,
                            csv_columns_for_features = [39,40,41,42],  
                            entropy_threshold = 0.01,
                            max_depth_desired = 8,
                            symbolic_to_numeric_cardinality_threshold = 10,
                            looking_for_needles_in_haystack = 1,
                            how_many_trees = 5,
                            csv_cleanup_needed = 1,
                 )
@endcode
    
    Note in particular the constructor parameters:

@begincode    
            looking_for_needles_in_haystack

            how_many_trees
@endcode

    The first of these parameters, looking_for_needles_in_haystack, invokes
    the logic for constructing an ensemble of decision trees, each based on
    a training dataset that uses all of the minority class samples, and a
    random drawing from the majority class samples.

    Here is how you'd call the RandomizedTreesForBigData constructor for a
    more general attempt at constructing an ensemble of decision trees,
    with each tree trained with randomly drawn samples from a large
    database of training data:

@begincode    
            import RandomizedTreesForBigData
            training_datafile = "MyLargeDatabase.csv"
            rt = RandomizedTreesForBigData.RandomizedTreesForBigData(
                            training_datafile = training_datafile,
                            csv_class_column_index = 48,
                            csv_columns_for_features = [39,40,41,42],  
                            entropy_threshold = 0.01,
                            max_depth_desired = 8,
                            symbolic_to_numeric_cardinality_threshold = 10,
                            how_many_training_samples_per_tree = 150,
                            how_many_trees = 5,
                            csv_cleanup_needed = 1,
                 )
@endcode
    
    Note in particular the constructor parameters:

@begincode    
            how_many_training_samples_per_tree

            how_many_trees
@endcode

    NOTE: When you set the 'how_many_training_samples_per_tree' parameter,
    you are not allowed to also set the 'looking_for_needles_in_haystack'
    parameter, and vice versa.

    Here is a list of the methods defined for the RandomizedTreesForBigData
    class that you can call in your own scripts:

@tag2
    get_training_data_for_N_trees()

        What this method does depends on which of the two constructor
        parameters --- 'looking_for_needles_in_haystack' or
        'how_many_training_samples_per_tree' --- is set.  When the former
        is set, it creates a collection of training datasets for
        'how_many_trees' number of decision trees, with each dataset being
        a mixture of the minority class and sample drawn randomly from the
        majority class.  However, when the latter option is set, all the
        datasets are drawn randomly from the training database with no
        particular attention given to the relative populations of the two
        classes.

@tag2
    show_training_data_for_all_trees()

        As the name implies, this method shows the training data being
        used for all the decision trees.  This method is useful for
        debugging purposes using small datasets.

@tag2
    calculate_first_order_probabilities()

        Calls on the appropriate method of the main DecisionTree class to
        estimate the first-order probabilities for the training dataset to
        be used for each decision tree.
        
@tag2
    calculate_class_priors()

        Calls on the appropriate method of the main DecisionTree class to
        estimate the class priors for the training dataset to be used for
        each decision tree.

@tag2
    construct_all_decision_trees()

        Calls on the appropriate method of the main DecisionTree class to
        construct the decision trees.

@tag2
    display_all_decision_trees()

        Displays all the decision trees in your terminal window. (The
        textual form of the decision trees is written out to the standard
        output.)

@tag2
    classify_with_all_trees( test_sample )

        The test_sample is sent to each decision tree for classification.

@tag2
    display_classification_results_for_all_trees()

        The classification decisions returned by the individual decision
        trees are written out to the standard output.

@tag2
    get_majority_vote_classification()

        This method aggregates the classification results returned by the
        individual decision trees and returns the majority decision.


@tag_regression
CONSTRUCTING REGRESSION TREES:

    Decision tree based modeling requires that the class labels be
    distinct.  That is, the training dataset must contain a relatively
    small number of discrete class labels for all of your data records if
    you want to model the data with one or more decision trees.  However,
    when one is trying to understand all of the associational
    relationships that exist in a large database, one often runs into
    situations where, instead of discrete class labels, you have a
    continuously valued variable as a dependent variable whose values are
    predicated on a set of feature values.  It is for such situations that
    you will find useful the new class RegressionTree that is now a part of
    the DecisionTree module.  The RegressionTree class has been programmed
    as a subclass of the main DecisionTree class.

    You can think of regression with a regression tree as a powerful
    generalization of the very commonly used Linear Regression algorithms.
    Although you can certainly carry out polynomial regression with
    run-of-the-mill Linear Regression algorithms for modeling
    nonlinearities between the predictor variables and the dependent
    variable, specifying the degree of the polynomial is often tricky.
    Additionally, a polynomial can inject continuities between the
    predictor and the predicted variables that may not really exist in the
    real data.  Regression trees, on the other hand, give you a piecewise
    linear relationship between the predictor and the predicted variables
    that is freed from the constraints of superimposed continuities at the
    joins between the different segments.  See the following tutorial for
    further information regarding the standard linear regression approach
    and the regression that can be achieved with the RegressionTree class
    in this module:

    https://engineering.purdue.edu/kak/Tutorials/RegressionTree.pdf

    The RegressionTree class in the current version of the module assumes
    that all of your data is numerical.  That is, unlike what is possible
    with the DecisionTree class (and the other more closely related classes
    in this module) that allow your training file to contain a mixture of
    numerical and symbolic data, the RegressionTree class requires that ALL
    of your data be numerical.  I hope to relax this constraint in future
    versions of this module.  Obviously, the dependent variable will always
    be numerical for regression.

    See the example scripts in the directory ExamplesRegression if you wish
    to become more familiar with the regression capabilities of the module.

    Here is how you'd call the RegressionTree constructor:

@begincode    
            import RegressionTree
            training_datafile = "gendata6.csv"
            rt = RegressionTree.RegressionTree( 
                                    training_datafile = training_datafile,
                                    dependent_variable_column = 3,
                                    predictor_columns = [1,2],
                                    mse_threshold = 0.01,
                                    max_depth_desired = 2,
                                    jacobian_choice = 0,
                                    csv_cleanup_needed = 1,
                 )
@endcode
    
    Note in particular the constructor parameters:

@begincode    
        dependent_variable
        predictor_columns
        mse_threshold
        jacobian_choice
@endcode

    The first of these parameters, dependent_variable, is set to the column
    index in the CSV file for the dependent variable.  The second
    constructor parameter, predictor_columns, tells the system as to which
    columns contain values for the predictor variables.  The third
    parameter, mse_threshold, is for deciding when to partition the data at
    a node into two child nodes as a regression tree is being constructed.
    If the minmax of MSE (Mean Squared Error) that can be achieved by
    partitioning any of the features at a node is smaller than
    mse_threshold, that node becomes a leaf node of the regression tree.

    The last parameter, jacobian_choice, must be set to either 0 or 1 or 2.
    Its default value is 0. When this parameter equals 0, the regression
    coefficients are calculated using the linear least-squares method and
    no further "refinement" of the coefficients is carried out using
    gradient descent.  This is the fastest way to calculate the regression
    coefficients.  When jacobian_choice is set to 1, you get a weak version
    of gradient descent in which the Jacobian is set to the "design matrix"
    itself. Choosing 2 for jacobian_choice results in a more reasonable
    approximation to the Jacobian.  That, however, is at a cost of much
    longer computation time.  NOTE: For most cases, using 0 for
    jacobian_choice is the best choice.  See my tutorial "Linear Regression
    and Regression Trees" for why that is the case.

    Here is a list of the methods defined for the RegressionTree class that
    you can call in your own scripts:

@tag2
    get_training_data_for_regression()

        Only CSV training datafiles are allowed. Additionally, the first
        record in the file must list the names of the fields, and the first
        column must contain an integer ID for each record.

@tag2
    construct_regression_tree()

        As the name implies, this is the method that construct a regression
        tree.

@tag2
    display_regression_tree("     ")

        Displays the regression tree, as the name implies.  The white-space
        string argument specifies the offset to use in displaying the child
        nodes in relation to a parent node.

@tag2
    prediction_for_single_data_point( root_node, test_sample )

        You call this method after you have constructed a regression tree
        if you want to calculate the prediction for one sample.  The
        parameter root_node is what is returned by the call
        construct_regression_tree().  The formatting of the argument bound
        to the test_sample parameter is important.  To elaborate, let's say
        you are using two variables named xvar1 and xvar2 as your predictor
        variables. In this case, the test_sample parameter will be bound to
        a Python list that will look like 
@begincode 
            ['xvar1 = 23.4', 'xvar2 = 12.9'] 
@endcode 
        Arbitrary amount of white space, including none, on the two sides
        of the equality symbol is allowed in the construct shown above.  A
        call to this method returns a dictionary with two key-value pairs.
        One of the keys is called 'solution_path' and the other
        'prediction'.  The value associated with key 'solution_path' is the
        path in the regression tree to the leaf node that yielded the
        prediction.  And the value associated with the key 'prediction' is
        the answer you are looking for.

@tag2
    predictions_for_all_data_used_for_regression_estimation( root_node )

        This call calculates the predictions for all of the predictor
        variables data in your training file.  The parameter root_node is
        what is returned by the call to construct_regression_tree().  The
        values for the dependent variable thus predicted can be seen by
        calling display_all_plots(), which is the method mentioned below.

@tag2
    display_all_plots()

        This method displays the results obtained by calling the prediction
        method of the previous entry.  This method also creates a hardcopy
        of the plots and saves it as a '.png' disk file. The name of this
        output file is always "regression_plots.png".

@tag2
    mse_for_tree_regression_for_all_training_samples( root_node )

        This method carries out an error analysis of the predictions for
        the samples in your training datafile.  It shows you the overall
        MSE (Mean Squared Error) with tree-based regression, the MSE for
        the data samples at each of the leaf nodes of the regression tree,
        and the MSE for the plain old Linear Regression as applied to all
        of the data.  The parameter root_node in the call syntax is what is
        returned by the call to construct_regression_tree().

@tag2
    bulk_predictions_for_data_in_a_csv_file(root_node, filename, columns)

        Call this method if you want to apply the regression tree to all
        your test data in a disk file.  The predictions for all of the test
        samples in the disk file are written out to another file whose name
        is the same as that of the test file except for the addition of
        '_output' in the name of the file.  The parameter 'filename' is the
        name of the disk file that contains the test data. And the
        parameter 'columns' is a Python list of the column indices for the
        predictor variables in the test file.


@tag_generating
GENERATING SYNTHETIC TRAINING DATA:

    To generate synthetic training data, you first construct an instance of
    the class TrainingDataGenerator that is incorporated in the
    DecisionTree module.  A call to the constructor of this class will look
    like:

@begincode
        parameter_file = "param_numeric.txt"
        output_csv_file = "training.csv";
        training_data_gen = TrainingDataGeneratorNumeric(
                              output_csv_file   = output_csv_file,
                              parameter_file    = parameter_file,
                              number_of_samples_per_class = some_number,
                            )
        training_data_gen.read_parameter_file_numeric()
        training_data_gen.gen_numeric_training_data_and_write_to_csv()
@endcode

    The training data that is generated is according to the specifications
    described in the parameter file.  The structure of this file must be as
    shown in the file `param_numeric.txt' for the numeric training data and
    as shown in `param_symbolic.txt' for the case of symbolic training
    data.  Both these example parameter files are in the 'Examples'
    subdirectory.  The parameter file names the classes, the features for
    the classes, and the possible values for the features.

    If you want to generate purely symbolic training data, here is the
    constructor call to make:

@begincode
        parameter_file = "param_symbolic.txt"
        output_data_file = "training_symbolic2.csv";
        training_data_gen = TrainingDataGeneratorSymbolic(
                              output_datafile   = output_data_file,
                              parameter_file    = parameter_file,
                              write_to_file     = 1,
                              number_of_training_samples = some_number,
                            )
        training_data_gen.read_parameter_file_symbolic()
        training_data_gen.gen_symbolic_training_data()
        training_data_gen.write_training_data_to_file()
@endcode


@tag_examples
THE Examples DIRECTORY:

    See the 'Examples' directory in the distribution for how to construct a
    decision tree, and how to then classify new data using the decision
    tree.  To become more familiar with the module, run the scripts

@begincode
        construct_dt_and_classify_one_sample_case1.py
        construct_dt_and_classify_one_sample_case2.py
        construct_dt_and_classify_one_sample_case3.py
        construct_dt_and_classify_one_sample_case4.py
@endcode

    The first script is for the purely symbolic case, the second for the
    case that involves both numeric and symbolic features, the third for
    the case of purely numeric features, and the last for the case when the
    training data is synthetically generated by the script
    generate_training_data_numeric.py

    Next, run the following script as it is for bulk classification of data
    records placed in a CSV file:

@begincode
      classify_test_data_in_a_file.py  training4.csv  test4.csv  out4.csv
@endcode

    The script first constructs a decision tree using the training data in
    the first-argument file, `training4.csv'.  Subsequently, the script
    calculates the class labels for each of the test records in the file
    `test4.csv'.  The class labels are written out the file `out4.csv'.  An
    important thing to note here that your test file --- in this case
    `test4.csv' --- must have a column for the class labels.  Obviously, in
    real-life situations, there will be no class labels in this column.
    When that is the case, you can place the empty string "" for each data
    record in this column.  A demonstration of that is give by the
    following variation of the above call:

@begincode
      classify_test_data_in_a_file.py  training4.csv  test4_no_class_labels.csv  out4.csv 
@endcode

    The following script in the 'Examples' directory 

@begincode
        classify_by_asking_questions.py
@endcode

    shows how you can use a decision-tree classifier interactively.  In
    this mode, you first construct the decision tree from the training data
    and then the user is prompted for answers to the feature tests at the
    nodes of the tree.

    If your training data has a feature whose values span a large range
    and, at the same time, are characterized by a heavy-tail distribution,
    you should look at the script 

@begincode
        construct_dt_for_heavytailed.py
@endcode

    to see how to use the option number_of_histogram_bins in the call to
    the constructor.  This option was introduced in Version 2.2.4 for
    dealing with such features.  If you do not set this option, the module
    will use the default value of 500 for the number of points at which to
    sample the value range for such a feature.

    The 'Examples' directory also contains the following scripts:

@begincode
        generate_training_data_numeric.py
        generate_training_data_symbolic.py
@endcode

    that show how you can use the module to generate synthetic training
    data.  Synthetic training and test data are generated according to the
    specifications laid out in a parameter file.  There are constraints on
    how the information is laid out in the parameter file.  See the files
    `param_numeric.txt' and `param_symbolic.txt' in the 'Examples'
    directory for how to structure these files.

    The Examples directory of Versions 2.2 and higher of the DecisionTree
    module also contains the following two scripts:

@begincode
       evaluate_training_data1.py
       evaluate_training_data2.py
@endcode

    that illustrate how the Python class EvalTrainingData can be used to
    evaluate the quality of your training data (as long as it resides in a
    `.csv' file.)  This new class is a subclass of the DecisionTree class
    in the module file.  See the README in the Examples directory for
    further information regarding these two scripts.

    The Examples directory of Versions 2.3.2 and higher of the module
    contains the following three scripts:

@begincode
        introspection_in_a_loop_interactive.py

        introspection_show_training_samples_at_all_nodes_direct_influence.py

        introspection_show_training_samples_to_nodes_influence_propagation.py
@endcode

    The first script illustrates how to use the DTIntrospection class of
    the module interactively for generating explanations for the
    classification decisions made at the nodes of the decision tree.  In
    the interactive session you are first asked for the node number you are
    interested in.  Subsequently, you are asked for whether or not you are
    interested in specific questions that the introspector can provide
    answers for. The second script generates a tabular display that shows
    for each node of the decision tree a list of the training samples that
    fall directly in the portion of the feature space assigned that node.
    (As mentioned elsewhere in this documentation, when this list is empty
    for a node, that means the node is a result of the generalization
    achieved by probabilistic modeling of the data.  Note that this module
    constructs a decision tree NOT by partitioning the set of training
    samples, BUT by partitioning the domains of the probability density
    functions.)  The third script listed above also generates a tabular
    display, but one that shows how the influence of each training sample
    propagates in the tree.  This display first shows the list of nodes
    that are affected directly by the data in a training sample. This list
    is followed by an indented display of the nodes that are affected
    indirectly by the training sample.  A training sample affects a node
    indirectly if the node is a descendant of one of the nodes affected
    directly.

    The latest addition to the Examples directory is the script:

@begincode
        get_indexes_associated_with_fields.py
@endcode

    As to why you may find this script useful, note that large database
    files may have hundreds of fields and it is not always easy to figure
    out what numerical index is associated with a given field.  At the same
    time, the constructor of the DecisionTree module requires that the
    field that holds the class label and the fields that contain the
    feature values be specified by their numerical zero-based indexes.  If
    you have a very large database and you are faced with such a problem,
    you can run this script to see the zero-based numerical index values
    associated with the different columns of your CSV file.


@tag_baggingexamples
THE ExamplesBagging DIRECTORY:

    The ExamplesBagging subdirectory in the main installation directory
    contains the following scripts:

@begincode
        bagging_for_classifying_one_test_sample.py

        bagging_for_bulk_classification.py
@endcode

    As the names of the scripts imply, the first shows how to call the
    different methods of the DecisionTreeWithBagging class for classifying
    a single test sample.  When you are classifying a single test sample,
    you can also see how each bag is classifying the test sample.  You can,
    for example, display the training data used in each bag, the decision
    tree constructed for each bag, etc.

    The second script is for the case when you place all of the test
    samples in a single file.  The demonstration script displays for each
    test sample a single aggregate classification decision that is obtained
    through majority voting by all the decision trees.


@tag_boostingexamples
THE ExamplesBoosting DIRECTORY:

    The ExamplesBoosting subdirectory in the main installation directory
    contains the following three scripts:

@begincode
        boosting_for_classifying_one_test_sample_1.py

        boosting_for_classifying_one_test_sample_2.py

        boosting_for_bulk_classification.py
@endcode

    As the names of the first two scripts imply, these show how to call the
    different methods of the BoostedDecisionTree class for classifying a
    single test sample.  When you are classifying a single test sample, you
    can see how each stage of the cascade of decision trees is classifying
    the test sample.  You can also view each decision tree separately and
    also see the trust factor associated with the tree.

    The third script is for the case when you place all of the test samples
    in a single file.  The demonstration script outputs for each test
    sample a single aggregate classification decision that is obtained
    through trust-factor weighted majority voting by all the decision
    trees.


@tag_examplesrandomized
THE ExamplesRandomizedTrees DIRECTORY:

    The ExamplesRandomizedTrees subdirectory in the main installation
    directory shows example scripts that you can use to become more
    familiar with the RandomizedTreesForBigData class for solving
    needle-in-a-haystack and big-data data classification problems. These
    scripts are:

@begincode
        randomized_trees_for_classifying_one_test_sample_1.py

        randomized_trees_for_classifying_one_test_sample_2.py

        classify_database_records.py
@endcode

    The first script shows the constructor options to use for solving a
    needle-in-a-haystack problem --- that is, a problem in which a vast
    majority of the training data belongs to just one class.  The second
    script shows the constructor options for using randomized decision
    trees for the case when you have access to a very large database of
    training samples and you'd like to construct an ensemble of decision
    trees using training samples pulled randomly from the training
    database.  The last script illustrates how you can evaluate the
    classification power of an ensemble of decision trees as constructed by
    RandomizedTreesForBigData by classifying a large number of test samples
    extracted randomly from the training database.


@tag_examplesregression
THE ExamplesRegression DIRECTORY:

    The ExamplesRegression subdirectory in the main installation directory
    shows example scripts that you can use to become familiar with
    regression trees and how they can be used for nonlinear regression.  If
    you are new to the concept of regression trees, start by executing the
    following scripts without changing them and see what sort of output is
    produced by them:

@begincode
        regression4.py

        regression5.py

        regression6.py

        regression8.py
@endcode

    With all these scripts, you may wish to try all three values, 0, 1, or
    2, for the constructor parameter jacobian_choice to see the difference
    made by including gradient descent in the calculation of the regression
    coefficients.  Recall, when jacobian_choice is 0, which is the default,
    you get no gradient descent.

    The 'regression4.py' script involves only one predictor variable and
    one dependent variable. The training data for this exercise is drawn
    from the file 'gendata4.csv'.  This data file contains strongly
    nonlinear data.  When you run the script regression4.py, you will see
    how much better the result from tree regression is compared to what you
    can get with linear regression.

    The 'regression5.py' script is essentially the same as the previous
    script except for the fact that the training datafile used in this
    case, 'gendata5.csv', consists of three noisy segments, as opposed to
    just two in the previous case.

    The script 'regression6.py' deals with the case when we have two
    predictor variables and one dependent variable.  You can think of the
    data as consisting of noisy height values over an (x1,x2) plane.  The
    data used in this script is drawn from the csv file 'gen3Ddata1.csv'

    Finally, the script 'regression8.py' shows how you can carry out bulk
    prediction for all your test data records in a disk file.  The script
    writes all the calculated predictions into another disk file whose name
    is derived from the name of the test data file.


@tag_install
INSTALLATION:

    The DecisionTree class was packaged using setuptools.  For
    installation, execute the following command-line in the source
    directory (this is the directory that contains the setup.py file after
    you have downloaded and uncompressed the package):
 
@begincode
            sudo python setup.py install
@endcode

    and/or

@begincode
            sudo python3 setup.py install
@endcode

    On Linux distributions, this will install the module file at a location
    that looks like

@begincode
             /usr/local/lib/python2.7/dist-packages/
@endcode

    and for Python3 at a location like

@begincode
             /usr/local/lib/python3.4/dist-packages/
@endcode

    If you do not have root access, you have the option of working directly
    off the directory in which you downloaded the software by simply
    placing the following statements at the top of your scripts that use
    the DecisionTree class:

@begincode
            import sys
            sys.path.append( "pathname_to_DecisionTree_directory" )
@endcode

    To uninstall the module, simply delete the source directory, locate
    where the DecisionTree module was installed with "locate DecisionTree"
    and delete those files.  As mentioned above, the full pathname to the
    installed version is likely to look like
    /usr/local/lib/python2.7/dist-packages/DecisionTree*

    If you want to carry out a non-standard install of the DecisionTree
    module, look up the on-line information on Disutils by pointing your
    browser to

              http://docs.python.org/dist/dist.html


@tag_bugs
BUGS:

    Please notify the author if you encounter any bugs.  When sending
    email, please place the string 'DecisionTree' in the subject line.


@tag_ack
ACKNOWLEDGMENTS:

    The importance of the 'sentiment' feature in the "What Practical Problem
    is Solved by this Module" section was mentioned to the author by John
    Gorup.  Thanks John.

    Thanks go to Wenshuai Hou for discovering and reporting the bug that
    resulted in Version 2.2.3.  I should also thank Wenshuai for sending me
    a training data file with heavy-tailed values for one of the
    features. This datafile became the reason for the modifications to the
    code that are incorporated in Version 2.2.4.

    I owe many thanks to Eugene Kolotinsky for the bug discovery whose fix
    was the primary reason for Version 2.2.6.  This bug related to the
    erroneous calculation of the probability of a feature acquiring a
    certain value if the training data contained zeros for the feature
    values.

    The bug whose fix led to Version 3.2.2 was reported by Erez Segal. He
    noticed that I had left out the keyword "raise" in the "else" clause
    when checking for the allowed suffixes in the name of the
    user-specified training data file. Thanks, Erez!

    With regard to Version 3.3.2, I wish to thank Alexandre Drouin-Picaro
    for bringing to my attention the problem created by the nested class
    DTNode when using pickle to save a decision tree to a disk file.


@tag_author
ABOUT THE AUTHOR:

    The author, Avinash Kak, recently finished a 17-year long "Objects
    Trilogy Project" with the publication of the book "Designing with
    Objects" by John-Wiley. If interested, visit his web page at Purdue to
    find out what this project was all about. You might like "Designing
    with Objects" especially if you enjoyed reading Harry Potter as a kid
    (or even as an adult, for that matter).

    For all issues related to this module, contact the author at
    kak@purdue.edu

    If you send email, please place the string "DecisionTree" in your
    subject line to get past the author's spam filter.


@tag_copyright
COPYRIGHT:

    Python Software Foundation License

    Copyright 2016 Avinash Kak

@endofdocs
'''

import math
import re
import sys
import functools 
import operator
import itertools
import string
import os

#-----------------------------------  Utility Functions  ------------------------------------

def sample_index(sample_name):
    '''
    When the training data is read from a CSV file, we assume that the first column
    of each data record contains a unique integer identifier for the record in that
    row. This training data is stored in a dictionary whose keys are the prefix
    'sample_' followed by the identifying integers. The purpose of this function is to
    return the identifying integer associated with a data record.
    '''
    m = re.search('_(.+)$', sample_name)
    return int(m.group(1))

def deep_copy_array(array_in):
    '''
    Meant only for an array of scalars (no nesting):
    '''
    array_out = []
    for i in range(len(array_in)):
        array_out.append( array_in[i] )
    return array_out

def minimum(arr):
    '''
    Returns simultaneously the minimum value and its positional index in an
    array. [Could also have used min() and index() defined for Python's
    sequence types.]
    '''
    minval,index = None,None
    for i in range(0, len(arr)):  
        if minval is None or arr[i] < minval:
            index = i
            minval = arr[i]
    return minval,index

def convert(value):
    try:
        answer = float(value)
        return answer
    except:
        return value

def closest_sampling_point(value, arr):
    if value == 'NA': return None                # in version 3.3.0
    compare = [abs(x - value) for x in arr]
    minval,index = minimum(compare)
    return arr[index]

def cleanup_csv(line):
    line = line.translate(bytes.maketrans(b":?/()[]{}'",b"          ")) \
           if sys.version_info[0] == 3 else line.translate(string.maketrans(":?/()[]{}'","          "))
    double_quoted = re.findall(r'"[^\"]+"', line[line.find(',') : ])
    for item in double_quoted:
        clean = re.sub(r',', r'', item[1:-1].strip())
        parts = re.split(r'\s+', clean.strip())
        line = str.replace(line, item, '_'.join(parts))
    white_spaced = re.findall(r',(\s*[^,]+)(?=,|$)', line)
    for item in white_spaced:
        litem = item
        litem = re.sub(r'\s+', '_', litem)
        litem = re.sub(r'^\s*_|_\s*$', '', litem) 
        line = str.replace(line, "," + item, "," + litem) if line.endswith(item) else str.replace(line, "," + item + ",", "," + litem + ",") 
    fields = re.split(r',', line)
    newfields = []
    for field in fields:
        newfield = field.strip()
        if newfield == '':
            newfields.append('NA')
        else:
            newfields.append(newfield)
    line = ','.join(newfields)
    return line

#----------------------------------- DecisionTree Class Definition ------------------------------------

class DecisionTree(object):
    def __init__(self, *args, **kwargs ):
        if kwargs and args:
            raise SyntaxError(  
                   '''DecisionTree constructor can only be called with keyword arguments for the 
                      following keywords: training_datafile,entropy_threshold, max_depth_desired, 
                      csv_class_column_index,symbolic_to_numeric_cardinality_threshold, number_of_histogram_bins,
                      csv_columns_for_features,csv_cleanup_needed,number_of_histogram_bins, debug1, debug2, and debug3''') 
        allowed_keys = 'training_datafile','entropy_threshold','max_depth_desired','csv_class_column_index',\
                       'symbolic_to_numeric_cardinality_threshold','csv_columns_for_features',\
                       'number_of_histogram_bins','csv_cleanup_needed','debug1','debug2','debug3'
        keywords_used = kwargs.keys()
        for keyword in keywords_used:
            if keyword not in allowed_keys:
                raise SyntaxError(keyword + ":  Wrong keyword used --- check spelling") 
        training_datafile=entropy_threshold=max_depth_desired=csv_class_column_index=number_of_histogram_bins= None
        symbolic_to_numeric_cardinality_threshold=csv_columns_for_features=csv_cleanup_needed=debug1=debug2=debug3=None
        if kwargs and not args:
            if 'training_datafile' in kwargs : training_datafile = kwargs.pop('training_datafile')
            if 'entropy_threshold' in kwargs : entropy_threshold = kwargs.pop('entropy_threshold')
            if 'max_depth_desired' in kwargs : max_depth_desired = kwargs.pop('max_depth_desired')
            if 'csv_class_column_index' in kwargs: csv_class_column_index = kwargs.pop('csv_class_column_index')
            if 'csv_columns_for_features' in kwargs: \
                                  csv_columns_for_features = kwargs.pop('csv_columns_for_features')
            if 'symbolic_to_numeric_cardinality_threshold' in kwargs: \
                           symbolic_to_numeric_cardinality_threshold = \
                                                         kwargs.pop('symbolic_to_numeric_cardinality_threshold')
            if 'number_of_histogram_bins' in kwargs: \
                                               number_of_histogram_bins = kwargs.pop('number_of_histogram_bins')
            if 'csv_cleanup_needed' in kwargs: csv_cleanup_needed = kwargs.pop('csv_cleanup_needed')
            if 'debug1' in kwargs  :  debug1 = kwargs.pop('debug1')
            if 'debug2' in kwargs  :  debug2 = kwargs.pop('debug2')
            if 'debug3' in kwargs  :  debug3 = kwargs.pop('debug3')
        if not args and training_datafile:
            self._training_datafile = training_datafile
        elif not args and not training_datafile:
            raise SyntaxError('''You must specify a training datafile''')
        else:
            if (args[0] != 'evalmode') and (args[0] != 'boostingmode'):
                raise SyntaxError("""When supplying non-keyword arg, it can only be 'evalmode' or 'boostingmode'""")
        if entropy_threshold: 
            self._entropy_threshold                         =      entropy_threshold
        else:
            self._entropy_threshold                         =      0.01        
        if max_depth_desired:
            self._max_depth_desired                         =      max_depth_desired 
        else:
            self._max_depth_desired                         =      None
        if number_of_histogram_bins:
            self._number_of_histogram_bins                  =      number_of_histogram_bins
        else:
            self._number_of_histogram_bins                  =      None
        if csv_class_column_index:
            self._csv_class_column_index                    =      csv_class_column_index
        else:
            self._csv_class_column_index                    =      None
        if csv_columns_for_features:
            self._csv_columns_for_features                  =      csv_columns_for_features
        else: 
            self._csv_columns_for_features                  =      None            
        if symbolic_to_numeric_cardinality_threshold:
            self._symbolic_to_numeric_cardinality_threshold =      symbolic_to_numeric_cardinality_threshold
        else:
            self._symbolic_to_numeric_cardinality_threshold =      10
        if csv_cleanup_needed:
            self._csv_cleanup_needed                        =      csv_cleanup_needed
        else:
            self._csv_cleanup_needed                        =      0
        if debug1:
            self._debug1                                    =      debug1
        else:
            self._debug1                                    =      0
        if debug2:
            self._debug2                                    =      debug2
        else:
            self._debug2                                    =      0
        if debug3:
            self._debug3                                    =      debug3
        else:
            self._debug3                                    =      0
        self._root_node                                     =      None
        self._probability_cache                             =      {}
        self._entropy_cache                                 =      {}
        self._training_data_dict                            =      {}
        self._features_and_values_dict                      =      {}
        self._features_and_unique_values_dict               =      {}
        self._samples_class_label_dict                      =      {}  
        self._class_names                                   =      []
        self._class_priors_dict                             =      {}
        self._feature_names                                 =      []
        self._numeric_features_valuerange_dict              =      {}
        self._sampling_points_for_numeric_feature_dict      =      {}
        self._feature_values_how_many_uniques_dict          =      {}
        self._prob_distribution_numeric_features_dict       =      {}
        self._histogram_delta_dict                          =      {}
        self._num_of_histogram_bins_dict                    =      {}

    def get_training_data(self):
        if not self._training_datafile.endswith('.csv'): 
            TypeError("Aborted. get_training_data_from_csv() is only for CSV files")
        class_names = []
        all_record_ids_with_class_labels = {}
        firstline = None
        data_dict = {}
        with open(self._training_datafile) as f:
            for i,line in enumerate(f):
                record = cleanup_csv(line) if self._csv_cleanup_needed else line
                if i == 0:
                    firstline = record
                    continue
                parts = record.rstrip().split(r',')
                data_dict[parts[0].strip('"')] = parts[1:]
                class_names.append(parts[self._csv_class_column_index])
                all_record_ids_with_class_labels[parts[0].strip('"')] = parts[self._csv_class_column_index]
                if i%10000 == 0:
                    print('.'),
                    sys.stdout.flush()
                sys.stdout = sys.__stdout__
            f.close() 
        self._how_many_total_training_samples = i   # i is less by 1 from total num of records; but that's okay
        unique_class_names = list(set(class_names))
        if self._debug1:
            print("\n\nTotal number of training samples: %d\n" % self._how_many_total_training_samples)
        all_feature_names = firstline.rstrip().split(',')[1:]
        class_column_heading = all_feature_names[self._csv_class_column_index - 1]
        feature_names = [all_feature_names[i-1] for i in self._csv_columns_for_features]
        class_for_sample_dict = { "sample_" + key : 
               class_column_heading + "=" + data_dict[key][self._csv_class_column_index - 1] for key in data_dict}
        sample_names = ["sample_" + key for key in data_dict]
        feature_values_for_samples_dict = {"sample_" + key :         
                  list(map(operator.add, list(map(operator.add, feature_names, "=" * len(feature_names))), 
           [str(convert(data_dict[key][i-1])) for i in self._csv_columns_for_features])) 
                           for key in data_dict}
        features_and_values_dict = {all_feature_names[i-1] :
            [convert(data_dict[key][i-1]) for key in data_dict] for i in self._csv_columns_for_features}
        all_class_names = sorted(list(set(class_for_sample_dict.values())))
        if self._debug1: print("\n All class names: "+ str(all_class_names))
        numeric_features_valuerange_dict = {}
        feature_values_how_many_uniques_dict = {}
        features_and_unique_values_dict = {}
        for feature in features_and_values_dict:
            unique_values_for_feature = list(set(features_and_values_dict[feature]))
            unique_values_for_feature = sorted(list(filter(lambda x: x != 'NA', unique_values_for_feature)))
            feature_values_how_many_uniques_dict[feature] = len(unique_values_for_feature)
            if all(isinstance(x,float) for x in unique_values_for_feature):
                numeric_features_valuerange_dict[feature] = \
                                             [min(unique_values_for_feature), max(unique_values_for_feature)]
                unique_values_for_feature.sort(key=float)
            features_and_unique_values_dict[feature] = sorted(unique_values_for_feature)
        if self._debug1:
            print("\nAll class names: " + str(all_class_names))
            print("\nEach sample data record:")
            for item in sorted(feature_values_for_samples_dict.items(), key = lambda x: sample_index(x[0]) ):
                print(item[0]  + "  =>  "  + str(item[1]))
            print("\nclass label for each data sample:")
            for item in sorted(class_for_sample_dict.items(), key=lambda x: sample_index(x[0])):
                print(item[0]  + "  =>  "  + str(item[1]))
            print("\nfeatures and the values taken by them:")
            for item in sorted(features_and_values_dict.items()):
                print(item[0]  + "  =>  "  + str(item[1]))
            print("\nnumeric features and their ranges:")
            for item in sorted(numeric_features_valuerange_dict.items()):
                print(item[0]  + "  =>  "  + str(item[1]))
            print("\nnumber of unique values in each feature:")
            for item in sorted(feature_values_how_many_uniques_dict.items()):
                print(item[0]  + "  =>  "  + str(item[1]))
        self._class_names = all_class_names
        self._feature_names = feature_names
        self._samples_class_label_dict    =  class_for_sample_dict
        self._training_data_dict          =  feature_values_for_samples_dict
        self._features_and_values_dict    =  features_and_values_dict
        self._features_and_unique_values_dict    =  features_and_unique_values_dict
        self._numeric_features_valuerange_dict = numeric_features_valuerange_dict
        self._feature_values_how_many_uniques_dict = feature_values_how_many_uniques_dict

    def calculate_first_order_probabilities(self):
        print("\nEstimating probabilities...")
        for feature in self._feature_names:
            print("CALLING CALCULATE FIRST ORDER PROBABILITIES FOR FEATURE: " + feature)
            self.probability_of_feature_value(feature,None)
            if self._debug2:
                if feature in self._prob_distribution_numeric_features_dict:
                    print("\nPresenting probability distribution for a feature considered to be numeric:")
                    for sampling_point in \
                         sorted(self._prob_distribution_numeric_features_dict[feature].keys()):
                        print(feature + '::' + str(sampling_point) + ' = ' +  "{0:.5f}".format(
                               self._prob_distribution_numeric_features_dict[feature][sampling_point]))  
                else:
                    print("\nPresenting probabilities for the values of a feature considered to be symbolic:")
                    values_for_feature = self._features_and_unique_values_dict[feature]
                    for value in values_for_feature:
                        prob = self.probability_of_feature_value(feature,value) 
                        print(feature + '::' + str(value) + ' = ' +  "{0:.5f}".format(prob))

    def show_training_data(self):
        print("Class names: %s" % str(self._class_names))
        print("\n\nFeatures and Their Values:\n\n")
        features = self._features_and_values_dict.keys()
        for feature in sorted(features):
            nums,strings = [x for x in self._features_and_values_dict[feature] if isinstance(x,(float,int))],[x for x in self._features_and_values_dict[feature] if not isinstance(x,(float,int))]
            print("%s ---> %s"  % (feature, sorted(nums) + strings))
        print("\n\nSamples vs. Class Labels:\n\n")
        for item in sorted(self._samples_class_label_dict.items(), key = lambda x: sample_index(x[0]) ):
            print(item)
        print("\n\nTraining Samples:\n\n")
        for item in sorted(self._training_data_dict.items(), key = lambda x: sample_index(x[0]) ):
            print(item)

    def print_class_variables(self):
        # print the type of each class variable
        class_vars = self.__dict__.items()

        # Filter out non-class variables and print their types
        print(f'NEW OBJ')
        for var_name, value in class_vars:
            if not callable(value) and not var_name.startswith('__'):
                print(f"{var_name}: {type(value)}")
                if isinstance(value, list):
                    for i, item in enumerate(value):
                        print(f'  {var_name}[{i}]: {type(item)}')
                    # if empty list
                    if not value:
                        print(f'  empty list')
        print(f'END OBJ\n')

#-----------------------------    Classify with Decision Tree  --------------------------------

    def classify(self, root_node, features_and_values):
        '''
        Classifies one test sample at a time using the decision tree constructed from
        your training file.  The data record for the test sample must be supplied as
        shown in the scripts in the `Examples' subdirectory.  See the scripts
        construct_dt_and_classify_one_sample_caseX.py in that subdirectory.
        '''
        if not self._check_names_used(features_and_values):
            raise SyntaxError("\n\nError in the names you have used for features and/or values.  Try using the csv_cleanup_needed option in the constructor call.") 
        new_features_and_values = []
        pattern = r'(\S+)\s*=\s*(\S+)'
        for feature_and_value in features_and_values:
            m = re.search(pattern, feature_and_value)
            feature,value = m.group(1),m.group(2)
            value = convert(value)
            newvalue = value
            if ((feature not in self._prob_distribution_numeric_features_dict) and
                (all(isinstance(x,float) for x in self._features_and_unique_values_dict[feature]))):
                newvalue = closest_sampling_point(value, self._features_and_unique_values_dict[feature])
            new_features_and_values.append(feature + " = " + str(newvalue))
        features_and_values = new_features_and_values       
        if self._debug3: print("\nCL1 New feature and values: "+ str(features_and_values))
        answer = {class_name : None for class_name in self._class_names}
        answer['solution_path'] = []
        classification = self.recursive_descent_for_classification(root_node,features_and_values, answer)
        answer['solution_path'].reverse()
        if self._debug3: 
            print("\nCL2 The classification:")
            for class_name in self._class_names:
                print("    " + class_name + " with probability " + str(classification[class_name]))
        classification_for_display = {}
        for item in classification:
            if isinstance(classification[item], float):
                classification_for_display[item] = "%0.3f" % classification[item]
            else:
                classification_for_display[item] =  ["NODE" + str(x) for x in classification[item]]
        return classification_for_display

    def recursive_descent_for_classification(self, node, feature_and_values, answer):
        children = node.get_children()
        if len(children) == 0:
            leaf_node_class_probabilities = node.get_class_probabilities()
            for i in range(len(self._class_names)):
                answer[self._class_names[i]] = leaf_node_class_probabilities[i]
            answer['solution_path'].append(node.get_serial_num())
            return answer
        feature_tested_at_node = node.get_feature()
        if self._debug3: print("\nCLRD1 Feature tested at node for classification: " + feature_tested_at_node)
        value_for_feature = None
        path_found = None
        pattern = r'(\S+)\s*=\s*(\S+)'
        feature,value = None,None
        for feature_and_value in feature_and_values:
            m = re.search(pattern, feature_and_value)
            feature,value = m.group(1),m.group(2)
            if feature == feature_tested_at_node:
                value_for_feature = convert(value)
        # added in 3.2.0:
        if value_for_feature is None:
            leaf_node_class_probabilities = node.get_class_probabilities()
            for i in range(len(self._class_names)):
                answer[self._class_names[i]] = leaf_node_class_probabilities[i]
            answer['solution_path'].append(node.get_serial_num())
            return answer
        if feature_tested_at_node in self._prob_distribution_numeric_features_dict:
            if self._debug3: print( "\nCLRD2 In the truly numeric section")
            for child in children:
                branch_features_and_values = child.get_branch_features_and_values_or_thresholds()
                last_feature_and_value_on_branch = branch_features_and_values[-1] 
                pattern1 = r'(.+)<(.+)'
                pattern2 = r'(.+)>(.+)'
                if re.search(pattern1, last_feature_and_value_on_branch):
                    m = re.search(pattern1, last_feature_and_value_on_branch)
                    feature,threshold = m.group(1),m.group(2)
                    if value_for_feature <= float(threshold):
                        path_found = True
                        answer = self.recursive_descent_for_classification(child, feature_and_values, answer)
                        answer['solution_path'].append(node.get_serial_num())
                        break
                if re.search(pattern2, last_feature_and_value_on_branch):
                    m = re.search(pattern2, last_feature_and_value_on_branch)
                    feature,threshold = m.group(1),m.group(2)
                    if value_for_feature > float(threshold):
                        path_found = True
                        answer = self.recursive_descent_for_classification(child, feature_and_values, answer)
                        answer['solution_path'].append(node.get_serial_num())
                        break
            if path_found: return answer 
        else:
            feature_value_combo = "".join([feature_tested_at_node,"=", str(convert(value_for_feature))])
            if self._debug3: 
                print("\nCLRD3 In the symbolic section with feature_value_combo: " + feature_value_combo)
            for child in children:
                branch_features_and_values = child.get_branch_features_and_values_or_thresholds()
                if self._debug3: print("\nCLRD4 branch features and values: "+str(branch_features_and_values))
                last_feature_and_value_on_branch = branch_features_and_values[-1] 
                if last_feature_and_value_on_branch == feature_value_combo:
                    answer = self.recursive_descent_for_classification(child, feature_and_values, answer)
                    answer['solution_path'].append(node.get_serial_num())
                    path_found = True
                    break
            if path_found: return answer
        if not path_found:
            leaf_node_class_probabilities = node.get_class_probabilities()
            for i in range(0, len(self._class_names)):
                answer[self._class_names[i]] = leaf_node_class_probabilities[i]
            answer['solution_path'].append(node.get_serial_num())
        return answer

    def classify_by_asking_questions(self, root_node):
        '''
        If you want classification to be carried out by engaging a human user in a
        question-answer session, this is the method to use for that purpose.  See the
        script classify_by_asking_questions.py in the Examples subdirectory for an
        illustration of how to do that.
        '''
        answer = {class_name : None for class_name in self._class_names}
        answer['solution_path'] = []
        scratchpad_for_numeric_answers = {feature : None 
                                   for feature in self._prob_distribution_numeric_features_dict}
        classification = self.interactive_recursive_descent_for_classification(root_node, 
                                                                   answer, scratchpad_for_numeric_answers)
        classification['solution_path'].reverse()
        classification_for_display = {}
        for item in classification:
            if isinstance(classification[item], float):
                classification_for_display[item] = "%0.3f" % classification[item]
            else:
                classification_for_display[item] =  ["NODE" + str(x) for x in classification[item]]
        return classification_for_display

    def interactive_recursive_descent_for_classification(self, node, answer, scratchpad_for_numerics):
        pattern1 = r'(.+)<(.+)'
        pattern2 = r'(.+)>(.+)'
        user_value_for_feature = None
        children = node.get_children()
        if len(children) == 0:
            leaf_node_class_probabilities = node.get_class_probabilities()
            for i in range(len(self._class_names)):
                answer[self._class_names[i]] = leaf_node_class_probabilities[i]
            answer['solution_path'].append(node.get_serial_num())
            return answer
        list_of_branch_attributes_to_children = []
        for child in children:
            branch_features_and_values = child.get_branch_features_and_values_or_thresholds()
            feature_and_value_on_branch = branch_features_and_values[-1] 
            list_of_branch_attributes_to_children.append(feature_and_value_on_branch)
        feature_tested_at_node = node.get_feature()
        feature_value_combo = None
        path_found = None
        if feature_tested_at_node in self._prob_distribution_numeric_features_dict:
            if scratchpad_for_numerics[feature_tested_at_node]:
                user_value_for_feature = scratchpad_for_numerics[feature_tested_at_node]
            else:
                valuerange =  self._numeric_features_valuerange_dict[feature_tested_at_node]
                while True: 
                    if sys.version_info[0] == 3:
                        user_value_for_feature = input( "\nWhat is the value for the feature '" + 
                       feature_tested_at_node + "'?" + "\n" +  
                       "Enter a value in the range: " + str(valuerange) + " => " )
                    else:
                        user_value_for_feature = raw_input( "\nWhat is the value for the feature '" + 
                       feature_tested_at_node + "'?" + "\n" +    
                       "Enter a value in the range: " + str(valuerange) + " => " )
                    user_value_for_feature = convert(user_value_for_feature.strip())
                    answer_found = 0
                    if valuerange[0] <= user_value_for_feature <= valuerange[1]:
                        answer_found = 1
                        break    
                    if answer_found == 1: break
                    print("You entered illegal value. Let's try again")
                scratchpad_for_numerics[feature_tested_at_node] = user_value_for_feature
            for i in range(len(list_of_branch_attributes_to_children)):
                branch_attribute = list_of_branch_attributes_to_children[i]
                if re.search(pattern1, branch_attribute):
                    m = re.search(pattern1, branch_attribute)
                    feature,threshold = m.group(1),m.group(2)
                    if user_value_for_feature <= float(threshold):
                        answer = self.interactive_recursive_descent_for_classification(children[i], 
                                                                          answer, scratchpad_for_numerics)
                        path_found = True
                        answer['solution_path'].append(node.get_serial_num())
                        break
                if re.search(pattern2, branch_attribute):
                    m = re.search(pattern2, branch_attribute)
                    feature,threshold = m.group(1),m.group(2)
                    if user_value_for_feature > float(threshold):
                        answer = self.interactive_recursive_descent_for_classification(children[i], 
                                                                          answer, scratchpad_for_numerics)
                        answer['solution_path'].append(node.get_serial_num())
                        break
            if path_found: return answer
        else:
            possible_values_for_feature = self._features_and_unique_values_dict[feature_tested_at_node]
            while True:
                if sys.version_info[0] == 3:
                    user_value_for_feature = \
                       input( "\nWhat is the value for the feature '" + feature_tested_at_node + 
                              "'?" + "\n" + "Enter one of: " + str(possible_values_for_feature) + " => " )
                else:
                    user_value_for_feature = \
                       raw_input( "\nWhat is the value for the feature '" + feature_tested_at_node + 
                                  "'?" + "\n" + "Enter one of: " + str(possible_values_for_feature) + " => " )
                user_value_for_feature = convert(user_value_for_feature.strip())
                answer_found = 0
                for value in possible_values_for_feature:
                    if value == user_value_for_feature: 
                        answer_found = 1
                        break
                if answer_found == 1: break
                print("You entered illegal value. Let's try again")
            feature_value_combo = "".join([feature_tested_at_node,"=",str(user_value_for_feature)])
            for i in range(len(list_of_branch_attributes_to_children)):
                branch_attribute = list_of_branch_attributes_to_children[i]
                if branch_attribute == feature_value_combo:
                    answer = self.interactive_recursive_descent_for_classification(children[i], 
                                                                          answer, scratchpad_for_numerics)
                    path_found = True
                    answer['solution_path'].append(node.get_serial_num())
                    break
            if path_found: return answer    
        if not path_found:
            leaf_node_class_probabilities = node.get_class_probabilities()
            for i in range(0, len(self._class_names)):
                answer[self._class_names[i]] = leaf_node_class_probabilities[i]
            answer['solution_path'].append(node.get_serial_num())

        return answer        

##-------------------------------  Construct Decision Tree  ------------------------------------

    def construct_decision_tree_classifier(self):
        '''
        Construct the root node object and set its entropy value as derived from the priors
        associated with the different classes.
        '''        
        print("\nConstructing a decision tree...")
        if self._debug3:        
            self.determine_data_condition() 
            print("\nStarting construction of the decision tree:\n") 
        class_probabilities = list(map(lambda x: self.prior_probability_for_class(x), self._class_names))
        if self._debug3:         
            print("\nPrior class probabilities: " + str(class_probabilities))
            print("\nClass names: " + str(self._class_names))
        entropy = self.class_entropy_on_priors()
        if self._debug3: print("\nClass entropy on priors: "+ str(entropy))
#        root_node = self.DTNode(None, entropy, class_probabilities, [], self, 'root')
        root_node = DTNode(None, entropy, class_probabilities, [], self, 'root')
        root_node.set_class_names(self._class_names)
        self._root_node = root_node
        self.recursive_descent(root_node)
        return root_node        

    def recursive_descent(self, node):
        '''
        After the root node of the decision tree is constructed by the previous method, we
        find  at that node the feature that yields the greatest reduction in class entropy
        from the entropy based on just the class priors. The logic for finding this
        feature is different for symbolic features and for numeric features (that logic is
        built into the best feature calculator). We then invoke this method recursively to 
        create the rest of the tree.  
        '''
        if self._debug3:
            print("\n==================== ENTERING RECURSIVE DESCENT ==========================")
        node_serial_number = node.get_serial_num()
        features_and_values_or_thresholds_on_branch = node.get_branch_features_and_values_or_thresholds()
        existing_node_entropy = node.get_node_entropy()
        if self._debug3: 
            print("\nRD1 NODE SERIAL NUMBER: "+ str(node_serial_number))
            print("\nRD2 Existing Node Entropy: " + str(existing_node_entropy))
            print("\nRD3 features_and_values_or_thresholds_on_branch: " + 
                                                        str(features_and_values_or_thresholds_on_branch))
            class_probs = node.get_class_probabilities()
            print("\nRD4 Class probabilities: " + str(class_probs))
        if existing_node_entropy < self._entropy_threshold: 
            if self._debug3: print("\nRD5 returning because existing node entropy is below threshold")
            return
        copy_of_path_attributes = deep_copy_array(features_and_values_or_thresholds_on_branch)
        best_feature,best_feature_entropy,best_feature_val_entropies,decision_val = \
                       self.best_feature_calculator(copy_of_path_attributes, existing_node_entropy)
        node.set_feature(best_feature)
        if self._debug3: node.display_node() 
        if self._max_depth_desired is not None and \
         len(features_and_values_or_thresholds_on_branch) >= self._max_depth_desired:
            if self._debug3: print("\nRD6 REACHED LEAF NODE AT MAXIMUM DEPTH ALLOWED")
            return
        if best_feature is None: return
        if self._debug3:
            print("\nRD7 Existing entropy at node: " + str(existing_node_entropy))
            print("\nRD8 Calculated best feature is %s and its value %s" % (best_feature, decision_val))
            print("\nRD9 Best feature entropy: "+ str(best_feature_entropy))
            print("\nRD10 Calculated entropies for different values of best feature: %s" % str(best_feature_val_entropies))
        entropy_gain = existing_node_entropy - best_feature_entropy
        if self._debug3: print("\nRD11 Expected entropy gain at this node: " + str(entropy_gain))
        if entropy_gain > self._entropy_threshold:
            if best_feature in self._numeric_features_valuerange_dict and \
                         self._feature_values_how_many_uniques_dict[best_feature] > \
                                             self._symbolic_to_numeric_cardinality_threshold:
                best_threshold = decision_val                 # as returned by best feature calculator
                best_entropy_for_less, best_entropy_for_greater = best_feature_val_entropies
                extended_branch_features_and_values_or_thresholds_for_lessthan_child = \
                                  deep_copy_array(features_and_values_or_thresholds_on_branch)
                extended_branch_features_and_values_or_thresholds_for_greaterthan_child  = \
                                  deep_copy_array(features_and_values_or_thresholds_on_branch)
                feature_threshold_combo_for_less_than = \
                                            "".join([best_feature,"<",str(convert(best_threshold))])
                feature_threshold_combo_for_greater_than = \
                                            "".join([best_feature,">",str(convert(best_threshold))])
                extended_branch_features_and_values_or_thresholds_for_lessthan_child.append( 
                                                              feature_threshold_combo_for_less_than)
                extended_branch_features_and_values_or_thresholds_for_greaterthan_child.append( 
                                                           feature_threshold_combo_for_greater_than)
                if self._debug3:
                    print("\nRD12 extended_branch_features_and_values_or_thresholds_for_lessthan_child: "+ 
                             str(extended_branch_features_and_values_or_thresholds_for_lessthan_child))
                    print("\nRD13 extended_branch_features_and_values_or_thresholds_for_greaterthan_child: " +
                             str(extended_branch_features_and_values_or_thresholds_for_greaterthan_child))
                class_probabilities_for_lessthan_child_node = list(map(lambda x: 
                       self.probability_of_a_class_given_sequence_of_features_and_values_or_thresholds(\
                      x, extended_branch_features_and_values_or_thresholds_for_lessthan_child), self._class_names))
                class_probabilities_for_greaterthan_child_node = list(map(lambda x: 
                   self.probability_of_a_class_given_sequence_of_features_and_values_or_thresholds(
                      x, extended_branch_features_and_values_or_thresholds_for_greaterthan_child), 
                                                           self._class_names))
                if self._debug3:
                    print("\nRD14 class entropy for going down lessthan child: " + str(best_entropy_for_less))
                    print("\nRD15 class_entropy_for_going_down_greaterthan_child: " + 
                                                                        str(best_entropy_for_greater))
                if best_entropy_for_less < existing_node_entropy - self._entropy_threshold:
#                    left_child_node = self.DTNode(None, best_entropy_for_less, 
                    left_child_node = DTNode(None, best_entropy_for_less, 
                                      class_probabilities_for_lessthan_child_node,
                                         extended_branch_features_and_values_or_thresholds_for_lessthan_child, self)
                    node.add_child_link(left_child_node)
                    self.recursive_descent(left_child_node)
                if best_entropy_for_greater < existing_node_entropy - self._entropy_threshold:
#                    right_child_node = self.DTNode(None, best_entropy_for_greater,
                    right_child_node = DTNode(None, best_entropy_for_greater,
                                      class_probabilities_for_greaterthan_child_node, 
                                      extended_branch_features_and_values_or_thresholds_for_greaterthan_child, self)
                    node.add_child_link(right_child_node)
                    self.recursive_descent(right_child_node)
            else:
                if self._debug3:
                    print("\nRD16 RECURSIVE DESCENT: In section for symbolic features for creating children")
                values_for_feature = self._features_and_unique_values_dict[best_feature]
                if self._debug3:
                    print("\nRD17 Values for feature %s are %s" % (best_feature, str(values_for_feature)))
                feature_value_combos = \
                  map(lambda x: "".join([best_feature,"=",x]), map(str, map(convert, values_for_feature)))
                feature_value_combos = sorted(feature_value_combos)
                class_entropies_for_children = []
                for feature_and_value_index in range(len(feature_value_combos)):
                    if self._debug3:
                        print("\nRD18 Creating a child node for: " + 
                                                       str(feature_value_combos[feature_and_value_index])) 
                    extended_branch_features_and_values_or_thresholds = None
                    if features_and_values_or_thresholds_on_branch is None:
                        extended_branch_features_and_values_or_thresholds = \
                                                 [feature_value_combos[feature_and_value_index]]
                    else:
                        extended_branch_features_and_values_or_thresholds = \
                            deep_copy_array(features_and_values_or_thresholds_on_branch)
                        extended_branch_features_and_values_or_thresholds.append(
                                                 feature_value_combos[feature_and_value_index])
                    class_probabilities = list(map(lambda x: 
                       self.probability_of_a_class_given_sequence_of_features_and_values_or_thresholds(
                                 x, extended_branch_features_and_values_or_thresholds), self._class_names))
                    class_entropy_for_child = \
                          self.class_entropy_for_a_given_sequence_of_features_and_values_or_thresholds(
                                                        extended_branch_features_and_values_or_thresholds)
                    if self._debug3: 
                        print("\nRD19 branch attributes: "+ 
                                                 str(extended_branch_features_and_values_or_thresholds))
                        print("\nRD20 class entropy for child: " + str(class_entropy_for_child)) 
                    if existing_node_entropy - class_entropy_for_child > self._entropy_threshold:
#                        child_node = self.DTNode(None, class_entropy_for_child, 
                        child_node = DTNode(None, class_entropy_for_child, 
                             class_probabilities, extended_branch_features_and_values_or_thresholds, self)
                        node.add_child_link( child_node )
                        self.recursive_descent(child_node)
                    elif self._debug3: print("\nRD21 This child will NOT result in a node")
        else:
            if self._debug3:
                print("\nRD22 REACHED LEAF NODE NATURALLY for: " + 
                                      str(features_and_values_or_thresholds_on_branch))
            return   

    def best_feature_calculator(self, features_and_values_or_thresholds_on_branch, existing_node_entropy):
        '''
        This is the heart of the decision tree constructor.  Its main job is to figure
        out the best feature to use for partitioning the training data samples that
        correspond to the current node.  The search for the best feature is carried
        out differently for symbolic features and for numeric features.  For a
        symbolic feature, the method estimates the entropy for each value of the
        feature and then averages out these entropies as a measure of the
        discriminatory power of that features.  For a numeric feature, on the other
        hand, it estimates the entropy reduction that can be achieved if we were to
        partition the set of training samples at each possible threshold for that
        numeric feature.  For a numeric feature, all possible sampling points
        relevant to the node in question are considered as candidates for thresholds.
        '''
        pattern1 = r'(.+)=(.+)'
        pattern2 = r'(.+)<(.+)'
        pattern3 = r'(.+)>(.+)'
        all_symbolic_features = []
        for feature_name in self._feature_names:
            if feature_name not in self._prob_distribution_numeric_features_dict:
                all_symbolic_features.append(feature_name)
        symbolic_features_already_used = []
        for feature_and_value_or_threshold in features_and_values_or_thresholds_on_branch:
            if re.search(pattern1, feature_and_value_or_threshold):
                m = re.search(pattern1, feature_and_value_or_threshold)
                feature = m.group(1)
                symbolic_features_already_used.append(feature)
        symbolic_features_not_yet_used = [x for x in all_symbolic_features if x not in symbolic_features_already_used]
        true_numeric_types = []        
        symbolic_types = []
        true_numeric_types_feature_names = []
        symbolic_types_feature_names = []
        for item in features_and_values_or_thresholds_on_branch:
            if re.search(pattern2, item):
                true_numeric_types.append(item)
                m = re.search(pattern2, item)
                feature,value = m.group(1),m.group(2)
                true_numeric_types_feature_names.append(feature)
            elif re.search(pattern3, item): 
                true_numeric_types.append(item)
                m = re.search(pattern3, item)
                feature,value = m.group(1),m.group(2)
                true_numeric_types_feature_names.append(feature)
            else:
                symbolic_types.append(item) 
                m = re.search(pattern1, item)
                feature,value = m.group(1),m.group(2)
                symbolic_types_feature_names.append(feature)
        true_numeric_types_feature_names = list(set(true_numeric_types_feature_names))
        symbolic_types_feature_names = list(set(symbolic_types_feature_names))
        bounded_intervals_numeric_types = self.find_bounded_intervals_for_numeric_features(true_numeric_types)
        # Calculate the upper and the lower bounds to be used when searching for the best
        # threshold for each of the numeric features that are in play at the current node:
        upperbound = {feature : None for feature in true_numeric_types_feature_names}
        lowerbound = {feature : None for feature in true_numeric_types_feature_names}
        for item in bounded_intervals_numeric_types:
            if item[1] == '>':
                lowerbound[item[0]] = float(item[2])
            else:
                upperbound[item[0]] = float(item[2])
        entropy_values_for_different_features = {}
        partitioning_point_child_entropies_dict = {feature : {} for feature in self._feature_names}
        partitioning_point_threshold = {feature : None for feature in self._feature_names}
        entropies_for_different_values_of_symbolic_feature ={feature : [] for feature in self._feature_names}
        for i in range(len(self._feature_names)):
            feature_name = self._feature_names[i]
            if self._debug3: 
                print("\n\nBFC1          FEATURE BEING CONSIDERED: " + feature_name)
            if feature_name in symbolic_features_already_used:
                continue
            elif feature_name in self._numeric_features_valuerange_dict and \
                              self._feature_values_how_many_uniques_dict[feature_name] > \
                                                  self._symbolic_to_numeric_cardinality_threshold:
                values = self._sampling_points_for_numeric_feature_dict[feature_name]
                if self._debug3: print("\nBFC2 values for %s are %s                " % (feature_name, values))
                newvalues = []
                if feature_name in true_numeric_types_feature_names:
                    if upperbound[feature_name] is not None and lowerbound[feature_name] is not None and \
                                                 lowerbound[feature_name] >= upperbound[feature_name]:
                        continue
                    elif upperbound[feature_name] is not None and lowerbound[feature_name] is not None and \
                                                       lowerbound[feature_name] < upperbound[feature_name]:
                        newvalues = [x for x in values \
                                             if lowerbound[feature_name] < x <= upperbound[feature_name]]
                    elif upperbound[feature_name] is not None:
                        newvalues = [x for x in values if x <= upperbound[feature_name]]
                    elif lowerbound[feature_name] is not None:
                        newvalues = [x for x in values if x > lowerbound[feature_name]]
                    else:
                        raise Exception("Error in bound specifications in best feature calculator")
                else:
                    newvalues = values
                if len(newvalues) == 0:
                    continue
                partitioning_entropies = []
                for value in newvalues:
                    feature_and_less_than_value_string = "".join([feature_name,"<",str(convert(value))]) 
                    feature_and_greater_than_value_string = "".join([feature_name,">",str(convert(value))])
                    for_left_child = for_right_child = None
                    if features_and_values_or_thresholds_on_branch:
                        for_left_child = deep_copy_array(features_and_values_or_thresholds_on_branch)
                        for_left_child.append(feature_and_less_than_value_string)
                        for_right_child = deep_copy_array(features_and_values_or_thresholds_on_branch)
                        for_right_child.append(feature_and_greater_than_value_string)
                    else:
                        for_left_child = [feature_and_less_than_value_string]
                        for_right_child = [feature_and_greater_than_value_string]
                    entropy1 = self.class_entropy_for_less_than_threshold_for_feature(\
                                        features_and_values_or_thresholds_on_branch,feature_name,value)
                    entropy2 = self.class_entropy_for_greater_than_threshold_for_feature(\
                                        features_and_values_or_thresholds_on_branch,feature_name, value)
                    partitioning_entropy = entropy1 * \
                       self.probability_of_a_sequence_of_features_and_values_or_thresholds(for_left_child) \
                           + entropy2 * \
                       self.probability_of_a_sequence_of_features_and_values_or_thresholds(for_right_child)
                    partitioning_entropies.append(partitioning_entropy)
                    partitioning_point_child_entropies_dict[feature_name][value] = (entropy1, entropy2)
                min_entropy,best_partition_point_index = minimum(partitioning_entropies)         
                if min_entropy < existing_node_entropy:
                    partitioning_point_threshold[feature_name] = newvalues[best_partition_point_index]
                    entropy_values_for_different_features[feature_name] = min_entropy
            else:          
                if self._debug3:
                    print("\nBFC3 Best feature calculator: Entering section reserved for symbolic features")
                    print("\nBFC4 Feature name: " + feature_name)
                values =  self._features_and_unique_values_dict[feature_name]
                values = sorted(list(set(filter(lambda x: x != 'NA', values))))
                if self._debug3: print("\nBFC5 values for feature %s are %s" % (feature_name, values))
                entropy = 0
                for value in values:                              
                    feature_value_string = feature_name + "=" + str(convert(value))
                    if self._debug3: print("\nBFC6 feature_value_string: " + feature_value_string)
                    extended_attributes = deep_copy_array(features_and_values_or_thresholds_on_branch)
                    if features_and_values_or_thresholds_on_branch:
                        extended_attributes.append(feature_value_string)
                    else:
                        extended_attributes = [feature_value_string]
                    entropy += \
         self.class_entropy_for_a_given_sequence_of_features_and_values_or_thresholds(extended_attributes) * \
         self.probability_of_a_sequence_of_features_and_values_or_thresholds(extended_attributes)
                    if self._debug3:
                        print("\nBFC7 Entropy calculated for symbolic feature value choice (%s, %s) is %s" % \
                                                                   (feature_name,value,entropy))       
                    entropies_for_different_values_of_symbolic_feature[feature_name].append(entropy)
                if entropy < existing_node_entropy:
                    entropy_values_for_different_features[feature_name] = entropy
        min_entropy_for_best_feature = None
        best_feature_name = None
        # sorting introduced in 3.2.0:
        for feature_nom in sorted(entropy_values_for_different_features):
            if not best_feature_name:
                best_feature_name = feature_nom
                min_entropy_for_best_feature = entropy_values_for_different_features[feature_nom]
            else:
                if entropy_values_for_different_features[feature_nom] < min_entropy_for_best_feature:
                    best_feature_name = feature_nom                    
                    min_entropy_for_best_feature = entropy_values_for_different_features[feature_nom]
        if  best_feature_name in partitioning_point_threshold:
            threshold_for_best_feature = partitioning_point_threshold[best_feature_name]
        else:
            threshold_for_best_feature = None
        best_feature_entropy = min_entropy_for_best_feature
        val_based_entropies_to_be_returned = None
        decision_val_to_be_returned = None
        if best_feature_name in self._numeric_features_valuerange_dict and \
                              self._feature_values_how_many_uniques_dict[best_feature_name] > \
                                                    self._symbolic_to_numeric_cardinality_threshold:
            val_based_entropies_to_be_returned = \
                 partitioning_point_child_entropies_dict[best_feature_name][threshold_for_best_feature]
        else:
            val_based_entropies_to_be_returned = None
        if  best_feature_name in partitioning_point_threshold:
            decision_val_to_be_returned = partitioning_point_threshold[best_feature_name]
        else:
            decision_val_to_be_returned = None
        if self._debug3:
            print("\nBFC8 Val based entropies to be returned for feature %s are %s" % 
                             (best_feature_name, str(val_based_entropies_to_be_returned)))
        return best_feature_name, best_feature_entropy, val_based_entropies_to_be_returned, decision_val_to_be_returned

#-----------------------------------  Entropy Calculators  ------------------------------------

    def class_entropy_on_priors(self):
        if 'priors' in self._entropy_cache:
            return self._entropy_cache['priors']
        entropy = None
        for class_name in self._class_names:
            prob = self.prior_probability_for_class(class_name)
            if (prob >= 0.0001) and (prob <= 0.999):
                log_prob = math.log(prob,2)
            if prob < 0.0001:
                log_prob = 0 
            if prob > 0.999:
                log_prob = 0 
            if entropy is None:
                entropy = -1.0 * prob * log_prob
                continue
            entropy += -1.0 * prob * log_prob
        if abs(entropy) < 0.0000001: entropy = 0.0
        self._entropy_cache['priors'] = entropy
        return entropy

    def entropy_scanner_for_a_numeric_feature(self, feature):
        all_sampling_points = self._sampling_points_for_numeric_feature_dict[feature]
        entropies_for_less_than_thresholds = []
        entropies_for_greater_than_thresholds = []
        for point in  all_sampling_points:
            entropies_for_less_than_thresholds.append(
                    self.class_entropy_for_less_than_threshold_for_feature([], feature, point))
            entropies_for_greater_than_thresholds.append(
                    self.class_entropy_for_greater_than_threshold_for_feature([], feature, point))
        print("\nSCANNER: All entropies less than thresholds for feature %s are: %s" % 
                                (feature, entropies_for_less_than_thresholds))
        print("\nSCANNER: All entropies greater than thresholds for feature %s are: %s" % 
                                (feature, entropies_for_greater_than_thresholds))

    def class_entropy_for_less_than_threshold_for_feature(self, \
                array_of_features_and_values_or_thresholds, feature, threshold):
        threshold = convert(threshold)
        feature_threshold_combo = feature + '<' + str(threshold)
        sequence = ":".join(array_of_features_and_values_or_thresholds) + ":" + feature_threshold_combo
        if sequence in self._entropy_cache:
            return self._entropy_cache[sequence]
        copy_of_array_of_features_and_values_or_thresholds = \
                        deep_copy_array(array_of_features_and_values_or_thresholds)
        copy_of_array_of_features_and_values_or_thresholds.append(feature_threshold_combo)
        entropy = None
        for class_name in self._class_names:
            prob = self.probability_of_a_class_given_sequence_of_features_and_values_or_thresholds(\
                                        class_name, copy_of_array_of_features_and_values_or_thresholds)
            if (prob >= 0.0001) and (prob <= 0.999):
                log_prob = math.log(prob,2)
            if prob < 0.0001:
                log_prob = 0 
            if prob > 0.999:
                log_prob = 0 
            if entropy is None:
                entropy = -1.0 * prob * log_prob
                continue
            entropy += -1.0 * prob * log_prob
        self._entropy_cache[sequence] = entropy
        if abs(entropy) < 0.0000001: entropy = 0.0
        return entropy

    def class_entropy_for_greater_than_threshold_for_feature(self, 
                    array_of_features_and_values_or_thresholds, feature, threshold):
        threshold = convert(threshold)
        feature_threshold_combo = feature + '>' + str(threshold)
        sequence = ":".join(array_of_features_and_values_or_thresholds) + ":" + feature_threshold_combo
        if sequence in self._entropy_cache:
            return self._entropy_cache[sequence]
        copy_of_array_of_features_and_values_or_thresholds = \
                               deep_copy_array(array_of_features_and_values_or_thresholds)
        copy_of_array_of_features_and_values_or_thresholds.append(feature_threshold_combo)
        entropy = None
        for class_name in self._class_names:
            prob = self.probability_of_a_class_given_sequence_of_features_and_values_or_thresholds(\
                                        class_name, copy_of_array_of_features_and_values_or_thresholds)
            if (prob >= 0.0001) and (prob <= 0.999):
                log_prob = math.log(prob,2)
            if prob < 0.0001:
                log_prob = 0 
            if prob > 0.999:
                log_prob = 0 
            if entropy is None:
                entropy = -1.0 * prob * log_prob
                continue
            entropy += -1.0 * prob * log_prob
        if abs(entropy) < 0.0000001: entropy = 0.0
        self._entropy_cache[sequence] = entropy
        return entropy

    def class_entropy_for_a_given_sequence_of_features_and_values_or_thresholds(self, 
                                             array_of_features_and_values_or_thresholds):
        sequence = ":".join(array_of_features_and_values_or_thresholds)
        if sequence in self._entropy_cache:
            return self._entropy_cache[sequence]
        entropy = None    
        for class_name in self._class_names:
            prob = self.probability_of_a_class_given_sequence_of_features_and_values_or_thresholds(\
                                                class_name, array_of_features_and_values_or_thresholds)
            if (prob >= 0.0001) and (prob <= 0.999):
                log_prob = math.log(prob,2)
            if prob < 0.0001:
                log_prob = 0 
            if prob > 0.999:
                log_prob = 0 
            if entropy is None:
                entropy = -1.0 * prob * log_prob
                continue
            entropy += -1.0 * prob * log_prob
        if abs(entropy) < 0.0000001: entropy = 0.0
        self._entropy_cache[sequence] = entropy
        return entropy


#---------------------------------  Probability Calculators  ----------------------------------

    def prior_probability_for_class(self, class_name):
        class_name_in_cache = "".join(["prior::", class_name])
        if class_name_in_cache in self._probability_cache:
            return self._probability_cache[class_name_in_cache]
        total_num_of_samples = len( self._samples_class_label_dict )
        all_values = self._samples_class_label_dict.values()
        for this_class_name in self._class_names:
            trues = list(filter( lambda x: x == this_class_name, all_values ))
            prior_for_this_class = (1.0 * len(trues)) / total_num_of_samples
            self._class_priors_dict[this_class_name] = prior_for_this_class
            this_class_name_in_cache = "".join(["prior::", this_class_name])
            self._probability_cache[this_class_name_in_cache] = prior_for_this_class
        return self._probability_cache[class_name_in_cache]

    def calculate_class_priors(self):
        print("\nEstimating class priors...")
        if len(self._class_priors_dict) > 1: return
        for class_name in self._class_names:
            class_name_in_cache = "".join(["prior::", class_name])
            total_num_of_samples = len( self._samples_class_label_dict )
            all_values = self._samples_class_label_dict.values()
            trues = list(filter( lambda x: x == class_name, all_values ))
            prior_for_this_class = (1.0 * len(trues)) / total_num_of_samples
            self._class_priors_dict[class_name] = prior_for_this_class
            this_class_name_in_cache = "".join(["prior::", class_name])
            self._probability_cache[this_class_name_in_cache] = prior_for_this_class
        if self._debug2: print(str(self._class_priors_dict))

    def probability_of_feature_value(self, feature_name, value):
        value = convert(value)
        if (value is not None) and (feature_name in self._sampling_points_for_numeric_feature_dict):
            value = closest_sampling_point(convert(value), self._sampling_points_for_numeric_feature_dict[feature_name])
        if value is not None:
            feature_and_value = "".join([feature_name, "=", str(convert(value))])
        if (value is not None) and (feature_and_value in self._probability_cache):
            return self._probability_cache[feature_and_value]
        histogram_delta = num_of_hist_bins = valuerange = diffrange = None


        print(f'ALL FAVD for feature {feature_name}')
        sorte = sorted(self._features_and_values_dict[feature_name])
        for f in sorte:
            print(f)

        if feature_name in self._numeric_features_valuerange_dict:
            if self._feature_values_how_many_uniques_dict[feature_name] > self._symbolic_to_numeric_cardinality_threshold:
                if feature_name not in self._sampling_points_for_numeric_feature_dict:
                    print(f'inside triple if')
                    valuerange = self._numeric_features_valuerange_dict[feature_name] 
                    diffrange = valuerange[1] - valuerange[0]
                    unique_values_for_feature = sorted(list(set(filter(lambda x: x != 'NA', 
                                   self._features_and_values_dict[feature_name]))))
                

                    diffs = sorted([unique_values_for_feature[i] - unique_values_for_feature[i-1] 
                                            for i in range(1,len(unique_values_for_feature))])

                    print(f'diffs for feature {feature_name}')
                    for d in diffs:
                        print("DIFF: ", d)
                    median_diff = diffs[int(len(diffs)/2) - 1]
                    histogram_delta =  median_diff * 2
                    print("MEDIAN DIFF: ", median_diff)
                    if histogram_delta < diffrange / 500:  
                        if self._number_of_histogram_bins:
                            histogram_delta = diffrange / self._number_of_histogram_bins
                        else:
                            histogram_delta = diffrange / 500; 
                    self._histogram_delta_dict[feature_name] = histogram_delta
                    num_of_histogram_bins = int(diffrange / histogram_delta) + 1
                    self._num_of_histogram_bins_dict[feature_name] = num_of_histogram_bins
                    sampling_points_for_feature = [valuerange[0] + histogram_delta * j for j in range(num_of_histogram_bins)]
                    self._sampling_points_for_numeric_feature_dict[feature_name] = sampling_points_for_feature
        if feature_name in self._numeric_features_valuerange_dict:
            if self._feature_values_how_many_uniques_dict[feature_name] > self._symbolic_to_numeric_cardinality_threshold:
                sampling_points_for_feature = self._sampling_points_for_numeric_feature_dict[feature_name]
                counts_at_sampling_points = [0] * len(sampling_points_for_feature)
                actual_values_for_feature = self._features_and_values_dict[feature_name]
                actual_values_for_feature = list(filter(lambda x: x != 'NA', actual_values_for_feature))
                for i in range(len(sampling_points_for_feature)):
                    for j in range(len(actual_values_for_feature)):
                        if abs(sampling_points_for_feature[i] - 
                           convert(actual_values_for_feature[j])) < (histogram_delta):
                            print("incrementing count for ", sampling_points_for_feature[i], "due to ", actual_values_for_feature[j])
                            counts_at_sampling_points[i] += 1
                total_counts =  functools.reduce(lambda x,y:x+y, counts_at_sampling_points)

                print("TOTAL COUNTS: ", total_counts)

                probs = [x / (1.0 * total_counts) for x in counts_at_sampling_points]
                sum_probs =  functools.reduce(lambda x,y:x+y, probs)
                bin_prob_dict = {sampling_points_for_feature[i] : probs[i] 
                               for i in range(len(sampling_points_for_feature))}
                self._prob_distribution_numeric_features_dict[feature_name] = bin_prob_dict
                values_for_feature = list(map(lambda x: feature_name + "=" + x, map(str, sampling_points_for_feature)))
                for i in range(0, len(values_for_feature)):
                    print(f'CACHE: {values_for_feature[i]}: {probs[i]}')
                    self._probability_cache[values_for_feature[i]] = probs[i]
                if (value is not None) and (feature_and_value in self._probability_cache):
                    return self._probability_cache[feature_and_value]
                else:
                    return 0
            else:
                # This section is for those numeric features treated symbolically:
                values_for_feature = list(set(self._features_and_values_dict[feature_name]))
                values_for_feature = list(filter(lambda x: x != 'NA', values_for_feature))
                values_for_feature = list(map(lambda x: "".join([feature_name,"=",x]), map(str,values_for_feature)))
                value_counts = [0] * len(values_for_feature)
                for sample in sorted(self._training_data_dict.keys(), key = lambda x: sample_index(x) ):
                    features_and_values = self._training_data_dict[sample]
                    for i in range(0, len(values_for_feature)):
                        for current_value in (features_and_values):
                            if values_for_feature[i] == current_value:
                                value_counts[i] += 1 
                total_counts = functools.reduce(lambda x,y:x+y, value_counts)
                if total_counts == 0:
                    raise Exception('''PFV Something is wrong with your training file. '''
                                    '''It contains no training samples for feature named %s ''' % feature_name)
                probs = [x / (1.0 * total_counts) for x in value_counts]
                for i in range(0, len(values_for_feature)):
                    self._probability_cache[values_for_feature[i]] = probs[i]
                if (value is not None) and (feature_and_value in self._probability_cache):
                    return self._probability_cache[feature_and_value]
                else:
                    return 0
        else:
            # This section is only for purely symbolic features:  
            values_for_feature = self._features_and_values_dict[feature_name]
            values_for_feature = list(map(lambda x: feature_name + "=" + x, values_for_feature))
            value_counts = [0] * len(values_for_feature)
            for sample in sorted(self._training_data_dict.keys(), key = lambda x: sample_index(x) ):
                features_and_values = self._training_data_dict[sample]
                for i in range(0, len(values_for_feature)):
                    for current_value in features_and_values:
                        if values_for_feature[i] == current_value:
                            value_counts[i] += 1 
            for i in range(0, len(values_for_feature)):
                self._probability_cache[values_for_feature[i]] = value_counts[i] / (1.0 * len(self._training_data_dict))
            if (value is not None) and (feature_and_value in self._probability_cache):
                return self._probability_cache[feature_and_value]
            else:
                return 0

    def probability_of_feature_value_given_class(self,feature_name,feature_value,class_name):
        feature_value = convert(feature_value)
        histogram_delta = num_of_histogram_bins = valuerange = diffrange = None
        if feature_name in self._sampling_points_for_numeric_feature_dict:
            feature_value = closest_sampling_point(convert(feature_value), \
                       self._sampling_points_for_numeric_feature_dict[feature_name])
        feature_value_class = "".join([feature_name,"=",str(feature_value),"::",class_name])
        if feature_value_class in self._probability_cache:
            return self._probability_cache[feature_value_class]
        if feature_name in self._numeric_features_valuerange_dict:
            if self._feature_values_how_many_uniques_dict[feature_name] > self._symbolic_to_numeric_cardinality_threshold:
                histogram_delta = self._histogram_delta_dict[feature_name]
                num_of_histogram_bins = self._num_of_histogram_bins_dict[feature_name]
                valuerange = self._numeric_features_valuerange_dict[feature_name]
                diffrange = valuerange[1] - valuerange[0]
        samples_for_class = []
        # Accumulate all samples names for the given class:
        for sample_name in self._samples_class_label_dict:
            if self._samples_class_label_dict[sample_name] == class_name:
                samples_for_class.append(sample_name) 
        if feature_name in self._numeric_features_valuerange_dict:
            if self._feature_values_how_many_uniques_dict[feature_name] > self._symbolic_to_numeric_cardinality_threshold:
                sampling_points_for_feature = self._sampling_points_for_numeric_feature_dict[feature_name]
                counts_at_sampling_points = [0] * len(sampling_points_for_feature)
                actual_feature_values_for_samples_in_class = []
                for sample in samples_for_class:
                    for feature_and_value in self._training_data_dict[sample]:
                        pattern = r'(.+)=(.+)'
                        m = re.search(pattern, feature_and_value)
                        feature,value = m.group(1),m.group(2)
                        if feature == feature_name and value != 'NA':
                            actual_feature_values_for_samples_in_class.append(convert(value))
                for i in range(len(sampling_points_for_feature)):
                    for j in range(len(actual_feature_values_for_samples_in_class)):
                        if abs(sampling_points_for_feature[i] - 
                     actual_feature_values_for_samples_in_class[j]) < histogram_delta:
                            counts_at_sampling_points[i] += 1
                total_counts =  functools.reduce(lambda x,y:x+y, counts_at_sampling_points)
                probs = [x / (1.0 * total_counts) for x in counts_at_sampling_points]
                if total_counts == 0:
                    raise Exception('''PFVC1 Something is wrong with your training file. It contains no training '''
                                    '''samples for Class %s and Feature %s''' % class_name, feature_name)
                values_for_feature_and_class = list(map(lambda x: feature_name + "=" + x + 
                                "::" + class_name, map(str, sampling_points_for_feature)))
                for i in range(0, len(values_for_feature_and_class)):
                    self._probability_cache[values_for_feature_and_class[i]] = probs[i]
                if feature_value_class in self._probability_cache:
                    return self._probability_cache[feature_value_class]
                else:
                    return 0
            else:
                # We now take care of numeric features with a small number of unique values
                values_for_feature = list(set(self._features_and_values_dict[feature_name]))
                values_for_feature = list(filter(lambda x: x != 'NA', values_for_feature))
                values_for_feature =list(map(lambda x: "".join([feature_name,"=",x]), map(str,map(convert, values_for_feature))))
                value_counts = [0] * len(values_for_feature)
                for sample in samples_for_class:
                    features_and_values = self._training_data_dict[sample]
                    for i in range(0, len(values_for_feature)):
                        for current_value in (features_and_values):
                            if values_for_feature[i] == current_value:
                                value_counts[i] += 1 
                total_count = functools.reduce(lambda x,y:x+y, value_counts)
                if total_count == 0:
                    raise Exception('''PFVC2 Something is wrong with your training file. It contains no training samples '''
                                    '''for Class %s and Feature %s''' % (class_name, feature_name))
                # We normalize by total_count because the probabilities are
                # conditioned on a given class
                for i in range(len(values_for_feature)):
                    feature_and_value_and_class =  values_for_feature[i] + "::" + class_name
                    self._probability_cache[feature_and_value_and_class] = value_counts[i] / (1.0 * total_count)
                if feature_value_class in self._probability_cache:
                    return self._probability_cache[feature_value_class]
                else:
                    return 0
        else:
            # This section is for purely symbolic features
            values_for_feature = list(set(self._features_and_values_dict[feature_name]))
            values_for_feature = \
                        list(map(lambda x: "".join([feature_name,"=",x]), map(str,values_for_feature)))
            value_counts = [0] * len(values_for_feature)
            for sample in samples_for_class:
                features_and_values = self._training_data_dict[sample]
                for i in range(len(values_for_feature)):
                    for current_value in (features_and_values):
                        if values_for_feature[i] == current_value:
                            value_counts[i] += 1 
            total_count = functools.reduce(lambda x,y:x+y, value_counts)
            if total_count == 0:
                raise Exception('''PFVC3 Something is wrong with your training file. It contains no training samples '''
                              '''for Class %s and Feature %s''' % (class_name, feature_name))
            # We normalize by total_count because the probabilities are
            # conditioned on a given class
            for i in range(0, len(values_for_feature)):
                feature_and_value_for_class = "".join([values_for_feature[i],"::",class_name])
                self._probability_cache[feature_and_value_for_class] = value_counts[i] / (1.0 * total_count)
            feature_and_value_and_class = "".join([feature_name,"=", feature_value,"::",class_name])
            if feature_and_value_and_class in self._probability_cache:
                return self._probability_cache[feature_and_value_and_class]
            else:
                return 0

    def probability_of_feature_less_than_threshold(self, feature_name, threshold):
        threshold = convert(threshold)
        feature_threshold_combo = feature_name + '<' + str(threshold)
        if feature_threshold_combo in self._probability_cache:
            return self._probability_cache[feature_threshold_combo]
        all_values = list(filter(lambda x: x != 'NA', self._features_and_values_dict[feature_name]))
        all_values_less_than_threshold = list(filter(lambda x: x <= threshold, all_values))
        probability = 1.0 * len(all_values_less_than_threshold) / len(all_values)
        self._probability_cache[feature_threshold_combo] = probability
        return probability

    def probability_of_feature_less_than_threshold_given_class(self, feature_name, threshold, class_name):
        threshold = convert(threshold)
        feature_threshold_class_combo = "".join([feature_name,'<',str(threshold),"::",class_name])
        if feature_threshold_class_combo in self._probability_cache:
            return self._probability_cache[feature_threshold_class_combo]
        data_samples_for_class = []
        # Accumulate all samples names for the given class:
        for sample_name in self._samples_class_label_dict:
            if self._samples_class_label_dict[sample_name] == class_name:
                data_samples_for_class.append(sample_name) 
        actual_feature_values_for_samples_in_class = []
        for sample in data_samples_for_class:
            for feature_and_value in self._training_data_dict[sample]:
                pattern = r'(.+)=(.+)'
                m = re.search(pattern, feature_and_value)
                feature,value = m.group(1),m.group(2)
                if feature == feature_name and value != 'NA':
                    actual_feature_values_for_samples_in_class.append(convert(value))
        actual_points_for_feature_less_than_threshold = list(filter(lambda x: x <= threshold, 
                                          actual_feature_values_for_samples_in_class))
        # The conditional in the assignment shown below introduced in Version 3.2.1:
        probability = ((1.0 * len(actual_points_for_feature_less_than_threshold)) / len(actual_feature_values_for_samples_in_class)) if len(actual_feature_values_for_samples_in_class) > 0 else 0.0
        self._probability_cache[feature_threshold_class_combo] = probability
        return probability

    def probability_of_a_sequence_of_features_and_values_or_thresholds(self,array_of_features_and_values_or_thresholds):
        '''
        This method requires that all truly numeric types only be expressed as '<' or '>'
        constructs in the array of branch features and thresholds
        '''
        if len(array_of_features_and_values_or_thresholds) == 0: return
        sequence = ":".join(array_of_features_and_values_or_thresholds)
        if sequence in self._probability_cache:
            return self._probability_cache[sequence]
        probability = None
        pattern1 = r'(.+)=(.+)'
        pattern2 = r'(.+)<(.+)'
        pattern3 = r'(.+)>(.+)'
        true_numeric_types = []        
        true_numeric_types_feature_names = []
        symbolic_types = []
        symbolic_types_feature_names = []
        for item in array_of_features_and_values_or_thresholds:
            if re.search(pattern2, item):
                true_numeric_types.append(item)
                m = re.search(pattern2, item)
                feature,value = m.group(1),m.group(2)
                true_numeric_types_feature_names.append(feature)
            elif re.search(pattern3, item): 
                true_numeric_types.append(item)
                m = re.search(pattern3, item)
                feature,value = m.group(1),m.group(2)
                true_numeric_types_feature_names.append(feature)
            else:
                symbolic_types.append(item) 
                m = re.search(pattern1, item)
                feature,value = m.group(1),m.group(2)
                symbolic_types_feature_names.append(feature)
        true_numeric_types_feature_names = list(set(true_numeric_types_feature_names))
        symbolic_types_feature_names = list(set(symbolic_types_feature_names))
        bounded_intervals_numeric_types = self.find_bounded_intervals_for_numeric_features(true_numeric_types)
        # Calculate the upper and the lower bounds to be used when searching for the best
        # threshold for each of the numeric features that are in play at the current node:
        upperbound = {feature : None for feature in true_numeric_types_feature_names}
        lowerbound = {feature : None for feature in true_numeric_types_feature_names}
        for item in bounded_intervals_numeric_types:
            if item[1] == '>':
                lowerbound[item[0]] = float(item[2])
            else:
                upperbound[item[0]] = float(item[2])
        for feature_name in true_numeric_types_feature_names:
            if lowerbound[feature_name] is not None and upperbound[feature_name] is not None \
                          and upperbound[feature_name] <= lowerbound[feature_name]:
                return 0
            elif lowerbound[feature_name] is not None and upperbound[feature_name] is not None:
                if not probability:
                    probability = self.probability_of_feature_less_than_threshold(feature_name, upperbound[feature_name]) - \
                          self.probability_of_feature_less_than_threshold(feature_name, lowerbound[feature_name])
                else:
                    probability *= (self.probability_of_feature_less_than_threshold(feature_name, upperbound[feature_name]) - \
                          self.probability_of_feature_less_than_threshold(feature_name, lowerbound[feature_name]))
            elif upperbound[feature_name] is not None and lowerbound[feature_name] is None:
                if not probability:
                    probability = self.probability_of_feature_less_than_threshold(feature_name, upperbound[feature_name])
                else:
                    probability *= self.probability_of_feature_less_than_threshold(feature_name, upperbound[feature_name])
            elif lowerbound[feature_name] is not None and upperbound[feature_name] is None:
                if not probability:
                    probability = 1.0 -self.probability_of_feature_less_than_threshold(feature_name, lowerbound[feature_name])
                else:
                    probability *= (1.0 - self.probability_of_feature_less_than_threshold(feature_name,
                                                                                          lowerbound[feature_name]))
            else:
                raise SyntaxError("Ill formatted call to 'probability_of_sequence' method")
        for feature_and_value in symbolic_types:
            if re.search(pattern1, feature_and_value):      
                m = re.search(pattern1, feature_and_value)
                feature,value = m.group(1),m.group(2)
                if not probability:
                    probability = self.probability_of_feature_value(feature, value)
                else:
                    probability *= self.probability_of_feature_value(feature, value)
        self._probability_cache[sequence] = probability
        return probability

    def probability_of_a_sequence_of_features_and_values_or_thresholds_given_class(self, 
                                              array_of_features_and_values_or_thresholds, class_name):
        '''
        This method requires that all truly numeric types only be expressed as '<' or '>'
        constructs in the array of branch features and thresholds
        '''
        if len(array_of_features_and_values_or_thresholds) == 0: return
        sequence = ":".join(array_of_features_and_values_or_thresholds)
        sequence_with_class = sequence + "::" + class_name
        if sequence_with_class in self._probability_cache:
            return self._probability_cache[sequence_with_class]
        probability = None
        pattern1 = r'(.+)=(.+)'
        pattern2 = r'(.+)<(.+)'
        pattern3 = r'(.+)>(.+)'
        true_numeric_types = []        
        true_numeric_types_feature_names = []        
        symbolic_types = []
        symbolic_types_feature_names = []
        for item in array_of_features_and_values_or_thresholds:
            if re.search(pattern2, item):
                true_numeric_types.append(item)
                m = re.search(pattern2, item)
                feature,value = m.group(1),m.group(2)
                true_numeric_types_feature_names.append(feature)
            elif re.search(pattern3, item): 
                true_numeric_types.append(item)
                m = re.search(pattern3, item)
                feature,value = m.group(1),m.group(2)
                true_numeric_types_feature_names.append(feature)
            else:
                symbolic_types.append(item) 
                m = re.search(pattern1, item)
                feature,value = m.group(1),m.group(2)
                symbolic_types_feature_names.append(feature)
        true_numeric_types_feature_names = list(set(true_numeric_types_feature_names))
        symbolic_types_feature_names = list(set(symbolic_types_feature_names))
        bounded_intervals_numeric_types = self.find_bounded_intervals_for_numeric_features(true_numeric_types)
        # Calculate the upper and the lower bounds to be used when searching for the best
        # threshold for each of the numeric features that are in play at the current node:
        upperbound = {feature : None for feature in true_numeric_types_feature_names}
        lowerbound = {feature : None for feature in true_numeric_types_feature_names}
        for item in bounded_intervals_numeric_types:
            if item[1] == '>':
                lowerbound[item[0]] = float(item[2])
            else:
                upperbound[item[0]] = float(item[2])
        for feature_name in true_numeric_types_feature_names:
            if lowerbound[feature_name] is not None and upperbound[feature_name] is not None \
                          and upperbound[feature_name] <= lowerbound[feature_name]:
                return 0
            elif lowerbound[feature_name] is not None and upperbound[feature_name] is not None:
                if not probability:
                    probability = self.probability_of_feature_less_than_threshold_given_class(feature_name, 
                                                upperbound[feature_name], class_name) - \
                          self.probability_of_feature_less_than_threshold_given_class(feature_name, 
                                                lowerbound[feature_name], class_name)
                else:
                    probability *= (self.probability_of_feature_less_than_threshold_given_class(feature_name, 
                                  upperbound[feature_name], class_name) - \
                          self.probability_of_feature_less_than_threshold_given_class(feature_name, 
                                                        lowerbound[feature_name], class_name))
            elif upperbound[feature_name] is not None and lowerbound[feature_name] is None:
                if not probability:
                    probability = self.probability_of_feature_less_than_threshold_given_class(feature_name, 
                                                         upperbound[feature_name], class_name)
                else:
                    probability *= self.probability_of_feature_less_than_threshold_given_class(feature_name, 
                                                         upperbound[feature_name], class_name)
            elif lowerbound[feature_name] is not None and upperbound[feature_name] is None:
                if not probability:
                    probability = 1.0 - \
                          self.probability_of_feature_less_than_threshold_given_class(feature_name, 
                                                         lowerbound[feature_name], class_name)
                else:
                    probability *= (1.0 - 
                          self.probability_of_feature_less_than_threshold_given_class(feature_name, 
                                                         lowerbound[feature_name], class_name))
            else:
                raise SyntaxError("Ill formatted call to 'probability of sequence with class' method")
        for feature_and_value in symbolic_types:
            if re.search(pattern1, feature_and_value):      
                m = re.search(pattern1, feature_and_value)
                feature,value = m.group(1),m.group(2)
                if not probability:
                    probability = self.probability_of_feature_value_given_class(feature, value, class_name)
                else:
                    probability *= self.probability_of_feature_value_given_class(feature, value, class_name)
        self._probability_cache[sequence_with_class] = probability
        return probability

    def probability_of_a_class_given_sequence_of_features_and_values_or_thresholds(self, 
                                  class_name, array_of_features_and_values_or_thresholds):
        sequence = ":".join(array_of_features_and_values_or_thresholds)
        class_and_sequence = "".join([class_name, "::", sequence])
        if class_and_sequence in self._probability_cache:
            return self._probability_cache[class_and_sequence]
        array_of_class_probabilities = [0] * len(self._class_names)
        for i in range(len(self._class_names)):
            class_name = self._class_names[i]
            prob = self.probability_of_a_sequence_of_features_and_values_or_thresholds_given_class(
                                     array_of_features_and_values_or_thresholds, class_name) 
            if prob < 0.000001:
                array_of_class_probabilities[i] = 0.0
                continue
            prob_of_feature_sequence = self.probability_of_a_sequence_of_features_and_values_or_thresholds(
                                                      array_of_features_and_values_or_thresholds)
            ##Commented out in 2.2.4
            #if not prob_of_feature_sequence: 
            #    sys.exit('''PCS Something is wrong with your sequence of feature values and thresholds in '''
            #       '''probability_of_a_class_given_sequence_of_features_and_values_or_thresholds()''')
            prior = self._class_priors_dict[self._class_names[i]] 
            if prob_of_feature_sequence: 
                array_of_class_probabilities[i] = prob * prior / prob_of_feature_sequence
            else:
                array_of_class_probabilities[i] = prior
        sum_probability = functools.reduce(lambda x,y:x+y, array_of_class_probabilities)
        if sum_probability == 0:
            array_of_class_probabilities = [1.0/len(self._class_names)] * len(self._class_names)
        else:
            array_of_class_probabilities = \
                             list(map(lambda x: x / sum_probability, array_of_class_probabilities))
        for i in range(len(self._class_names)):
            this_class_and_sequence = "".join([self._class_names[i], "::", sequence])
            self._probability_cache[this_class_and_sequence] = array_of_class_probabilities[i]
        return self._probability_cache[class_and_sequence]

    
#---------------------------------  Class Based Utilities  ------------------------------------

    def determine_data_condition(self):
        '''
        This method estimates the worst-case fan-out of the decision tree taking into
        account the number of values (and therefore the number of branches emanating
        from a node) for the symbolic features.
        '''
        num_of_features = len(self._feature_names)
        values = []
        for feature in self._features_and_unique_values_dict:  
            if feature not in self._numeric_features_valuerange_dict:
                values.append(self._features_and_unique_values_dict[feature])
        # return if no symbolic features found:
        if not values: return
        print("Number of features: " + str(num_of_features))
        max_num_values = max(list(map(len, values)))
        print("Largest number of values for symbolic features is: " + str(max_num_values))
        estimated_number_of_nodes = max_num_values ** num_of_features
        print('''\nWORST CASE SCENARIO: The decision tree COULD have as many as %s '''
              ''' nodes. The exact number of nodes created depends critically on '''
              '''the entropy_threshold used for node expansion (the default value '''
              '''for this threshold is 0.01) and on the value set for max_depth_desired '''
              '''for the depth of the tree\n''' % estimated_number_of_nodes)
        if estimated_number_of_nodes > 10000:
            print('''THIS IS WAY TOO MANY NODES. Consider using a relatively '''
                  '''large value for entropy_threshold and/or a small value for '''
                  '''for max_depth_desired to reduce the number of nodes created''')
            ans = None
            if sys.version_info[0] == 3:
                ans = input("\nDo you wish to continue? Enter 'y' if yes:  ")
            else:
                ans = raw_input("\nDo you wish to continue? Enter 'y' if yes:  ")
            ans = ans.strip()
            if ans != 'y':
                sys.exit(0)
 
    def _check_names_used(self, features_and_values_test_data):
        '''
        This method is used to verify that you used legal feature names in the test
        sample that you want to classify with the decision tree.
        '''
        for feature_and_value in features_and_values_test_data:
            pattern = r'(\S+)\s*=\s*(\S+)'
            m = re.search(pattern, feature_and_value)
            feature,value = m.group(1),m.group(2)
            if feature is None or value is None:
                raise Exception("Your test data has formatting error")
            if feature not in self._feature_names:
                return 0
        return 1

    def get_class_names(self):
        return self._class_names

    def find_bounded_intervals_for_numeric_features(self, arr):
        '''
        Given a list of branch attributes for the numeric features of the form, say,
        ['g2<1','g2<2','g2<3','age>34','age>36','age>37'], this method returns the
        smallest list that is relevant for the purpose of calculating the
        probabilities.  To explain, the probability that the feature `g2' is less
        than 1 AND, at the same time, less than 2, AND, at the same time, less than
        3, is the same as the probability that the feature less than 1. Similarly,
        the probability that 'age' is greater than 34 and also greater than 37 is the
        same as `age' being greater than 37.
        '''       
        features = self._feature_names
        arr1 = list(map(lambda x: re.split(r'(>|<)', x, 0), arr))
        # make a separate list for each feature name:                                    
        arr3 = list(filter(lambda x: len(x)>0, [list(filter(lambda x: x[0]==y, arr1)) for y in features]))
        # Sort each list so that '<' entries occur before '>' entries:                   
        arr4 = [sorted(li, key=lambda x: x[1]) for li in arr3]
        arr5 = [[list(filter(lambda x: x[1]==y, alist))] for alist in arr4 for y in ['<','>']]
        arr6 = []
        for i in range(len(arr5)):
            arr6 += [sorted(li, key=lambda x: float(x[2])) for li in arr5[i]]
        arr7 = list(itertools.chain(*arr6))
        arr8 = list(filter(lambda x: len(x)>0, [list(filter(lambda x: x[0]==y, arr7)) for y in features]))
        arr9 = []
        for alist in arr8:
            newalist = []
            if alist[0][1] == '<':
                newalist.append(alist[0])
            else:
                newalist.append(alist[-1])
            if alist[0][1] != alist[-1][1]:
                newalist.append(alist[-1])
            arr9.append(newalist)
        arr10 = list(itertools.chain(*arr9))
        return arr10


#-------------------------------------------  Class DTNode   ------------------------------------------

class DTNode(object):
    '''
    The nodes of a decision tree are instances of this class:
    '''
    def __init__(self, feature, entropy, class_probabilities, branch_features_and_values_or_thresholds, dt,
                          root_or_not = None):
        if root_or_not == 'root':
             dt.nodes_created = -1 
             dt.class_names = None
        self._dt = dt
        self._serial_number               = self.get_next_serial_num()
        self._feature                     = feature
        self._node_creation_entropy       = entropy
        self._class_probabilities = class_probabilities
        self._branch_features_and_values_or_thresholds = branch_features_and_values_or_thresholds
        self._linked_to = []

    def print_class_vars(self):
        # print the type of each class variable
        class_vars = self.__dict__.items()

        # Filter out non-class variables and print their types
        print(f'NEW OBJ')
        for var_name, value in class_vars:
            if not callable(value) and not var_name.startswith('__'):
                print(f"{var_name}: {type(value)}")
                if isinstance(value, list):
                    for i, item in enumerate(value):
                        print(f'  {var_name}[{i}]: {type(item)}')
                    # if empty list
                    if not value:
                        print(f'  empty list')
        print(f'END OBJ')

    def how_many_nodes(self):
        return self._dt.nodes_created + 1

    def set_class_names(self, class_names_list):
        self._dt.class_names = class_names_list

    def get_class_names(self):
        return self._dt.class_names

    def set_node_creation_entropy(self, entropy):
        self._node_creation_entropy = entropy

    def get_next_serial_num(self):
        self._dt.nodes_created += 1
        return self._dt.nodes_created

    def get_serial_num(self):
        return self._serial_number

    def get_feature(self):
        '''
        Returns the feature test at the current node
        '''
        return self._feature

    def set_feature(self, feature):
        self._feature = feature

    def get_node_entropy(self):
        return self._node_creation_entropy

    def get_class_probabilities(self):
        return self._class_probabilities

    def get_branch_features_and_values_or_thresholds(self):
        return self._branch_features_and_values_or_thresholds

    def get_children(self):
        return self._linked_to

    def add_child_link(self, new_node):
        self._linked_to.append(new_node)                  

    def delete_all_links(self):
        self._linked_to = None

    def display_node(self):
        feature_at_node = self.get_feature() or " "
        node_creation_entropy_at_node = self.get_node_entropy()
        print_node_creation_entropy_at_node = "%.3f" % node_creation_entropy_at_node
        class_probabilities = self.get_class_probabilities()
        class_probabilities_for_display = ["%0.3f" % x for x in class_probabilities]
        serial_num = self.get_serial_num()
        branch_features_and_values_or_thresholds = self.get_branch_features_and_values_or_thresholds()
        print("\n\nNODE " + str(serial_num) + 
              ":\n   Branch features and values to this node: " + 
              str(branch_features_and_values_or_thresholds) + 
              "\n   Class probabilities at current node: " + 
              str(class_probabilities_for_display) + 
              "\n   Entropy at current node: " + 
              print_node_creation_entropy_at_node + 
              "\n   Best feature test at current node: " + feature_at_node + "\n\n")

    def display_decision_tree(self, offset):
        serial_num = self.get_serial_num()
        if len(self.get_children()) > 0:
            feature_at_node = self.get_feature() or " "
            node_creation_entropy_at_node = self.get_node_entropy()
            print_node_creation_entropy_at_node = "%.3f" % node_creation_entropy_at_node
            branch_features_and_values_or_thresholds = self.get_branch_features_and_values_or_thresholds()
            class_probabilities = self.get_class_probabilities()
            print_class_probabilities = ["%.3f" % x for x in class_probabilities]
            print_class_probabilities_with_class = [self.get_class_names()[i] + " => " + 
                         print_class_probabilities[i] for i in range(len(self.get_class_names()))]
            print("NODE " + str(serial_num) + ":  " + offset +  "BRANCH TESTS TO NODE: " +
                                           str(branch_features_and_values_or_thresholds))
            second_line_offset = offset + " " * (8 + len(str(serial_num)))
            print(second_line_offset +  "Decision Feature: " + 
                  feature_at_node + "   Node Creation Entropy: " + print_node_creation_entropy_at_node +  
                  "   Class Probs: " + str(print_class_probabilities_with_class) + "\n")
            offset += "   "
            for child in self.get_children():
                child.display_decision_tree(offset)
        else:
            node_creation_entropy_at_node = self.get_node_entropy()
            print_node_creation_entropy_at_node = "%.3f" % node_creation_entropy_at_node
            branch_features_and_values_or_thresholds = self.get_branch_features_and_values_or_thresholds()
            class_probabilities = self.get_class_probabilities()
            print_class_probabilities = ["%.3f" % x for x in class_probabilities]
            print_class_probabilities_with_class = [self.get_class_names()[i] + " => " + 
                   print_class_probabilities[i] for i in range(len(self.get_class_names()))]
            print("NODE " + str(serial_num) + ":  " + offset +  "BRANCH TESTS TO LEAF NODE: " +
                                           str(branch_features_and_values_or_thresholds))
            second_line_offset = offset + " " * (8 + len(str(serial_num)))
            print(second_line_offset + "Node Creation Entropy: " + print_node_creation_entropy_at_node +  
                  "   Class Probs: " + str(print_class_probabilities_with_class) + "\n")

    

#-----------------------------  Evaluate Quality of Training Data  ----------------------------

class EvalTrainingData(DecisionTree):
    def __init__(self, *args, **kwargs ):
        DecisionTree.__init__(self, *args, **kwargs)

    def evaluate_training_data(self):
        evaldebug = 0
        if not self._training_datafile.endswith('.csv'):
            raise SyntaxError('''The data evaluation function in the module can only be used when your '''
                              '''training data is in a CSV file''')
        print('''\nWill run a 10-fold cross-validation test on your training data to test its '''
              '''class-discriminatory power:''')
        all_training_data = self._training_data_dict
        all_sample_names = sorted(all_training_data.keys(), key = lambda x: sample_index(x))
        fold_size = int(0.1 * len(all_training_data))
        confusion_matrix = {class_name : {class_name : 0 for class_name in self._class_names} \
                                                                  for class_name in self._class_names}
        
        print("training data dict")
        print(self._features_and_values_dict)
        print(self._feature_names)

        for fold_index in range(10):
            print("\nStarting the iteration indexed %d of the 10-fold cross-validation test" % fold_index)
            testing_samples = all_sample_names[fold_size * fold_index : fold_size * (fold_index+1)]
            training_samples = all_sample_names[0 : fold_size * fold_index] + \
                                                      all_sample_names[fold_size * (fold_index+1):] 
            testing_data = { x : all_training_data[x] for x in testing_samples }
            training_data = { x : all_training_data[x] for x in training_samples }
            
            print("length of training data")
            print(len(training_data))

            trainingDT = DecisionTree('evalmode')
            trainingDT._training_data_dict = training_data
            trainingDT._class_names = self._class_names
            trainingDT._feature_names = self._feature_names
            trainingDT._entropy_threshold = self._entropy_threshold
            trainingDT._max_depth_desired = self._max_depth_desired
            trainingDT._symbolic_to_numeric_cardinality_threshold =  self._symbolic_to_numeric_cardinality_threshold
            trainingDT._samples_class_label_dict = {sample_name : self._samples_class_label_dict[sample_name] 
                                                           for sample_name in training_samples}
            trainingDT._features_and_values_dict = {feature : [] for feature in self._features_and_values_dict}
            pattern = r'(\S+)\s*=\s*(\S+)'        
        
            
            for item in sorted(trainingDT._training_data_dict.items(), key = lambda x: sample_index(x[0])):
                for feature_and_value in item[1]:
                    m = re.search(pattern, feature_and_value)
                    feature,value = m.group(1),m.group(2)
                    if value != 'NA':
                        if feature == "g2": print("we shall append to feature: " + feature + " the value: " + value)
                        trainingDT._features_and_values_dict[feature].append(convert(value))

            print(f'len of class vs training dt: {len(self._features_and_values_dict["g2"])} vs {len(trainingDT._features_and_values_dict["g2"])}')
            for item in sorted(trainingDT._features_and_values_dict['g2'], key = lambda x: x):
                print(f'item: {item}')

            trainingDT._features_and_unique_values_dict = {feature : 
                                      sorted(list(set(trainingDT._features_and_values_dict[feature]))) for 
                                                            feature in trainingDT._features_and_values_dict}
            trainingDT._numeric_features_valuerange_dict = {feature : [] 
                                                         for feature in self._numeric_features_valuerange_dict}
            trainingDT._numeric_features_valuerange_dict = {feature : 
                                      [min(trainingDT._features_and_unique_values_dict[feature]), 
                                       max(trainingDT._features_and_unique_values_dict[feature])] 
                                                         for feature in self._numeric_features_valuerange_dict}
            
            
            
            if evaldebug:
                print("\n\nprinting samples in the testing set: " + str(testing_samples))            
                print("\n\nPrinting features and their values in the training set:\n")
                for item in sorted(trainingDT._features_and_values_dict.items()):
                    print(item[0]  + "  =>  "  + str(item[1]))
                print("\n\nPrinting unique values for features:\n")
                for item in sorted(trainingDT._features_and_unique_values_dict.items()):
                    print(item[0]  + "  =>  "  + str(item[1]))
                print("\n\nPrinting unique value ranges for features:\n")
                for item in sorted(trainingDT._numeric_features_valuerange_dict.items()):
                    print(item[0]  + "  =>  "  + str(item[1]))
            trainingDT._feature_values_how_many_uniques_dict = {feature : [] 
                                                    for  feature in self._features_and_unique_values_dict}
            trainingDT._feature_values_how_many_uniques_dict = {feature : 
                                     len(trainingDT._features_and_unique_values_dict[feature]) 
                                                    for  feature in self._features_and_unique_values_dict}
            if evaldebug: trainingDT._debug2 = 1

            # prob cache shape
            print("probability cache shape")
            print(len(trainingDT._probability_cache))

            print(f'FEATSHAPE: {len(trainingDT._features_and_values_dict["g2"])}')

            for val in trainingDT._features_and_values_dict["g2"]:
                print(f'VAL: {val}')



            trainingDT.calculate_first_order_probabilities()
            trainingDT.calculate_class_priors()
            # #root_node = trainingDT.construct_decision_tree_classifier()


            # we now wish to display all results
            print("\n\nResults of the 10-fold cross-validation test for run indexed %s" % str(fold_index + 1))

            print("feature values how many uniques dict")
            print(trainingDT._feature_values_how_many_uniques_dict)
            print("numeric features value range dict")
            print(trainingDT._numeric_features_valuerange_dict)
            print("features and unique values dict")
            print(trainingDT._features_and_unique_values_dict)
            print("features and values dict")
            print(trainingDT._features_and_values_dict)
            print(f'shapes: uniques: {len(trainingDT._feature_values_how_many_uniques_dict)}, numeric features: {len(trainingDT._numeric_features_valuerange_dict)}, values: {len(trainingDT._features_and_unique_values_dict)}')

            # also print first order probabilities and class priors
            print("probability cache shape")
            print(len(trainingDT._probability_cache))
            print("Class priors")
            print(trainingDT._class_priors_dict)

            if evaldebug:
                root_node.display_decision_tree("     ")
            for test_sample_name in testing_samples:
                test_sample_data = all_training_data[test_sample_name]
                if evaldebug: 
                    print("original data in test sample:", str(test_sample_data))  
                test_sample_data = [x for x in test_sample_data if not x.endswith('=NA')]
                if evaldebug: 
                    print("data in test sample:", str(test_sample_data))  
                classification = trainingDT.classify(root_node, test_sample_data)
                solution_path = classification['solution_path']                                  
                del classification['solution_path']                                              
                which_classes = list( classification.keys() )                                    
                which_classes = sorted(which_classes, key=lambda x: classification[x], reverse=True)
                most_likely_class_label = which_classes[0]
                if evaldebug:
                    print("\nClassification:\n")                                                     
                    print("     "  + str.ljust("class name", 30) + "probability")                    
                    print("     ----------                    -----------")                          
                    for which_class in which_classes:                                                
                        if which_class is not 'solution_path':                                       
                            print("     "  + str.ljust(which_class, 30) +  str(classification[which_class])) 
                    print("\nSolution path in the decision tree: " + str(solution_path))             
                    print("\nNumber of nodes created: " + str(root_node.how_many_nodes()))
                true_class_label_for_test_sample = self._samples_class_label_dict[test_sample_name]
                if evaldebug: 
                    print("%s:   true_class: %s    estimated_class: %s\n" % \
                             (test_sample_name, true_class_label_for_test_sample, most_likely_class_label))
                confusion_matrix[true_class_label_for_test_sample][most_likely_class_label] += 1    
        
        print("\n\n       DISPLAYING THE CONFUSION MATRIX FOR THE 10-FOLD CROSS-VALIDATION TEST:\n")
        matrix_header = " " * 30
        for class_name in self._class_names:  
            matrix_header += '{:^30}'.format(class_name)
        print("\n" + matrix_header + "\n")
        for row_class_name in sorted(confusion_matrix.keys()):
            row_display = str.rjust(row_class_name, 30)
            for col_class_name in sorted(confusion_matrix[row_class_name].keys()):
                row_display += '{:^30}'.format(str(confusion_matrix[row_class_name][col_class_name]) )
            print(row_display + "\n")
        diagonal_sum, off_diagonal_sum = 0,0
        for row_class_name in sorted(confusion_matrix.keys()):
            for col_class_name in sorted(confusion_matrix[row_class_name].keys()):
                if row_class_name == col_class_name:
                    diagonal_sum += confusion_matrix[row_class_name][col_class_name]
                else:
                    off_diagonal_sum += confusion_matrix[row_class_name][col_class_name]
        data_quality_index = 100.0 * diagonal_sum / (diagonal_sum + off_diagonal_sum)
        print("\nTraining Data Quality Index: %s   (out of a possible maximum of 100)" % data_quality_index)
        if data_quality_index <= 80:
            print( '''\nYour training data does not possess much class discriminatory '''
                   '''information.  It could be that the classes are inherently not well '''
                   '''separable or that your constructor parameter choices are not appropriate.''')
        elif 80 < data_quality_index <= 90:
            print( '''\nYour training data possesses some class discriminatory information '''
                   '''but it may not be sufficient for real-world applications.  You might '''
                   '''try tweaking the constructor parameters to see if that improves the '''
                   '''class discriminations.''')
        elif 90 < data_quality_index <= 95:
            print( '''\nYour training data appears to possess good class discriminatory '''
                   '''information.  Whether or not it is acceptable would depend on your '''
                   '''application.''')
        elif 95 < data_quality_index < 98:        
            print( '''\nYour training data is of very high quality.''')
        else:
            print('''\nYour training data is excellent.''')


#------------------------------  Generate Your Own Numeric Training Data  -----------------------------

class TrainingDataGeneratorNumeric(object):
    '''
    See the example script generate_training_data_numeric.py on how to use this class
    for generating your numeric training data.  The training data is generator in
    accordance with the specifications you place in a parameter file.
    '''
    def __init__(self, *args, **kwargs ):
        if args:
            raise SyntaxError('''TrainingDataGeneratorNumeric can only be called with keyword arguments '''
                              '''for the following keywords: output_csv_file, parameter_file, '''
                              '''number_of_samples_per_class, and debug''') 
        allowed_keys = 'output_csv_file','parameter_file','number_of_samples_per_class','debug'
        keywords_used = kwargs.keys()
        for keyword in keywords_used:
            if keyword not in allowed_keys:
                raise SyntaxError("Wrong keyword used --- check spelling") 
        output_csv_file = parameter_file = number_of_samples_per_class = debug = None
        if 'output_csv_file' in kwargs : output_csv_file = kwargs.pop('output_csv_file')
        if 'parameter_file' in kwargs : parameter_file = kwargs.pop('parameter_file')
        if 'number_of_samples_per_class' in kwargs:
            number_of_samples_per_class = kwargs.pop('number_of_samples_per_class')
        if 'debug' in kwargs:  debug = kwargs.pop('debug')
        if output_csv_file:
            self._output_csv_file = output_csv_file
        else:
            raise SyntaxError('''You must specify the name for a csv file for the training data''')
        if parameter_file: 
            self._parameter_file =  parameter_file
        else:
            raise SyntaxError('''You must specify a parameter file''')
        if number_of_samples_per_class:
            self._number_of_samples_per_class = number_of_samples_per_class
        else:
            raise SyntaxError('''You forgot to specify the number of training samples needed per class''')
        if debug:
            self._debug = debug
        else:
            self._debug = 0
        self._class_names                 = []
        self._features_ordered =            []
        self._class_names_and_priors      = {}
        self._features_with_value_range   = {}
        self._classes_and_their_param_values = {}

    def read_parameter_file_numeric( self ):
        '''
        The training data generated by an instance of the class
        TrainingDataGeneratorNumeric is based on the specs you place in a parameter
        that you supply to the class constructor through a constructor variable
        called `parameter_file.  This method is for parsing the parameter file in
        order to order to determine the names to be used for the different data
        classes, their means, and their variances.
        '''
        class_names = []
        class_names_and_priors = {}
        features_with_value_range = {}
        classes_and_their_param_values = {}
#        regex8 =  '[+-]?\ *(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?';
        FILE = open(self._parameter_file)
        params = FILE.read()
        regex = r'class names: ([\w\s+]+)\W*class priors: ([\d.\s+]+)'
        classes = re.search(regex, params, re.DOTALL | re.IGNORECASE)
        if (classes != None):
            class_names = list( filter( None, classes.group(1).strip().split(' ') ) )
            class_priors = list( filter( None, classes.group(2).strip().split(' ') ) )
        class_names_and_priors = {class_names[i] : float(class_priors[i]) for i in range(len(class_names))}
        if self._debug: print("\nClass names and priors: " + str(class_names_and_priors))
        regex = r'feature name: \w*.*?value range: [\d\. -]+'
        features = re.findall(regex, params, re.DOTALL | re.IGNORECASE)
        regex = r'feature name: (\w+)\W*?value range:\s*([\d. -]+)'
        features_ordered = []
        for feature in features:
            feature_groups = re.match(regex, feature, re.IGNORECASE)
            feature_name = feature_groups.group(1)
            features_ordered.append(feature_name)
            value_range = feature_groups.group(2).split()
            value_range = [float(value_range[0]), float(value_range[2])]
            features_with_value_range[feature_name] = value_range
        if self._debug: print("\nFeatures and their value ranges: "+ str(features_with_value_range))
        classes_and_their_param_values = {class_names[i] : {} for i in range(len(class_names))}
        regex = r'params for class:\s+\w*?\W+?mean:[\d\.\s+]+\W*?covariance:\W+?(?:[\s+\d.]+\W+?)+'
        class_params = re.findall(regex, params, re.DOTALL | re.IGNORECASE)
        regex = r'params for class:\s+(\w+)\W*?mean:\s*([\d. -]+)\W*covariance:\s*([\s\d.]+)'
        for class_param in class_params:
            class_params_groups = re.match(regex, class_param, re.IGNORECASE)
            class_name = class_params_groups.group(1)
            class_mean = class_params_groups.group(2).split()
            class_mean = list(map(float, class_mean))
            classes_and_their_param_values[class_name]['mean'] =  class_mean
            vector_size = len(class_mean)
            class_param_string = class_params_groups.group(3)
            covar_rows = filter(None, \
                  list(map(lambda x: x.strip().split(), class_param_string.splitlines())))
            covar_matrix = []
            for row in covar_rows:
                row = list(map(float, row))
                covar_matrix.append(row)
            classes_and_their_param_values[class_name]['covariance'] =  covar_matrix
        if self._debug: print("\nThe class parameters are: "+ str(classes_and_their_param_values))
        self._class_names                 = class_names
        self._class_names_and_priors      = class_names_and_priors
        self._features_with_value_range   = features_with_value_range
        self._classes_and_their_param_values = classes_and_their_param_values
        self._features_ordered = features_ordered

    def gen_numeric_training_data_and_write_to_csv(self):
        '''
        After the parameter file is parsed by the previous method, this method calls
        on `numpy.random.multivariate_normal()' to generate the training data
        samples. Your training data can be of any number of of dimensions, can have
        any mean, and any covariance.
        '''
        import numpy
        import random
        samples_for_class = {class_name : [] for class_name in self._class_names}
        for class_name in self._classes_and_their_param_values:
            mean = self._classes_and_their_param_values[class_name]['mean']
            covariance = self._classes_and_their_param_values[class_name]['covariance']
            samples = numpy.random.multivariate_normal(mean, covariance, self._number_of_samples_per_class)
            samples = [map(float, map(lambda x: "%.3f" % x, list_of_samples)) for list_of_samples in samples]
            samples_for_class[class_name] = samples
        data_records = []     
        for class_name in samples_for_class:
            for sample_index in range(self._number_of_samples_per_class):
                data_record = class_name + "," + \
                             ','.join(map(lambda x: str(x), samples_for_class[class_name][sample_index]))
                if self._debug: print("data record: " + data_record)
                data_records.append(data_record)
        random.shuffle(data_records)
        sample_records = []
        sample_records.append('""' + ',' + 'class_name'  + ',' + ','.join(self._features_ordered) + "\n")
        for i in range(len(data_records)):
            i1 = i+1
            sample_records.append(str(i1) + ',' + data_records[i] +"\n")
        FILE = open(self._output_csv_file, 'w') 
        list(map(FILE.write, sample_records))
        FILE.close()    

#------------------------  Generate Your Own Symbolic Training Data  --------------------------------

class TrainingDataGeneratorSymbolic(object):
    '''
    See the sample script generate_training_data_symbolic.py for how to use this
    class for generating symbolic training data.  The data is generated according to
    the specifications you place in a parameter file.
    '''
    def __init__(self, *args, **kwargs ):
        if args:
            raise SyntaxError('''TrainingDataGeneratorSymbolic can only be called with keyword arguments for the '''
                             '''following keywords: output_datafile, parameter_file, number_of_training_samples, '''
                             '''write_to_file, debug1, and debug2''') 
        allowed_keys = 'output_datafile','parameter_file','number_of_training_samples', 'write_to_file','debug1','debug2'
        keywords_used = kwargs.keys()
        for keyword in keywords_used:
            if keyword not in allowed_keys:
                raise SyntaxError("Wrong keyword used --- check spelling") 
        output_datafile = parameter_file = number_of_training_samples = None
        write_to_file = debug1 = debug2 = None
        if 'output_datafile' in kwargs: output_datafile = kwargs.pop('output_datafile')
        if 'parameter_file' in kwargs:  parameter_file = kwargs.pop('parameter_file')
        if 'number_of_training_samples' in kwargs: 
            number_of_training_samples = kwargs.pop('number_of_training_samples')
        if 'write_to_file' in kwargs : write_to_file = kwargs.pop('write_to_file')
        if 'debug1' in kwargs:  debug1 = kwargs.pop('debug1')
        if 'debug2' in kwargs:  debug2 = kwargs.pop('debug2')
        if output_datafile:
            self._output_datafile = output_datafile
        else:
            raise SyntaxError('''You must specify an output datafile''')
        if parameter_file: 
            self._parameter_file =  parameter_file
        else:
            raise SyntaxError('''You must specify a parameter file''')
        if number_of_training_samples:
            self._number_of_training_samples = number_of_training_samples
        else:
            raise SyntaxError('''You forgot to specify the number of training samples needed''')
        if write_to_file:
            self._write_to_file = write_to_file
        else:
            self._write_to_file = 0          
        if debug1:
            self._debug1 = debug1
        else:
            self._debug1 = 0
        if debug2:
            self._debug2 = debug2
        else:
            self._debug2 = 0
        self._training_sample_records     = {}
        self._features_and_values_dict    = {}
        self._bias_dict                   = {}
        self._class_names                 = []
        self._class_priors                = []


    def read_parameter_file_symbolic( self ):
        '''
        Read a parameter file for generating symbolic training data. See the script
        generate_training_data_symbolic.py in the Examples directory for how to pass
        the name of the parameter file to the constructor of the
        TrainingDataGeneratorSymbolic class.
        '''
        debug1 = self._debug1
        debug2 = self._debug2
        write_to_file = self._write_to_file
        number_of_training_samples = self._number_of_training_samples
        input_parameter_file = self._parameter_file
        all_params = []
        param_string = ''
        FILE = open(input_parameter_file, 'r')
        all_params = FILE.read()
        all_params = re.split(r'\n', all_params)
        FILE.close()
        pattern = r'^(?![ ]*#)'
        try:
            regex = re.compile( pattern )
        except:
            print("error in your pattern")
            raise
        all_params = list( filter( regex.search, all_params ) )
        all_params = list( filter( None, all_params ) )
        all_params = [x.rstrip('\n') for x in all_params]
        param_string = ' '.join( all_params )
        pattern = '^\s*class names:(.*?)\s*class priors:(.*?)(feature: .*)'
        m = re.search( pattern, param_string )
        rest_params = m.group(3)
        self._class_names = list( filter(None, re.split(r'\s+', m.group(1))) )
        self._class_priors = list( filter(None, re.split(r'\s+', m.group(2))) )
        print(f'  PRINTING REST PARAMS {rest_params}')
        pattern = r'(feature:.*?) (bias:.*)'
        m = re.search( pattern, rest_params  )
        feature_string = m.group(1)
        bias_string = m.group(2)
        features_and_values_dict = {}
        features = list( filter( None, re.split( r'(feature[:])', feature_string ) ) )
        for item in features:
            if re.match(r'feature', item): continue
            splits = list( filter(None, re.split(r' ', item)) )
            for i in range(0, len(splits)):
                if i == 0: features_and_values_dict[splits[0]] = []
                else:
                    if re.match( r'values', splits[i] ): continue
                    features_and_values_dict[splits[0]].append( splits[i] )
        self._features_and_values_dict = features_and_values_dict
        bias_dict = {}
        biases = list( filter(None, re.split(r'(bias[:]\s*class[:])', bias_string )) )
        for item in biases:
            if re.match(r'bias', item): continue
            splits = list( filter(None, re.split(r' ', item)) )
            feature_name = ''
            for i in range(0, len(splits)):
                if i == 0:
                    bias_dict[splits[0]] = {}
                elif ( re.search( r'(^.+)[:]$', splits[i] ) ):
                    m = re.search(  r'(^.+)[:]$', splits[i] )
                    feature_name = m.group(1)
                    bias_dict[splits[0]][feature_name] = []
                else:
                    if not feature_name: continue
                    bias_dict[splits[0]][feature_name].append( splits[i] )
        self._bias_dict = bias_dict
        if self._debug1:
            print("\n\n") 
            print("Class names: " + str(self._class_names))
            print("\n") 
            num_of_classes = len(self._class_names)
            print("Number of classes: " + str(num_of_classes))
            print("\n")
            print("Class priors: " + str(self._class_priors))
            print("\n\n")
            print("Here are the features and their possible values")
            print("\n")
            items = self._features_and_values_dict.items()
            for item in items:
                print(item[0] + " ===> " + str(item[1]))
            print("\n")
            print("Here is the biasing for each class:")
            print("\n")          
            items = self._bias_dict.items()
            for item in items:
                print("\n")
                print(item[0])
                items2 = list( item[1].items() )
                for i in range(0, len(items2)):
                    print( items2[i])

    def gen_symbolic_training_data( self ):
        '''
        This method generates training data according to the specifications
        placed in a parameter file that is read by the previous method.
        '''
        class_names = self._class_names
        class_priors = self._class_priors
        training_sample_records = {}
        features_and_values_dict = self._features_and_values_dict
        bias_dict  = self._bias_dict
        how_many_training_samples = self._number_of_training_samples
        class_priors_to_unit_interval_map = {}
        accumulated_interval = 0
        for i in range(0, len(class_names)):
            class_priors_to_unit_interval_map[class_names[i]] = \
            (accumulated_interval, accumulated_interval+float(class_priors[i]))
            accumulated_interval += float(class_priors[i])
        if self._debug1:
            print("Mapping of class priors to unit interval:")
            print("\n")
            items = class_priors_to_unit_interval_map.items()
            for item in items:
                print(item[0] + " ===> " + str(item[1]))
        class_and_feature_based_value_priors_to_unit_interval_map = {}
        for class_name  in class_names:
            class_and_feature_based_value_priors_to_unit_interval_map[class_name] = {}
            for feature in features_and_values_dict:
                class_and_feature_based_value_priors_to_unit_interval_map[class_name][feature] = {}
        for class_name  in class_names:
            for feature in features_and_values_dict:
                values = features_and_values_dict[feature]
                if len(bias_dict[class_name][feature]) > 0:
                    bias_string = bias_dict[class_name][feature][0]
                else:
                    no_bias = 1.0 / len(values)
                    bias_string = values[0] +  "=" + str(no_bias)
                value_priors_to_unit_interval_map = {}
                splits = list( filter( None, re.split(r'\s*=\s*', bias_string) ) )
                chosen_for_bias_value = splits[0]
                chosen_bias = splits[1]
                remaining_bias = 1 - float(chosen_bias)
                remaining_portion_bias = remaining_bias / (len(values) -1)
                accumulated = 0;
                for i in range(0, len(values)):
                    if (values[i] == chosen_for_bias_value):
                        value_priors_to_unit_interval_map[values[i]] = \
                          [accumulated, accumulated + float(chosen_bias)]
                        accumulated += float(chosen_bias)
                    else:
                        value_priors_to_unit_interval_map[values[i]] = \
                          [accumulated, accumulated + remaining_portion_bias]
                        accumulated += remaining_portion_bias
                class_and_feature_based_value_priors_to_unit_interval_map[class_name][feature] = \
                                                                   value_priors_to_unit_interval_map
                if self._debug2:
                    print("\n")
                    print( "For class " + class_name + \
                       ": Mapping feature value priors for feature '" + \
                       feature + "' to unit interval: ")
                    print("\n")
                    items = value_priors_to_unit_interval_map.items()
                    for item in items:
                        print("    " + item[0] + " ===> " + str(item[1]))
        ele_index = 0
        while (ele_index < how_many_training_samples):
            sample_name = "sample" + "_" + str(ele_index)
            training_sample_records[sample_name] = []
            # Generate class label for this training sample:                
            import random
            ran = random.Random()
            roll_the_dice  = ran.randint(0,1000) / 1000.0
            class_label = ''
            for class_name  in class_priors_to_unit_interval_map:
                v = class_priors_to_unit_interval_map[class_name]
                if ( (roll_the_dice >= v[0]) and (roll_the_dice <= v[1]) ):
                    training_sample_records[sample_name].append( 
                                             "class=" + class_name )
                    class_label = class_name
                    break
            for feature in sorted(list(features_and_values_dict.keys())):
                roll_the_dice  = ran.randint(0,1000) / 1000.0
                value_label = ''
                value_priors_to_unit_interval_map = \
                  class_and_feature_based_value_priors_to_unit_interval_map[class_label][feature]
                for value_name in value_priors_to_unit_interval_map:
                    v = value_priors_to_unit_interval_map[value_name]
                    if ( (roll_the_dice >= v[0]) and (roll_the_dice <= v[1]) ):
                        training_sample_records[sample_name].append( \
                                            feature + "=" + value_name )
                        value_label = value_name;
                        break
            ele_index += 1
        self._training_sample_records = training_sample_records
        if self._debug2:
            print("\n\n")
            print("TERMINAL DISPLAY OF TRAINING RECORDS:")
            print("\n\n")
            sample_names = training_sample_records.keys()
            sample_names = sorted( sample_names, key=lambda x: int(x.lstrip('sample_')) )
            for sample_name in sample_names:
                print(sample_name + " = " + str(training_sample_records[sample_name]))

    def write_training_data_to_file( self ):
        features_and_values_dict = self._features_and_values_dict
        class_names = self._class_names
        output_file = self._output_datafile
        training_sample_records = self._training_sample_records
        FILE = open(self._output_datafile, 'w') 
        features = list( features_and_values_dict.keys() )
        features.sort()
        title_string = ',class'
        for feature_name in features:
            title_string += ',' + str(feature_name)
        FILE.write(title_string + "\n")
        sample_names = list( training_sample_records.keys() )
        sample_names = sorted( sample_names, key=lambda x: int(x.lstrip('sample_')) )
        record_string = ''
        for sample_name in sample_names:
            sample_name_string =  str(sample_index(sample_name))
            record_string += sample_name_string + ','
            record = training_sample_records[sample_name]
            item_parts_dict = {}
            for item in record:
                splits = list( filter(None, re.split(r'=', item)) )
                item_parts_dict[splits[0]] = splits[1]
            record_string += item_parts_dict["class"]
            del item_parts_dict["class"]
            kees = list(item_parts_dict.keys())
            kees.sort()
            for kee in kees:
                record_string += "," + item_parts_dict[kee]
            FILE.write(record_string + "\n")
            record_string = ''
        FILE.close()


#----------------------------------  Class DTIntrospection   ----------------------------------

class DTIntrospection(object):
    '''Instances constructed from this class can provide explanations for the
    classification decisions at the nodes of a decision tree.  

    When used in the interactive mode, the decision-tree introspection made possible
    by this class provides answers to the following three questions: (1) List of the
    training samples that fall in the portion of the feature space that corresponds
    to a node of the decision tree; (2) The probabilities associated with the last
    feature test that led to the node; and (3) The class probabilities predicated on
    just the last feature test on the path to that node.

    CAVEAT: It is possible for a node to exist even when there are no training
    samples in the portion of the feature space that corresponds to the node.  That
    is because a decision tree is based on the probability densities estimated from
    the training data. When training data is non-uniformly distributed, it is
    possible for the probability associated with a point in the feature space to be
    non-zero even when there are no training samples at or in the vicinity of that
    point.

    For a node to exist even where there are no training samples in the portion of
    the feature space that belongs to the node is an indication of the generalization
    ability of decision-tree based classification.

    When used in a non-interactive mode, an instance of this class can be used to
    create a tabular display that shows what training samples belong directly to the
    portion of the feature space that corresponds to each node of the decision tree.
    An instance of this class can also construct a tabular display that shows how the
    influence of each training sample propagates in the decision tree.  For each
    training sample, this display first shows the list of nodes that came into
    existence through feature test(s) that used the data provided by that sample.
    This list for each training sample is followed by a subtree of the nodes that owe
    their existence indirectly to the training sample. A training sample influences a
    node indirectly if the node is a descendant of another node that is affected
    directly by the training sample.

    '''
    def __init__(self, dt):
        if isinstance(dt, DecisionTree) is False:
            raise TypeError("The argument supplied to the DTIntrospector constructor must be of type DecisionTree")
        self._dt = dt
        self._root_dtnode =  dt._root_node
        self._samples_at_nodes_dict = {}
        self._branch_features_to_nodes_dict = {}
        self._sample_to_node_mapping_direct_dict = {}
        self._node_serial_num_to_node_dict = {} 
        self._awareness_raising_msg_shown = 0
        self._debug = 0

    def initialize(self):
        if self._root_dtnode is None:
            raise SyntaxError("You must first construct the decision tree before using the DT Introspection class.")
        root_node = self._root_dtnode
        self.recursive_descent(root_node)

    def recursive_descent(self, node):
        node_serial_number = node.get_serial_num()
        self._node_serial_num_to_node_dict[node_serial_number] = node
        branch_features_and_values_or_thresholds = node.get_branch_features_and_values_or_thresholds()
        if self._debug:
            print("\nat node " + str(node_serial_number) + ":  the branch features and values are: " + str(branch_features_and_values_or_thresholds))
        self._branch_features_to_nodes_dict[node_serial_number] = branch_features_and_values_or_thresholds
        samples_at_node = None
        for item in branch_features_and_values_or_thresholds:
            samples_for_feature_value_combo = self.get_samples_for_feature_value_combo(item)
            samples_at_node = samples_for_feature_value_combo if samples_at_node is None else \
                        [sample for sample in samples_at_node if sample in samples_for_feature_value_combo]
        if (samples_at_node is not None) and (len(samples_at_node) > 0): 
            samples_at_node = sorted(samples_at_node, key = lambda x: sample_index(x))
        if self._debug:
            print("Node: " + str(node_serial_number) + " the samples are: " + str(samples_at_node))
        self._samples_at_nodes_dict[node_serial_number] = samples_at_node
        if samples_at_node is not None:
            for sample in samples_at_node:
                if sample not in self._sample_to_node_mapping_direct_dict:
                    self._sample_to_node_mapping_direct_dict[sample] = [node_serial_number] 
                else:
                    self._sample_to_node_mapping_direct_dict[sample].append(node_serial_number)
        children = node.get_children()
        list(map(lambda x: self.recursive_descent(x), children))

    def display_training_samples_at_all_nodes_direct_influence_only(self):
        if self._root_dtnode is None:
            raise SyntaxError("You must first construct the decision tree before using the DT Introspection class.")
        root_node = self._root_dtnode
        self.recursive_descent_for_showing_samples_at_a_node(root_node)

    def recursive_descent_for_showing_samples_at_a_node(self, node):
        node_serial_number = node.get_serial_num()
        branch_features_and_values_or_thresholds = node.get_branch_features_and_values_or_thresholds()
        if node_serial_number in self._samples_at_nodes_dict:
            if self._debug:
                print("\nat node " + str(node_serial_number) + ":  the branch features and values are: " + str(branch_features_and_values_or_thresholds))
            print("Node " + str(node_serial_number) + ": the samples are: " + str(self._samples_at_nodes_dict[node_serial_number]))
        children = node.get_children()
        list(map(lambda x: self.recursive_descent_for_showing_samples_at_a_node(x), children))

    def display_training_samples_to_nodes_influence_propagation(self):
        for sample in sorted(self._dt._training_data_dict, key = lambda x: sample_index(x)):
            if sample in self._sample_to_node_mapping_direct_dict:
                nodes_directly_affected = self._sample_to_node_mapping_direct_dict[sample]
                print("\n" + sample + ":\n" + "   nodes affected directly: " + str(nodes_directly_affected))
                print("   nodes affected through probabilistic generalization:")
                list(map(lambda x: self.recursive_descent_for_sample_to_node_influence(x, nodes_directly_affected, "    "), nodes_directly_affected))

    def recursive_descent_for_sample_to_node_influence(self, node_serial_num, nodes_already_accounted_for, offset ):
        offset += "    "  
        node =  self._node_serial_num_to_node_dict[node_serial_num]
        children = [x.get_serial_num() for x in node.get_children()]
        children_affected = [child for child in children if child not in nodes_already_accounted_for]
        if len(children_affected) > 0:
            print(offset + str(node_serial_num) + "=> " + str(children_affected))
        list(map(lambda x: self.recursive_descent_for_sample_to_node_influence(x, children_affected, offset), children_affected))

    def get_samples_for_feature_value_combo(self, feature_value_combo):
        feature,op,value = self.extract_feature_op_val(feature_value_combo)
        samples = []
        if op == '=':
            samples = [sample for sample in self._dt._training_data_dict if feature_value_combo in \
                                                       self._dt._training_data_dict[sample]]
        elif op == '<':      
            for sample in self._dt._training_data_dict:
                features_and_values = self._dt._training_data_dict[sample]            
                for item in features_and_values:
                    feature_data,op_data,val_data = self.extract_feature_op_val(item)
                    value = convert(value)
                    val_data = convert(val_data)
                    if isinstance(val_data, (int,float)) and (feature == feature_data) and (val_data <= value):
                        samples.append(sample)
                        break
        elif op == '>':      
            for sample in self._dt._training_data_dict:
                features_and_values = self._dt._training_data_dict[sample]            
                for item in features_and_values:
                    feature_data,op_data,val_data = self.extract_feature_op_val(item)
                    value = convert(value)
                    val_data = convert(val_data)
                    if isinstance(val_data, (int,float)) and (feature == feature_data) and (val_data > value):
                        samples.append(sample)
                        break
        else:
            raise SyntaxError("Something is wrong with the feature-value syntax")
        return samples

    def extract_feature_op_val(self, feature_value_combo):
        pattern1 = r'(.+)=(.+)'
        pattern2 = r'(.+)<(.+)'
        pattern3 = r'(.+)>(.+)'
        feature = value = op = None
        if re.search(pattern2, feature_value_combo):
            m = re.search(pattern2, feature_value_combo)
            feature,op,value = m.group(1),'<',m.group(2)
        elif re.search(pattern3, feature_value_combo): 
            m = re.search(pattern3, feature_value_combo)
            feature,op,value = m.group(1),'>',m.group(2)
        else:
            m = re.search(pattern1, feature_value_combo)
            feature,op,value = m.group(1),'=',m.group(2)
        return (feature,op,value) 

    def explain_classifications_at_multiple_nodes_interactively(self):
        if not self._samples_at_nodes_dict:
            raise SyntaxError('''You called explain_classifications_at_multiple_nodes_interactively() without '''
                              '''first initializing the DTIntrospection instance in your code. Aborting.''')
        print("\n\nIn order for the decision tree to introspect\n")
        msg = '''DO YOU ACCEPT the fact that, in general, a region of the feature space
                 that corresponds to a node may have NON-ZERO probabilities associated
                 with it even when there are NO training data points in that region?
                 Enter 'y' for yes or any other character for no: '''
        ans = None
        if sys.version_info[0] == 3:
            ans = input(msg)
        else:
            ans = raw_input(msg)
        ans = ans.strip()
        if (ans != 'y') and (ans != 'yes'):
            raise Exception("\n  Since you answered 'no' to a very real theoretical possibility, no explanations possible for the classification decisions in the decision tree. Aborting")
        self._awareness_raising_msg_shown = 1
        while True:
            node_id = None
            while True:
                ans = None
                if sys.version_info[0] == 3:
                    ans = input("\nEnter the integer ID of a node: ")
                else:
                    ans = raw_input("\nEnter the integer ID of a node: ")
                ans = ans.strip()
                if ans == 'exit': return
                try:
                    ans = int(ans)
                except ValueError:
                    print("\nWhat you entered does not look like an integer ID for a node. Aborting!")
                    raise
                if ans in self._samples_at_nodes_dict:
                    node_id = ans
                    break;
                else:
                   print("\nYour answer must be an integer ID of a node. Try again or enter 'exit'.")
            self.explain_classification_at_one_node(node_id)   

    def explain_classification_at_one_node(self, node_id):
        if not self._samples_at_nodes_dict:
            raise SyntaxError('''You called explain_classification_at_one_node() without first initializing '''
                              '''the DTIntrospection instance in your code. Aborting."''')
        if node_id not in self._samples_at_nodes_dict:
            print("Node %d is not a node in the tree" % node_id)
            return
        if node_id == 0:
            print("Nothing useful to be explained at the root node")
            return
        if not self._awareness_raising_msg_shown: 
            print("\n\nIn order for the decision tree to introspect at Node %d: \n" % node_id)
            msg = "  DO YOU ACCEPT the fact that, in general, a region of the feature space\n" + \
                  "  that corresponds to a DT node may have NON-ZERO probabilities associated\n" + \
                  "  with it even when there are NO training data points in that region?\n" + \
                  "\nEnter 'y' for yes or any other character for no:  "
            ans = None
            if sys.version_info[0] == 3:
                ans = input(msg)
            else:
                ans = raw_input(msg)
            ans = ans.strip()
            if (ans != 'y') and (ans != 'yes'):
                raise Exception("\n  Since you answered 'no' to a very real theoretical possibility, no explanations possible for the classification decision at node %d" % node_id)
        samples_at_node = self._samples_at_nodes_dict[node_id]
        branch_features_to_node = self._branch_features_to_nodes_dict[node_id]
        class_names = self._root_dtnode.get_class_names()
        class_probabilities = self._root_dtnode.get_class_probabilities()
        feature,op,value = self.extract_feature_op_val( branch_features_to_node[-1] )
        if len(samples_at_node) > 0:
            msg2 = "\n    Samples in the portion of the feature space assigned to Node %d: %s\n" % (node_id,  str(samples_at_node))
        else:
            msg2 = "\n\n    There are NO training data samples directly in the region of the feature space assigned to node %d.\n" % node_id
        msg2 += "\n    Features tests on the branch to node %d: %s\n" % (node_id, str(branch_features_to_node))
        msg2 += "\n    Would you like to see the probability associated with the last feature test on the branch leading to Node %d\n?" % node_id
        msg2 += "\n\n    Enter 'y' if yes and `n' if no:  "
        ans = None
        if sys.version_info[0] == 3:
            ans = input(msg2)
        else:
            ans = raw_input(msg2)
        ans = ans.strip()
        if (ans == 'y') or (ans == 'yes'):
            sequence = [ branch_features_to_node[-1] ]
            prob = self._dt.probability_of_a_sequence_of_features_and_values_or_thresholds(sequence) 
            print("\n    Probability of %s is: %s" % (str(sequence), str(prob))) 
        msg3 = "\n    Using Bayes rule, would you like to see the class probabilities predicated on just the last feature test on the branch leading to Node %d\n?" % node_id
        msg3 += "\n\n    Enter 'y' if yes and `n' if no:  "
        ans = None
        if sys.version_info[0] == 3:
            ans = input(msg3)
        else:
            ans = raw_input(msg3)
        ans = ans.strip()
        if (ans == 'y') or (ans == 'yes'):
            sequence = [ branch_features_to_node[-1] ]
            for cls in class_names:
                prob = self._dt.probability_of_a_class_given_sequence_of_features_and_values_or_thresholds(cls, sequence)
                print("\n    Probability of class %s given just one feature test %s is: %s" % (cls, str(sequence), str(prob)))
        else:
            print("goodbye")
        print("\nFinished supplying information on Node %d\n\n" % node_id)

#------------------------------------  Test Code Follows  -------------------------------------

if __name__ == '__main__':

    dt = DecisionTree( training_datafile = "Examples/training_symbolic.csv",  
                        csv_class_column_index = 1,
                        csv_columns_for_features = [2,3,4,5],
                        max_depth_desired = 5,
                        entropy_threshold = 0.1,
                     )
    dt.get_training_data()

    dt.show_training_data()

    prob = dt.prior_probability_for_class( 'class=benign' )
    print("prior for benign: ", prob)
    prob = dt.prior_probability_for_class( 'class=malignant' )
    print("prior for malignant: ", prob)

    prob = dt.probability_of_feature_value( 'smoking', 'heavy')
    print(prob)

    dt.determine_data_condition()

    root_node = dt.construct_decision_tree_classifier()
    root_node.display_decision_tree("   ")

    test_sample = ['exercising=never', 'smoking=heavy', 'fatIntake=heavy', 'videoAddiction=heavy']
    classification = dt.classify(root_node, test_sample)
    print("Classification: " + str(classification))

    test_sample = ['videoAddiction=none', 'exercising=occasionally', 'smoking=never', 'fatIntake=medium']
    classification = dt.classify(root_node, test_sample)
    print("Classification: " + str(classification))

    dt.print_class_variables()

    print(f'\n\n')

    print("Number of nodes created: " + str(root_node.how_many_nodes()))

    #root_node.print_class_vars()
