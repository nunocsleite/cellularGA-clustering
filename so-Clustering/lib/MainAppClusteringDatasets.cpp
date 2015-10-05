#include <eo>

#include <algorithm>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <set>

#include "testset/TestSetDescription.h"
#include "init/eoClusteringInit.h"

#include "eval/eoClusteringEval.h"
#include "eval/eoNumberEvalsCounter.h"
#include "eval/statistics/eoClusteringEvalWithStatistics.h"
#include "testset/ClusteringTestSet.h"

#include "algorithms/eo/eoCellularEA.h"
#include "algorithms/eo/eoCellularEARing.h"
#include "algorithms/eo/eoCellularEAMatrix.h"
#include "algorithms/eo/Mutation.h"
#include "algorithms/eo/Crossover.h"

#include "algorithms/eo/eoGenerationContinuePopVector.h"
#include "eoSelectOne.h"
#include "algorithms/eo/eoDeterministicTournamentSelectorPointer.h" // eoDeterministicTournamentSelector using boost::shared_ptr

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

#include <armadillo>




using namespace std;
using namespace arma;


//#define MAINAPP_DEBUG
//#define MAINAPP_DEBUG_1

//#define CROSS_VALIDATION


extern int getSANumberEvaluations(double tmax, double r, double k, double tmin);


#include <cstdlib>
#include <iostream>
#include <boost/lambda/lambda.hpp>
#include <boost/random/mersenne_twister.hpp>


using namespace boost;
using namespace boost::lambda;
using namespace std;



struct Main {

    void buildSets(ProblemData const &_completeData, ProblemData &_trainData, ProblemData &_testData, double _percentageSplit);

    eoChromosome train(const string &_outputDir, const TestSet &_testSet, ProblemData const &_trainData);

    double classify(const string &_outputDir, eoChromosome const &_bestSolution, ProblemData &_testData);

    void assignToClosestCluster(vector<int> &_dataPointsAssignedClusters,
                                      vector<double> const &_clusterCentres,
                                      ProblemData &_testData);

    eoChromosome trainCV(const string &_outputDir, TestSet const& _testSet, const ProblemData &_trainData,
                         double _percentageSplit, int _numCycles);

    eoChromosome runCellularEA(const string &_outputDir,
                       TestSet const& _testSet,
                       ProblemData const &_trainData);


    eoChromosome trainIteration(const string &_outputDir, TestSet const& _testSet,
                                  ProblemData const &_trainDataCV, ProblemData const &_testDataCV, int _iterationNumber);


    void buildCellularEA(const string &_outputDir,
                         TestSet const& _testSet,
                         ProblemData const &_trainData);


    eoChromosome runCellularEAIteration();


    void buildPop(ProblemData const &_trainData);

    //
    // Fields
    //
    /**
     * @brief pop Individuals population
     */
    boost::shared_ptr<vector<boost::shared_ptr<eoChromosome> > > pop;
    /**
     * @brief cGA cGA object
     */
    boost::shared_ptr<eoCellularEAMatrix<eoChromosome> > cGA;
};





/**
 * @brief runClusteringDatasets
 * @param _datasetIndex
 * @param _testBenchmarksDir
 * @param _outputDir
 */
void runClusteringDatasets(int _datasetIndex, string const& _testBenchmarksDir, string const& _outputDir) {

    Main m;

    // Initialise random seed
    srand(time(NULL));

    vector<TestSetDescription> clusteringTestSet;
    //
    // TestSetDescription info: <name>, <description>, <directory name>, <data filename>,
    // <atribute class index>, <number of attributes>
    // There two data files, namely <data filename>.data and <data names>.names
    //
    clusteringTestSet.push_back(TestSetDescription("Balance", "Balance dataset", "balance", "balance-scale", 1, 5));
    // Has missing data
    clusteringTestSet.push_back(TestSetDescription("Credit", "Credit dataset", "credit", "crx", 16, 16));
    // Has missing data
    clusteringTestSet.push_back(TestSetDescription("Dermatology", "Dermatology dataset", "dermatology", "dermatology", 35, 35));
    clusteringTestSet.push_back(
                TestSetDescription("Diabetes", "Pima Indians Diabetes dataset", "diabetes", "pima-indians-diabetes", 9, 9));
    clusteringTestSet.push_back(TestSetDescription("E.Coli", "E.Coli dataset", "ecoli", "ecoli-filtered", 8, 8));
    clusteringTestSet.push_back(TestSetDescription("Glass", "Glass dataset", "glass", "glass", 10, 10));
    // Has missing data. Values 1, 2, 3, and 4 belong to class 1. Value 0 belong to class 0.
    clusteringTestSet.push_back(TestSetDescription("Heart", "Heart (Cleveland) dataset", "heart", "heart-disease", 14, 14));
    // Has missing data. Train and test set joined in one single set.
    clusteringTestSet.push_back(TestSetDescription("Horse Colic", "Horse Colic dataset", "horse-colic", "horse-colic-all", 28, 28));
    clusteringTestSet.push_back(TestSetDescription("Iris", "Iris dataset", "iris", "iris", 5, 5));
    clusteringTestSet.push_back(TestSetDescription("Thyroid", "New Thyroid dataset", "thyroid", "new-thyroid", 6, 6));
    clusteringTestSet.push_back(TestSetDescription("WDBCancer", "WDBCancer dataset", "wdbcancer", "wdbc_no_id", 1, 31));
    // Has missing data
    clusteringTestSet.push_back(
                TestSetDescription("WDBCancer-Int", "WDBCancer original dataset", "wdbcancer-int",
                                   "breast-cancer-wisconsin-without-id", 10, 10));
    clusteringTestSet.push_back(TestSetDescription("Wine", "Wine dataset", "wine", "wine", 1, 14));
    // Synthetic Rings data
    clusteringTestSet.push_back(TestSetDescription("Rings", "Rings synthetic dataset", "rings", "rings", 3, 3));

    /////////////////////////////////////////////////////////////////////////////////////////////////
    copy(clusteringTestSet.begin(), clusteringTestSet.end(), ostream_iterator<TestSetDescription>(cout, "\n"));

    vector<TestSetDescription>::iterator it = clusteringTestSet.begin() + _datasetIndex;
    // Create TestSet instance
    ClusteringTestSet testSet(*it, _testBenchmarksDir);
    // Load dataset
    ClusteringTestSet* ptr = &testSet;
    ptr->load();

    // Print testset info
    cout << testSet << endl;
    cout << "Press any key to continue..." << endl;
    cin.get();


#ifdef CROSS_VALIDATION
    //
    // Divide into train and test sets
    //
    // Percentage split
//    const double percentageSplit = 0.50; // Balance - 10 (no shuffling)!
    const double PERCENT_SPLIT = 0.75;
//    const double PERCENT_SPLIT = 1.0;
//    const double PERCENT_SPLIT_CV = 0.80;

    // Do 20 runs
    for (int i = 0; i < 20; ++i) {

        // Build train and test sets
        ProblemData trainData, testData;
        ProblemData const &completeData = testSet.getProblemData();
        m.buildSets(completeData, trainData, testData, PERCENT_SPLIT);

        const double PERCENT_SPLIT_CV = 0.84;
    //    const double PERCENT_SPLIT_CV = 0.99;
        // Cross-validation
        const int CV_NUM_CYCLES = 10;
    //    const int CV_NUM_CYCLES = 1;

        // Train classifier
    //    eoChromosome bestSolution = train(_outputDir, testSet, trainData);
        ////////////////////////////////////////////////////////////////////////////////////////////////
        // Train classifier using Cross-validation
        eoChromosome bestSolution = m.trainCV(_outputDir, testSet, trainData, PERCENT_SPLIT_CV, CV_NUM_CYCLES);

        // Classify test set
        double cep = m.classify(_outputDir, bestSolution, testData);

        cout << cep << endl;
        ////////////////////////////////////////////////////////////////////////////////////////////////

        /// TEST
    //    classify(_outputDir, bestSolution, trainData);
    }
#else

    //
    // Divide into train and test sets
    //
    // Percentage split
    const double PERCENT_SPLIT = 0.75;

    // Build train and test sets
    ProblemData trainData, testData;
    ProblemData const &completeData = testSet.getProblemData();
    m.buildSets(completeData, trainData, testData, PERCENT_SPLIT);


    /////////////////////////////////////////////////////////////////////////////
///
/// TODO - USE BUILDCELLULAREA METHOD
///
    // Population reference
//    boost::shared_ptr<vector<boost::shared_ptr<eoChromosome> > > pop = buildPop(trainData);
    m.buildPop(trainData);
    // Build cellular GA object
    // cGA reference
//    boost::shared_ptr<eoCellularEAMatrix<eoChromosome> > cGA = buildCellularEA(_outputDir,
//                    testSet,
//                    trainData,
//                    pop);

//    m.buildCellularEA(_outputDir, testSet, trainData);

    //////////////////////////////////////////////////////////////////////////////
    //
    // cEA parameters
    //

    // Matrix
//    const int NLINES = 50;
//    const int NCOLS = 50;

//    const int NLINES = 100;
//    const int NCOLS = 1000;

    ////// Pop size = 16  ////////
    // Rect
//    const int NLINES = 2;
//    const int NCOLS = 8;
    // Matrix
//    const int NLINES = 4;
//    const int NCOLS = 4;
//    // Ring
//    const int NLINES = 1;
//    const int NCOLS = 16;
    //////////////////////////////

    ////// Pop size = 36  ////////
    // Rect
//    const int NLINES = 3;
//    const int NCOLS = 12;
    // Matrix
//    const int NLINES = 6;
//    const int NCOLS = 6;
    // Ring
//    const int NLINES = 1;
//    const int NCOLS = 36;
    //////////////////////////////

    ////// Pop size = 64  ////////
    // Rect
//    const int NLINES = 2;
//    const int NCOLS = 32;
    // Matrix
//    const int NLINES = 8;
//    const int NCOLS = 8;
    // Ring
//    const int NLINES = 1;
//    const int NCOLS = 64;
    //////////////////////////////

    ////// Pop size = 100  ////////
//    // Rect
    const int NLINES = 5;
    const int NCOLS = 20;

//    // Matrix
//    const int NLINES = 10;
//    const int NCOLS = 10;
//    // Ring
//    const int NLINES = 1;
//    const int NCOLS = 100;
    //////////////////////////////

    ////// Pop size = 225  ////////
    // Rect
//    const int NLINES = 9;
//    const int NCOLS = 25;
//    // Matrix
//    const int NLINES = 15;
//    const int NCOLS = 15;

    // Ring
//    const int NLINES = 1;
//    const int NCOLS = 225;
    //////////////////////////////

//    const int NLINES = 10;
//    const int NCOLS = 100;

//    const int NLINES = 50;
//    const int NCOLS = 200;

    const int POP_SIZE = NLINES*NCOLS;  // Population size
//    const int L = 5000000; // Number of generations
//    const int L = 2500; // Number of generations
//    const int L = 500; // Number of generations
//    const int L = 1000; // Number of generations
    const int L = 1; // Number of generations

    // Crossover probability
//    const double cp = 1;
//        const double cp = 0.8;
    const double cp = 0.6;
//    const double cp = 0.4; // TEST
//    const double cp = 0.15;
//    const double cp = 0;

    // Mutation probability
//    const double mp = 1;
//        const double mp = 0.5;
//    const double mp = 0.2; // 20%
    const double mp = 0.1; // 10%
//    const double mp = 0.01; // 1%
//    const double mp = 0.001; // 0.1%
//    const double mp = 0; // TEST


    // Creating the output file in the specified output directory
    stringstream sstream;
    sstream << _outputDir << testSet.getDescription().getName() << "_NLINES_" << NLINES << "_NCOLS_" << NCOLS
            << "_cp_" << cp << "_mp_" << mp << ".txt";

    string filename;
    sstream >> filename;
    ofstream outFile(filename);
    // # evaluations counter
    eoNumberEvalsCounter numEvalCounter;
    // Objective function evaluation
    eoClusteringEvalWithStatistics<eoChromosome> eval(numEvalCounter);
    // Get population
    auto &solutionPop = *m.pop.get();
    // Population size
    int popSize = solutionPop.size();

    /////////////////////////////////////////////////////////////////////////////////
    cout << "pop size = " << popSize <<  endl;
    // Print population information to output and
    std::cout << std::endl << "Initial population" << std::endl;
    int k = 0;
    for (int k = 0; k < popSize; ++k) {
        std::cout << (*solutionPop[k].get()).fitness() << "\t";
    }
    /////////////////////////////////////////////////////////////////////////////////

    //
    // Build CellularGA
    //
    // Terminate after concluding L time loops or 'Ctrl+C' signal is received
    // The eoGenerationContinuePopVector object, instead of using an eoPop to represent the population,
    // uses a vector. A vector is used in order to swap offspring and population efficiently using pointers
    eoGenerationContinuePopVector<eoChromosome> terminator(L);

    // Declare 1-selectors
    //
    // Binary deterministic tournament selector used in neighbour selection
    // Work with pointers for efficiency
    eoDetTournamentSelectSharedPtr<eoChromosome> detSelectNeighbourPtr;

    // Crossover and mutation
    Mutation<eoChromosome> mutation;
    Crossover<eoChromosome> crossover;

//    boost::shared_ptr<eoCellularEA<eoChromosome> > cGA;
    // Build the corresponding type of cGA object depending on the layout (Ring or Matrix)
//    if (NLINES == 1) { // Ring cGA
//    eoCellularEARing<eoChromosome> cEA(checkpoint, eval, detSelectNeighbour,
//                      crossover, mutation,
//                     selectBestOne,
//                     selectBestOne
//                    );
//    }
//    else {
        m.cGA = boost::make_shared<eoCellularEAMatrix<eoChromosome> >(
                    outFile, NLINES, NCOLS,
                    cp, mp,
                    terminator,
                    eval,
                    detSelectNeighbourPtr, // Work with pointers for efficiency
                    crossover, mutation,
                    numEvalCounter
        );

//    }
    ////////////////////////////////////////////////////////////////////////////////////////////////

    // Vector for registering the train set errors
    vector<double> trainSetErrors(1000);
    // Vector for registering the test set errors
    vector<double> testSetErrors(1000);


    double minValue = numeric_limits<double>::max();
    int minIteration = 0;
    for (int i = 1; i <= 1000; ++i) {
        eoChromosome solution = m.runCellularEAIteration();

        // Classify train set
        double trainError = m.classify(_outputDir, solution, trainData);
        // Register the train error
        trainSetErrors[i-1] = trainError;

        // Classify test set
        double mVal = m.classify(_outputDir, solution, testData);
        if (mVal < minValue) {
            minValue = mVal;
            minIteration = i;
        }
        // Register the test error
        testSetErrors[i-1] = mVal;

/*
        // Last iteration
        if (i == 1000) {
            // Print centers
            vector<double> const &centers = solution.getClusterCentres();
            int numClasses = solution.getNumClasses();
            int numFeatures = solution.getNumFeatures();
            for (int k = 0; k < K_CLUSTERS; ++k) {
                for (int i = 0; i < numClasses; ++i) {
                    cout << "(";
                    int j;
                    for (j = 0; j < numFeatures-1; ++j) {
                        cout << centers[k*numFeatures*numClasses + i*numFeatures+j] << ", ";
                    }
                    // Last coordinate
                    cout << centers[k*numFeatures*numClasses + i*numFeatures+j] << ")" << endl;
                }
                cout << endl;
            }
            ////////////////////////////////////////////////////////////////////////////
            // Print test set assigned classes
            // Get data points ground thruth clusters
            vector<int> const &dataPointsClusters = testData.getSamplesClasses();
            auto const& testSamples = testData.getSamples();
            // Get data points assigned clusters
            vector<int> dataPointsAssignedClusters(testData.getNumSamples());
            m.assignToClosestCluster(dataPointsAssignedClusters, solution.getClusterCentres(), testData);

            int numMisclassified = 0;
            // Print cluster index
            for (int i = 0; i < testData.getNumSamples(); ++i) {
                // x_i cluster index
                int cluster_x_i_index = dataPointsClusters[i];
                numMisclassified += (dataPointsAssignedClusters[i] != cluster_x_i_index ? 1 : 0);

                cout << "i = " << i << ", sample: " << testSamples[i] << ", assigned Cluster = " << dataPointsAssignedClusters[i] <<
                        ", real cluster = " << cluster_x_i_index << endl;
            }
            // Dataset size
            int sizeDataset = testData.getNumSamples();
            // CEP
            double cep = ((double)numMisclassified / sizeDataset) * 100;

            cout << "numMisclassified = " << numMisclassified << ", sizeDataset = " << sizeDataset << endl;
            cout << "CEP = " << cep << endl;
            cin.get();

            //
            // Print the class of each sample
            //
            // Class 0
            cout << "%Class 0" << endl;
            for (int i = 0; i < testData.getNumSamples(); ++i) {
                // x_i cluster index
                int cluster_x_i_index = dataPointsClusters[i];
                if (dataPointsAssignedClusters[i] == 0) {
//                    cout << "i = " << i << ", sample: " << testSamples[i] <<  ", assigned Cluster = "
//                         << dataPointsAssignedClusters[i] << ", real cluster = " << cluster_x_i_index << endl;
                    cout << "(";
                    int j;
                    for (j = 0; j < numFeatures-1; ++j) {
                        cout << testSamples[i][j] << ",";
                    }
                    cout << testSamples[i][j] << ")" << endl;
                }
            }
            // Class 1
            cout << "%Class 1" << endl;
            for (int i = 0; i < testData.getNumSamples(); ++i) {
                // x_i cluster index
                int cluster_x_i_index = dataPointsClusters[i];
                if (dataPointsAssignedClusters[i] == 1) {
//                    cout << "i = " << i << ", sample: " << testSamples[i] <<  ", assigned Cluster = "
//                         << dataPointsAssignedClusters[i] << ", real cluster = " << cluster_x_i_index << endl;
                    cout << "(";
                    int j;
                    for (j = 0; j < numFeatures-1; ++j) {
                        cout << testSamples[i][j] << ",";
                    }
                    cout << testSamples[i][j] << ")" << endl;
                }
            }
            // Class 2
            cout << "%Class 2" << endl;
            for (int i = 0; i < testData.getNumSamples(); ++i) {
                // x_i cluster index
                int cluster_x_i_index = dataPointsClusters[i];
                if (dataPointsAssignedClusters[i] == 2) {
//                    cout << "i = " << i << ", sample: " << testSamples[i] <<  ", assigned Cluster = "
//                         << dataPointsAssignedClusters[i] << ", real cluster = " << cluster_x_i_index << endl;
                    cout << "(";
                    int j;
                    for (j = 0; j < numFeatures-1; ++j) {
                        cout << testSamples[i][j] << ",";
                    }
                    cout << testSamples[i][j] << ")" << endl;
                }
            }
            ////////////////////////////////////////////////////////////////////////////
        }
//        cin.get();
*/
    }

    cout << "Min value = " << minValue << " at iteration " << minIteration << endl;

//    cout << "Train set errors" << endl;
//    for (int i = 0; i < trainSetErrors.size(); ++i) {
//        cout << "(" << (i+1) << ", " << trainSetErrors[i] << ")" << endl;
//    }
//    cout << endl;
//    cout << "Test set errors" << endl;
//    for (int i = 0; i < testSetErrors.size(); ++i) {
//        cout << "(" << (i+1) << ", " << testSetErrors[i] << ")" << endl;
//    }
    /////////////////////////////////////////////////////////////////////////////

#endif
}







/**
 * @brief buildSets Builds a train set and a test set from the given data, according % split
 * @param _completeData
 * @param _trainData
 * @param _testData
 * @param _percentageSplit
 */
void Main::buildSets(ProblemData const &_completeData, ProblemData &_trainData,
               ProblemData &_testData, double _percentageSplit)
{
    // Size of original data set
    int dataSize = _completeData.getNumSamples();
    // Train set size
    int trainDataSize = ceil(_percentageSplit * dataSize);
    // Test set size
    int testDataSize = dataSize - trainDataSize;

    /////////////////////////////////////////////////////////////////////////////////////
    //
    // Shuffle test set
    //
    vector<arma::vec> const &auxSamples = _completeData.getSamples();
    vector<int> const &auxSamplesClasses = _completeData.getSamplesClasses();

    vector<arma::vec> samples(_completeData.getNumSamples());
    vector<int> classes(_completeData.getNumSamples());

    // Perform shufling
    ClusteringTestSet::shuffleTestSet(auxSamples, auxSamplesClasses, samples, classes);
    /////////////////////////////////////////////////////////////////////////////////////

#ifdef MAINAPP_DEBUG
    cout << "dataSize = " << dataSize << endl;
    cout << "trainDataSize = " << trainDataSize << endl;
    cout << "testDataSize = " << testDataSize << endl;
    cout << "trainDataSize + testDataSize == dataSize ? " << (trainDataSize+testDataSize == dataSize ? "yes" : "no") << endl;
#endif
    //////////////////////////////////////////////////////////////////////////////////////
    //
    // Copy constant fields in _completeData
    //
    // Copy class labels
    _trainData.setClassLabels(boost::make_shared<vector<string>>(_completeData.getClassLabels()));
    _testData.setClassLabels(boost::make_shared<vector<string>>(_completeData.getClassLabels()));
    // Copy pointer to class map
    _trainData.setClassMap(boost::make_shared<boost::unordered_map<string, int>>(_completeData.getClassMap()));
    _testData.setClassMap(boost::make_shared<boost::unordered_map<string, int>>(_completeData.getClassMap()));
    // Copy NumClasses
    _trainData.setNumClasses(_completeData.getNumClasses());
    _testData.setNumClasses(_completeData.getNumClasses());
    // Copy NumFeaturesPerSample
    _trainData.setNumFeaturesPerSample(_completeData.getNumFeaturesPerSample());
    _testData.setNumFeaturesPerSample(_completeData.getNumFeaturesPerSample());
    // Copy Attributes
    _trainData.setAttributes(_completeData.getAttributes());
    _testData.setAttributes(_completeData.getAttributes());
    //////////////////////////////////////////////////////////////////////////////////////
    //
    // Define non-const fields
    //
    // Define NumSamples
    _trainData.setNumSamples(trainDataSize);
    _testData.setNumSamples(testDataSize);
    //
    // Define samples and classes in each set
    //
    // Vector containing the samples (each sample is represented by a arma::vec)
    boost::shared_ptr<std::vector<arma::vec>> ptrSamplesTrain = boost::make_shared<std::vector<arma::vec>>();
    // Classes
    boost::shared_ptr<std::vector<int> > ptrSamplesClassesTrain = boost::make_shared<std::vector<int> >();
    // Vector containing the samples (each sample is represented by a arma::vec)
    boost::shared_ptr<std::vector<arma::vec>> ptrSamplesTest = boost::make_shared<std::vector<arma::vec>>();
    // Classes
    boost::shared_ptr<std::vector<int> > ptrSamplesClassesTest = boost::make_shared<std::vector<int> >();
    // Auxiliary vars
    // Train samples
    auto &trainSamples = *ptrSamplesTrain.get();
    // Train classes
    auto &trainClasses = *ptrSamplesClassesTrain.get();
    // Test samples
    auto &testSamples = *ptrSamplesTest.get();
    // Test classes
    auto &testClasses = *ptrSamplesClassesTest.get();

    //////////////////////////////////////////////////////////////////////////////////////
    // Split samples maintaining an equal distribution of classes between train and test sets
//    auto const& samples = _completeData.getSamples();
//    auto const& classes = _completeData.getSamplesClasses();
    // Number of classes
    int numberClasses = _completeData.getNumClasses();
    // Mark chosen samples to true and false otherwise
    vector<bool> chosenSamples(_completeData.getNumSamples());
    // Compute number of samples in each class
    vector<int> numSamplesPerClass(numberClasses);
    for (int k = 0; k < classes.size(); ++k) {
        ++numSamplesPerClass[classes[k]];
    }
    // Compute number of samples in each class for train set and test set
    vector<int> numSamplesPerClassTrain(numberClasses);
    vector<int> numSamplesPerClassTest(numberClasses);
    for (int k = 0; k < numberClasses; ++k) {
        numSamplesPerClassTrain[k] = _percentageSplit*numSamplesPerClass[k];
        numSamplesPerClassTest[k] = numSamplesPerClass[k] - numSamplesPerClassTrain[k];
    }

    int i = 0;
    // Counter of the number of samples in each class for the train set
    vector<int> countOfSamplesPerClassTrain(numberClasses);
    // Counter of # samples in the train set
    int numSamplesTrain = 0;
    for (; i < dataSize; ++i) {
        // Get sample class
        int sampleClass = classes[i];
        // If the maximum number of samples in this class was not reached, insert it
        if (countOfSamplesPerClassTrain[sampleClass] < numSamplesPerClassTrain[sampleClass]) {
            // Push sample
            trainSamples.push_back(samples[i]);
            // Push class
            trainClasses.push_back(sampleClass);
            // Increment # samples in the current class
            ++countOfSamplesPerClassTrain[sampleClass];
            // Mark sample as chosen
            chosenSamples[i] = true;
            // Increment counter of samples
            ++numSamplesTrain;
        }
    }
    // Insert remainder samples if classes are unbalanced
    i = 0;
    for (; numSamplesTrain < trainDataSize && i < dataSize; ++i) {
        // If this sample was not chosen yet
        if (!chosenSamples[i]) {
            // Get sample class
            int sampleClass = classes[i];
            // Push sample
            trainSamples.push_back(samples[i]);
            // Push class
            trainClasses.push_back(sampleClass);
            // Increment # samples in the current class
            ++countOfSamplesPerClassTrain[sampleClass];
            // Mark sample as chosen
            chosenSamples[i] = true;
            // Increment counter of samples
            ++numSamplesTrain;
        }
    }

    // Insert the remainder into the test set
    // Counter of # samples in the test set
    int numSamplesTest = 0;
    i = 0;
    // Counter of the number of samples in each class for the test set
    vector<int> countOfSamplesPerClassTest(numberClasses);
    for (; i < dataSize; ++i) {
        // If this sample was not chosen yet
        if (!chosenSamples[i]) {
            // Get sample class
            int sampleClass = classes[i];
            // If the maximum number of samples in this class was not reached, insert it
            if (countOfSamplesPerClassTest[sampleClass] < numSamplesPerClassTest[sampleClass]) {
                // Push sample
                testSamples.push_back(samples[i]);
                // Push class
                testClasses.push_back(sampleClass);
                // Increment # samples in the current class
                ++countOfSamplesPerClassTest[sampleClass];
                // Mark sample as chosen
                chosenSamples[i] = true;
                // Increment counter of samples
                ++numSamplesTest;
            }
        }
    }
    // Insert remainder samples if classes are unbalanced
    i = 0;
    for (; numSamplesTest < testDataSize && i < dataSize; ++i) {
        // If this sample was not chosen yet
        if (!chosenSamples[i]) {
            // Get sample class
            int sampleClass = classes[i];
            // Push sample
            testSamples.push_back(samples[i]);
            // Push class
            testClasses.push_back(sampleClass);
            // Increment # samples in the current class
            ++countOfSamplesPerClassTest[sampleClass];
            // Mark sample as chosen
            chosenSamples[i] = true;
            // Increment counter of samples
            ++numSamplesTest;
        }
    }


#ifdef MAINAPP_DEBUG
    cout << "numSamplesPerClass in the complete set: " << endl;
    for (int k = 0; k < numberClasses; ++k) {
        cout << "class " << k << " - " << numSamplesPerClass[k] << endl;
    }
    cout << "numSamplesPerClass in the train set: " << endl;
    for (int k = 0; k < numberClasses; ++k) {
        cout << "class " << k << " - " << countOfSamplesPerClassTrain[k] << endl;
    }
    cout << "numSamplesPerClass in the test set: " << endl;
    for (int k = 0; k < numberClasses; ++k) {
        cout << "class " << k << " - " << countOfSamplesPerClassTest[k] << endl;
    }
#endif
    //////////////////////////////////////////////////////////////////////////////////////

    // Train set samples
    _trainData.setSamples(ptrSamplesTrain);
    // Train set classes
    _trainData.setSamplesClasses(ptrSamplesClassesTrain);
    // Test set samples
    _testData.setSamples(ptrSamplesTest);
    // Test set classes
    _testData.setSamplesClasses(ptrSamplesClassesTest);

    //////////////////////////////////////////////////////////////////////////////////////

#ifdef MAINAPP_DEBUG
    cout << "_trainData.getClassLabels().size() = " << _trainData.getClassLabels().size() << endl;
    cout << "_trainData.getNumClasses() = " << _trainData.getNumClasses() << endl;
    cout << "_trainData.getNumFeaturesPerSample() = " << _trainData.getNumFeaturesPerSample() << endl;
    cout << "_trainData.getNumSamples() = " << _trainData.getNumSamples() << endl;
    cout << "trainSamples.size() = " << trainSamples.size() << endl;
    cout << "trainClasses.size() = " << trainClasses.size() << endl;
    // Compute percentage of samples in each class
    cout << "Train set distribution" << endl;
    vector<double> counts(_trainData.getNumClasses());
    for (int i = 0; i < trainClasses.size(); ++i) {
        counts[trainClasses[i]]++;
    }
    for (int i = 0; i < counts.size(); ++i) {
        counts[i] /= trainClasses.size();
        counts[i] *= 100;
        cout << "class " << i << " - " << counts[i] << " %" << endl;
    }
    cout << endl;

    cout << "_testData.getClassLabels().size() = " << _testData.getClassLabels().size() << endl;
    cout << "_testData.getNumClasses() = " << _testData.getNumClasses() << endl;
    cout << "_testData.getNumFeaturesPerSample() = " << _testData.getNumFeaturesPerSample() << endl;
    cout << "_testData.getNumSamples() = " << _testData.getNumSamples() << endl;
    cout << "testSamples.size() = " << testSamples.size() << endl;
    cout << "testClasses.size() = " << testClasses.size() << endl;

    // Compute percentage of samples in each class
    cout << "Test set distribution" << endl;
    std::fill(counts.begin(), counts.end(), 0);
    for (int i = 0; i < testClasses.size(); ++i) {
        counts[testClasses[i]]++;
    }
    for (int i = 0; i < counts.size(); ++i) {
        counts[i] /= testClasses.size();
        counts[i] *= 100;
        cout << "class " << i << " - " << counts[i] << " %" << endl;
    }
    cout << endl;
    cout << "Press any key to continue..." << endl;
    cin.get();
#endif
}







/**
 * @brief train Train classifier
 * @param _outputDir
 * @param _testSet
 * @param _trainData
 * @return
 */
//eoChromosome train(const string &_outputDir, TestSet const& _testSet, const ProblemData &_trainData) {
//    eoChromosome bestSolution = runCellularEA(_outputDir, _testSet, _trainData);

//    return bestSolution;
//}





/**
 * @brief trainIteration
 * @param _outputDir
 * @param _testSet
 * @param _trainDataCV
 * @param _testDataCV
 * @param _iterationNumber
 * @return
// */
//eoChromosome trainIteration(const string &_outputDir, TestSet const& _testSet,
//                              ProblemData const &_trainDataCV, ProblemData const &_testDataCV, int _iterationNumber)
//{
//    // Find best clusters for train samples
////    eoChromosome solution = runCellularEA(_outputDir, _testSet, _trainDataCV);

//    // Find best clusters for train samples
//    eoChromosome solution = runCellularEAIteration(_outputDir, _testSet, _trainDataCV, _iterationNumber);

//    // Classify on the validation set
//    double cep = classify(_outputDir, solution, _testDataCV);

//    cout << "Iteration #" << _iterationNumber << " - CEP = " << cep << endl;
//    return solution;
//}







/**
 * @brief trainCV Train classifier using Cross-validation
 * @param _outputDir
 * @param _testSet
 * @param _trainData
 * @param _percentageSplit
 * @param _numCycles
 * @return
 */
eoChromosome Main::trainCV(const string &_outputDir, TestSet const& _testSet, ProblemData const &_trainData,
                     double _percentageSplit, int _numCycles)
{
    double bestCEP = -1;
    eoChromosome bestSolution;

    // Cross-validation loop
    for (int i = 0; i < _numCycles; ++i) {
//        cout << "CV iteration #" << (i+1) << endl;
        // Generate random train and validation partitions
        ProblemData trainDataCV, testDataCV;
        ProblemData const &completeData = _trainData;
        buildSets(completeData, trainDataCV, testDataCV, _percentageSplit);
        // Find best clusters for train samples
        eoChromosome solution = runCellularEA(_outputDir, _testSet, trainDataCV);
        // Classify on the validation set
        double cep = classify(_outputDir, solution, testDataCV);

        if (bestCEP == -1 || cep < bestCEP) {
            bestCEP = cep;
            bestSolution = solution;
        }

    }
//    cout << "After training classifier with CV" << endl;
//    cout << "best CEP = " << bestCEP << endl;
//    cin.get();
    return bestSolution;
}






eoChromosome Main::runCellularEA(const string &_outputDir,
                     TestSet const& _testSet,
                     ProblemData const &_trainData) {
    //
    // cEA parameters
    //

    // Matrix
//    const int NLINES = 50;
//    const int NCOLS = 50;

//    const int NLINES = 100;
//    const int NCOLS = 1000;

    ////// Pop size = 16  ////////
    // Rect
//    const int NLINES = 2;
//    const int NCOLS = 8;
    // Matrix
//    const int NLINES = 4;
//    const int NCOLS = 4;
//    // Ring
//    const int NLINES = 1;
//    const int NCOLS = 16;
    //////////////////////////////

    ////// Pop size = 36  ////////
    // Rect
//    const int NLINES = 3;
//    const int NCOLS = 12;
    // Matrix
//    const int NLINES = 6;
//    const int NCOLS = 6;
    // Ring
//    const int NLINES = 1;
//    const int NCOLS = 36;
    //////////////////////////////

    ////// Pop size = 64  ////////
    // Rect
//    const int NLINES = 2;
//    const int NCOLS = 32;
    // Matrix
//    const int NLINES = 8;
//    const int NCOLS = 8;
    // Ring
//    const int NLINES = 1;
//    const int NCOLS = 64;
    //////////////////////////////

    ////// Pop size = 100  ////////
//    // Rect
    const int NLINES = 5;
    const int NCOLS = 20;

//    // Matrix
//    const int NLINES = 10;
//    const int NCOLS = 10;
//    // Ring
//    const int NLINES = 1;
//    const int NCOLS = 100;
    //////////////////////////////

    ////// Pop size = 225  ////////
    // Rect
//    const int NLINES = 9;
//    const int NCOLS = 25;
//    // Matrix
//    const int NLINES = 15;
//    const int NCOLS = 15;

    // Ring
//    const int NLINES = 1;
//    const int NCOLS = 225;
    //////////////////////////////

//    const int NLINES = 10;
//    const int NCOLS = 100;

//    const int NLINES = 50;
//    const int NCOLS = 200;


    const int POP_SIZE = NLINES*NCOLS;  // Population size
//    const int L = 5000000; // Number of generations
//    const int L = 2500; // Number of generations
//    const int L = 500; // Number of generations
    const int L = 1000; // Number of generations

    // Crossover probability
//    const double cp = 1;
//        const double cp = 0.8;
    const double cp = 0.6;
//    const double cp = 0.4; // TEST
//    const double cp = 0.15;
//    const double cp = 0;

    // Mutation probability
//    const double mp = 1;
//        const double mp = 0.5;
//    const double mp = 0.2; // 20%
    const double mp = 0.1; // 10%
//    const double mp = 0.01; // 1%
//    const double mp = 0.001; // 0.1%
//    const double mp = 0; // TEST

    // Creating the output file in the specified output directory
    stringstream sstream;
    sstream << _outputDir << _testSet.getDescription().getName() << "_NLINES_" << NLINES << "_NCOLS_" << NCOLS
            << "_cp_" << cp << "_mp_" << mp << ".txt";

    string filename;
    sstream >> filename;
    ofstream outFile(filename);

    // Solution initializer
    eoClusteringInit<eoChromosome> init(_trainData);
    // Generate initial population
    // We can't work with eoPop of shared_ptr because shared_ptr is not an EO
    // Solution: Work with vector<shared_ptr<EOT> > directly
    boost::shared_ptr<vector<boost::shared_ptr<eoChromosome> > > solutionPopPtr
            = boost::make_shared<vector<boost::shared_ptr<eoChromosome> > >();
    auto &solutionPop = *solutionPopPtr.get();
    for (int i = 0; i < POP_SIZE; ++i) {
        // Create solution object and insert it in the vector
        solutionPop.push_back(boost::make_shared<eoChromosome>());
        // Initialize chromosome
        init((*solutionPop.back().get()));
    }
    // # evaluations counter
    eoNumberEvalsCounter numEvalCounter;
     // Objective function evaluation
    //    eoETTPEval<eoChromosome> eval;
    // Objective function evaluation
    eoClusteringEvalWithStatistics<eoChromosome> eval(numEvalCounter);
    // Evaluate population
    for (int i = 0; i < solutionPop.size(); ++i) {
        eval(*solutionPop[i].get());
    }
    int popSize = solutionPop.size();

//    cout << "[In runCellularEA method]" << endl;
//    cout << "pop size = " << popSize <<  endl;
//    /////////////////////////////////////////////////////////////////////////////////
//    // Print population information to output and
//    std::cout << std::endl << "Initial population" << std::endl;
//    int k = 0;
//    for (int k = 0; k < popSize; ++k) {
//        std::cout << (*solutionPop[k].get()).fitness() << "\t";
//    }
//    cin.get();
    /////////////////////////////////////////////////////////////////////////////////


    //
    // Build CellularGA
    //
    // Terminate after concluding L time loops or 'Ctrl+C' signal is received
    // The eoGenerationContinuePopVector object, instead of using an eoPop to represent the population,
    // uses a vector. A vector is used in order to swap offspring and population efficiently using pointers
    eoGenerationContinuePopVector<eoChromosome> terminator(L);

    // Declare 1-selectors
    //
    // Binary deterministic tournament selector used in neighbour selection
    // Work with pointers for efficiency
    eoDetTournamentSelectSharedPtr<eoChromosome> detSelectNeighbourPtr;

    // Crossover and mutation
    Mutation<eoChromosome> mutation;
    Crossover<eoChromosome> crossover;

//    boost::shared_ptr<eoCellularEA<eoChromosome> > cGA;
    // Build the corresponding type of cGA object depending on the layout (Ring or Matrix)
//    if (NLINES == 1) { // Ring cGA
//    eoCellularEARing<eoChromosome> cEA(checkpoint, eval, detSelectNeighbour,
//                      crossover, mutation,
//                     selectBestOne,
//                     selectBestOne
//                    );
//    }
//    else {
        cGA = boost::make_shared<eoCellularEAMatrix<eoChromosome> >(
                    outFile, NLINES, NCOLS,
                    cp, mp,
                    terminator,
                    eval,
                    detSelectNeighbourPtr, // Work with pointers for efficiency
                    crossover, mutation,
                    numEvalCounter
        );

//    }

        // Run the algorithm
        (*cGA.get())(solutionPopPtr);

        // Write best solution to file
        eoChromosome const *bestSolution = (*cGA.get()).getBestSolution();

        return *bestSolution; // Create a copy of the chromosome because the container (cGA) is local to this method
}









/**
 * @brief classify
 * @param _outputDir
 * @param _bestSolution
 * @param _testData
 * @return
 */
double Main::classify(const string &_outputDir, eoChromosome const &_bestSolution, ProblemData &_testData) {
    //
    // Compute CEP (Classification Error Percentage) which is the percentage of incorrectly classified
    // patterns of the test data sets, given by
    // CEP = (number of misclassified samples in the test set / total size of test data set) × 100.
    // The classification of each pattern is done by assigning it to
    // the class whose distance is closest to the center of the clusters.
    // Then, the classified output is compared with the desired output
    // and if they are not exactly the same, the pattern is separated as
    // misclassified.

    // Get data points ground thruth clusters
    vector<int> const &dataPointsClusters = _testData.getSamplesClasses();

    // Get data points assigned clusters
    vector<int> dataPointsAssignedClusters(_testData.getNumSamples());
    assignToClosestCluster(dataPointsAssignedClusters, _bestSolution.getClusterCentres(), _testData);

    int numMisclassified = 0;
    // Print cluster index
    for (int i = 0; i < _testData.getNumSamples(); ++i) {
        // x_i cluster index
        int cluster_x_i_index = dataPointsClusters[i];
        numMisclassified += (dataPointsAssignedClusters[i] != cluster_x_i_index ? 1 : 0);

//        cout << "i = " << i << ", assigned Cluster = " << dataPointsAssignedClusters[i] <<
//                ", real cluster = " << cluster_x_i_index << endl;
    }

    // Dataset size
    int sizeDataset = _testData.getNumSamples();
    // CEP
    double cep = ((double)numMisclassified / sizeDataset) * 100;

    cout << "numMisclassified = " << numMisclassified << ", sizeDataset = " << sizeDataset << endl;
    cout << "CEP = " << cep << endl;
//    cout << cep << endl;
    return cep;
}






/**
 * @brief assignToClosestCluster
 * @param _dataPointsAssignedClusters
 * @param _clusterCentres
 * @param _testData
 */
void Main::assignToClosestCluster(vector<int> &_dataPointsAssignedClusters,
                            vector<double> const &_clusterCentres,
                            ProblemData &_testData) {

///
/// TODO - OPTIMIZE BY DEFINING THE CLUSTER CENTRES USING ARMA::VEC
///
    /////////////////////////////////////////////////////////////
    // Convert clusters to arma::vec
    vector<vector<arma::vec>> clusters(_testData.getNumClasses());

    for (int k = 0; k < K_CLUSTERS; ++k) {
        for (int i = 0; i < _testData.getNumClasses(); ++i) {
            // vec to hold the i cluster
            arma::vec cluster_i(_testData.getNumFeaturesPerSample());
            // Copy cluster center
            for (int j = 0; j < _testData.getNumFeaturesPerSample(); ++j) {
                cluster_i(j) = _clusterCentres[k*_testData.getNumClasses()*_testData.getNumFeaturesPerSample() +
                        i*_testData.getNumFeaturesPerSample()+j];
            }
            // Add cluster
            clusters[i].push_back(cluster_i); // Push_back for each k
        }
    }
    /////////////////////////////////////////////////////////////


//    // Convert clusters to arma::vec
//    vector<arma::vec> clusters(_testData.getNumClasses());
//    for (int i = 0; i < _testData.getNumClasses(); ++i) {
//        // vec to hold the i cluster
//        arma::vec cluster_i(_testData.getNumFeaturesPerSample());
//        // Copy cluster center
//        for (int j = 0; j < _testData.getNumFeaturesPerSample(); ++j) {
//            cluster_i(j) = _clusterCentres[i*_testData.getNumFeaturesPerSample()+j];
//        }
//        // Add cluster
//        clusters[i] = cluster_i;
//    }
    // Get samples x_i
    vector<arma::vec> const &samples = _testData.getSamples();

    // Assign each point x_i, i = 1, 2, ..., n to one of the clusters C_j with centre
    // z_j such that norm2(x_i - z_j) < norm2(x_i - z_p), p = 1, 2, ..., K,
    // and p neq j. All ties are resolved arbitrarily.
    for (int i = 0; i < _testData.getNumSamples(); ++i) {
        // Get point xi
        arma::vec const &xi = samples[i];

#ifdef MAINAPP_DEBUG
//        cout << "xi = " << xi << endl;
#endif
        // Minimum norm value and cluster index
        double min_norm = -1;
        int cluster_idx = -1;
        int cluster_idx_k = -1; /// ADD
        for (int j = 0; j < _testData.getNumClasses(); ++j) {
            for (int k = 0; k < K_CLUSTERS; ++k) { /// ADD
                // Get cluster zj
                arma::vec &zj = clusters[j][k];
//            // Get cluster zj
//            arma::vec &zj = clusters[j];
#ifdef MAINAPP_DEBUG
//        cout << "zj = " << zj << endl;
//        cin.get();
#endif
                /////////////////////////////////////////////////////////////////////////
                // Compute l2-norm
                double calc_norm = norm(xi - zj, 2);

                /////////////////////////////////////////////////////////////////////////

                if (min_norm == -1) {
                    // l2-norm
                    min_norm = calc_norm;
                    // Cluster index
                    cluster_idx = j;
                    //
                    cluster_idx_k = k;
                }
                else if (calc_norm <= min_norm) {
                    bool apply = true;
                    // All ties are resolved arbitrarily.
                    // We flip a coin and if prob < 0.5 we don't set xi to this cluster.
                    if (calc_norm == min_norm) {
                        if (rng.flip() < 0.5)
                            apply = false;
                    }
                    if (apply) {
                        // l2-norm
                        min_norm = calc_norm;
                        // Cluster index
                        cluster_idx = j;
                        //
                        cluster_idx_k = k;
                    }
                }
            }
        }
        // Assign xi to cluster zj
        _dataPointsAssignedClusters[i] = cluster_idx;
        // 0% error - testing
//        _dataPointsAssignedClusters[i] = _testData.getSamplesClasses()[i];
    }
}





/**
 * @brief Main::buildPop
 * @param _trainData
 */
void Main::buildPop(ProblemData const &_trainData) {

    ////// Pop size = 100  ////////
    // Rect
    const int NLINES = 5;
    const int NCOLS = 20;

    const int POP_SIZE = NLINES*NCOLS;  // Population size

    // Solution initializer
    eoClusteringInit<eoChromosome> init(_trainData);
    // Generate initial population
    // We can't work with eoPop of shared_ptr because shared_ptr is not an EO
    // Solution: Work with vector<shared_ptr<EOT> > directly
    pop = boost::make_shared<vector<boost::shared_ptr<eoChromosome> > >();
    auto &solutionPop = *pop.get();
    for (int i = 0; i < POP_SIZE; ++i) {
        // Create solution object and insert it in the vector
        solutionPop.push_back(boost::make_shared<eoChromosome>());
        // Initialize chromosome
        init((*solutionPop.back().get()));
    }
    // # evaluations counter
    eoNumberEvalsCounter numEvalCounter;
     // Objective function evaluation
    //    eoETTPEval<eoChromosome> eval;
    // Objective function evaluation
    eoClusteringEvalWithStatistics<eoChromosome> eval(numEvalCounter);
    // Evaluate population
    for (int i = 0; i < solutionPop.size(); ++i) {
        eval(*solutionPop[i].get());
    }
}






/*
void Main::buildCellularEA(const string &_outputDir,
                     TestSet const& _testSet,
                     ProblemData const &_trainData) {
    //
    // cEA parameters
    //

    // Matrix
//    const int NLINES = 50;
//    const int NCOLS = 50;

//    const int NLINES = 100;
//    const int NCOLS = 1000;

    ////// Pop size = 16  ////////
    // Rect
//    const int NLINES = 2;
//    const int NCOLS = 8;
    // Matrix
//    const int NLINES = 4;
//    const int NCOLS = 4;
//    // Ring
//    const int NLINES = 1;
//    const int NCOLS = 16;
    //////////////////////////////

    ////// Pop size = 36  ////////
    // Rect
//    const int NLINES = 3;
//    const int NCOLS = 12;
    // Matrix
//    const int NLINES = 6;
//    const int NCOLS = 6;
    // Ring
//    const int NLINES = 1;
//    const int NCOLS = 36;
    //////////////////////////////

    ////// Pop size = 64  ////////
    // Rect
//    const int NLINES = 2;
//    const int NCOLS = 32;
    // Matrix
//    const int NLINES = 8;
//    const int NCOLS = 8;
    // Ring
//    const int NLINES = 1;
//    const int NCOLS = 64;
    //////////////////////////////

    ////// Pop size = 100  ////////
//    // Rect
    const int NLINES = 5;
    const int NCOLS = 20;

//    // Matrix
//    const int NLINES = 10;
//    const int NCOLS = 10;
//    // Ring
//    const int NLINES = 1;
//    const int NCOLS = 100;
    //////////////////////////////

    ////// Pop size = 225  ////////
    // Rect
//    const int NLINES = 9;
//    const int NCOLS = 25;
//    // Matrix
//    const int NLINES = 15;
//    const int NCOLS = 15;

    // Ring
//    const int NLINES = 1;
//    const int NCOLS = 225;
    //////////////////////////////

//    const int NLINES = 10;
//    const int NCOLS = 100;

//    const int NLINES = 50;
//    const int NCOLS = 200;


    const int POP_SIZE = NLINES*NCOLS;  // Population size
//    const int L = 5000000; // Number of generations
//    const int L = 2500; // Number of generations
//    const int L = 500; // Number of generations
    const int L = 1000; // Number of generations


    // Crossover probability
//    const double cp = 1;
//        const double cp = 0.8;
    const double cp = 0.6;
//    const double cp = 0.4; // TEST
//    const double cp = 0.15;
//    const double cp = 0;

    // Mutation probability
//    const double mp = 1;
//        const double mp = 0.5;
//    const double mp = 0.2; // 20%
    const double mp = 0.1; // 10%
//    const double mp = 0.01; // 1%
//    const double mp = 0.001; // 0.1%
//    const double mp = 0; // TEST

    // Creating the output file in the specified output directory
    stringstream sstream;
    sstream << _outputDir << _testSet.getDescription().getName() << "_NLINES_" << NLINES << "_NCOLS_" << NCOLS
            << "_cp_" << cp << "_mp_" << mp << ".txt";

    string filename;
    sstream >> filename;
    ofstream outFile(filename);

    auto &solutionPop = *pop.get();

    // # evaluations counter
    eoNumberEvalsCounter numEvalCounter;
    // Objective function evaluation
    eoClusteringEvalWithStatistics<eoChromosome> eval(numEvalCounter);
    int popSize = solutionPop.size();

    cout << "[In buildCellularEA method]" << endl;
    cout << "pop size = " << popSize <<  endl;
    /////////////////////////////////////////////////////////////////////////////////
    // Print population information to output and
    std::cout << std::endl << "Initial population" << std::endl;
    int k = 0;
    for (int k = 0; k < popSize; ++k) {
        std::cout << (*solutionPop[k].get()).fitness() << "\t";
    }
    /////////////////////////////////////////////////////////////////////////////////


    //
    // Build CellularGA
    //
    // Terminate after concluding L time loops or 'Ctrl+C' signal is received
    // The eoGenerationContinuePopVector object, instead of using an eoPop to represent the population,
    // uses a vector. A vector is used in order to swap offspring and population efficiently using pointers
    eoGenerationContinuePopVector<eoChromosome> terminator(L);

    // Declare 1-selectors
    //
    // Binary deterministic tournament selector used in neighbour selection
    // Work with pointers for efficiency
    eoDetTournamentSelectSharedPtr<eoChromosome> detSelectNeighbourPtr;

    // Crossover and mutation
    Mutation<eoChromosome> mutation;
    Crossover<eoChromosome> crossover;

//    boost::shared_ptr<eoCellularEA<eoChromosome> > cGA;
    // Build the corresponding type of cGA object depending on the layout (Ring or Matrix)
//    if (NLINES == 1) { // Ring cGA
//    eoCellularEARing<eoChromosome> cEA(checkpoint, eval, detSelectNeighbour,
//                      crossover, mutation,
//                     selectBestOne,
//                     selectBestOne
//                    );
//    }
//    else {
        cGA = boost::make_shared<eoCellularEAMatrix<eoChromosome> >(
                    outFile, NLINES, NCOLS,
                    cp, mp,
                    terminator,
                    eval,
                    detSelectNeighbourPtr, // Work with pointers for efficiency
                    crossover, mutation,
                    numEvalCounter
        );

//    }

}
*/






/**
 * @brief Main::runCellularEAIteration Runs a single iteration (configured in cGA object) of cellular GA
 * @return
 */
eoChromosome Main::runCellularEAIteration()
{
    vector<boost::shared_ptr<eoChromosome> > &p = *pop.get();
    int popSize = p.size();

    // Run the algorithm
    (*cGA.get())(pop);

    // Write best solution to file
    eoChromosome const *bestSolution = (*cGA.get()).getBestSolution();

    return *bestSolution; // Create a copy of the chromosome because the container (cGA) is local to this method
}




