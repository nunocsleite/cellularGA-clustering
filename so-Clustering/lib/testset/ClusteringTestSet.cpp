
#include "testset/ClusteringTestSet.h"
#include <iostream>
#include <random>
#include <boost/make_shared.hpp>
#include <string>
#include "data/Attribute.h"
#include <utils/eoRNG.h>

using namespace std;
using namespace boost;


// For debugging purposes
//#define CLUSTERING_TESTSET_DEBUG
//#define CLUSTERING_TESTSET_DEBUG1


///////////////////////////////////////////////////////////////////////////////
//  Helper function reading a file into a string
///////////////////////////////////////////////////////////////////////////////
std::string read_from_file(char const* infile)
{
    std::ifstream instream(infile);
    if (!instream.is_open()) {
        std::cerr << "Couldn't open file: " << infile << std::endl;
        exit(-1);
    }
    instream.unsetf(std::ios::skipws);      // No white space skipping!
    return std::string(std::istreambuf_iterator<char>(instream.rdbuf()),
                       std::istreambuf_iterator<char>());
}




////////////////////////////////////////////////////////////
// Clustering Dataset Methods
//
////////////////////////////////////////////////////////////

/**
 * @brief ClusteringTestSet::load Overriden method
 * @param _testSetName
 * @param _rootDir
 */
void ClusteringTestSet::load()
{

#ifdef CLUSTERING_TESTSET_DEBUG
    cout << "ClusteringTestSet::load" << endl;
#endif

    readFileContents();

}



void ClusteringTestSet::readFileContents() {
    // Directory name
    std::string dirName = this->getDescription().getDirName();
    // Data filename
    std::string dataFilename = this->getDescription().getDataFilename();
    // Full path filename
    string filename = this->getRootDirectory() + "/" + dirName + "/" + dataFilename + ".data";

#ifdef CLUSTERING_TESTSET_DEBUG
    cout << filename << endl;
#endif
    // Read whole file into memory, for fast processing
    string wholeFile = read_from_file(filename.c_str());
    int numLines = 0;
    double feature;
    /// Feature related //////////////////
    // Vector containing the samples (each sample is represented by a arma::vec)
    boost::shared_ptr<std::vector<arma::vec>> ptrSamples = boost::make_shared<std::vector<arma::vec>>();
    // Classes
    boost::shared_ptr<std::vector<int> > ptrSamplesClasses = boost::make_shared<std::vector<int> >();
    // Class labels
    boost::shared_ptr<std::vector<std::string> > ptrClassLabels = boost::make_shared<std::vector<std::string> >();
    // Map of class label -> class index
    boost::shared_ptr<boost::unordered_map<std::string,int> > ptrClassMap =
            boost::make_shared<boost::unordered_map<std::string,int> >();
    // # features + class
    int numFeaturesAndClass = this->getDescription().getNumAttributes();
    // # features
    int numFeaturesPerSample = numFeaturesAndClass-1;
    // # samples
    int numSamples;
    // # distinct classes
    int numClasses;
    // String representing the sample class
    string sampleClass;
    // Class index
    int classIndex = 0;
    // Attribute class index (zero based index)
    int attributeClassIndex = this->getDescription().getClassIndex()-1;
    /////////////////////////////////////
    // Auxiliary variables
    //
    // Vector containing the samples (each sample is represented by a arma::vec)
    std::vector<arma::vec> &samples = *ptrSamples.get();
    // Classes
    std::vector<int> &samplesClasses = *ptrSamplesClasses.get();
    // Class labels
    std::vector<std::string> &classLabels = *ptrClassLabels.get();
    // Class map class label -> class index
    boost::unordered_map<std::string,int> &classMap = *ptrClassMap.get();
    // Auxiliary vectors used for shuffling the data
    // Vector containing the samples (each sample is represented by a arma::vec)
    std::vector<arma::vec> auxSamples;
    // Classes
    std::vector<int> auxSamplesClasses;
    // Attributes
    std::vector<Attribute> &attributes = this->getProblemData().getAttributes();
    // Resize attributes vector
    attributes.resize(numFeaturesPerSample);
//    cout << "attributes.size() = " << attributes.size() << endl;
//    for (auto & att : attributes)
//        cout << "att.min() = " << att.min() << ", att.max() = " << att.max() << endl;
//    cin.get();
    /////////////////////////////////////

    // Missing values flag
    bool missingValues = false;

    // Process whole file escaping '\n'
    typedef boost::tokenizer<boost::char_separator<char> > CharTokenizer;
    boost::char_separator<char> sep(
      "\n", // dropped delimiters
      "",  // kept delimiters
      boost::/*drop_empty_tokens*/keep_empty_tokens); // empty token policy
    CharTokenizer tok(wholeFile, sep);
    for (CharTokenizer::iterator it = tok.begin(); it != tok.end(); ++it, ++numLines) {
        std::string line(*it);

        if (line.empty()) {
#ifdef CLUSTERING_TESTSET_DEBUG
        cout << "Empty line. Stopping..." << endl;
#endif
            break;
        }

#ifdef CLUSTERING_TESTSET_DEBUG
        cout << numLines << ": " << line << endl;
#endif
        // Get number tokenizer
        boost::tokenizer<escaped_list_separator<char> > numberTok(line, escaped_list_separator<char>('\\', ','));
        // Get number iterator
        boost::tokenizer<escaped_list_separator<char> >::iterator numberItr = numberTok.begin();
        // Process each line - extract attributes
        vector<string> tokens;
        for (; numberItr != numberTok.end(); ++numberItr) {
            tokens.push_back((*numberItr).c_str());
        }
        // Extract attributes and class for this sample.
        // The class is the last attribute
        // Save the sample in an arma::vec
        arma::vec sample(numFeaturesPerSample);
        for (int i = 0, k = 0; i < tokens.size(); ++i) {
            // Test if the feature is the class or not
            if (i != attributeClassIndex) {
                // Determine if attribute value is missing
                if (tokens[i] == "?") {
                    missingValues = true;
                    // Set default value
                    feature = 1234321.0;
                    // Insert missing attribute sample index into auxiliary vector
                    // We store a pair with the sample index (numLines), and the missing attribute index (k)
                    missingAttributeInfo.push_back(std::make_pair(numLines, k));
                }
                // Determine the feature type.
                // If it is not numeric (integer of double),
                // then it is nominal, string, or date
                else {
                    if (!isNumeric(tokens[i])) {
                        attributes[k].setType(Attribute::NOMINAL);
                        // Get feature value
                        feature = attributes[k].getIndex(tokens[i]);
                    }
                    else { // Is numeric
#ifdef CLUSTERING_TESTSET_DEBUG
//                cout << "tokens[i] = " << tokens[i] << endl;
//                cout << "Is numeric -> ";
#endif
                        // Attribute type is numeric
                        attributes[k].setType(Attribute::NUMERIC);
                        // Get feature value
                        feature = std::stof(tokens[i]);
                    }

                    // Update attribute's min and max values
                    attributes[k].updateMinMax(feature);
                }
#ifdef CLUSTERING_TESTSET_DEBUG
//            cout << feature << " ";
#endif
                // Save feature
                sample(k) = feature;
                ++k;
            }
        } // 2nd For
        // Push sample to samples vector
        auxSamples.push_back(sample);

#ifdef CLUSTERING_TESTSET_DEBUG
//        cout << "[ClusteringTestSet] Attribute info:"  << endl;
//        cout << "atts.size() = " << attributes.size() << endl;
//        for (auto & att : attributes)
//            cout << "att.min() = " << att.min() << ", att.max() = " << att.max() << endl;
//        cin.get();
#endif
        /////////////////////////////////////////////
        // Get sample class
        sampleClass = tokens[attributeClassIndex];

///////////////////////////////////////////////////////////////
///
/// TODO - CHANGE IN DATABASE DIRECTLY
///
        if (getDescription().getName() == "Heart") {
            // Values 1, 2, 3, and 4 belong to class 1. Value 0 belong to class 0.
            if (sampleClass == "2" || sampleClass == "3" || sampleClass == "4")
                sampleClass = "1";
        }
///
///////////////////////////////////////////////////////////////


        // Insert current class, given by the index 'numLines', in the class map
        // If the class doesn't exist yet, create entry <class label, class index>
        boost::unordered_map<string, int>::iterator mapIt = classMap.find(sampleClass);
        if (mapIt == classMap.end()) {
            // Add map entry
            classMap.insert(pair<string,int>(sampleClass, classIndex));
            // Add class label
            classLabels.push_back(sampleClass);
            // Increment class index
            classIndex++;
        }
        // Push class into the classes vector
        mapIt = classMap.find(sampleClass);
        if (mapIt == classMap.end())
            throw runtime_error("[Clustering Test set] class doesn't exist");
        else
            auxSamplesClasses.push_back((*mapIt).second);
        /////////////////////////////////////////////

#ifdef CLUSTERING_TESTSET_DEBUG
//        cout << " - class: " << sampleClass;
//        cout << endl;
//        cin.get();
#endif
    } // 1st For

    // # samples
    numSamples = numLines;
    // # distinct classes
    numClasses = classLabels.size();

    /////////////////////////////////////////////////////////
    //
    // Shuffle test set
    //
    // Resize samples and classes vectors
    samples.resize(auxSamples.size());
    samplesClasses.resize(auxSamplesClasses.size());
    // Perform shufling
    shuffleTestSet(auxSamples, auxSamplesClasses, samples, samplesClasses);
    /////////////////////////////////////////////////////////


    /////////////////////////////////////////////////////////
    if (missingValues) {
        // Fill attributes missing values
        fillAttributeMissingValues(samples);
    }



    /////////////////////////////////////////////////////////
    // Min-max normalization performs a linear transformation on the original data.
    // Suppose that mina and maxa are the minimum and the maximum values for attribute A.
    // Min-max normalization maps a value v of A-v in the range (0, 1) by computing:
    //
    // v' = (v-min) / (max - min)
    //
    // Normalize data
    for (int i = 0; i < samples.size(); ++i) {
        for (int j = 0; j < numFeaturesPerSample; ++j) {
            // Min-max Normalization
            samples[i](j) = (samples[i][j]-attributes[j].min()) /
                    (attributes[j].max() - attributes[j].min());

        }
    }

    /////////////////////////////////////////////////////////
    // Set fields in ProblemData object
    //
    // Set vector containing the samples (each sample is represented by a arma::vec)
    this->getProblemData().setSamples(ptrSamples);
    // Set classes
    this->getProblemData().setSamplesClasses(ptrSamplesClasses);
    // Set class labels
    this->getProblemData().setClassLabels(ptrClassLabels);
    // Set class map: class label -> class index
    this->getProblemData().setClassMap(ptrClassMap);
    // Set # features per sample
    this->getProblemData().setNumFeaturesPerSample(numFeaturesPerSample);
    // Set # samples
    this->getProblemData().setNumSamples(numSamples);
    // Set # classes
    this->getProblemData().setNumClasses(numClasses);


#ifdef CLUSTERING_TESTSET_DEBUG
    cout << "Attribute info:"  << endl;
    cout << "atts.size() = " << attributes.size() << endl;
    for (auto & att : attributes)
        cout << "att.min() = " << att.min() << ", att.max() = " << att.max() << endl;
    cin.get();

    for (int i = 0; i < samples.size(); ++i) {
        cout << "Sample # " << (i+1) << ": ";
        for (int j = 0; j < numFeaturesPerSample; ++j) {
            cout << samples[i][j] << " ";
        }
        cout << " - class label: " << classLabels[samplesClasses[i]] << ", class index: " << samplesClasses[i];
        cout << endl;
        cin.get();
    }

//    cout << *this->getProblemData() << endl;
//    cout << this->getProblemData() << endl;
#endif

}




/**
 * @brief ClusteringTestSet::shuffleTestSet
 * @param _auxSamples
 * @param _auxSamplesClasses
 * @param _samples
 * @param _samplesClasses
 */
void ClusteringTestSet::shuffleTestSet(std::vector<arma::vec> const &_auxSamples,
                                       std::vector<int> const &_auxSamplesClasses,
                                       std::vector<arma::vec> &_samples,
                                       std::vector<int> &_samplesClasses)
{
    // Auxiliary index vector used for shuffling
    vector<int> indexes(_auxSamples.size());
    for (int i = 0; i < indexes.size(); ++i)
        indexes[i] = i; // Fill with indexes

    ///
    /// COMMENT THIS LINE IF YOU WANT NO SHUFFLING OF DATA SAMPLES
    ///
    ///
    std::random_shuffle(indexes.begin(), indexes.end());


//    std::random_device rd;
//    std::mt19937 g(rd());
//    std::shuffle(indexes.begin(), indexes.end(), g);

    #ifdef CLUSTERING_TESTSET_DEBUG
    //    std::copy(indexes.begin(), indexes.end(), ostream_iterator<int>(cout, " "));
    //    cin.get();
    #endif

    // Fill samples and classes with random entries
    for (int i = 0; i < indexes.size(); ++i) {
        _samples[i] = _auxSamples[indexes[i]];
        _samplesClasses[i] = _auxSamplesClasses[indexes[i]];
    }
}




ostream& operator<<(ostream& _os, const TestSet& _t) {
//    _os << "Test set\t # features per sample\t # samples\t # classes" << endl;
//    _os << _t.getDescription().getName() << "\t "
//        << _t.getProblemData()->getNumFeaturesPerSample() << "\t "
//        << _t.getProblemData()->getNumSamples() << "\t "
//        << _t.getProblemData()->getNumClasses() << endl;

    _os << "Test set\t # features per sample\t # samples\t # classes" << endl;
    _os << _t.getDescription().getName() << "\t "
        << _t.getProblemData().getNumFeaturesPerSample() << "\t "
        << _t.getProblemData().getNumSamples() << "\t "
        << _t.getProblemData().getNumClasses() << endl;


    return _os;
}




/**
 * @brief fillAttributeMissingValues
 * @param _samples
 */
void ClusteringTestSet::fillAttributeMissingValues(std::vector<arma::vec> &_samples) {
    // Go over missingAttributeInfo vector and replace attribute value for its mean value
    for (int i = 0; i < missingAttributeInfo.size(); ++i) {
        // Get sample index
        int sampleIdx = missingAttributeInfo[i].first;
        // Get missing attribute index
        int missAttrIdx = missingAttributeInfo[i].second;
        double attrMeanVal = getAttrMeanValue(_samples, missAttrIdx);
        // Set sample missing attribute value
        _samples[sampleIdx](missAttrIdx) = attrMeanVal;

#ifdef CLUSTERING_TESTSET_DEBUG
        std::cout << "sampleIdx = " << sampleIdx << ", missAttrIdx = " << missAttrIdx
                  << ", attrMeanVal = " << attrMeanVal << std::endl;
        cin.get();
#endif
    }
}



/**
 * @brief ClusteringTestSet::getAttrMeanValue
 * @param _missAttrIdx
 * @return
 */
double ClusteringTestSet::getAttrMeanValue(std::vector<arma::vec> const &_samples, int _missAttrIdx) {
    double attrMeanVal = -1;
    // Insert missing attribute index in the class map.
    // If the attribute doesn't exist yet, create entry <attribute index, attribute info (pair)>
//    boost::unordered_map<int, double>::iterator mapIt = missingAttributeHashtable.find(_missAttrIdx);
//    if (mapIt == missingAttributeHashtable.end()) {
        // Attribute mean value was not computed yet. Compute it now.
        std::multiset<double> attValues;
        // Get attribute values
        for (int i = 0; i < _samples.size(); ++i) {
            double attVal = _samples[i][_missAttrIdx];
///
/// TODO - USE BOOL MATRIX
///
            // Test if it's not a missing value
            if (attVal != 1234321.0) {
                // If the attribute value doesn't exist in the set, insert it.
                if (attValues.find(attVal) == attValues.end())
                    attValues.insert(attVal);
            }
        }
#ifdef CLUSTERING_TESTSET_DEBUG1
        cout << "Attribute values:" << endl;
        for (double d : attValues)
            cout << d << " ";
        cout << endl;
#endif
        /////////////////
        // Compute attribute mean value
        double sum = 0.0;
        int count = 0;
        for (double const& val : attValues) {
            sum += val;
            ++count;
        }
        attrMeanVal = sum / count;
//        /////////////////
//        // Add map entry
//        missingAttributeHashtable.insert(pair<int,double>(_missAttrIdx, attrMeanVal));
//    }
//    else {
//        // Attribute exist, so return the mean value
//        attrMeanVal = mapIt->second;
//    }
    return attrMeanVal;
}















