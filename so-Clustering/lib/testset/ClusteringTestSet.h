#ifndef CLUSTERINGTESTSET_H
#define CLUSTERINGTESTSET_H

#include <iostream>
#include <fstream>
#include <vector>
#include <iterator>
#include <string>
#include <sstream>
#include <boost/tokenizer.hpp>
#include <boost/unordered_map.hpp>
#include "testset/TestSet.h"
#include "testset/TestSetDescription.h"
#include <boost/unordered_map.hpp>



/////////////////////////////
// Clustering Dataset
//
/////////////////////////////
class ClusteringTestSet : public TestSet {

public:
    /**
     * @brief ClusteringTestSet Constructor
     * @param _testSetName
     * @param _description
     * @param _rootDir
     */
    ClusteringTestSet(TestSetDescription const& _testSetDescription, std::string _rootDir)
        : TestSet(_testSetDescription, _rootDir, boost::shared_ptr<ProblemData>(new ProblemData()))
    {

#ifdef CLUSTERING_TESTSET_DEBUG
        std::cout << "ClusteringTestSet ctor" << std::endl;
#endif

        // After creating the instance, the caller should invoke the load method
    }


public:
    // Overriden method
    virtual void load() override;



    /**
     * @brief shuffleTestSet Shuffle test set samples
     * @param _auxSamples
     * @param _auxSamplesClasses
     * @param _samples
     * @param _samplesClasses
     */
    static void shuffleTestSet(std::vector<arma::vec> const &_auxSamples,
                        std::vector<int> const &_auxSamplesClasses,
                        std::vector<arma::vec> &_samples,
                        std::vector<int> &_samplesClasses);

protected:
    /**
     * Protected members
     */
    void readFileContents();

    inline bool isInteger(const std::string & s)
    {
//        std::cout << "isInteger: string s = " << s << ", isdigit(s[0]) = " << isdigit(s[0]) << std::endl;
        if (s.empty() || ((!isdigit(s[0])) && (s[0] != '-') && (s[0] != '+')))
            return false ;

        char * p ;
        strtol(s.c_str(), &p, 10) ;

        return (*p == 0) ;
    }

    inline bool isFloat(const std::string & s)
    {
//        std::cout << "isFloat: string s = " << s << ", isdigit(s[0]) = " << isdigit(s[0]) << std::endl;
        if (s.empty() || ((!isdigit(s[0])) && (s[0] != '-') && (s[0] != '+') && (s[0] != '.')))
            return false;

//        std::cout << s << std::endl;

        char * p ;
        strtof(s.c_str(), &p) ;

        return (*p == 0) ;
    }

    inline bool isNumeric(const std::string & s) {
//        std::cout << "isNumeric" << std::endl;
        return isInteger(s) || isFloat(s);
    }

    /**
     * @brief fillAttributeMissingValues
     * @param _samples
     */
    void fillAttributeMissingValues(std::vector<arma::vec> &_samples);
    /**
     * @brief getAttrMeanValue
     * @param _samples
     * @param _missAttrIdx
     * @return
     */
    double getAttrMeanValue(const std::vector<arma::vec> &_samples, int _missAttrIdx);


    //
    // Fields
    //
    /**
     * @brief missingAttributeInfo
     */
    std::vector<std::pair<int, int> > missingAttributeInfo;
    /**
     * @brief missingAttributeHashtable Hash table containing the missing attributes values and average value
     */
    boost::unordered_map<int, double> missingAttributeHashtable;
};



#endif // CLUSTERINGTESTSET_H




