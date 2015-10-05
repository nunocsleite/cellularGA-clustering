#ifndef PROBLEMDATA_H
#define PROBLEMDATA_H

#include <vector>
#include <string>
#include <iostream>
#include <boost/shared_ptr.hpp>
#include <boost/unordered_map.hpp>
#include <armadillo>
#include <boost/make_shared.hpp>
#include "Attribute.h"


class ProblemData {

public:
    // Constructors
    ProblemData() { }

    // Public interface
    //
    friend std::ostream& operator<<(std::ostream& _os, const ProblemData& _problemData);

    // Getters & setters

//    std::vector<arma::vec> &getSamples();

    std::vector<arma::vec> const &getSamples() const;
    void setSamples(const boost::shared_ptr<std::vector<arma::vec> > &value);

    std::vector<int> const &getSamplesClasses() const;
    void setSamplesClasses(const boost::shared_ptr<std::vector<int> > &value);

    std::vector<std::string> const &getClassLabels() const;
    void setClassLabels(const boost::shared_ptr<std::vector<std::string> > &value);

    boost::unordered_map<std::string, int> const &getClassMap() const;
    void setClassMap(const boost::shared_ptr<boost::unordered_map<std::string, int> > &value);

    int getNumFeaturesPerSample() const;
    void setNumFeaturesPerSample(int value);

    int getNumSamples() const;
    void setNumSamples(int value);

    int getNumClasses() const;
    void setNumClasses(int value);

    std::vector<Attribute> const &getAttributes() const;
    std::vector<Attribute> &getAttributes();
//    void setAttributes(const boost::shared_ptr<std::vector<Attribute> > &value);
    void setAttributes(const std::vector<Attribute> &value);

private:
    //--
    // Fields
    //--
    // Vector containing the samples (each sample is represented by a arma::vec)
    boost::shared_ptr<std::vector<arma::vec>> ptrSamples;
    // Classes
    boost::shared_ptr<std::vector<int> > ptrSamplesClasses;
    // Class labels
    boost::shared_ptr<std::vector<std::string> > ptrClassLabels;
    // Class map class label -> class index
    boost::shared_ptr<boost::unordered_map<std::string,int> > ptrClassMap;
    // # features
    int numFeaturesPerSample;
    // # samples
    int numSamples;
    // # distinct classes
    int numClasses;
    // Attribute info
//    boost::shared_ptr<std::vector<Attribute>> ptrAttributes;
    std::vector<Attribute> attributes;
};


//// ADDED 10/AUG/2015
//inline std::vector<arma::vec> &ProblemData::getSamples()
//{
//    return *ptrSamples.get();
//}

inline std::vector<arma::vec> const &ProblemData::getSamples() const
{
    return *ptrSamples.get();
}
inline void ProblemData::setSamples(const boost::shared_ptr<std::vector<arma::vec> > &value)
{
    ptrSamples = value;
}

inline std::vector<int> const &ProblemData::getSamplesClasses() const
{
    return *ptrSamplesClasses.get();
}
inline void ProblemData::setSamplesClasses(const boost::shared_ptr<std::vector<int> > &value)
{
    ptrSamplesClasses = value;
}

inline std::vector<std::string> const &ProblemData::getClassLabels() const
{
    return *ptrClassLabels.get();
}
inline void ProblemData::setClassLabels(const boost::shared_ptr<std::vector<std::string> > &value)
{
    ptrClassLabels = value;
}

inline boost::unordered_map<std::string, int> const &ProblemData::getClassMap() const
{
    return *ptrClassMap.get();
}
inline void ProblemData::setClassMap(const boost::shared_ptr<boost::unordered_map<std::string, int> > &value)
{
    ptrClassMap = value;
}

inline int ProblemData::getNumFeaturesPerSample() const
{
    return numFeaturesPerSample;
}
inline void ProblemData::setNumFeaturesPerSample(int value)
{
    numFeaturesPerSample = value;
}

inline int ProblemData::getNumSamples() const
{
    return numSamples;
}
inline void ProblemData::setNumSamples(int value)
{
    numSamples = value;
}

inline int ProblemData::getNumClasses() const
{
    return numClasses;
}
inline void ProblemData::setNumClasses(int value)
{
    numClasses = value;
}

inline std::vector<Attribute> const &ProblemData::getAttributes() const
{
    return attributes;
}
inline std::vector<Attribute> &ProblemData::getAttributes()
{
    return attributes;
}

inline void ProblemData::setAttributes(const std::vector<Attribute> &value)
{
    attributes = value;
}

//inline void ProblemData::setAttributes(const boost::shared_ptr<std::vector<Attribute> > &value)
//{
//    ptrAttributes = value;
//}


#endif // PROBLEMDATA_H





