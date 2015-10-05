#ifndef EOCHROMOSOME_H
#define EOCHROMOSOME_H

#include <EO.h>

#include "data/ProblemData.hpp"
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <vector>
#include <string>
#include <ostream>
#include <armadillo>


#define EOCHROMOSOME_DEBUG_COPY_CTOR


#define K_CLUSTERS 1
//#define K_CLUSTERS 3
//#define K_CLUSTERS 4


// EO (Evolving Object) chromosome implementation.
// A chromosome encodes a sequence of real numbers representing
// the K cluster centres. For an N-dimensional space, the
// length of a chromosome is N*K words, where the first
// N positions (or, genes) represent the N dimensions of the
// first cluster centre, the next N positions represent those of
// the second cluster centre, and so on.
////
class eoChromosome : public EO<double> {

public:
    /**
        * @brief Chromosome Default chromosome constructor
        */
       eoChromosome()
           :
             clusterCentres(boost::make_shared<std::vector<double>>()),
             problemData(nullptr) { }

       /**
        * @brief eoChromosome Copy constructor
        * @param _chrom
        */
       eoChromosome(const eoChromosome &_chrom)
           : clusterCentres(boost::make_shared<std::vector<double>>(
                        _chrom.getNumFeatures()*_chrom.getNumClasses()*K_CLUSTERS)),
             problemData(_chrom.getProblemData()) {

           // Copy timetable data
           copyData(_chrom);

           // Set fitness
           fitness(_chrom.fitness());

#ifdef EOCHROMOSOME_DEBUG_COPY_CTOR
//            std::cout << "eoChromosome::Copy ctor" << std::endl;
#endif
       }

       /**
        * @brief operator =
        * @param _chrom
        * @return
        */
       eoChromosome& operator=(const eoChromosome &_chrom) {
#ifdef EOCHROMOSOME_DEBUG_COPY_CTOR
//           std::cout << "eoChromosome::operator=" << std::endl;
//           std::cin.get();
#endif

           if (&_chrom != this) {
//               clusterCentres = boost::make_shared<std::vector<double>>(
//                                       _chrom.getNumFeatures()*_chrom.getNumClasses());
               clusterCentres = boost::make_shared<std::vector<double>>(
                                       _chrom.getNumFeatures()*_chrom.getNumClasses()*K_CLUSTERS);

               problemData = _chrom.getProblemData();
               // Copy data
               copyData(_chrom);
               // Set fitness
               fitness(_chrom.fitness());
           }
           return *this;
       }

   protected:
       // Copy timetable data
       void copyData(const eoChromosome &_chrom) {
           // Copy cluster centres
           *clusterCentres.get() = _chrom.getClusterCentres();
           // Copy points clusters
       }

public:

    /**
     * @brief init Initialise the chromosome
     * @param _problemData Timetable problem data
     */
    void init(ProblemData const* _problemData);

    ////////// ProblemData methods ////////////////////////////////////////////////
    /**
     * @brief getNumFeatures
     * @return
     */
    inline int getNumFeatures() const { return problemData->getNumFeaturesPerSample(); }

    /**
     * @brief getNumSamples
     * @return
     */
    int getNumSamples() const { return problemData->getNumSamples(); }

    /**
     * @brief getSamples
     * @return
     */
    const std::vector<arma::vec> &getSamples() const { return problemData->getSamples(); }

    /**
     * @brief getClassesLabels
     * @return
     */
    const std::vector<std::string> &getClassLabels() const { return problemData->getClassLabels(); }

    /**
     * @brief getNumClasses
     * @return
     */
    int getNumClasses() const { return problemData->getNumClasses(); }

    /**
     * @brief getSamplesClasses
     * @return
     */
    const std::vector<int> &getSamplesClasses() const { return problemData->getSamplesClasses(); }

    ///////////////////////////////////////////////////////////////////////////////////////////////

    ////////// Debug methods /////////////////////////////////////////////////////////////////////
    /**
     * @brief validate Validate a chromosome solution
     */
    void validate() const;
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * @brief className +Overriden method+
     * @return The class name
     */
    std::string className() const override { return "EO Chromosome"; }
    /**
     * @brief operator << Print chromosome contents
     * @param os
     * @param timetable
     * @return
     */
    friend std::ostream& operator<<(std::ostream& _os, const eoChromosome& _chrom);

    void printToFile(std::ostream &_os) const;

    inline const ProblemData *getProblemData() const { return problemData; }
    inline void setProblemData(const ProblemData *value) { problemData = value; }

    // Const version
    std::vector<double> const &getClusterCentres() const { return *clusterCentres.get(); }
    // Non-const version
    std::vector<double> &getClusterCentres() { return *clusterCentres.get(); }

    /**
     * @brief computeFitness
     * @return  The computed fitness
     */
    double computeFitness();

    double fitness1();
    double fitness2();
    double fitness3();
//    double fitness4();

    void assignToClosestCluster(std::vector<int> &_dataPointsAssignedClusters) const;

    /**
     * @brief length
     * @return Chromosome length
     */
    inline int length() const { return clusterCentres.get()->size(); }

protected:
    /**
     * @brief clusterCentres String containing the representation of the cluster centres
     */
    boost::shared_ptr<std::vector<double>> clusterCentres;

    /**
     * @brief timetableProblemData The problem data
     */
    ProblemData const* problemData;
};





#endif // EOCHROMOSOME_H
