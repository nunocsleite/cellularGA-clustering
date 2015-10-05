#ifndef EOCLUSTERINGINIT_H
#define EOCLUSTERINGINIT_H

#include <eoInit.h>
#include <iostream>
#include "data/ProblemData.hpp"
#include <utils/eoRNG.h>

#include "eoChromosome.h"

// For debugging purposes
//#define EOINIT_DEBUG


// Example 1. Let N=2 and K=3, i.e., the space is two-
// dimensional and the number of clusters being considered
// is three. Then the chromosome
//
// 51.6 72.3 18.3 15.7 29.1 32.2
//
// represents the three cluster centres (51.6, 72.3), (18.3, 15.7)
// and (29.1, 32.2). Note that each real number in the chro-
// mosome is an indivisible gene.
//
// Population initialization
// The K cluster centres encoded in each chromosome
// are initialized to K randomly chosen points from the
// data set. This process is repeated for each of the P chro-
// mosomes in the population, where P is the size of the
// population.
//
template <typename EOT>
class eoClusteringInit : public eoInit<EOT> {

public:
    eoClusteringInit(ProblemData const& _problemData)
        : problemData(_problemData) { }

    /**
     * @brief operator ()
     * @param _chrom
     */
    virtual void operator()(EOT& _chrom) {
#ifdef EOINIT_DEBUG
        std::cout << "EOInit::operator(chrom)" << std::endl;
#endif
        // Initialise chromosome
        _chrom.init(&problemData);
        // Generate a random solution.
        // The K cluster centres encoded in each chromosome
        // are initialized to K randomly chosen points from the
        // data set.
        // # clusters
        int numClusters = _chrom.getNumClasses();
        // # features
        int numFeatures = _chrom.getNumFeatures();
        int randSampleIdx, k;
        // Get cluster centres
        std::vector<double> &clusterCentres = _chrom.getClusterCentres();
        // Get samples vector
        const std::vector<arma::vec> &samples = _chrom.getSamples();

        for (int i = 0; i < numClusters; ++i) {
            // Generate a random sample index
            randSampleIdx = rng.uniform(_chrom.getNumSamples());
#ifdef EOINIT_DEBUG
            std::cout << "randSampleIdx = " << randSampleIdx << std::endl;
            std::cout << "samples[randSampleIdx] = " << samples[randSampleIdx] << std::endl;
#endif
            // Copy the sample to the current cluster center
//            for (int j = 0; j < numFeatures; ++j)
//                clusterCentres[i*numFeatures + j] = samples[randSampleIdx][j];

//            for (int j = 0; j < numFeatures; ++j)
//                clusterCentres[i*numFeatures + j] = rng.uniform();
//                clusterCentres[i*numFeatures + j] = rng.uniform()*5; // LAST
//                clusterCentres[i*numFeatures + j] = 0;//rng.uniform()*100;
//                clusterCentres[i*numFeatures + j] = rng.uniform()*200 - 100;
            for (int k = 0; k < K_CLUSTERS; ++k) {
                for (int j = 0; j < numFeatures; ++j) {
//                    clusterCentres[k*numFeatures*numClusters + i*numFeatures + j] = rng.uniform()*5;
                    clusterCentres[k*numFeatures*numClusters + i*numFeatures + j] = rng.uniform();
//                      clusterCentres[k*numFeatures*numClusters + i*numFeatures + j] = samples[randSampleIdx][j];

//                    std::cout << clusterCentres[k*numFeatures*numClusters + i*numFeatures + j] << std::endl;
                }
            }
        }

#ifdef EOINIT_DEBUG
    std::cout << "numClusters = " << numClusters << std::endl;
    std::cout << "numFeatures = " << numFeatures << std::endl;
    for (int i = 0; i < clusterCentres.size(); ++i)
        std::cout << clusterCentres[i] << " ";
    std::cout << std::endl;
    std::cin.get();
#endif
    }


private:
    /**
     * @brief problemData Problem data info
     */
    ProblemData const& problemData;
};



#endif // EOCLUSTERINGINIT_H
