#include "eoChromosome.h"
#include <iostream>
#include <armadillo>
#include "utils/eoRNG.h"
#include <vector>

using namespace std;
using namespace arma;


// For debugging purposes
#define EOCHROMOSOME_DEBUG
//#define EOCHROMOSOME_DEBUG_1


/**
 * @brief eoChromosome::init
 * @param _problemData
 */
void eoChromosome::init(ProblemData const* _problemData) {
    // Instantiate chromosome's cluster centres vector with size (NxK),
    // where N is the number of dimensions and K is the number of clusters
    clusterCentres = boost::make_shared<vector<double>>(
                    _problemData->getNumFeaturesPerSample()*_problemData->getNumClasses()*K_CLUSTERS);
    // Set timetable problem data
    problemData = _problemData;
}




///**
// * @brief computeFitness
// * @return  The computed fitness
// */
//double eoChromosome::computeFitness() {
//    // The fitness computation process consists of two
//    // phases. In the first phase, the clusters are formed accord-
//    // ing to the centres encoded in the chromosome under
//    // consideration. This is done by assigning each point
//    // x_i, i = 1, 2, ..., n to one of the clusters C_j with centre
//    // z_j such that
//    //
//    // norm2(x_i - z_j) < norm2(x_i - z_p), p = 1, 2, ..., K,
//    // and p neq j.
//    //
//    // All ties are resolved arbitrarily. After the clustering is
//    // done, the cluster centres encoded in the chromosome are
//    // replaced by the mean points of the respective clusters. In
//    // other words, for cluster C_i, the new centre znew_i is
//    // computed as
//    //
//    // znew_i = 1/n_i * sum_{xj in C_i} x_j, i = 1, 2, ..., K
//    //
//    // These znew_i now replace the previous z_i in the chromo-
//    // some.
//    //
//    // Subsequently, the clustering metric M is computed as
//    // follows:
//    //
//    // M = sum_{i=1}^K M_i
//    //
//    // M_i = sum_{x_j in C_i} norm2(x_j - z_i)
//    //
//    // Goal: Minimise M.
//    ///////////////////////////////////////////////////////////
///*
//    ///////////////////////////////////////////////////////////
//    // First phase. The clusters are formed accord-
//    // ing to the centres encoded in the chromosome under
//    // consideration. This is done by assigning each point
//    // x_i, i = 1, 2, ..., n to one of the clusters C_j with centre
//    // z_j such that
//    //
//    // norm2(x_i - z_j) < norm2(x_i - z_p), p = 1, 2, ..., K,
//    // and p neq j.
//    //
//    // All ties are resolved arbitrarily.
//    //
//    // Get samples x_i
//    vector<arma::vec> const &samples = getSamples();
//    // Get data points clusters
//    vector<int> &dataPointsClusters = *pointsClusters.get();
/////
///// TODO - OPTIMIZE, AVOID THIS CONVERSION
/////
//    // Convert clusters to arma::vec
//    vector<arma::vec> clusters(this->getNumClasses());
//    // Get cluster centres
//    vector<double> clusterCentres = this->getClusterCentres();
//#ifdef EOCHROMOSOME_DEBUG_1
//    // Print clusters
//    cout << "[Before compute fitness]clusters:" << endl;
//    for (auto & c : clusterCentres) {
//        cout << c << " ";
//    }
//    cout << endl;
////    cout << "Press a key..." << endl;
////    cin.get();
//#endif

//    for (int i = 0; i < getNumClasses(); ++i) {
//        // vec to hold the i cluster
//        arma::vec cluster_i(getNumFeatures());
//        // Copy cluster center
//        for (int j = 0; j < getNumFeatures(); ++j) {
//            cluster_i(j) = clusterCentres[i*getNumFeatures()+j];
//        }
//        // Add cluster
//        clusters[i] = cluster_i;
//    }

//#ifdef EOCHROMOSOME_DEBUG
////    // Print samples
////    cout << "Samples:" << endl;
////    for (auto & v : samples) {
////        cout << v << endl;
////    }
////    // Print clusters
////    cout << "clusters:" << endl;
////    for (auto & c : clusters) {
////        cout << c << endl;
////    }
////    cout << "Press a key..." << endl;
////    cin.get();
//#endif

//    // Assign each point x_i, i = 1, 2, ..., n to one of the clusters C_j with centre
//    // z_j such that norm2(x_i - z_j) < norm2(x_i - z_p), p = 1, 2, ..., K,
//    // and p neq j. All ties are resolved arbitrarily.
//    for (int i = 0; i < getNumSamples(); ++i) {
//#ifdef EOCHROMOSOME_DEBUG
//            // Print norm
//            cout << endl << "i = " << i << " norms: " << endl;
//#endif
//        // Get point xi
//        vec const &xi = samples[i];
//        // Minumum norm value and cluster index
//        double min_norm = -1;
//        int cluster_idx = -1;
//        for (int j = 0; j < getNumClasses(); ++j) {
//            // Get cluster zj
//            vec const &zj = clusters[j];
//            // Compute l2-norm
//            double calc_norm = norm(xi-zj, 2);
//#ifdef EOCHROMOSOME_DEBUG
//            cout << calc_norm << " ";
//#endif
//            if (min_norm == -1) {
//                // l2-norm
//                min_norm = calc_norm;
//                // Cluster index
//                cluster_idx = j;
//            }
//            else if (calc_norm <= min_norm) {
//                bool apply = true;
//                // All ties are resolved arbitrarily.
//                // We flip a coin and if prob < 0.5 we don't set xi to this cluster.
//                if (calc_norm == min_norm) {
//                    if (rng.flip() < 0.5)
//                        apply = false;
//                }
//                if (apply) {
//                    // l2-norm
//                    min_norm = calc_norm;
//                    // Cluster index
//                    cluster_idx = j;
//                }
//            }
//        }
//        // Assign xi to cluster zj
//        dataPointsClusters[i] = cluster_idx;
//    }
//    ///////////////////////////////////////////////////////////


//    ///////////////////////////////////////////////////////////
//    // Second phase. The cluster centres encoded in the chromosome are
//    // replaced by the mean points of the respective clusters. In
//    // other words, for cluster C_i, the new centre znew_i is
//    // computed as
//    //
//    // znew_i = 1/n_i * sum_{xj in C_i} x_j, i = 1, 2, ..., K
//    //
//    // These znew_i now replace the previous z_i in the chromo-
//    // some.
//    //
//    // In order to compute the mean points of the clusters,
//    // we set sum and count vectors. Then, we divide the sum
//    // by the counter to get the mean points
//    vector<arma::vec> sums(getNumClasses());
//    // Vector of bools used to verify the first addition
//    vector<bool> added(getNumClasses(), false); // Initialise to false
//    vector<int> counts(getNumClasses(), 0); // Initialise to 0
//    // For cluster C_i, the new centre znew_i is
//    // computed as:
//    // znew_i = 1/n_i * sum_{xj in C_i} x_j, i = 1, 2, ..., K
//    for (int i = 0; i < getNumSamples(); ++i) {
//        // Get point xi
//        vec const &x_i = samples[i];
//        // x_i cluster
//        int cluster_x_i = dataPointsClusters[i];
//        // Sum x_i
//        if (!added[cluster_x_i]) {
//            sums[cluster_x_i] = x_i;
//            added[cluster_x_i] = true;
//        }
//        else
//            sums[cluster_x_i] += x_i;
//        // Increment counter
//        counts[cluster_x_i]++;
//    }
//    // Compute means and replace old cluster centres by the new ones
//    for (int k = 0; k < getNumClasses(); ++k) {
//        sums[k] /= counts[k];
//        // Replace old cluster centres by the new ones
//        clusters[k] = sums[k];
//        // Copy cluster center
//        for (int j = 0; j < getNumFeatures(); ++j) {
//            clusterCentres[k*getNumFeatures()+j] = clusters[k][j];
//        }
//    }
//    ///////////////////////////////////////////////////////////


//#ifdef EOCHROMOSOME_DEBUG_1
//    // Print clusters
//    cout << "[After compute fitness]clusters:" << endl;
//    for (auto & c : clusterCentres) {
//        cout << c << " ";
//    }
//    cout << endl;
//    cout << "Press a key..." << endl;
//    cin.get();
//#endif

//*/
//    /*

//    /// TEST////

//    // Get samples x_i
//    vector<arma::vec> const &samples = getSamples();
//    // Get data points clusters
//    vector<int> &dataPointsClusters = *pointsClusters.get();
/////
///// TODO - OPTIMIZE, AVOID THIS CONVERSION
/////
//    // Convert clusters to arma::vec
//    vector<arma::vec> clusters(this->getNumClasses());
//    // Get cluster centres
//    vector<double> clusterCentres = this->getClusterCentres();
//    for (int i = 0; i < getNumClasses(); ++i) {
//        // vec to hold the i cluster
//        arma::vec cluster_i(getNumFeatures());
//        // Copy cluster center
//        for (int j = 0; j < getNumFeatures(); ++j) {
//            cluster_i(j) = clusterCentres[i*getNumFeatures()+j];
//        }
//        // Add cluster
//        clusters[i] = cluster_i;
//    }

//    // Assign each point x_i, i = 1, 2, ..., n to one of the clusters C_j with centre
//    // z_j such that norm2(x_i - z_j) < norm2(x_i - z_p), p = 1, 2, ..., K,
//    // and p neq j. All ties are resolved arbitrarily.
//    for (int i = 0; i < getNumSamples(); ++i) {
//#ifdef EOCHROMOSOME_DEBUG
//            // Print norm
//            cout << endl << "i = " << i << " norms: " << endl;
//#endif
//        // Get point xi
//        vec const &xi = samples[i];
//        // Minimum norm value and cluster index
//        double min_norm = -1;
//        int cluster_idx = -1;
//        for (int j = 0; j < getNumClasses(); ++j) {
//            // Get cluster zj
//            vec const &zj = clusters[j];
//            // Compute l2-norm
//            double calc_norm = norm(xi-zj, 2);
//#ifdef EOCHROMOSOME_DEBUG
//            cout << calc_norm << " ";
//#endif
//            if (min_norm == -1) {
//                // l2-norm
//                min_norm = calc_norm;
//                // Cluster index
//                cluster_idx = j;
//            }
////            else if (calc_norm <= min_norm) {
//            else if (calc_norm < min_norm) {
////                bool apply = true;
////                // All ties are resolved arbitrarily.
////                // We flip a coin and if prob < 0.5 we don't set xi to this cluster.
////                if (calc_norm == min_norm) {
////                    if (rng.flip() < 0.5)
////                        apply = false;
////                }
////                if (apply) {
//                    // l2-norm
//                    min_norm = calc_norm;
//                    // Cluster index
//                    cluster_idx = j;
////                }
//            }
//        }
//        // Assign xi to cluster zj
//        dataPointsClusters[i] = cluster_idx;
//    }

//////

//    ///////////////////////////////////////////////////////////
//    // Compute the clustering metric M as
//    // follows:
//    //
//    // M = sum_{i=1}^K M_i
//    //
//    // M_i = sum_{x_j in C_i} norm2(x_j - z_i)
//    //
//    arma::vec m_i(getNumSamples());
//    m_i.zeros();
//    for (int i = 0; i < getNumSamples(); ++i) {
//        // Get point xi
//        vec const &x_i = samples[i];
//        // x_i cluster index
//        int cluster_x_i_index = dataPointsClusters[i];
//        // x_i cluster center
//        vec const &z_i = clusters[cluster_x_i_index];
//        m_i(i) += norm(x_i-z_i, 2);
//    }
//    // Compute the clustering metric
//    double m = sum(m_i);
//    ///////////////////////////////////////////////////////////

//    return m;
//    */

//    return 0.5*(fitness1()/100 + fitness2());
////    return fitness1();

//    //return fitness2();
//}





/**
 * @brief computeFitness
 * @return  The computed fitness
 */
double eoChromosome::computeFitness() {

//    return 0.5*(fitness4()/100 + fitness1()); // GOOD

//    return fitness4()/100;
//    return fitness4();
    return fitness3();

//    return fitness1();

//    return fitness2();
}



/**
 * @brief eoChromosome::fitness3
 * @return
 */
double eoChromosome::fitness3() {
    return 0.5*(fitness1()/100 + fitness2());
}




///**
// * @brief eoChromosome::fitness4
// * @return
// */
//double eoChromosome::fitness4() {
//    // The fitness computation process consists of two
//    // phases. In the first phase, the clusters are formed accord-
//    // ing to the centres encoded in the chromosome under
//    // consideration. This is done by assigning each point
//    // x_i, i = 1, 2, ..., n to one of the clusters C_j with centre
//    // z_j such that
//    //
//    // norm2(x_i - z_j) < norm2(x_i - z_p), p = 1, 2, ..., K,
//    // and p neq j.
//    //
//    // All ties are resolved arbitrarily. After the clustering is
//    // done, the cluster centres encoded in the chromosome are
//    // replaced by the mean points of the respective clusters. In
//    // other words, for cluster C_i, the new centre znew_i is
//    // computed as
//    //
//    // znew_i = 1/n_i * sum_{xj in C_i} x_j, i = 1, 2, ..., K
//    //
//    // These znew_i now replace the previous z_i in the chromo-
//    // some.
//    //
//    // Subsequently, the clustering metric M is computed as
//    // follows:
//    //
//    // M = sum_{i=1}^K M_i
//    //
//    // M_i = sum_{x_j in C_i} norm2(x_j - z_i)
//    //
//    // Goal: Minimise M.
//    ///////////////////////////////////////////////////////////

//    ///////////////////////////////////////////////////////////
//    // First phase. The clusters are formed accord-
//    // ing to the centres encoded in the chromosome under
//    // consideration. This is done by assigning each point
//    // x_i, i = 1, 2, ..., n to one of the clusters C_j with centre
//    // z_j such that
//    //
//    // norm2(x_i - z_j) < norm2(x_i - z_p), p = 1, 2, ..., K,
//    // and p neq j.
//    //
//    // All ties are resolved arbitrarily.
//    //
//    // Get samples x_i
//    vector<arma::vec> const &samples = getSamples();
//    // Get data points clusters
//    vector<int> dataPointsAssignedClusters(getNumSamples());

/////
///// TODO - USE ARMA::VEC FOR CLUSTERS
/////
//    /////////////////////////////////////////////////////////////
//    // Convert clusters to arma::vec
//    vector<vector<arma::vec>> clusters(this->getNumClasses());
//    // Get cluster centres
//    vector<double> &clusterCentres = this->getClusterCentres();
//    for (int k = 0; k < K_CLUSTERS; ++k) {
//        for (int i = 0; i < getNumClasses(); ++i) {
//            // vec to hold the i cluster
//            arma::vec cluster_i(getNumFeatures());
//            // Copy cluster center
//            for (int j = 0; j < getNumFeatures(); ++j) {
//                cluster_i(j) = clusterCentres[k*getNumClasses()*getNumFeatures() + i*getNumFeatures()+j];
//            }
//            // Add cluster
//            clusters[i].push_back(cluster_i); // Push_back for each k
//        }
//    }
//    /////////////////////////////////////////////////////////////


//    // Assign each point x_i, i = 1, 2, ..., n to one of the clusters C_j with centre
//    // z_j such that norm2(x_i - z_j) < norm2(x_i - z_p), p = 1, 2, ..., K,
//    // and p neq j. All ties are resolved arbitrarily.
//    for (int i = 0; i < getNumSamples(); ++i) {
//        // Get point xi
//        vec const &xi = samples[i];
//        // Minimum norm value and cluster index
//        double min_norm = -1;
//        int cluster_idx = -1;
//        for (int j = 0; j < getNumClasses(); ++j) {
//            for (int k = 0; k < K_CLUSTERS; ++k) {
//                // Get cluster zj
//                arma::vec &zj = clusters[j][k];
//                // Compute l2-norm
//                double calc_norm = norm(xi - zj, 2);

//                if (min_norm == -1) {
//                    // l2-norm
//                    min_norm = calc_norm;
//                    // Cluster index
//                    cluster_idx = j;
//                }
//                else if (calc_norm <= min_norm) {
//                    bool apply = true;
//                    // All ties are resolved arbitrarily.
//                    // We flip a coin and if prob < 0.5 we don't set xi to this cluster.
//                    if (calc_norm == min_norm) {
//                        if (rng.flip() < 0.5)
//                            apply = false;
//                    }
//                    if (apply) {
//                        // l2-norm
//                        min_norm = calc_norm;
//                        // Cluster index
//                        cluster_idx = j;
//                    }
//                }
//            }
//        }
//        // Assign xi to cluster zj
//        dataPointsAssignedClusters[i] = cluster_idx;
//    }
//    ///////////////////////////////////////////////////////////



//    ///////////////////////////////////////////////////////////
//    // Second phase. The cluster centres encoded in the chromosome are
//    // replaced by the mean points of the respective clusters. In
//    // other words, for cluster C_i, the new centre znew_i is
//    // computed as
//    //
//    // znew_i = 1/n_i * sum_{xj in C_i} x_j, i = 1, 2, ..., K
//    //
//    // These znew_i now replace the previous z_i in the chromo-
//    // some.
//    //
//    // In order to compute the mean points of the clusters,
//    // we set sum and count vectors. Then, we divide the sum
//    // by the counter to get the mean points
//    vector<arma::vec> sums(getNumClasses());
//    // Vector of bools used to verify the first addition
//    vector<bool> added(getNumClasses(), false); // Initialise to false
//    vector<int> counts(getNumClasses(), 0); // Initialise to 0
//    // For cluster C_i, the new centre znew_i is
//    // computed as:
//    // znew_i = 1/n_i * sum_{xj in C_i} x_j, i = 1, 2, ..., K
//    for (int i = 0; i < getNumSamples(); ++i) {
//        // Get point xi
//        vec const &x_i = samples[i];
//        // x_i cluster
//        int cluster_x_i = dataPointsAssignedClusters[i];
//        // Sum x_i
//        if (!added[cluster_x_i]) {
//            sums[cluster_x_i] = x_i;
//            added[cluster_x_i] = true;
//        }
//        else
//            sums[cluster_x_i] += x_i;
//        // Increment counter
//        counts[cluster_x_i]++;
//    }
//    // Compute means and replace old cluster centres by the new ones
//    for (int k = 0; k < getNumClasses(); ++k) {
//        sums[k] /= counts[k];
//        // Replace old cluster centres by the new ones
////        clusters[k] = sums[k];
//        if (added[k]) {
//            clusters[k][0] = sums[k];
//        }
//        else {
//            arma::vec r(clusters[k][0].n_elem);
//            r.randn();
//            clusters[k][0] = r;
//        }
//        // Copy cluster center
//        for (int j = 0; j < getNumFeatures(); ++j) {
////            clusterCentres[k*getNumFeatures()+j] = clusters[k][j];
//            clusterCentres[k*getNumFeatures()+j] = clusters[k][0][j];
//        }
//    }
//    ///////////////////////////////////////////////////////////

////    cout << "clusters:" << endl;
////    for (int k = 0; k < getNumClasses(); ++k) {
////        cout << clusters[k][0] << endl;
////    }


//    // Get data points clusters
//    vector<int> const &dataPointsClusters = getSamplesClasses();


//    ///////////////////////////////////////////////////////////
//    // Compute the clustering metric M as
//    // follows:
//    //
//    // M = sum_{i=1}^K M_i
//    //
//    // M_i = sum_{x_j in C_i} norm2(x_j - z_i)
//    //
//    double sum_norm = 0.0;
//    for (int i = 0; i < getNumSamples(); ++i) {
//        // Get point xi
//        vec const &x_i = samples[i];
//        // x_i cluster index
////        int cluster_x_i_index = dataPointsAssignedClusters[i];
//        // TEST
//        int cluster_x_i_index = dataPointsClusters[i];
//        // Get x_i updated cluster center
//        vec const &z_i = clusters[cluster_x_i_index][0];

////        cout << "clusters:" << endl;
////        for (int k = 0; k < getNumClasses(); ++k) {
////            cout << clusters[k][0] << endl;
////        }

////        cout << "z_i = " << endl << z_i << endl;
////        cout << "x_i = " << endl << x_i << endl;
////        cin.get();


//        // Compute the clustering metric
//        sum_norm += norm(x_i - z_i, 2);
//    }
//    ///////////////////////////////////////////////////////////

//    return sum_norm;


//}






/**
 * @brief eoChromosome::fitness1
 * @return
 */
double eoChromosome::fitness1() {
    // The fitness computation process consists of two
    // phases. In the first phase, the clusters are formed accord-
    // ing to the centres encoded in the chromosome under
    // consideration. This is done by assigning each point
    // x_i, i = 1, 2, ..., n to one of the clusters C_j with centre
    // z_j such that
    //
    // norm2(x_i - z_j) < norm2(x_i - z_p), p = 1, 2, ..., K,
    // and p neq j.
    //
    // All ties are resolved arbitrarily. After the clustering is
    // done, the cluster centres encoded in the chromosome are
    // replaced by the mean points of the respective clusters. In
    // other words, for cluster C_i, the new centre znew_i is
    // computed as
    //
    // znew_i = 1/n_i * sum_{xj in C_i} x_j, i = 1, 2, ..., K
    //
    // These znew_i now replace the previous z_i in the chromo-
    // some.
    //
    // Subsequently, the clustering metric M is computed as
    // follows:
    //
    // M = sum_{i=1}^K M_i
    //
    // M_i = sum_{x_j in C_i} norm2(x_j - z_i)
    //
    // Goal: Minimise M.
    ///////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////
    // First phase. The clusters are formed accord-
    // ing to the centres encoded in the chromosome under
    // consideration. This is done by assigning each point
    // x_i, i = 1, 2, ..., n to one of the clusters C_j with centre
    // z_j such that
    //
    // norm2(x_i - z_j) < norm2(x_i - z_p), p = 1, 2, ..., K,
    // and p neq j.
    //
    // All ties are resolved arbitrarily.
    //
    // Get samples x_i
    vector<arma::vec> const &samples = getSamples();
    // Get data points clusters
    vector<int> dataPointsAssignedClusters(getNumSamples());

///
/// TODO - USE ARMA::VEC FOR CLUSTERS
///
    /////////////////////////////////////////////////////////////
    // Convert clusters to arma::vec
    vector<vector<arma::vec>> clusters(this->getNumClasses());
    // Get cluster centres
    vector<double> &clusterCentres = this->getClusterCentres();
    for (int k = 0; k < K_CLUSTERS; ++k) {
        for (int i = 0; i < getNumClasses(); ++i) {
            // vec to hold the i cluster
            arma::vec cluster_i(getNumFeatures());
            // Copy cluster center
            for (int j = 0; j < getNumFeatures(); ++j) {
                cluster_i(j) = clusterCentres[k*getNumClasses()*getNumFeatures() + i*getNumFeatures()+j];
            }
            // Add cluster
            clusters[i].push_back(cluster_i); // Push_back for each k
        }
    }
    /////////////////////////////////////////////////////////////


    // Assign each point x_i, i = 1, 2, ..., n to one of the clusters C_j with centre
    // z_j such that norm2(x_i - z_j) < norm2(x_i - z_p), p = 1, 2, ..., K,
    // and p neq j. All ties are resolved arbitrarily.
    for (int i = 0; i < getNumSamples(); ++i) {
    #ifdef EOCHROMOSOME_DEBUG
            // Print norm
//            cout << endl << "i = " << i << " norms: " << endl;
    #endif
        // Get point xi
        vec const &xi = samples[i];
        // Minimum norm value and cluster index
        double min_norm = -1;
        int cluster_idx = -1;
        for (int j = 0; j < getNumClasses(); ++j) {
            for (int k = 0; k < K_CLUSTERS; ++k) {
                // Get cluster zj
                arma::vec &zj = clusters[j][k];
                // Compute l2-norm
                double calc_norm = norm(xi - zj, 2);

                if (min_norm == -1) {
                    // l2-norm
                    min_norm = calc_norm;
                    // Cluster index
                    cluster_idx = j;
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
                    }
                }
            }
        }
        // Assign xi to cluster zj
        dataPointsAssignedClusters[i] = cluster_idx;
    }
    ///////////////////////////////////////////////////////////

    //
    // Second phase (De Falco paper, fitness function #1)
    //

    // Get data points clusters
    vector<int> const &dataPointsClusters = getSamplesClasses();
    int numMisclassified = 0;
    // Print cluster index
    for (int i = 0; i < getNumSamples(); ++i) {
        // x_i cluster index
        int cluster_x_i_index = dataPointsClusters[i];
        numMisclassified += (dataPointsAssignedClusters[i] != cluster_x_i_index ? 1 : 0);
    }

    // Dataset size
    int sizeDataset = getNumSamples();
    // CEP
    double cep = ((double)numMisclassified / sizeDataset) * 100;

    return cep;

}





/**
 * @brief eoChromosome::fitness2
 *  Fitness function 2 (De Falco et al. paper, 2007)
 *  The fitness function F2 is computed in one step as the sum on
 *  all training set instances of Euclidean distance in N-
 *  dimensional space between generic instance x_j and the centroid
 *  of the class it belongs to according to the database (p_i^{CL_known(x_j)}).
 *  This sum is normalized with respect to D_Train.
 *  In symbols, the i-th individual fitness is given by
 *  F2(i) = 1/D_Train * sum_{j=1}^{D_Train} d[x_j, p_i^{CL_known(x_j)}]
 * @return
 */
double eoChromosome::fitness2() {

///
/// TODO - USE ARMA::VEC FOR CLUSTERS
///

    /////////////////////////////////////////////////////////////
    // Convert clusters to arma::vec
    vector<vector<arma::vec>> clusters(this->getNumClasses());
    // Get cluster centres
    vector<double> const &clusterCentres = this->getClusterCentres();
    for (int k = 0; k < K_CLUSTERS; ++k) {
        for (int i = 0; i < getNumClasses(); ++i) {
            // vec to hold the i cluster
            arma::vec cluster_i(getNumFeatures());
            // Copy cluster center
            for (int j = 0; j < getNumFeatures(); ++j) {
                cluster_i(j) = clusterCentres[k*getNumClasses()*getNumFeatures() + i*getNumFeatures()+j];
            }
            // Add cluster
            clusters[i].push_back(cluster_i); // Push_back for each k
        }
    }
    /////////////////////////////////////////////////////////////

    int dTrain = getNumSamples();
    double sum = 0;
    // Get samples x
    vector<arma::vec> const &samples = getSamples();
    // Get data points clusters
    vector<int> const &dataPointsClusters = this->getSamplesClasses();
    for (int j = 0; j < dTrain; ++j) {
        // Get point x_j
        vec const &x_j = samples[j];
        // Get cluster vector and class where x_j belongs to according to database (p_i^{CL_known(x_j)}).
        int cluster_idx = dataPointsClusters[j];
        /////////////////////////////////////////////////////////////////////////
        // Get cluster vector
        // Choose closer cluster center
        // Minimum norm value and cluster index
        double min_norm = -1;
        int min_cluster_idx = -1;
        for (int j = 0; j < K_CLUSTERS; ++j) {
            // Get cluster zj
            arma::vec &z_j = clusters[cluster_idx][j];
            // Compute l2-norm
            double calc_norm = norm(x_j - z_j, 2);

            if (min_norm == -1) {
                // l2-norm
                min_norm = calc_norm;
                // Cluster index
                min_cluster_idx = j;
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
                    min_cluster_idx = j;
                }
            }
        }
        /////////////////////////////////////////////////////////////////////////

        vec &z_j = clusters[cluster_idx][min_cluster_idx];

        // Compute l2-norm - Euclidean distance
//        double calc_norm = norm(x_j - z_j, 2);
        double calc_norm = min_norm;

        sum += calc_norm;
    }
    sum /= dTrain;

    return sum;
}






/*
// Fitness function 2 (De Falco et al. paper, 2007)
// The fitness function F2 is computed in one step as the sum on
// all training set instances of Euclidean distance in N-
// dimensional space between generic instance x_j and the centroid
// of the class it belongs to according to the database (p_i^{CL_known(x_j)}).
// This sum is normalized with respect to D_Train.
// In symbols, the i-th individual fitness is given by
// F2(i) = 1/D_Train * sum_{j=1}^{D_Train} d[x_j, p_i^{CL_known(x_j)}]
//
double eoChromosome::fitness2() {

///
/// TODO - USE ARMA::VEC FOR CLUSTERS
///

    /////////////////////////////////////////////////////////////
    // Convert clusters to arma::vec
    vector<arma::vec> clusters(this->getNumClasses());
    // Get cluster centres
    vector<double> const &clusterCentres = this->getClusterCentres();
    for (int i = 0; i < getNumClasses(); ++i) {
        // vec to hold the i cluster
        arma::vec cluster_i(getNumFeatures());
        // Copy cluster center
        for (int j = 0; j < getNumFeatures(); ++j) {
            cluster_i(j) = clusterCentres[i*getNumFeatures()+j];
        }
        // Add cluster
        clusters[i] = cluster_i;
    }
    /////////////////////////////////////////////////////////////

    int dTrain = getNumSamples();
    double sum = 0;
    // Get samples x
    vector<arma::vec> const &samples = getSamples();
    // Get data points clusters
    vector<int> const &dataPointsClusters = this->getSamplesClasses();
    for (int j = 0; j < dTrain; ++j) {
        // Get point x_j
        vec const &x_j = samples[j];
        // Get cluster vector and class where x_j belongs to according to database (p_i^{CL_known(x_j)}).
        int cluster_idx = dataPointsClusters[j];
        // Get cluster vector
        vec &z_j = clusters[cluster_idx];

        ////////////////////////////////////////////////////////////////////////////
        // Compute l2-norm - Euclidean distance
        double calc_norm = norm(x_j - z_j, 2);

        // l1-norm
//        double calc_norm = norm(x_j - z_j, 1);

        //  NormalizedSquaredEuclideanDistance[u,v] = 0.5*Var(u - v)/[Var(u) + Var(v)]
//        double calc_norm = 0.5*var(x_j - z_j) / (var(x_j) + var(z_j));
        ////////////////////////////////////////////////////////////////////////////

        sum += calc_norm;
    }
    sum /= dTrain;

    return sum;
}
*/











/**
 * @brief eoChromosome::printToFile
 * @param _os
 */
void eoChromosome::printToFile(ostream &_os) const {
    // Print cluster centres
    vector<double> clusterCentres = this->getClusterCentres();
//    for (double c : clusterCentres)
//        _os << c << " ";
//    _os << endl;


    // Print fitness
    _os << fitness() << endl;

    //
    // Print data points clusters
    //
    // Convert clusters to arma::vec
    vector<arma::vec> clusters(this->getNumClasses());
    for (int i = 0; i < getNumClasses(); ++i) {
        // vec to hold the i cluster
        arma::vec cluster_i(getNumFeatures());
        // Copy cluster center
        for (int j = 0; j < getNumFeatures(); ++j) {
            cluster_i(j) = clusterCentres[i*getNumFeatures()+j];
        }
        // Add cluster
        clusters[i] = cluster_i;
        // Print cluster center vector
        _os << cluster_i << endl;
    }
    // Get samples x_i
    vector<arma::vec> const &samples = getSamples();
    // Get data points clusters
    vector<int> const &dataPointsClusters = getSamplesClasses();
    for (int i = 0; i < getNumSamples(); ++i) {
        // Get point xi
        vec const &x_i = samples[i];
        // x_i cluster index
        int cluster_x_i_index = dataPointsClusters[i];
        // x_i cluster center
        vec const &z_i = clusters[cluster_x_i_index];

        // Print cluster index and vector
//        _os << cluster_x_i_index << " " << z_i << endl;
//        _os << cluster_x_i_index << " ";
        for (int j = 0; j < getNumFeatures(); ++j)
            _os << x_i[j] << " ";
        _os << endl;
    }

    _os << endl;

    // Print cluster index
    for (int i = 0; i < getNumSamples(); ++i) {
        // x_i cluster index
        int cluster_x_i_index = dataPointsClusters[i];
        // Print cluster index
        _os << cluster_x_i_index << endl;
    }

    // Print cluster vector
    for (int i = 0; i < getNumSamples(); ++i) {
        // x_i cluster index
        int cluster_x_i_index = dataPointsClusters[i];
        // x_i cluster center
        vec const &z_i = clusters[cluster_x_i_index];

        // Print cluster vector
        for (int j = 0; j < getNumFeatures(); ++j)
            _os << z_i[j] << " ";
        _os << endl;
    }
}



/**
 * @brief eoChromosome::assignToClosestCluster
 * @param _dataPointsAssignedClusters
 */
void eoChromosome::assignToClosestCluster(vector<int> &_dataPointsAssignedClusters) const {
    vector<double> clusterCentres = this->getClusterCentres();

    // Convert clusters to arma::vec
    vector<arma::vec> clusters(this->getNumClasses());
    for (int i = 0; i < getNumClasses(); ++i) {
        // vec to hold the i cluster
        arma::vec cluster_i(getNumFeatures());
        // Copy cluster center
        for (int j = 0; j < getNumFeatures(); ++j) {
            cluster_i(j) = clusterCentres[i*getNumFeatures()+j];
        }
        // Add cluster
        clusters[i] = cluster_i;
    }
    // Get samples x_i
    vector<arma::vec> const &samples = getSamples();

    // Assign each point x_i, i = 1, 2, ..., n to one of the clusters C_j with centre
    // z_j such that norm2(x_i - z_j) < norm2(x_i - z_p), p = 1, 2, ..., K,
    // and p neq j. All ties are resolved arbitrarily.
//    for (int i = 0; i < getNumSamples(); ++i) {

    for (int i = 113; i < getNumSamples(); ++i) {
        // Get point xi
        vec const &xi = samples[i];
        // Minimum norm value and cluster index
        double min_norm = -1;
        int cluster_idx = -1;
        for (int j = 0; j < getNumClasses(); ++j) {
            // Get cluster zj
            vec const &zj = clusters[j];
            // Compute l2-norm
            double calc_norm = norm(xi-zj, 2);
            if (min_norm == -1) {
                // l2-norm
                min_norm = calc_norm;
                // Cluster index
                cluster_idx = j;
            }
            else if (calc_norm <= min_norm) {
//            else if (calc_norm < min_norm) {
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
                }
            }
        }
        // Assign xi to cluster zj
        _dataPointsAssignedClusters[i] = cluster_idx;
    }
}




/**
 * @brief eoChromosome::validate
 */
void eoChromosome::validate() const {

    cout << endl << "[validate() method]" << endl;


}





/**
 * @brief operator <<
 * @param _os
 * @param _chrom
 * @return
 */
ostream& operator<<(ostream &_os, const eoChromosome &_chrom) {
    _os << endl << "eoChromosome::operator<<" << endl;
    _os << endl;

    // Get cluster centres
    vector<double> clusterCentres = _chrom.getClusterCentres();
    for (double c : clusterCentres)
        _os << c << " ";
    _os << endl;

    return _os;
}
