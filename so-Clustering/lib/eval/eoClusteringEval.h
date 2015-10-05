#ifndef EOCLUSTERINGEVAL_H
#define EOCLUSTERINGEVAL_H

#include <eoEvalFunc.h>


// For debugging purposes
//#define EOCLUSTERINGEVAL_DEBUG



/**
 * Evaluation of objective function
 */
template <class EOT>
class eoClusteringEval : public eoEvalFunc<EOT> {
public:
    void operator()(EOT& _chrom) {

#ifdef EOCLUSTERINGEVAL_DEBUG
        std::cout << "eoClusteringEval::operator()()" << std::endl;
#endif

        // Compute chromosome fitness
        double computedFitness = _chrom.computeFitness();
        // Set chromosome fitness
        _chrom.fitness(computedFitness);
    }
};




#endif // EOCLUSTERINGEVAL_H
