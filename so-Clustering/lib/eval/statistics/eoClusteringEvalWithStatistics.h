#ifndef EOCLUSTERINGEVALWITHSTATISTICS_H
#define EOCLUSTERINGEVALWITHSTATISTICS_H


#include "eval/eoClusteringEval.h"
#include "eval/eoNumberEvalsCounter.h"



/**
 * Evaluation of objective function and # evals statistics computation
 */
template <class EOT>
class eoClusteringEvalWithStatistics : public eoClusteringEval<EOT> {
public:

    eoClusteringEvalWithStatistics(eoNumberEvalsCounter &_numberEvalsCounter)
        : numberEvalsCounter(_numberEvalsCounter) { }

    void operator()(EOT& _chrom) {
        // Invoke base class method to perform chromosome evaluation
        eoClusteringEval<EOT>::operator ()(_chrom);
        // # evals statistics computation. Add 1 to # evals
        numberEvalsCounter.addNumEvalsToGenerationTotal(1);
    }


protected:

    eoNumberEvalsCounter &numberEvalsCounter;
};



#endif // EOCLUSTERINGEVALWITHSTATISTICS_H
