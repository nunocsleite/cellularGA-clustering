#ifndef EOCELLULARGA_H
#define EOCELLULARGA_H

#include <fstream>
#include "algorithms/eo/eoGenerationContinuePopVector.h"
#include <eoEvalFunc.h>
#include <eoSelectOne.h>
#include "algorithms/eo/eoDeterministicTournamentSelectorPointer.h"
#include <eoPopEvalFunc.h>
#include "algorithms/eo/eoAlgoPointer.h"
#include <eoOp.h>

#include "eval/eoClusteringEval.h"

#include "utils/CurrentDateTime.h"
#include <boost/make_shared.hpp>

// Using boost accumulators framework for computing the variance
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/variance.hpp>

// I'm importing the accumulators namespace as "a" to make clear which
// elements belong to it
namespace a = boost::accumulators;


#define EOCELLULARGA_DEBUG


/**
   The abstract cellular evolutionary algorithm.

   @ingroup Algorithms
 */
template <class EOT>
class eoCellularEA : public eoAlgoPointer<EOT> {

public :

    /**
     * Constructor
     */
    eoCellularEA(std::ofstream & _outFile, int _nrows, int _ncols, double _cp, double _mp,
                 eoGenerationContinuePopVector<EOT> &_cont, // Stop. criterion
                 eoEvalFunc<EOT> & _eval, // Evaluation function
                 eoDetTournamentSelectSharedPtr<EOT> &_sel_neigh, // To choose a partner. // Work with pointers for efficiency
                 eoQuadOp<EOT> & _cross,
                 eoMonOp<EOT> & _mut, // Mutation operator
                 eoNumberEvalsCounter &_numEvalCounter // # evaluations counter
                 ) :
        outFile(_outFile),
        nrows(_nrows), ncols(_ncols),
        cp(_cp), mp(_mp),
        cont (_cont),
        fullEval(_eval),
        popEval(_eval),
        sel_neigh(_sel_neigh),
        cross(_cross),
        mut(_mut),
        bestSolution(nullptr),
        popVariance(0),
        numEvalsCounter(_numEvalCounter)
    { }

    /**
     *   Evolve a given population
     */
//    void operator() (eoPop<EOT> & _pop) override {
    virtual void operator() (boost::shared_ptr<std::vector<boost::shared_ptr<EOT> > >&_pop) override {
        //
        // We apply a Synchronous cGA.
        // In the synchronous cGA the reproductive cycle is applied to all the
        // individuals simulataneously, that is, the individuals of the population
        // of the next generation are formally created at the same time, in a concurrent way.
        //
        // Create pointer to offspring population of chromosome pointers (empty)
        boost::shared_ptr<std::vector<boost::shared_ptr<EOT> > > offspringPop =
                boost::make_shared<std::vector<boost::shared_ptr<EOT> > >();


        int genNumber = 1;

#ifdef EOCELLULARGA_DEBUG
//      std::cout << std::endl << "Running cGA" << std::endl;

#endif

        do {
            // Clear the offspring population produced in the previous generation
            (*offspringPop.get()).clear();

            // Get reference to original population
            std::vector<boost::shared_ptr<EOT> > &originalPop = *_pop.get();

            // Produce the generation offspring
            for (int i = 0; i < originalPop.size(); ++i) {
                // Who are neighbouring to the current individual?
                //
                // The neighbours method return a vector containing const pointers
                // the neighbour solutions
                std::vector<boost::shared_ptr<EOT> > neighs = neighbours(originalPop, i);

                // Create, in the heap, object copies of current individual and its neighbour
                boost::shared_ptr<EOT> solCopy(new EOT(*originalPop[i].get())); // Invoke the copy ctor
                boost::shared_ptr<EOT> part(new EOT(*sel_neigh(neighs).get())); // Invoke the copy ctor

                // To perform cross-over
                if (rng.uniform() < cp) {
                    // Change the _pop[i] and part solutions directly
                    cross(*solCopy.get(), *part.get());
                    // # evals statistics computation. Add 2 to # evals
                    numEvalsCounter.addNumEvalsToGenerationTotal(2);
                }
                // To perform mutation
                if (rng.uniform() < mp) {
                    // Change the solutions directly
                    mut(*solCopy.get());
                    mut(*part.get());
                    // # evals statistics computation. Add 2 to # evals
                    numEvalsCounter.addNumEvalsToGenerationTotal(2);
                }



//#ifdef EOCELLULARGA_DEBUG
//                std::cout << "After variation operator" << std::endl;
//                std::cout << "sol.fitness() = " << (*solCopy.get()).fitness() << std::endl;
//                std::cout << "part.fitness() = " << (*part.get()).fitness() << std::endl;
//                std::cout << "Press a key" << std::endl;
//                std::cin.get();
//#endif

                // To choose the best of the two children
                boost::shared_ptr<EOT> offspringSol;
                if ((*solCopy.get()).fitness() < (*part.get()).fitness())
                    offspringSol = solCopy;
                else
                    offspringSol = part;


                // To choose the best between the new made child and the old individual
                boost::shared_ptr<EOT> bestOffspringSol;
                if ((*originalPop[i].get()).fitness() < (*offspringSol.get()).fitness()) {
                    boost::shared_ptr<EOT> originalSol(new EOT(*originalPop[i].get())); // Invoke the copy ctor
                    bestOffspringSol = originalSol;
                }
                else
                    bestOffspringSol = offspringSol;

                // Insert into the offspring vector
                (*offspringPop.get()).push_back(bestOffspringSol);


            } // End of generation

            // Swap offspring and original populations
            offspringPop.swap(_pop);
            // Add to total evaluations the generation # evals
            numEvalsCounter.addNumEvalsToTotal(numEvalsCounter.getGenerationNumEvals());
            // Get reference to population
            std::vector<boost::shared_ptr<EOT> > &finalPop = *_pop.get();

            ///////////////////////////////////////////////////////////////////////////////////////////////////
            // Compute population variance
            // The accumulator set which will calculate the properties for us:
            a::accumulator_set< double, a::stats<a::tag::mean, a::tag::variance> > acc_variance;
            for (unsigned i = 0; i < finalPop.size(); ++i)
                acc_variance((*finalPop[i].get()).fitness());
            // Set variance value
            popVariance = a::variance(acc_variance);

            ///////////////////////////////////////////////////////////////////////////////////////////////////
            // Determine the best solution
            for (unsigned i = 0; i < finalPop.size(); ++i) {
                if (bestSolution.get() == nullptr || (*finalPop[i].get()).fitness() < bestSolution->fitness())
                    bestSolution = finalPop[i];
            }


//            ///////////////////////////////////////////////////////////////////////////////////////////////////
//            // Print population information to output and
//            // save population information into file
//            std::cout << std::endl << "==============================================================" << std::endl;
//            std::cout << "Generation # " << genNumber << ", Date/Time = " << currentDateTime() << std::endl;
////            outFile << std::endl << "==============================================================" << std::endl;
////            outFile << "Generation # " << genNumber << ", Date/Time = " << currentDateTime() << std::endl;
//            int k = 0;
//            for (int i = 0; i < nrows; ++i) {
//                for (int j = 0; j < ncols; ++j, ++k) {
//                    std::cout << (*finalPop[k].get()).fitness() << "\t";
////                    outFile << (*finalPop[k].get()).fitness() << "\t";
//                }
//                std::cout << std::endl;
////                outFile << std::endl;
//            }
             // Just print population vector
            for (int i = 0; i < nrows*ncols; ++i) {
                outFile << (*finalPop[i].get()).fitness() << "\t";
            }
            outFile << std::endl;

//            std::cout << "popVariance = " << popVariance << ", best sol = " << bestSolution->fitness() << std::endl
//                      << "# evaluations generation: " << numEvalsCounter.getGenerationNumEvals()
//                      << ", Total # evaluations: " << numEvalsCounter.getTotalNumEvals() << std::endl;
////            outFile << "popVariance = " << popVariance << ", best sol = " << bestSolution->fitness() << std::endl
////                      << "# evaluations generation: " << numEvalsCounter.getGenerationNumEvals()
////                      << ", Total # evaluations: " << numEvalsCounter.getTotalNumEvals() << std::endl;
////            // Save best solution to file
////            outFile << *getBestSolution() << std::endl;
////            std::cout << "==============================================================" << std::endl;
////            outFile << "==============================================================" << std::endl;
////            ///////////////////////////////////////////////////////////////////////////////////////////////////

////            std::cin.get();

            // Increment # generations
            ++genNumber;

            // Reset # evaluations generation counter
            numEvalsCounter.setGenerationNumEvals(0);

        } while (cont(*_pop.get()));

//        std::cout << std::endl << "End of evolution cycle" << std::endl
//             << "Writing best solution to file..." << std::endl;
//        // Write best solution to file
//        outFile << std::endl << "End of evolution cycle" << std::endl
//                << "Best solution: " << std::endl;
//        outFile << *getBestSolution() << std::endl;
    }


    EOT* getBestSolution() { return bestSolution.get(); }

protected :

    virtual std::vector<boost::shared_ptr<EOT> > neighbours (
            const std::vector<boost::shared_ptr<EOT> > &_pop, int _rank) const = 0;

    std::ofstream & outFile;
    int nrows, ncols;
    double cp, mp;
    eoGenerationContinuePopVector<EOT> &cont; // Stop. criterion
    eoEvalFunc<EOT> & fullEval;
    eoPopLoopEval<EOT> popEval;
    eoDetTournamentSelectSharedPtr<EOT> &sel_neigh; // To choose a partner. // Work with pointers for efficiency
    eoBF<EOT &, EOT &, bool> & cross;
    eoMonOp<EOT> & mut;
    boost::shared_ptr<EOT> bestSolution; // Reference to the best solution
    double popVariance; // Population variance
    eoNumberEvalsCounter &numEvalsCounter;
};




#endif // EOCELLULARGA_H
