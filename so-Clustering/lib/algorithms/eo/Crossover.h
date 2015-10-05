#ifndef CROSSOVER_H
#define CROSSOVER_H

#include "chromosome/eoChromosome.h"
#include <eoOp.h>


// For debugging purposes
//#define _CROSSOVER_DEBUG



template <typename EOT>
class Crossover : public eoQuadOp<EOT>
{

public:

    /**
    * the class name (used to display statistics)
    */
    std::string className() const;

    /**
    * eoQuad crossover - _chromosome1 and _chromosome2 are the (future) offspring, i.e. _copies_ of the parents
    * @param _chromosome1 the first parent
    * @param _chromosome2 the second parent
    */
    bool operator()(EOT& _chromosome1, EOT& _chromosome2);

protected:

    /**
     * @brief generateOffspring generation of the offspring
     * @param _parent1
     * @param _parent2
     * @param _offspring1
     * @param _offspring2
     */
    void generateOffspring(const EOT &_parent1, const EOT &_parent2, EOT &_offspring1, EOT &_offspring2);

};





template <typename EOT>
std::string Crossover<EOT>::className() const
{
    return "Clustering Crossover";
}


template <typename EOT>
bool Crossover<EOT>::operator()(EOT &_chromosome1, EOT &_chromosome2)
{
//    cout << "Crossover" << endl;

    bool oneAtLeastIsModified = true;
    EOT chrom1 = _chromosome1; // Copy of original chromosome 1
    EOT chrom2 = _chromosome2; // Copy of original chromosome 2
    EOT &offspring1 = _chromosome1; // Reference to chromosome 1
    EOT &offspring2 = _chromosome2; // Reference to chromosome 2
    // Computation of the offspring
    generateOffspring(chrom1, chrom2, offspring1, offspring2); // Offspring1 and offspring2 are out parameters


    // Compute chromosome fitness
    double computedFitness = _chromosome1.computeFitness();
    // Set chromosome fitness
    _chromosome1.fitness(computedFitness);

    // Compute chromosome fitness
    computedFitness = _chromosome2.computeFitness();
    // Set chromosome fitness
    _chromosome2.fitness(computedFitness);


    return oneAtLeastIsModified;
}




/**
 * @brief generateOffspring generation of the offspring
 * @param _parent1
 * @param _parent2
 * @param _offspring1
 * @param _offspring2
 */
template <typename EOT>
void Crossover<EOT>::generateOffspring(EOT const &_parent1, EOT const &_parent2,
                                       EOT &_offspring1, EOT &_offspring2)
{
    //
    // Crossover is a probabilistic process that exchanges
    // information between two parent chromosomes for gen-
    // erating two child chromosomes. In this article single-
    // point crossover with a fixed crossover probability of miu_c is
    // used. For chromosomes of length l, a random integer,
    // called the crossover point, is generated in the range
    // [1, l-1]. The portions of the chromosomes lying to the
    // right of the crossover point are exchanged to produce
    // two offspring.
    //
    // Get cluster centres of parents and offsprings
    // Parent 1
    std::vector<double> const &par1ClusterCentres = _parent1.getClusterCentres();
    // Parent 2
    std::vector<double> const &par2ClusterCentres = _parent2.getClusterCentres();
    // Offspring 1
    std::vector<double> &off1ClusterCentres = _offspring1.getClusterCentres();
    // Offspring 2
    std::vector<double> &off2ClusterCentres = _offspring2.getClusterCentres();
    // Generate crossover point
    int idx = rng.uniform(_parent1.length()-1);
    // Generate offspring 1
    std::copy(par2ClusterCentres.begin()+idx, par2ClusterCentres.end(), off1ClusterCentres.begin()+idx);
    // Generate offspring 2
    std::copy(par1ClusterCentres.begin()+idx, par1ClusterCentres.end(), off2ClusterCentres.begin()+idx);
}


#endif // CROSSOVER_H














