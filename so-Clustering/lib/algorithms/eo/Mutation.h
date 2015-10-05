#ifndef MUTATION_H
#define MUTATION_H

#include "chromosome/eoChromosome.h"
#include <eoOp.h>
#include <string>




//#define MUTATION_DEBUG



template <typename EOT>
class Mutation : public eoMonOp<EOT>
{

public:

    /**
     * the class name (used to display statistics)
     */
    std::string className() const;

    /**
     * eoMon mutation - _chromosome is the chromosome to mutate.
     * @param _chromosome the chromosome
     */
    bool operator()(EOT& _chromosome);
};


/**
 * the class name (used to display statistics)
 */
template <typename EOT>
std::string Mutation<EOT>::className() const
{
    return "Clustering Mutation";
}



/**
 * eoMon mutation - _chromosome is the chromosome to mutate.
 * @param _chromosome the chromosome
 */
template <typename EOT>
bool Mutation<EOT>::operator()(EOT& _chrom)
{
    //
    // Each chromosome undergoes mutation with a fixed
    // probability miu_m. For binary representation of chromo-
    // somes, a bit position (or gene) is mutated by simply
    // flipping its value. Since we are considering floating point
    // representation in this article, we use the following muta-
    // tion. A number delta in the range [0, 1] is generated with
    // uniform distribution. If the value at a gene position is v,
    // after mutation it becomes
    // v +- 2 * delta * v, if v neq 0
    // v +- 2 * delta, if v eq 0
    // The '+' or '-' sign occurs with equal probability.
    //
//    cout << "Mutation" << endl;

    bool chromosomeIsModified = true;

    // Offspring
    EOT &offspring = _chrom; // Reference to chromosome
    // Cluster centres
    std::vector<double> &offClusterCentres = offspring.getClusterCentres();

    // Generate bit position (or gene)
    int idx = rng.uniform(_chrom.length());
    // Generate number delta in the range [0, 1]
    double delta = rng.uniform();
    // Gene v
    double &v = offClusterCentres[idx];
#ifdef MUTATION_DEBUG
    std::cout << "delta = " << delta << std::endl;
    std::cout << "v = " << v << std::endl;
    std::cin.get();
#endif

    // If the value at a gene position is v,
    // after mutation it becomes
    // v +- 2 * delta * v, if v neq 0
    // v +- 2 * delta, if v eq 0
    // The '+' or '-' sign occurs with equal probability.
    bool sign = true; // true means the '+' sign, and false means the '-' sign
    if (rng.flip() < 0.5)
        sign = false;

    // Original
    v = v + (2*delta)*(sign ? 1 : -1) * (v == 0 ? 1 : v);


#ifdef MUTATION_DEBUG
    std::cout << "v = " << v << std::endl;
    std::cin.get();
#endif

    // Compute chromosome fitness
    double computedFitness = _chrom.computeFitness();
    // Set chromosome fitness
    _chrom.fitness(computedFitness);

    // Return 'true' if at least one genotype has been modified
    return chromosomeIsModified;
}



#endif // MUTATION_H
