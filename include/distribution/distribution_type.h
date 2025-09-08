#ifndef DISTRIBUTION_TYPE_H
#define DISTRIBUTION_TYPE_H

#include <distribution.h>
#include <memory>
#include <vector>

// Enum to represent different types of statistical distributions.
enum class DistributionType {
    Bernoulli,
    Binomial,
    Categorical,
    Exponential,
    Gamma,
    InverseGaussian,
    Laplace,
    Multinomial,
    Normal,
    Poisson,
    // Add any other distribution types you need here.
};

std::unique_ptr<Distribution> createDistribution(DistributionType type, const std::vector<double>& params);

#endif // DISTRIBUTION_TYPE_H