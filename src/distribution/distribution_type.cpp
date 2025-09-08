#include <distribution_type.h>

#include <bernoulli_distribution.h>
#include <binomial_distribution.h>
#include <categorical_distribution.h>
#include <exponential_distribution.h>
#include <gamma_distribution.h>
#include <inverse_gaussian_distribution.h>
#include <laplace_distribution.h>
#include <multinomial_distribution.h>
#include <normal_distribution.h>
#include <poisson_distribution.h>
#include <stdexcept>

std::unique_ptr<Distribution> createDistribution(DistributionType type, const std::vector<double>& params) {
    switch (type) {
        case DistributionType::Bernoulli:
            if (params.size() != 1) {
                throw std::invalid_argument("Bernoulli distribution requires 1 parameter: p.");
            }
            return std::make_unique<BernoulliDistribution>(params[0]);
        case DistributionType::Binomial:
            if (params.size() != 2) {
                throw std::invalid_argument("Binomial distribution requires 2 parameters: n and p.");
            }
            return std::make_unique<BinomialDistribution>(static_cast<int>(params[0]),
                                                            params[1]);
        case DistributionType::Categorical:
            if (params.size() < 1) {
                throw std::invalid_argument("Categorical distribution requires at least 1 parameter: probabilities.");
            }
            return std::make_unique<CategoricalDistribution>(params);
        case DistributionType::Exponential:
            if (params.size() != 1) {
                throw std::invalid_argument("Exponential distribution requires 1 parameter: rate (lambda).");
            }
            return std::make_unique<ExponentialDistribution>(params[0]);
        case DistributionType::Gamma:
            if (params.size() != 2) {
                throw std::invalid_argument("Gamma distribution requires 2 parameters: shape and scale.");
            }
            return std::make_unique<GammaDistribution>(params[0], params[1]);
        case DistributionType::InverseGaussian:
            if (params.size() != 2) {
                throw std::invalid_argument("Inverse Gaussian distribution requires 2 parameters: mean and shape.");
            }
            return std::make_unique<InverseGaussianDistribution>(params[0], params[1]);
        case DistributionType::Laplace:
            if (params.size() != 2) {
                throw std::invalid_argument("Laplace distribution requires 2 parameters: location and scale.");
            }
            return std::make_unique<LaplaceDistribution>(params[0], params[1]);
        case DistributionType::Multinomial:
            if (params.size() < 2) {
                throw std::invalid_argument("Multinomial distribution requires at least 2 parameters: n and probabilities.");
            }
            return std::make_unique<MultinomialDistribution>(static_cast<int>(params[0]),
                                                             std::vector<double>(params.begin() + 1, params.end()));
        case DistributionType::Normal:
            if (params.size() != 2) {
                throw std::invalid_argument("Normal distribution requires 2 parameters: mean and stddev.");
            }
            return std::make_unique<NormalDistribution>(params[0], params[1]);
        case DistributionType::Poisson:
            if (params.size() != 1) {
                throw std::invalid_argument("Poisson distribution requires 1 parameter: rate (lambda).");
            }
            return std::make_unique<PoissonDistribution>(params[0]);
        default:
            throw std::invalid_argument("Unsupported distribution type.");
    }
}