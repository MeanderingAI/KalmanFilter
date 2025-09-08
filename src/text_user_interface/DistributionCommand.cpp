#include <text_user_interface/Command.hpp>
#include <distribution/distribution_type.h>
#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <stdexcept>
#include <memory>
#include <sstream>
#include <discrete_distribution.h>


// A helper map to convert string input to the enum type.
// This allows the user to type "Normal" instead of a number.
std::map<std::string, DistributionType> distributionTypeMap = {
    {"Bernoulli", DistributionType::Bernoulli},
    {"Binomial", DistributionType::Binomial},
    {"Categorical", DistributionType::Categorical},
    {"Exponential", DistributionType::Exponential},
    {"Gamma", DistributionType::Gamma},
    {"InverseGaussian", DistributionType::InverseGaussian},
    {"Laplace", DistributionType::Laplace},
    {"Multinomial", DistributionType::Multinomial},
    {"Normal", DistributionType::Normal},
    {"Poisson", DistributionType::Poisson},
};

// --- Your Commands ---
// The new 'generate' command for creating samples.
REGISTER_COMMAND(generate, "Generates samples from a distribution. Usage: generate <type> <num_samples> <params...>", [](const std::vector<std::string>& args) {
    if (args.size() < 2) {
        std::cout << "Error: Not enough arguments provided." << std::endl;
        std::cout << "Usage: generate <type> <num_samples> <params...>" << std::endl;
        return;
    }

    std::string type_string = args[0];
    if (distributionTypeMap.find(type_string) == distributionTypeMap.end()) {
        std::cout << "Error: Unknown distribution type '" << type_string << "'." << std::endl;
        return;
    }

    int num_samples;
    try {
        num_samples = std::stoi(args[1]);
        if (num_samples <= 0) {
            throw std::invalid_argument("Number of samples must be positive.");
        }
    } catch (const std::exception& e) {
        std::cout << "Error: Invalid number of samples provided." << std::endl;
        return;
    }

    std::vector<double> params;
    try {
        for (size_t i = 2; i < args.size(); ++i) {
            params.push_back(std::stod(args[i]));
        }
    } catch (const std::exception& e) {
        std::cout << "Error: Invalid parameter provided." << std::endl;
        return;
    }

    try {
        std::unique_ptr<Distribution> dist = createDistribution(distributionTypeMap[type_string], params);

        // Use dynamic_cast to determine if the distribution is discrete.
        DiscreteDistribution* discrete_dist = dynamic_cast<DiscreteDistribution*>(dist.get());

        if (discrete_dist) {
            // It's a discrete distribution, call sample_discrete().
            std::cout << "Generated discrete samples:" << std::endl;
            for (int i = 0; i < num_samples; ++i) {
                std::cout << discrete_dist->sample_discrete() << " ";
            }
            std::cout << std::endl;
        } else {
            // It's a continuous distribution, call a generic sample() method.
            std::cout << "Generated continuous samples:" << std::endl;
            // You will need to implement a function like this on your
            // Distribution base class.
            // Example: std::vector<double> samples = dist->sample(num_samples);
            for (int i = 0; i < num_samples; ++i) {
                std::cout << dist->sample() << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "Successfully generated " << num_samples << " samples from a " << type_string << " distribution." << std::endl;
        std::cout << "Parameters: ";
        for (double p : params) {
            std::cout << p << " ";
        }
        std::cout << std::endl;
    } catch (const std::invalid_argument& e) {
        std::cout << "Error: " << e.what() << std::endl;
    }
});