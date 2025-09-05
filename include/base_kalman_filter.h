#ifndef BASE_KALMAN_FILTER_H
#define BASE_KALMAN_FILTER_H

#include <Eigen/Dense>

/**
 * @brief Abstract base class for Kalman filter variants.
 */
class BaseKalmanFilter {
public:
    virtual ~BaseKalmanFilter() = default;

    /**
     * @brief Predicts the next state.
     */
    virtual void predict() = 0;

    /**
     * @brief Updates the state with a new measurement.
     * @param z Measurement vector.
     */
    virtual void update(const Eigen::VectorXd& z) = 0;

    /**
     * @brief Returns the current state estimate.
     */
    virtual const Eigen::VectorXd& state() const = 0;

    /**
     * @brief Returns the current state covariance.
     */
    virtual const Eigen::MatrixXd& covariance() const = 0;
};

#endif // BASE_KALMAN_FILTER_H