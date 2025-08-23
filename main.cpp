#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include "kalman_filter.h"

int main() {
    // Define the state space for a constant velocity model
    // State: [position, velocity]
    int n = 2; // Number of states
    int m = 1; // Number of measurements
    double dt = 1.0; // Time step

    // System dynamics matrix (A)
    Eigen::MatrixXd A(n, n);
    A << 1, dt,
         0, 1;

    // Observation matrix (C) - only observing position
    Eigen::MatrixXd C(m, n);
    C << 1, 0;

    // Process noise covariance (Q)
    Eigen::MatrixXd Q(n, n);
    Q << 0.001, 0,
         0, 0.001;

    // Measurement noise covariance (R)
    Eigen::MatrixXd R(m, m);
    R << 0.1;

    // Initial estimate error covariance (P)
    Eigen::MatrixXd P(n, n);
    P << 1, 0,
         0, 1;

    // Create the Kalman Filter
    KalmanFilter kf(dt, A, C, Q, R, P);

    // Initial state guess
    Eigen::VectorXd x0(n);
    x0 << 0, 1; // Start at position 0 with velocity 1
    kf.init(x0);

    // Some example noisy measurements
    std::vector<double> measurements = {1.2, 2.1, 3.5, 4.3, 5.8};

    std::cout << "Kalman Filter Tracking:" << std::endl;
    for (double z : measurements) {
        // Predict the next state
        kf.predict();

        // Update with the new measurement
        Eigen::VectorXd y(m);
        y << z;
        kf.update(y);

        // Print the estimated state
        Eigen::VectorXd estimated_state = kf.state();
        std::cout << "Measurement: " << z << "\t";
        std::cout << "Estimated state: [pos: " << estimated_state(0) << ", vel: " << estimated_state(1) << "]" << std::endl;
    }

    return 0;
}