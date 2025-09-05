#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include "kalman_filter.h"
#include "extended_kalman_filter.h"
#include "unscented_kalman_filter.h"

void testHarnessStandard() {
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
}


void testHarnessEKF() {
    // State: [position, velocity]
    int n = 2;
    int m = 1;
    double dt = 1.0;

    // Initial state and covariance
    Eigen::VectorXd x0(n);
    x0 << 0, 1;
    Eigen::MatrixXd P0 = Eigen::MatrixXd::Identity(n, n);

    // Process and measurement noise
    Eigen::MatrixXd Q = 0.001 * Eigen::MatrixXd::Identity(n, n);
    Eigen::MatrixXd R = 0.1 * Eigen::MatrixXd::Identity(m, m);

    // Nonlinear process model: x_k+1 = [x + v*dt, v]
    auto f = [dt](const Eigen::VectorXd& x) -> Eigen::VectorXd {
        Eigen::VectorXd x_new(2);
        x_new(0) = x(0) + x(1) * dt;
        x_new(1) = x(1);
        return x_new;
    };

    // Jacobian of process model
    auto F = [dt](const Eigen::VectorXd& x) -> Eigen::MatrixXd {
        Eigen::MatrixXd J(2,2);
        J << 1, dt,
             0, 1;
        return J;
    };

    // Nonlinear measurement model: z = sqrt(x^2 + 1)
    auto h = [](const Eigen::VectorXd& x) -> Eigen::VectorXd {
        Eigen::VectorXd z(1);
        z(0) = std::sqrt(x(0)*x(0) + 1.0);
        return z;
    };

    // Jacobian of measurement model
    auto H = [](const Eigen::VectorXd& x) -> Eigen::MatrixXd {
        Eigen::MatrixXd J(1,2);
        J(0,0) = x(0) / std::sqrt(x(0)*x(0) + 1.0);
        J(0,1) = 0;
        return J;
    };

    // Create EKF and set models
    ExtendedKalmanFilter ekf(x0, P0, Q, R);
    ekf.setProcessModel(f, F);
    ekf.setMeasurementModel(h, H);

    // Example nonlinear measurements (simulate noisy sqrt(position^2 + 1))
    std::vector<double> measurements = {1.1, 1.6, 2.2, 2.8, 3.3};

    std::cout << "Extended Kalman Filter Tracking:" << std::endl;
    for (double z_val : measurements) {
        ekf.predict();

        Eigen::VectorXd z(1);
        z << z_val;
        ekf.update(z);

        Eigen::VectorXd estimated_state = ekf.state();
        std::cout << "Measurement: " << z_val << "\t";
        std::cout << "Estimated state: [pos: " << estimated_state(0) << ", vel: " << estimated_state(1) << "]" << std::endl;
    }
}

void testHarnessUKF() {
    // State: [position, velocity]
    int n = 2;
    int m = 1;
    double dt = 1.0;

    // Initial state and covariance
    Eigen::VectorXd x0(n);
    x0 << 0, 1;
    Eigen::MatrixXd P0 = Eigen::MatrixXd::Identity(n, n);

    // Process and measurement noise
    Eigen::MatrixXd Q = 0.001 * Eigen::MatrixXd::Identity(n, n);
    Eigen::MatrixXd R = 0.1 * Eigen::MatrixXd::Identity(m, m);

    // Nonlinear process model: x_k+1 = [x + v*dt, v]
    auto f = [dt](const Eigen::VectorXd& x) -> Eigen::VectorXd {
        Eigen::VectorXd x_new(2);
        x_new(0) = x(0) + x(1) * dt;
        x_new(1) = x(1);
        return x_new;
    };

    // Nonlinear measurement model: z = sqrt(x^2 + 1)
    auto h = [](const Eigen::VectorXd& x) -> Eigen::VectorXd {
        Eigen::VectorXd z(1);
        z(0) = std::sqrt(x(0)*x(0) + 1.0);
        return z;
    };

    // Create UKF and set models
    UnscentedKalmanFilter ukf(n, m);
    ukf.initialize(x0, P0);
    ukf.setProcessModel(f, Q);
    ukf.setMeasurementModel(h, R);

    // Example nonlinear measurements (simulate noisy sqrt(position^2 + 1))
    std::vector<double> measurements = {1.1, 1.6, 2.2, 2.8, 3.3};

    std::cout << "Unscented Kalman Filter Tracking:" << std::endl;
    for (double z_val : measurements) {
        ukf.predict();

        Eigen::VectorXd z(1);
        z << z_val;
        ukf.update(z);

        Eigen::VectorXd estimated_state = ukf.state();
        std::cout << "Measurement: " << z_val << "\t";
        std::cout << "Estimated state: [pos: " << estimated_state(0) << ", vel: " << estimated_state(1) << "]" << std::endl;
    }
}

int main() {
    testHarnessStandard();
    testHarnessEKF();
    testHarnessUKF();
    return 0;
}