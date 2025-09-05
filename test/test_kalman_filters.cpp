#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "kalman_filter.h"
#include "extended_kalman_filter.h"
#include "unscented_kalman_filter.h"

TEST(KalmanFilterTest, LinearPrediction) {
    int n = 2, m = 1;
    double dt = 1.0;
    Eigen::MatrixXd A(n, n); A << 1, dt, 0, 1;
    Eigen::MatrixXd C(m, n); C << 1, 0;
    Eigen::MatrixXd Q = 0.001 * Eigen::MatrixXd::Identity(n, n);
    Eigen::MatrixXd R = 0.1 * Eigen::MatrixXd::Identity(m, m);
    Eigen::MatrixXd P = Eigen::MatrixXd::Identity(n, n);
    Eigen::VectorXd x0(n); x0 << 0, 1;

    KalmanFilter kf(dt, A, C, Q, R, P);
    kf.init(x0);
    kf.predict();
    Eigen::VectorXd x_pred = kf.state();
    EXPECT_NEAR(x_pred(0), 1.0, 1e-6);
    EXPECT_NEAR(x_pred(1), 1.0, 1e-6);
}

TEST(ExtendedKalmanFilterTest, NonlinearPrediction) {
    int n = 2, m = 1;
    double dt = 1.0;
    Eigen::VectorXd x0(n); x0 << 0, 1;
    Eigen::MatrixXd P0 = Eigen::MatrixXd::Identity(n, n);
    Eigen::MatrixXd Q = 0.001 * Eigen::MatrixXd::Identity(n, n);
    Eigen::MatrixXd R = 0.1 * Eigen::MatrixXd::Identity(m, m);

    auto f = [dt](const Eigen::VectorXd& x) {
        Eigen::VectorXd x_new(2);
        x_new(0) = x(0) + x(1) * dt;
        x_new(1) = x(1);
        return x_new;
    };
    auto F = [dt](const Eigen::VectorXd& x) {
        Eigen::MatrixXd J(2,2);
        J << 1, dt, 0, 1;
        return J;
    };
    auto h = [](const Eigen::VectorXd& x) {
        Eigen::VectorXd z(1);
        z(0) = std::sqrt(x(0)*x(0) + 1.0);
        return z;
    };
    auto H = [](const Eigen::VectorXd& x) {
        Eigen::MatrixXd J(1,2);
        J(0,0) = x(0) / std::sqrt(x(0)*x(0) + 1.0);
        J(0,1) = 0;
        return J;
    };

    ExtendedKalmanFilter ekf(x0, P0, Q, R);
    ekf.setProcessModel(f, F);
    ekf.setMeasurementModel(h, H);
    ekf.predict();
    Eigen::VectorXd x_pred = ekf.state();
    EXPECT_NEAR(x_pred(0), 1.0, 1e-6);
    EXPECT_NEAR(x_pred(1), 1.0, 1e-6);
}

TEST(UnscentedKalmanFilterTest, NonlinearPrediction) {
    int n = 2, m = 1;
    double dt = 1.0;
    Eigen::VectorXd x0(n); x0 << 0, 1;
    Eigen::MatrixXd P0 = Eigen::MatrixXd::Identity(n, n);
    Eigen::MatrixXd Q = 0.001 * Eigen::MatrixXd::Identity(n, n);
    Eigen::MatrixXd R = 0.1 * Eigen::MatrixXd::Identity(m, m);

    auto f = [dt](const Eigen::VectorXd& x) {
        Eigen::VectorXd x_new(2);
        x_new(0) = x(0) + x(1) * dt;
        x_new(1) = x(1);
        return x_new;
    };
    auto h = [](const Eigen::VectorXd& x) {
        Eigen::VectorXd z(1);
        z(0) = std::sqrt(x(0)*x(0) + 1.0);
        return z;
    };

    UnscentedKalmanFilter ukf(n, m);
    ukf.initialize(x0, P0);
    ukf.setProcessModel(f, Q);
    ukf.setMeasurementModel(h, R);
    ukf.predict();
    Eigen::VectorXd x_pred = ukf.state();
    EXPECT_NEAR(x_pred(0), 1.0, 1e-6);
    EXPECT_NEAR(x_pred(1), 1.0, 1e-6);
}