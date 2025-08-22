#include "ukf.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);
  x_.setZero();

  // initial covariance matrix
  P_ = MatrixXd(5, 5);
  P_.setZero();

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 2;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.5;

  is_initialized_ = false;

  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */

    n_x_ = 5;

    // Augmented state dimension
    n_aug_ = 7;

    // Sigma point spreading parameter
    lambda_ = 3-n_aug_;

    weights_ = VectorXd((2*n_aug_)+1);

    for(int i =0; i< 2*n_aug_+1; i++){
        if(i==0){
            weights_(i) = lambda_ / (lambda_ + n_aug_);
        }
        else{
            weights_(i) = 1 / (2*(lambda_ + n_aug_));
        }
    }

    x_aug_ = VectorXd(n_aug_);
    // Augmented state covariance_matrix
    P_aug_ = MatrixXd(n_aug_,n_aug_);
    P_aug_.setZero();

    Q_ = MatrixXd(2,2);
    Q_.setZero();
    Q_(0,0) = pow(std_a_,2);
    Q_(1,1) = pow(std_yawdd_,2);

    A_ = MatrixXd(n_aug_,n_aug_);

    Xsig_aug_ = MatrixXd(n_aug_, 2 * n_aug_ + 1);
    Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

     n_z_radar_ = 3;
     n_z_lidar_ = 2;
     Zsig_radar_ = MatrixXd(n_z_radar_, 2 * n_aug_ + 1);
     Zsig_lidar_ = MatrixXd(n_z_lidar_, 2 * n_aug_ + 1);

     S_radar_ = MatrixXd(n_z_radar_,n_z_radar_);
     S_lidar_ = MatrixXd (n_z_lidar_, n_z_lidar_);

     z_pred_radar_ = VectorXd(n_z_radar_);

    R_radar_ = MatrixXd::Zero(n_z_radar_,n_z_radar_);
    R_radar_.diagonal() << pow(std_radr_,2), pow(std_radphi_,2), pow(std_radrd_,2);

    Tc_radar_ = MatrixXd(n_x_, n_z_radar_);

    z_radar_ = VectorXd(n_z_radar_);
    z_lidar_ = VectorXd(n_z_lidar_);

    H_ =  MatrixXd(n_z_lidar_,n_x_);
    H_ << 1,0,0,0,0,
          0,1,0,0,0;

    R_lidar_ = MatrixXd::Zero(n_z_lidar_,n_z_lidar_);
    R_lidar_.diagonal() << pow(std_laspx_,2), pow(std_laspy_,2);
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */

  if(meas_package.sensor_type_ ==MeasurementPackage::LASER){
      if(is_initialized_){
          UpdateLidar(meas_package);
      }
      else {
          // initialize the state with lidar measurements
          x_(0) = meas_package.raw_measurements_(0) ;
          x_(1) = meas_package.raw_measurements_(1);

          // initialize the covariance matrix
          P_.diagonal() << pow(std_laspx_,2),pow(std_laspy_,2),10,0.5,0.5;
          is_initialized_ = true;
      }
  }
  else if(meas_package.sensor_type_ ==MeasurementPackage::RADAR) {
      if(is_initialized_) {
          UpdateRadar(meas_package);
      }
      else{
          //initialize the state with radar measurements
          double rho = meas_package.raw_measurements_(0);
          double phi = meas_package.raw_measurements_(1);
          double rho_dot = meas_package.raw_measurements_(2);

          x_(0) = rho*cos(phi);
          x_(1) = rho*sin(phi);
//          x_(2) = std::abs(rho_dot);

          P_.diagonal() << pow(std_radr_,2),pow(std_radr_,2),5,0.5,0.5;

          is_initialized_ = true;
      }
  }
}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */

  /// generate the Augmented sigma points

  x_aug_.setZero();
  x_aug_.head(n_x_) = x_;

   P_aug_.setZero();
   P_aug_.block(0,0,n_x_,n_x_) = P_;
   P_aug_.block(5,5,2,2) = Q_;
   A_ = ((lambda_ + n_aug_)*P_aug_).llt().matrixL();

   Xsig_aug_.col(0) = x_aug_;
   Xsig_aug_.block(0,1,n_aug_,n_aug_) = x_aug_.replicate(1, n_aug_) + A_;
   Xsig_aug_.block(0,n_aug_+1,n_aug_,n_aug_) = x_aug_.replicate(1, n_aug_) - A_;

   /// prediction step for the Augmented sigma points

    VectorXd Xsig_pred_vec = VectorXd(n_aug_);

    double p_x, p_y, v, psi, psi_dot, nu_a, nu_psi_dot_dot;
    double p_x_new, p_y_new, v_new ,psi_new, psi_dot_new;

    for(int i=0;i< 2 * n_aug_ + 1; i++) {

        Xsig_pred_vec = Xsig_aug_.col(i);

        p_x = Xsig_pred_vec(0);
        p_y = Xsig_pred_vec(1);
        v = Xsig_pred_vec(2);
        psi = Xsig_pred_vec(3);
        psi_dot = Xsig_pred_vec(4);
        nu_a = Xsig_pred_vec(5);
        nu_psi_dot_dot = Xsig_pred_vec(6);

        VectorXd Xnoise_vec = VectorXd(n_x_);
        Xnoise_vec << 0.5 * pow(delta_t, 2) * cos(psi) * nu_a,
                0.5 * pow(delta_t, 2) * sin(psi) * nu_a,
                delta_t * nu_a,
                0.5 * pow(delta_t, 2) * nu_psi_dot_dot,
                delta_t * nu_psi_dot_dot;

        if (fabs(psi_dot) < 0.001) {

            p_x_new = p_x + v * cos(psi) * delta_t;
            p_y_new = p_y + v * sin(psi) * delta_t;

        } else {

            p_x_new = p_x + (v / psi_dot) * (sin(psi + (psi_dot * delta_t)) - sin(psi));
            p_y_new = p_y + (v / psi_dot) * (-cos(psi + (psi_dot * delta_t)) + cos(psi));

        }

        v_new = v + 0;
        psi_new = psi + psi_dot * delta_t;
        psi_dot_new = psi_dot + 0;

        VectorXd Xsig_pred_vec_new = VectorXd(n_x_);
        VectorXd Xsig_pred_vec_noise = VectorXd(n_x_);

        Xsig_pred_vec_new << p_x_new, p_y_new, v_new, psi_new, psi_dot_new;
        Xsig_pred_vec_noise = Xsig_pred_vec_new + Xnoise_vec;

        Xsig_pred_.col(i) = Xsig_pred_vec_noise;

    }

        /// Predicted mean and covariance

        p_x = Xsig_pred_.row(0).transpose().dot(weights_);
        p_y = Xsig_pred_.row(1).transpose().dot(weights_);
        v =   Xsig_pred_.row(2).transpose().dot(weights_);
        psi = Xsig_pred_.row(3).transpose().dot(weights_);
        psi_dot = Xsig_pred_.row(4).transpose().dot(weights_);

        // predict state mean
        x_ <<p_x,p_y,v,psi,psi_dot;

        // predict state covariance matrix
        VectorXd diff = VectorXd(n_x_);

        P_.setZero();
        for (int i=0; i <2*n_aug_+1; i++ ){
            double w = weights_(i);
            diff = Xsig_pred_.col(i) - x_;

            while (diff(3)> M_PI) diff(3)-=2.*M_PI;   //wrapping around psi to be within limits
            while (diff(3)<-M_PI) diff(3)+=2.*M_PI;

            P_ += w* diff*diff.transpose();
        }

}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */

    z_lidar_ = meas_package.raw_measurements_ ;

    z_pred_lidar_ = H_ * x_;
    VectorXd innovation = z_lidar_ - z_pred_lidar_;
    MatrixXd Ht = H_.transpose();
    S_lidar_ = H_ * P_ * Ht + R_lidar_;
    MatrixXd K = P_ * Ht*(S_lidar_.inverse());

    //new estimate
    x_ = x_ + (K * innovation);

    while (x_(3) > M_PI) x_(3) -= 2. * M_PI;
    while (x_(3) < -M_PI) x_(3) += 2. * M_PI;

    MatrixXd I = MatrixXd::Identity(n_x_, n_x_);
    P_ = (I - K * H_) * P_;

}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */

    /// transform sigma points into measurement space

    for (int i=0; i< 2 * n_aug_ + 1; i++){

        VectorXd Xsig_vec = VectorXd(n_x_);
        VectorXd z_vec = VectorXd(n_z_radar_);
        Xsig_vec = Xsig_pred_.col(i);

        double p_x, p_y, v, psi,psi_dot, rho,phi,rho_dot;

        p_x = Xsig_vec(0);
        p_y = Xsig_vec(1);
        v = Xsig_vec(2);
        psi = Xsig_vec(3);
        psi_dot = Xsig_vec(4);

        rho = pow( (pow(p_x,2)+pow(p_y,2)),0.5);
        phi = std::atan2(p_y,p_x);
        if (rho < 1e-6) rho = 1e-6;
        rho_dot = (p_x*cos(psi)*v + p_y*sin(psi)*v )/rho;

        Zsig_radar_.col(i) << rho, phi, rho_dot;
    }

    /// calculate mean predicted measurement


    for(int i=0;i <n_z_radar_; i++){
        z_pred_radar_(i) = Zsig_radar_.row(i).dot(weights_);
    }


    /// calculate innovation covariance matrix S
    S_radar_.setZero();
    for (int i=0; i< 2 * n_aug_ + 1; i++){

        VectorXd diff = VectorXd(n_z_radar_);
        diff = Zsig_radar_.col(i) - z_pred_radar_;

        while (diff(1)> M_PI) diff(1)-=2.*M_PI;
        while (diff(1)<-M_PI) diff(1)+=2.*M_PI;

        S_radar_ += weights_(i)* diff*diff.transpose();
    }

    S_radar_ = S_radar_ + R_radar_;

// create and populate matrix for cross correlation Tc_radar_

    VectorXd x_diff = VectorXd(n_x_);
    VectorXd z_diff = VectorXd(n_z_radar_);

    Tc_radar_.setZero();
    for (int i=0; i<2*n_aug_+1; ++i) {

        x_diff = Xsig_pred_.col(i) - x_;

        while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
        while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

        z_diff = Zsig_radar_.col(i) - z_pred_radar_;

        while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
        while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

        Tc_radar_ += weights_(i)*x_diff*z_diff.transpose();
    }

    // calculate Kalman gain K;

    MatrixXd K = MatrixXd(n_x_,n_z_radar_);
    K = Tc_radar_*S_radar_.inverse();

    // update state mean and covariance matrix
    z_radar_ = meas_package.raw_measurements_ ;
    z_diff = z_radar_-z_pred_radar_;

    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    x_ = x_ + K*(z_diff);
    P_ = P_ - K*S_radar_*K.transpose();

}