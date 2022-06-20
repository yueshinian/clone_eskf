// ros
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/Pose.h>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
// C++
#include <vector>

#define stateNums 15

namespace ESKF{
class eskf {
public:
  eskf() {
    isSystemInit = false;
    stateX.position = Eigen::Vector3d::Zero();
    stateX.velocity = Eigen::Vector3d::Zero();
    stateX.rotation = Eigen::Matrix3d::Identity();
    stateX.accBias = 0.0004*Eigen::Vector3d::Identity();
    stateX.gyrBias = (2.0e-5)*Eigen::Vector3d::Identity();
    stateX.quaternion = Eigen::Quaterniond::Identity();
    //
    dt = 0.0;
    lastTime = 0.0;
    time = 0.0;
    lastAcc = Eigen::Vector3d::Zero();
    lastGyr = Eigen::Vector3d::Zero();
    lastQuat = Eigen::Quaterniond::Identity();
    averageAcc = Eigen::Vector3d::Zero();
    averageGyr = Eigen::Vector3d::Zero();
    g_ = -9.81;
    gravity << 0.0, 0.0, g_;
    isQuat = true;
    // eskf

    // Eigen::Matrix<double,stateNums,1> stateX;
    errorState = Eigen::MatrixXd::Zero(15, 1);
    Fx = Eigen::MatrixXd::Zero(stateNums, stateNums);
    Fi = Eigen::MatrixXd::Zero(stateNums, 12);
    P = Eigen::MatrixXd::Zero(stateNums, stateNums);
    Kk = Eigen::MatrixXd::Zero(stateNums, 6);
    H = Eigen::MatrixXd::Zero(6, 15);
    C = Eigen::MatrixXd::Zero(6, 6);
    // noise cov
    W_imuNoise = Eigen::MatrixXd::Zero(12, 1);
    //过程噪声
    Q_processNoiseCov = Eigen::MatrixXd::Zero(12, 12);
    sigmaAcc = 0.1;
    sigmaGyr = 0.01;
    sigmaAccBias = 0.0001;
    sigmaGyrBias = 0.0001;
    //观测噪声
    R_ObserveNoiseCov = Eigen::MatrixXd::Zero(6, 6);
    sigmaPos = 0.01;
    sigmaTheta = 0.01;
    //
    noisePos =0.5;
    noiseVel = 0.5;
    noiseQuat = 0.01;
    noiseBiasAcc = 3.5e-1;
    noiseBiasGyr = 1.5e-2;
    deltay_observe = Eigen::MatrixXd::Zero(6, 1);
    // utils
    I = Eigen::Matrix3d::Identity();
    Z = Eigen::Matrix3d::Zero();
    I_stateNums = Eigen::MatrixXd::Identity(stateNums, stateNums);
  }
  ~eskf() {}

  void initFilter()
   {
       P(0,0) = P(1,1) = P(2,2) = noisePos*noisePos;
       P(3,3) = P(4,4) = P(5,5) = noiseVel*noiseVel;
       P(6,6) = P(7,7) = P(8,8) = noiseQuat*noiseQuat;
       P(9,9) = P(10,10) = P(11,11) = noiseBiasAcc*noiseBiasAcc;
       P(12,12) = P(13,13) = P(14,14) =noiseBiasGyr*noiseBiasGyr;

       Q_processNoiseCov(0,0) = Q_processNoiseCov(1,1) = Q_processNoiseCov(2,2) = sigmaAcc*sigmaAcc;
       Q_processNoiseCov(3,3) = Q_processNoiseCov(4,4) = Q_processNoiseCov(5,5) = sigmaGyr*sigmaGyr;
       Q_processNoiseCov(6,6) = Q_processNoiseCov(7,7) = Q_processNoiseCov(8,8) = sigmaAccBias*sigmaAccBias;
       Q_processNoiseCov(9,9) = Q_processNoiseCov(10,10) = Q_processNoiseCov(11,11) = sigmaGyrBias*sigmaGyrBias;   

      H << I, Z, Z, Z, Z,
             Z, Z, I, Z, Z;
      C << I, Z, 
              Z, I;

       R_ObserveNoiseCov.block<3,3>(0,0) = sigmaPos*sigmaPos*I;
       R_ObserveNoiseCov.block<3,3>(3,3) = sigmaTheta*sigmaTheta*I;
   }

  void predict(sensor_msgs::Imu &imuMsg) {
    if (!isSystemInit) {
      lastTime = imuMsg.header.stamp.toSec();
      lastAcc(0) = imuMsg.linear_acceleration.x;
      lastAcc(1) = imuMsg.linear_acceleration.y;
      lastAcc(2) = imuMsg.linear_acceleration.z;
      lastGyr(0) = imuMsg.angular_velocity.x;
      lastGyr(1) = imuMsg.angular_velocity.y;
      lastGyr(2) = imuMsg.angular_velocity.z;
      lastQuat.x() = imuMsg.orientation.x;
      lastQuat.y() = imuMsg.orientation.y;
      lastQuat.z() = imuMsg.orientation.z;
      lastQuat.w() = imuMsg.orientation.w;
      initFilter();
      isSystemInit = true;
      return;
    } else {
      time = imuMsg.header.stamp.toSec();
      dt = time - lastTime;
      lastTime = time;
      if (dt < 0) {
        ROS_ERROR("imu time is error!");
        return;
      }
    }

    if (!isQuat) {
      Eigen::Quaterniond imuQuat;
      imuQuat.x() = imuMsg.orientation.x;
      imuQuat.y() = imuMsg.orientation.y;
      imuQuat.z() = imuMsg.orientation.z;
      imuQuat.w() = imuMsg.orientation.w;
      Eigen::Matrix3d rotationMatrix = imuQuat.toRotationMatrix();
      Eigen::Vector3d temp_gravity(0, 0, g_);
      Eigen::Vector3d acc_no_gravity = rotationMatrix.transpose() * temp_gravity;
      imuMsg.linear_acceleration.x += acc_no_gravity(0);
      imuMsg.linear_acceleration.y += acc_no_gravity(1);
      imuMsg.linear_acceleration.z += acc_no_gravity(2);
      gravity.setZero();
    }

    updatePCovariance();
    updateNominalState(imuMsg);
    updatePredictError();
  }

  geometry_msgs::Pose correct(const geometry_msgs::Pose &pose) {
    Eigen::Vector3d observePosetion(pose.position.x, pose.position.y,
                                                                        pose.position.z);
    Eigen::Quaterniond observeQuat;
    observeQuat.x() = pose.orientation.x;
    observeQuat.y() = pose.orientation.y;
    observeQuat.z() = pose.orientation.z;
    observeQuat.w() = pose.orientation.w;
    deltay_observe.block<3,1>(0,0) = stateX.position - observePosetion;
    Eigen::Matrix3d deltaR = observeQuat.toRotationMatrix().transpose()*stateX.rotation - I;
    Eigen::Vector3d deltaTheta((deltaR(2,1)-deltaR(1,2))/2.0,(deltaR(0,2)-deltaR(2,0))/2.0,(deltaR(1,0) - deltaR(0,1))/2.0);
    deltay_observe.block<3,1>(3,0) = deltaTheta;
    updateKGain();
    updateError();
    updateState();
    resetError();
    geometry_msgs::Pose correctPose;
    correctPose.position.x = stateX.position.x();
    correctPose.position.y = stateX.position.y();
    correctPose.position.z = stateX.position.z();
    correctPose.orientation.x = stateX.quaternion.x();
    correctPose.orientation.y = stateX.quaternion.y();
    correctPose.orientation.z = stateX.quaternion.z();
    correctPose.orientation.w = stateX.quaternion.w();
    return correctPose;
  }

  void updateKGain() {
    Kk = P * H.transpose() * (H * P * H.transpose() + C * R_ObserveNoiseCov * C.transpose()).inverse();
  }

  void updateError() {
    errorState = errorState + Kk * (deltay_observe - H * errorState);
    P = (I_stateNums - Kk * H) * P;
    P = (P + P.transpose())/2.0;
  }

  void updateState() {
    stateX.position -= errorState.block<3, 1>(0, 0);
    stateX.velocity -= errorState.block<3, 1>(3, 0);
    // stateX.quaternion *= errorState.block<3,1>(6,0);
    // stateX.rotation *= ()
    stateX.accBias -= errorState.block<3, 1>(9, 0);
    stateX.gyrBias -= errorState.block<3, 1>(12, 0);
  }

  void resetError() { errorState.setZero(); }

  void updatePredictError() {
    // think init error is 0
    errorState = Fx * errorState + Fi * W_imuNoise;
  }

  void updatePCovariance() {
    Eigen::Matrix<double, 3, 3> R = stateX.quaternion.toRotationMatrix();
    Eigen::Matrix<double, 3, 3> Ra = skewMatrix(averageAcc);
    Eigen::Matrix<double, 3, 3> Rw = skewMatrix(averageGyr*dt);
    //Eigen::Matrix<double, 3, 3> Rw = euler_angle_to_rotation_matrix(averageGyr*dt);
    // Eigen::Vector3d angle = averageGyr*dt;
    // Eigen::Matrix<double, 3, 3> Rw = a

    // Fx << I, I * dt, Z, Z, Z, 
    //             Z,  I, -R * Ra * dt, -R * dt, Z, 
    //             Z, Z, R.transpose() * Rw , Z, -I * dt,
    //              Z, Z, Z, I, Z, 
    //              Z, Z, Z, Z, I;
    // Fx << I, I * dt, Z, Z, Z, 
    //             Z,  I, -R * Ra * dt, -R * dt, Z, 
    //             Z, Z, I- R.transpose()*Rw , Z, -I * dt,
    //              Z, Z, Z, I, Z, 
    //              Z, Z, Z, Z, I;
    Fx = Eigen::MatrixXd::Identity(stateNums, stateNums);
    Fx.block<3,3>(0,3) = I*dt;
    Fx.block<3,3>(3,6) = -R*Ra*dt;
    Fx.block<3,3>(3,9) = -R*dt;
    Eigen::Vector3d delta_angle_axis = averageGyr*dt;
    if(delta_angle_axis.norm() > 1e-12){
      Fx.block<3,3>(6,6) = Eigen::AngleAxisd(delta_angle_axis.norm(), delta_angle_axis.normalized);
    }else{
      //Fx.block<3,3>(6,6) = I;
    }
    Fx.block<3,3>(6,12) = -I*dt;

    Fi << Z, Z, Z, Z,
             R, Z, Z, Z,
              Z, I, Z, Z,
               Z, Z, I, Z,
                Z, Z, Z, I;

    Q_processNoiseCov << sigmaAcc*dt*sigmaAcc*dt* I, Z, Z, Z, 
                                                    Z, sigmaGyr*dt*sigmaGyr*dt * I, Z, Z, 
                                                    Z, Z, sigmaAccBias*dt*sigmaAccBias*dt * I, Z,
                                                     Z, Z, Z, sigmaGyrBias*dt*sigmaGyrBias*dt * I;

    P = (Fx * P * Fx.transpose() + Fi * Q_processNoiseCov * Fi.transpose()).eval();
    P = (P + P.transpose())/2.0;
  }

  void updateNominalState(const sensor_msgs::Imu imu) {
    Eigen::Vector3d p = stateX.position;
    Eigen::Vector3d v = stateX.velocity;
    Eigen::Quaterniond q = stateX.quaternion;
    Eigen::Matrix3d r = stateX.rotation;
    Eigen::Vector3d acc_bias = stateX.accBias;
    Eigen::Vector3d gyr_bias = stateX.gyrBias;

    Eigen::Vector3d last_acc = q * (lastAcc - acc_bias) + gravity;
    lastAcc << imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z;

    Eigen::Quaterniond cur_quat;
    cur_quat.x() = imu.orientation.x;
    cur_quat.y() = imu.orientation.y;
    cur_quat.z() = imu.orientation.z;
    cur_quat.w() = imu.orientation.w;
    //Eigen::Matrix3d d_rotation(lastQuat.inverse()*cur_quat);
    Eigen::Quaterniond delta_quat = lastQuat.inverse()*cur_quat;
    lastQuat = cur_quat;
    Eigen::Vector3d cur_gyr{imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z};
    averageGyr = 0.5 * (lastGyr + cur_gyr) - gyr_bias;
    lastGyr = cur_gyr;
    //Eigen::Vector3d half_theta = averageGyr*dt/2.0;
    //Eigen::Quaterniond temp_q;
    //temp_q.w() = 1.0;
    //temp_q.x() = half_theta.x();
    //temp_q.y() = half_theta.y();
    //temp_q.z() = half_theta.z();
    //q = q * temp_q;  //旋转矢量转四元数，视觉slam的page60
    q *= delta_quat;
    q.normalize();
    Eigen::Vector3d d_theta = (stateX.gyrBias)*dt;
    if(d_theta.norm() >= 1e-12){
      Eigen::AngleAxisd d_rotation2(d_theta.norm(), d_theta.normalized());
      q *= Eigen::Quaterniond(d_rotation2);
    }else{
      d_theta = averageGyr*dt;
      if(d_theta.norm() >= 1e-12){
        Eigen::AngleAxisd d_rotation3(d_theta.norm(),d_theta.normalized());
        q *= Eigen::Quaterniond(d_rotation3);
      }
    }
    stateX.quaternion.normalize();

    Eigen::Vector3d cur_acc = stateX.quaternion * (lastAcc - acc_bias) + gravity;
    averageAcc = 0.5 * (last_acc + cur_acc);

    p = p + v * dt + 0.5 * averageAcc * dt * dt;
    v = v + averageAcc * dt;

    stateX.position = p;
    stateX.velocity = v;
    stateX.quaternion = q;
    stateX.rotation = stateX.quaternion.toRotationMatrix();
    stateX.accBias = acc_bias;
    stateX.gyrBias = gyr_bias;
  }

    //3维列向量反对称矩阵,视觉slam,page43
    Eigen::Matrix3d skewMatrix(const Eigen::Vector3d &vec)
    {
        Eigen::Matrix3d skew_matrix;
        skew_matrix << 0,-vec(2),vec(1),
                                            vec(2),0,-vec(0),
                                            -vec(1),vec(0),0;
        return skew_matrix;
    }

    Eigen::Matrix3d angular2matrix(const Eigen::Vector3d &vec)
    {

    }

    Eigen::Matrix3d euler_angle_to_rotation_matrix(const Eigen::Vector3d w) {
        Eigen::Matrix3d R;
        R = Eigen::AngleAxisd(w[2], Eigen::Vector3d::UnitZ()) *
            Eigen::AngleAxisd(w[1], Eigen::Vector3d::UnitY()) *
            Eigen::AngleAxisd(w[0], Eigen::Vector3d::UnitX());
        return R;
    }

    void setPosition(const Eigen::Vector3d &position)
    {   
        stateX.position = position;
    }

    void setOrientation(const Eigen::Quaterniond &quaternion)
    {
        stateX.quaternion = quaternion;
        stateX.rotation = stateX.quaternion.toRotationMatrix();
    }

    Eigen::Vector3d getPosition()
    {
      return stateX.position;
    }

    Eigen::Quaterniond getQuaternion()
    {
      return stateX.quaternion;
    }
private:
  struct state {
    Eigen::Vector3d position;
    Eigen::Vector3d velocity;
    Eigen::Quaterniond quaternion;
    Eigen::Matrix3d rotation;
    Eigen::Vector3d accBias;
    Eigen::Vector3d gyrBias;
  };
  state stateX;
  // ros
  std::vector<sensor_msgs::Imu> imuBuf;
  double lastTime;
  double time;
  bool isSystemInit;
  double dt;
  // imu set
  Eigen::Vector3d lastAcc;
  Eigen::Vector3d lastGyr;
  Eigen::Quaterniond lastQuat;
  Eigen::Vector3d averageAcc;
  Eigen::Vector3d averageGyr;
  Eigen::Vector3d gravity;
  double g_;
  bool isQuat;
  // eskf

  // Eigen::Matrix<double,stateNums,1> stateX;
  Eigen::Matrix<double, stateNums, 1> errorState;
  Eigen::Matrix<double, stateNums, stateNums> Fx;
  Eigen::Matrix<double, stateNums, 12> Fi;
  Eigen::Matrix<double, stateNums, stateNums> P;
  double noisePos,noiseVel,noiseQuat,noiseBiasAcc,noiseBiasGyr;//P矩阵初始化
  Eigen::Matrix<double, stateNums, 6> Kk;
  Eigen::Matrix<double, 6, 15> H;
  Eigen::Matrix<double, 6, 6> C;
  // noise cov
  Eigen::Matrix<double, 12, 12> Q_processNoiseCov;//
  double sigmaAcc;
  double sigmaGyr;
  double sigmaAccBias;
  double sigmaGyrBias;
  Eigen::Matrix<double, 6, 6> R_ObserveNoiseCov;//观测p,theta
  double sigmaPos;
  double sigmaTheta;
  Eigen::Matrix<double, 6, 1> deltay_observe;
  Eigen::Matrix<double, 12, 1> W_imuNoise;
  // utils
  Eigen::Matrix3d I;
  Eigen::Matrix3d Z;
  Eigen::MatrixXd I_stateNums;
};
}