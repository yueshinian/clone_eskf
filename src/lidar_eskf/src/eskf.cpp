/*
* Copyright (c) 2016 Carnegie Mellon University, Weikun Zhen <weikunz@andrew.cmu.edu>
*
* For License information please see the LICENSE file in the root directory.
*
*/

#include "lidar_eskf/eskf.h"

Eigen::Matrix3d skew(Eigen::Vector3d w) {
    Eigen::Matrix3d W;
    W <<    0.0, -w.z(),  w.y(),
          w.z(),    0.0, -w.x(),
         -w.y(),  w.x(),    0.0;
    return W;
}

Eigen::Matrix3d angle_axis_to_rotation_matrix(Eigen::Vector3d w) {
    double theta = w.norm();
    Eigen::Matrix3d  W;
    Eigen::Matrix3d  I;
    Eigen::Matrix3d  R;
    W = skew(w / theta);
    I = Eigen::Matrix3d::Identity();
    R = I * cos(theta) + W * sin(theta) + W * W * (1-cos(theta));
    return R;
}

Eigen::Matrix3d euler_angle_to_rotation_matrix(Eigen::Vector3d w) {
    Eigen::Matrix3d R;
    R = Eigen::AngleAxisd(w[2], Eigen::Vector3d::UnitZ()) *
        Eigen::AngleAxisd(w[1], Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(w[0], Eigen::Vector3d::UnitX());
    return R;
}

ESKF::ESKF(ros::NodeHandle &nh) {

    
    nh.param("imu_frequency",           _imu_freq,         50.0);
    nh.param("imu_frame",               _imu_frame,        std::string("/imu"));
    nh.param("robot_frame",             _robot_frame,      std::string("/base_frame"));
    nh.param("imu_enabled",             _imu_enabled,      false);
    nh.param("imu_has_quat",            _imu_has_quat,     false);
    nh.param("smooth_enabled",          _smooth_enabled,   false);
    nh.param("smooth_buf_size",         _smooth_buf_size,  5);
    nh.param("smooth_type",             _smooth_type,      std::string("mean"));
    
    nh.param("init_roll",               _init_roll,        0.0);
    nh.param("init_pitch",              _init_pitch,       0.0);
    nh.param("init_yaw",                _init_yaw,         0.0);
    
    nh.param("sigma_acceleration",      _sigma_acc,        0.1);
    nh.param("sigma_gyroscope",         _sigma_gyr,        0.01);
    nh.param("sigma_acceleration_bias", _sigma_bias_acc,   0.0001);
    nh.param("sigma_gyroscope_bias",    _sigma_bias_gyr,   0.00001);
    nh.param("gravity",                 _g,                9.82);
    nh.param("init_bias_acc_x",         _init_bias_acc_x,  0.0);
    nh.param("init_bias_acc_y",         _init_bias_acc_y,  0.0);
    nh.param("init_bias_acc_z",         _init_bias_acc_z,  0.0);
    nh.param("acc_queue_size",          _acc_queue_size,   5);
    nh.param("imu_transform",           _imu_transform,    false);

    // initialize nomial states
    _velocity.setZero();
    _quaternion = Eigen::AngleAxisd(_init_yaw,  Eigen::Vector3d::UnitZ())
                * Eigen::AngleAxisd(_init_pitch, Eigen::Vector3d::UnitY())
                * Eigen::AngleAxisd(_init_roll,   Eigen::Vector3d::UnitX());
    _rotation = _quaternion.toRotationMatrix();

    _position.setZero();
    _bias_acc << _init_bias_acc_x, _init_bias_acc_y, _init_bias_acc_z;
    _bias_gyr.setZero();

    // initialize error states
    _d_velocity.setZero();
    _d_theta.setZero();
    _d_rotation.setIdentity();
    _d_position.setZero();
    _d_bias_acc.setZero();
    _d_bias_gyr.setZero();

    // initialize imu
    _imu_acceleration.setZero();
    _imu_angular_velocity.setZero();
    _imu_orientation.setIdentity();

    // initialize measurements
    _m_position.setZero();
    _m_theta.setZero();
    _got_measurements = false;

    // initialize Jacobian matrix;
    _Fx.setZero();
    _Fn.setZero();

    // initialize covariance matrix
    _Sigma.setZero();
    _Q.setZero();

    // gravity
    _gravity << 0.0,0.0,_g;
    // time relatives
    _init_time = true;

    // subscriber and publisher
    _imu_sub  = nh.subscribe("/imu", 50, &ESKF::imu_callback, this);
    _odom_pub = nh.advertise<nav_msgs::Odometry>("odom", 50, true);
    _bias_pub = nh.advertise<geometry_msgs::TwistStamped>("bias", 50, true);

    // acc queue
    _acc_queue_count = 0;
    
    //smoother
    _smooth_buf_cnt = 0;
    _x_buf.resize(_smooth_buf_size);
    _y_buf.resize(_smooth_buf_size);
    _z_buf.resize(_smooth_buf_size);
    _vx_buf.resize(_smooth_buf_size);
    _vy_buf.resize(_smooth_buf_size);
    _vz_buf.resize(_smooth_buf_size);
}

ESKF::~ESKF() {
    //output_log();
}

void ESKF::imu_callback(const sensor_msgs::Imu &msg) {
    update_time(msg);
    update_imu(msg);

    propagate_state();
    propagate_error();
    propagate_covariance();

    // when a new measurement is available, update odometry
    if(_got_measurements) {
        // do measurements update
        update_error();
        update_state();
        reset_error();
        publish_bias();
        _got_measurements = false;
    }

    publish_odom();
}

void ESKF::update_time(const sensor_msgs::Imu &msg) {

    if(_init_time) {
        _dt = 1.0 / _imu_freq;
        _init_time = false;
    }
    else {
        _dt = msg.header.stamp.toSec() - _imu_time.toSec();
    }
    _imu_time = msg.header.stamp;

}

void ESKF::update_imu(const sensor_msgs::Imu &msg) {

    // stacking into a queue
    if(_acc_queue_count < _acc_queue_size) {
        _acc_queue.push_back(msg.linear_acceleration);
        _imu_acceleration[0] = msg.linear_acceleration.x;
        _imu_acceleration[1] = msg.linear_acceleration.y;
        _imu_acceleration[2] = msg.linear_acceleration.z;
    }
    else {
        _acc_queue[_acc_queue_count%_acc_queue_size] = msg.linear_acceleration;

        Eigen::Vector3d acc_avg;
        acc_avg.setZero();
        for(int i=0; i<_acc_queue_size; i++) {
            acc_avg[0] += _acc_queue[i].x / _acc_queue_size;
            acc_avg[1] += _acc_queue[i].y / _acc_queue_size;
            acc_avg[2] += _acc_queue[i].z / _acc_queue_size;
        }
        _imu_acceleration = acc_avg;
    }
    _acc_queue_count++; 

    _imu_angular_velocity[0] = msg.angular_velocity.x;
    _imu_angular_velocity[1] = msg.angular_velocity.y;
    _imu_angular_velocity[2] = msg.angular_velocity.z;

    _imu_orientation.x() = msg.orientation.x;
    _imu_orientation.y() = msg.orientation.y;
    _imu_orientation.z() = msg.orientation.z;
    _imu_orientation.w() = msg.orientation.w;

    // If imu disabled, set acc, grav to zero
    if(!_imu_enabled) {
        _imu_acceleration.setZero();
        _gravity.setZero();
    } else {
        // If imu has quaternion, remove gravity.
        if(_imu_has_quat) {
            Eigen::Matrix3d imu_rot = _imu_orientation.toRotationMatrix();
            Eigen::Vector3d grav(0.0,0.0,_g);
            _imu_acceleration += imu_rot.transpose() * grav;
            _gravity.setZero();
        }
    }

    // reproject imu to body frame
    if(_imu_transform) {
        tf::StampedTransform transform;
        try{
            _tf_listener.lookupTransform(_imu_frame, _robot_frame, ros::Time::now(), transform);
            Eigen::Matrix3d transform_body_to_imu;
            tf::matrixTFToEigen(transform.getBasis(),transform_body_to_imu);
            _imu_acceleration = transform_body_to_imu * _imu_acceleration;
            _imu_angular_velocity = transform_body_to_imu * _imu_angular_velocity;
        } catch (tf::TransformException ex){
            ROS_WARN("ESKF: imu to body transform not found. ");
        }
    }
}

void ESKF::propagate_state() {
    Eigen::Vector3d velocity;
    Eigen::Matrix3d rotation;
    Eigen::Vector3d position;
    Eigen::Vector3d bias_acc;
    Eigen::Vector3d bias_gyr;

    // system transition function for nominal state
    velocity = _velocity + (_rotation * (_imu_acceleration - _bias_acc) + _gravity ) * _dt;
    rotation = _rotation * euler_angle_to_rotation_matrix((_imu_angular_velocity - _bias_gyr) * _dt);
    position = _position + _velocity * _dt + 0.5 * (_rotation * (_imu_acceleration - _bias_acc) + _gravity) * _dt * _dt;
    bias_acc = _bias_acc;
    bias_gyr = _bias_gyr;

    // update norminal state to the next step
    _velocity   = velocity;
    _rotation   = rotation;
    _quaternion = Eigen::Quaterniond(_rotation);
    _position   = position;
    _bias_acc   = bias_acc;
    _bias_gyr   = bias_gyr;
}

void ESKF::propagate_error() {

    // system transition function for error state
    // this is not necessary because it is always zero with out measurement update
}

void ESKF::propagate_covariance() {
    // compute jacobian
    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d Z = Eigen::Matrix3d::Zero();

    Eigen::Matrix<double, 3, 3> R, R_1, R_2;
    R = _rotation;
    R_1 = skew(_imu_acceleration - _bias_acc);
    R_2 = euler_angle_to_rotation_matrix((_imu_angular_velocity - _bias_gyr) * _dt);

    _Fx <<     I,      Z,       -R*R_1*_dt,   -R*_dt,        Z,
           I*_dt,      I,                Z,        Z,        Z,
               Z,      Z,  R_2.transpose(),        Z,   -I*_dt,
               Z,      Z,                Z,        I,        Z,
               Z,      Z,                Z,        Z,        I;

    _Fn << R, Z, Z, Z,
           Z, Z, Z, Z,
           Z, I, Z, Z,
           Z, Z, I, Z,
           Z, Z, Z, I;

    _Q << pow(_sigma_acc * _dt, 2.0) * I, Z, Z, Z,
          Z, pow(_sigma_gyr * _dt, 2.0) * I, Z, Z,
          Z, Z, pow(_sigma_bias_acc * _dt, 2.0) * I, Z,
          Z, Z, Z, pow(_sigma_bias_gyr * _dt, 2.0) * I;

    // update covariance
    _Sigma = _Fx * _Sigma * _Fx.transpose() + _Fn * _Q * _Fn.transpose();
}

void ESKF::get_mean_pose(Eigen::Matrix<double, 6, 1> &mean_pose) {
    mean_pose[0] = _position.x();
    mean_pose[1] = _position.y();
    mean_pose[2] = _position.z();

    Eigen::Vector3d euler_angles = _rotation.eulerAngles(2,1,0);
    mean_pose[3] = euler_angles[2];
    mean_pose[4] = euler_angles[1];
    mean_pose[5] = euler_angles[0];
}

void ESKF::get_mean_pose(Eigen::Matrix<double, 7, 1> &mean_pose) {
    mean_pose[0] = _position.x();
    mean_pose[1] = _position.y();
    mean_pose[2] = _position.z();

    mean_pose[3] = _quaternion.w();
    mean_pose[4] = _quaternion.x();
    mean_pose[5] = _quaternion.y();
    mean_pose[6] = _quaternion.z();
}
void ESKF::get_cov_pose(Eigen::Matrix<double, 6, 6> &cov_pose) {
    cov_pose = _Sigma.block<6,6>(3,3);
}

void ESKF::publish_odom() {
    nav_msgs::Odometry msg;
    msg.header.frame_id = "world";
    msg.header.stamp = _imu_time;

    // smoother the trajectory
    _x_buf[_smooth_buf_cnt] = _position.x();
    _y_buf[_smooth_buf_cnt] = _position.y();
    _z_buf[_smooth_buf_cnt] = _position.z();
    _vx_buf[_smooth_buf_cnt] = _velocity.x();
    _vy_buf[_smooth_buf_cnt] = _velocity.y();
    _vz_buf[_smooth_buf_cnt] = _velocity.z();
    _smooth_buf_cnt = (_smooth_buf_cnt + 1) % _smooth_buf_size;

    // fill in pose and twist
    if(!_smooth_enabled) {
        msg.pose.pose.position.x = _position.x();
        msg.pose.pose.position.y = _position.y();
        msg.pose.pose.position.z = _position.z();
        msg.twist.twist.linear.x = _velocity.x();
        msg.twist.twist.linear.y = _velocity.y();
        msg.twist.twist.linear.z = _velocity.z();

    } else {
        if(_smooth_type.compare("median") == 1) {
	    std::vector<double> x_buf, y_buf, z_buf, vx_buf, vy_buf, vz_buf;
    	    x_buf = _x_buf;
      	    y_buf = _y_buf;
	    z_buf = _z_buf;
	    vx_buf = _vx_buf;
	    vy_buf = _vy_buf;
	    vz_buf = _vz_buf;
	    std::sort(x_buf.begin(), x_buf.end());
	    std::sort(y_buf.begin(), y_buf.end());
	    std::sort(z_buf.begin(), z_buf.end());
	    std::sort(vx_buf.begin(), vx_buf.end());
	    std::sort(vy_buf.begin(), vy_buf.end());
	    std::sort(vz_buf.begin(), vz_buf.end());
	    msg.pose.pose.position.x = x_buf[_smooth_buf_size/2];
            msg.pose.pose.position.y = y_buf[_smooth_buf_size/2];
	    msg.pose.pose.position.z = z_buf[_smooth_buf_size/2];
	    msg.twist.twist.linear.x = vx_buf[_smooth_buf_size/2];
	    msg.twist.twist.linear.y = vy_buf[_smooth_buf_size/2];
	    msg.twist.twist.linear.z = vz_buf[_smooth_buf_size/2];
        } else {
            msg.pose.pose.position.x = std::accumulate(_x_buf.begin(), _x_buf.end(), 0.0)/(double)_smooth_buf_size;
            msg.pose.pose.position.y = std::accumulate(_y_buf.begin(), _y_buf.end(), 0.0)/(double)_smooth_buf_size;
            msg.pose.pose.position.z = std::accumulate(_z_buf.begin(), _z_buf.end(), 0.0)/(double)_smooth_buf_size;
            msg.twist.twist.linear.x = std::accumulate(_vx_buf.begin(),_vx_buf.end(), 0.0)/(double)_smooth_buf_size;
            msg.twist.twist.linear.y = std::accumulate(_vy_buf.begin(),_vy_buf.end(), 0.0)/(double)_smooth_buf_size;
            msg.twist.twist.linear.z = std::accumulate(_vz_buf.begin(),_vz_buf.end(), 0.0)/(double)_smooth_buf_size; 
        }
    }


    msg.pose.pose.orientation.x = _quaternion.x();
    msg.pose.pose.orientation.y = _quaternion.y();
    msg.pose.pose.orientation.z = _quaternion.z();
    msg.pose.pose.orientation.w = _quaternion.w();

    msg.twist.twist.angular.x = _imu_angular_velocity.x();
    msg.twist.twist.angular.y = _imu_angular_velocity.y();
    msg.twist.twist.angular.z = _imu_angular_velocity.z();

    // fill in covariance
    Eigen::Matrix<double, 6, 6> pose_sigma, twist_sigma;

    pose_sigma << _Sigma.block<6,6>(3,3);

    twist_sigma << _Sigma.block<3,3>(0,0),      Eigen::MatrixXd::Zero(3,3),
                   Eigen::MatrixXd::Zero(3,3),  _sigma_gyr *_sigma_gyr * Eigen::MatrixXd::Identity(3,3);

    for(int i=0; i<6; i++) {
        for(int j=0; j<6; j++) {
            msg.pose.covariance[6*i+j] = pose_sigma(i, j);
            msg.twist.covariance[6*i+j] = twist_sigma(i, j);
        }
    }

    // publish message
    _odom_pub.publish(msg);
    _odom_vec.push_back(msg);
}

void ESKF::publish_bias() {
    geometry_msgs::TwistStamped msg;

    msg.header.frame_id = "world";
    msg.header.stamp = _imu_time;

    msg.twist.linear.x = _bias_acc.x();
    msg.twist.linear.y = _bias_acc.y();
    msg.twist.linear.z = _bias_acc.z();

    msg.twist.angular.x = _bias_gyr.x();
    msg.twist.angular.y = _bias_gyr.y();
    msg.twist.angular.z = _bias_gyr.z();

    _bias_pub.publish(msg);
}

void ESKF::update_meas_mean(Eigen::Matrix<double, 6, 1> &mean_meas) {
    _m_position = mean_meas.block<3,1>(0,0);
    _m_theta = mean_meas.block<3,1>(3,0);
}

void ESKF::update_meas_cov(Eigen::Matrix<double, 6, 6> &cov_meas) {
    _m_pose_sigma = cov_meas;
}

void ESKF::update_meas_flag() {
    _got_measurements = true;
}

void ESKF::update_error() {
    // assume only pose measurement is used
    Eigen::Matrix<double, 6, 15> H;
    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d Z = Eigen::Matrix3d::Zero();
    H << Z, I, Z, Z, Z,
         Z, Z, I, Z, Z;

    // measurements
    Eigen::Matrix<double, 6, 1> y;
    y[0] = _m_position.x(); y[1] = _m_position.y(); y[2] = _m_position.z();
    y[3] = _m_theta.x();    y[4] = _m_theta.y();    y[5] = _m_theta.z();

    // kalman gain matrix
    Eigen::Matrix<double, 15, 6> K;
    K = _Sigma * H.transpose() * (H * _Sigma * H.transpose() + 1.0 * _m_pose_sigma).inverse();

    // update error
    Eigen::Matrix<double, 15, 1> x;
    x = K * y;

    _d_velocity << x[0],  x[1],  x[2];
    _d_position << x[3],  x[4],  x[5];
    _d_theta    << x[6],  x[7],  x[8];
    _d_bias_acc << x[9],  x[10], x[11];
    _d_bias_gyr << x[12], x[13], x[14];
    _d_rotation << angle_axis_to_rotation_matrix(_d_theta);

    // update covariance
    Eigen::Matrix<double, 15, 15> M;
    M = Eigen::MatrixXd::Identity(15,15) - K*H;
    _Sigma = M * _Sigma;// * M.transpose() + K * _m_pose_sigma * K.transpose();

}

void ESKF::update_state() {
    _velocity += _d_velocity;
    _position += _d_position;
    _rotation *= _d_rotation;
    _bias_acc += _d_bias_acc;
    _bias_gyr += _d_bias_gyr;
    _quaternion = Eigen::Quaterniond(_rotation);
}

void ESKF::reset_error() {
    _d_velocity.setZero();
    _d_position.setZero();
    _d_theta.setZero();
    _d_rotation.setIdentity();
    _d_bias_acc.setZero();
    _d_bias_gyr.setZero();
}

void ESKF::output_log() {

}
