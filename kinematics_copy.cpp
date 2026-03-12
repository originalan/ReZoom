#include <kinematics.hpp>
#include <geometry_msgs/Twist.h>
#include <ros/ros.h>

geometry_msgs::Twist g_raw_cmd;
geometry_msgs::Twist g_safety_cmd;

ros::Time g_raw_cmd_time(0);
ros::Time g_safety_cmd_time(0);

bool g_have_raw_cmd = false;
bool g_have_safety_cmd = false;

const double RAW_TIMEOUT_SEC = 0.30;
const double SAFETY_TIMEOUT_SEC = 0.30;

// Extra callback for /cmd_vel_safety
void cmdVelSafetyCallback(const geometry_msgs::Twist::ConstPtr& msg) {
    g_safety_cmd = *msg;
    g_safety_cmd_time = ros::Time::now();
    g_have_safety_cmd = true;
}

Kinematics::Kinematics(ros::NodeHandle& nh) : kinematics_(nh){
    steerWheelRadius_ = 0.1090;
    driveWheelRadius_ = 0.1090;  // both same for Mi Scooter, 11cm when measured radially, but 10.9cm when calculated from circumference.
    baseLengthl_ = 0.84;
    steerInclination_ = 5;
    maxVelocity_ = 1.0;
    minVelocity_ = 0.0;
    maxSteerAngle_ = 0.108;
    minSteerAngle_ = -0.108;
    linearVelocity_ = 0;
    angularVelocity_ = 0;

    cmdSteer.data = 0.0;
    cmdWheel.data = 0.0;

    steerJointPublisher_ = kinematics_.advertise<std_msgs::Float64>("/escooter/front_steer_position_controller/command", 10);
    driveJointPublisher_ = kinematics_.advertise<std_msgs::Float64>("/escooter/drive_wheel_velocity_controller/command", 10);
  
    cmdVelSubscriber_ = kinematics_.subscribe("/cmd_vel", 10, &Kinematics::cmdVelCallback, this);

}

Kinematics::~Kinematics() {
}

void Kinematics::cmdVelCallback(geometry_msgs::Twist msg) {

    // Store raw /cmd_vel instead of directly commanding actuators
    g_raw_cmd = msg;
    g_raw_cmd_time = ros::Time::now();
    g_have_raw_cmd = true;
	// linearVelocity_ = msg.linear.x;
    // angularVelocity_ = msg.angular.z;

    // // angularVelocity is actually steer angle in radians from [-pi/2, pi/2]
    // cmdSteer.data = -1*steerAngle(angularVelocity_); //steerAngle(instantaneousCenterOfRotation());
    // cmdWheel.data = -1*linearToAngularVelocity(linearVelocity_);
}

double Kinematics::instantaneousCenterOfRotation() {
	if (angularVelocity_ == 0)
		return 0;
	else
		return linearVelocity_/angularVelocity_;
}

double Kinematics::steerAngle(double radius) {
	// if (radius == 0)
    // 	return 0;
    // double steerAngle = (atan((baseLengthl_)/(radius))*(1/(2*3.141)));
	// if(steerAngle < minSteerAngle_)
	// 	return minSteerAngle_*(2.77);
	// else if(steerAngle > maxSteerAngle_)
	// 	return maxSteerAngle_*(2.77);
	// else
	//     return steerAngle*(2.77);

    // big pulley = 72 teeth, small pulley = 36 teeth. Ratio = 2.
    return (double)radius * 2 / 6.283185;  

}

double Kinematics::linearToAngularVelocity(double linear) {
	if (((linear/driveWheelRadius_)*(1/(2*3.141))) < 5)
	   return ((linear/driveWheelRadius_)*(1/(2*3.141)));
	else
		return 5;
}


int main(int argc, char** argv)
{
    ros::init(argc, argv, "kinematics");
    ros::NodeHandle nh;
    Kinematics bicycle(nh);

    // Second subscriber created in main, no header changes needed
    ros::Subscriber cmdVelSafetySubscriber = nh.subscribe(
        "/cmd_vel_safety", 10, cmdVelSafetyCallback);

    ros::Rate loop(10);

    while (ros::ok()) {
        // ROS_INFO_STREAM("Throttle " << bicycle.cmdWheel);
        // ROS_INFO_STREAM("Steering " << bicycle.cmdSteer);

        ros::spinOnce();

        ros::Time now = ros::Time::now();

        bool safetyFresh = g_have_safety_cmd &&
                           ((now - g_safety_cmd_time).toSec() <= SAFETY_TIMEOUT_SEC);

        bool rawFresh = g_have_raw_cmd &&
                        ((now - g_raw_cmd_time).toSec() <= RAW_TIMEOUT_SEC);

        geometry_msgs::Twist active_cmd;
        active_cmd.linear.x = 0.0;
        active_cmd.angular.z = 0.0;

        // Policy A:
        // Prefer /cmd_vel_safety if available, else fallback to /cmd_vel
        if (safetyFresh) {
            active_cmd = g_safety_cmd;
            ROS_INFO_THROTTLE(1.0, "Kinematics using /cmd_vel_safety");
        } else if (rawFresh) {
            active_cmd = g_raw_cmd;
            ROS_WARN_THROTTLE(1.0, "Kinematics falling back to /cmd_vel");
        } else {
            ROS_WARN_THROTTLE(1.0, "No fresh command on /cmd_vel_safety or /cmd_vel, stopping");
        }

        bicycle.linearVelocity_ = active_cmd.linear.x;
        bicycle.angularVelocity_ = active_cmd.angular.z;

        // angularVelocity_ is being used as steer command in this stack
        bicycle.cmdSteer.data = -1 * bicycle.steerAngle(bicycle.angularVelocity_);
        bicycle.cmdWheel.data = -1 * bicycle.linearToAngularVelocity(bicycle.linearVelocity_);

        bicycle.steerJointPublisher_.publish(bicycle.cmdSteer);
        bicycle.driveJointPublisher_.publish(bicycle.cmdWheel);

        loop.sleep();
    }
    return 0;
}
