#include <memory>
#include <string>
#include <vector>
#include <array>
#include <cmath>
#include <map>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <nlohmann/json.hpp>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/pose_array.hpp"
#include "xrobo_ros2/msg/x_robo_state.hpp"
#include "PXREARobotSDK.h"

using json = nlohmann::json;

/**
 * @brief Helper struct for calculating velocity and angular velocity from pose differences.
 */
struct VelocityCalculator
{
    Eigen::Vector3d last_pos;
    Eigen::Quaterniond last_quat;
    uint64_t last_time_ns = 0;
    bool initialized = false;

    /**
     * @brief Computes angular velocity in the world frame given two rotations and delta time.
     */
    Eigen::Vector3d compute_world_angular_velocity(const Eigen::Quaterniond &q1, const Eigen::Quaterniond &q2, double dt)
    {
        Eigen::Quaterniond q1_norm = q1.normalized();
        Eigen::Quaterniond q2_norm = q2.normalized();

        if (q1_norm.dot(q2_norm) < 0.0)
        {
            q2_norm.coeffs() *= -1.0;
        }

        Eigen::Quaterniond q_rel = q2_norm * q1_norm.conjugate();
        Eigen::AngleAxisd angle_axis(q_rel);
        double angle = angle_axis.angle();
        Eigen::Vector3d axis = angle_axis.axis();

        if (std::abs(angle) < 1e-8)
        {
            return Eigen::Vector3d::Zero();
        }

        return axis * (angle / dt);
    }

    /**
     * @brief Updates the calculator with a new pose and returns the 6D twist.
     */
    std::array<double, 6> update(const std::array<double, 7> &current_pose, uint64_t current_time_ns)
    {
        std::array<double, 6> twist = {0, 0, 0, 0, 0, 0};
        Eigen::Vector3d current_pos(current_pose[0], current_pose[1], current_pose[2]);
        Eigen::Quaterniond current_quat(current_pose[3], current_pose[4], current_pose[5], current_pose[6]);

        if (initialized && current_time_ns > last_time_ns)
        {
            double dt = (current_time_ns - last_time_ns) * 1e-9;
            if (dt > 1e-6)
            {
                Eigen::Vector3d vel = (current_pos - last_pos) / dt;
                Eigen::Vector3d omega = compute_world_angular_velocity(last_quat, current_quat, dt);
                twist = {vel.x(), vel.y(), vel.z(), omega.x(), omega.y(), omega.z()};
            }
        }

        last_pos = current_pos;
        last_quat = current_quat;
        last_time_ns = current_time_ns;
        initialized = true;
        return twist;
    }
};

/**
 * @brief Helper struct for applying Exponential Moving Average (EMA) filter to pose data.
 */
struct PoseEMAFilter
{
    Eigen::Vector3d filtered_pos;
    Eigen::Quaterniond filtered_quat;
    bool initialized = false;

    /**
     * @brief Updates the filtered state with new pose data.
     *
     * @param pose_arr Reference to the current pose array (x, y, z, qw, qx, qy, qz).
     * @param alpha_pos Alpha value for position filtering (0.0 to 1.0, higher is less filtering).
     * @param alpha_quat Alpha value for orientation filtering (SLERP weight).
     */
    void update(std::array<double, 7> &pose_arr, double alpha_pos, double alpha_quat)
    {
        Eigen::Vector3d current_pos(pose_arr[0], pose_arr[1], pose_arr[2]);
        Eigen::Quaterniond current_quat(pose_arr[3], pose_arr[4], pose_arr[5], pose_arr[6]);

        if (!initialized)
        {
            filtered_pos = current_pos;
            filtered_quat = current_quat;
            initialized = true;
            return;
        }

        // Linear EMA for position
        filtered_pos = alpha_pos * current_pos + (1.0 - alpha_pos) * filtered_pos;

        // SLERP for orientation
        if (filtered_quat.dot(current_quat) < 0.0)
        {
            current_quat.coeffs() *= -1.0;
        }
        filtered_quat = filtered_quat.slerp(alpha_quat, current_quat);
        filtered_quat.normalize();

        // Update the array with filtered values
        pose_arr[0] = filtered_pos.x();
        pose_arr[1] = filtered_pos.y();
        pose_arr[2] = filtered_pos.z();
        pose_arr[3] = filtered_quat.w();
        pose_arr[4] = filtered_quat.x();
        pose_arr[5] = filtered_quat.y();
        pose_arr[6] = filtered_quat.z();
    }
};

/**
 * @brief Helper struct for applying Exponential Moving Average (EMA) filter to twist data (linear and angular velocity).
 */
struct TwistEMAFilter
{
    Eigen::Vector3d filtered_vel;
    Eigen::Vector3d filtered_omega;
    bool initialized = false;

    /**
     * @brief Updates the filtered state with new twist data.
     *
     * @param twist_arr Reference to the current twist array (vx, vy, vz, wx, wy, wz).
     * @param alpha_vel Alpha value for linear velocity filtering.
     * @param alpha_omega Alpha value for angular velocity filtering.
     */
    void update(std::array<double, 6> &twist_arr, double alpha_vel, double alpha_omega)
    {
        Eigen::Vector3d current_vel(twist_arr[0], twist_arr[1], twist_arr[2]);
        Eigen::Vector3d current_omega(twist_arr[3], twist_arr[4], twist_arr[5]);

        if (!initialized)
        {
            filtered_vel = current_vel;
            filtered_omega = current_omega;
            initialized = true;
            return;
        }

        filtered_vel = alpha_vel * current_vel + (1.0 - alpha_vel) * filtered_vel;
        filtered_omega = alpha_omega * current_omega + (1.0 - alpha_omega) * filtered_omega;

        // Update the array with filtered values
        twist_arr[0] = filtered_vel.x();
        twist_arr[1] = filtered_vel.y();
        twist_arr[2] = filtered_vel.z();
        twist_arr[3] = filtered_omega.x();
        twist_arr[4] = filtered_omega.y();
        twist_arr[5] = filtered_omega.z();
    }
};

/**
 * @brief ROS2 Node for interfacing with the XRobo SDK.
 */
class XRoboNode : public rclcpp::Node
{
public:
    XRoboNode() : Node("xrobo_node")
    {
        // Publishers
        auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort().durability_volatile();
        state_pub_ = this->create_publisher<xrobo_ros2::msg::XRoboState>("xrobo/state", qos);

        // Initialize SDK
        if (PXREAInit(this, &XRoboNode::on_sdk_callback, PXREAFullMask) != 0)
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to initialize XRobo SDK");
        }
        else
        {
            RCLCPP_INFO(this->get_logger(), "XRobo SDK initialized");
        }

        // Parameters
        /**
         * @brief Parameter to enable/disable frame rotation adjustments for better alignment
         * with the 0-joint-angle position of the Unitree G1. Note that only a subset of the
         * whole-body poses from XRobo are misaligned, namely, only the arms need adjustment.
         */
        this->declare_parameter("adjust_arm_rotations", true);
        adjust_arm_rotations_ = this->get_parameter("adjust_arm_rotations").as_bool();

        if (adjust_arm_rotations_)
        {
            RCLCPP_INFO(this->get_logger(), "Arm rotation adjustments enabled");
            // Left Shoulder: Rx(90)
            rotation_adjustments_[16] = Eigen::Quaterniond(Eigen::AngleAxisd(M_PI / 2.0, Eigen::Vector3d::UnitX()));

            // Left Elbow, Wrist, Hand: Ry(90) * Rz(90)
            Eigen::Quaterniond q_l_other = Eigen::Quaterniond(Eigen::AngleAxisd(M_PI / 2.0, Eigen::Vector3d::UnitY())) *
                                           Eigen::Quaterniond(Eigen::AngleAxisd(M_PI / 2.0, Eigen::Vector3d::UnitZ()));
            rotation_adjustments_[18] = q_l_other;
            rotation_adjustments_[20] = q_l_other;
            rotation_adjustments_[22] = q_l_other;

            // Right Shoulder: Rx(-90)
            rotation_adjustments_[17] = Eigen::Quaterniond(Eigen::AngleAxisd(-M_PI / 2.0, Eigen::Vector3d::UnitX()));

            // Right Elbow, Wrist, Hand: Rz(-90) * Rx(-90)
            Eigen::Quaterniond q_r_other = Eigen::Quaterniond(Eigen::AngleAxisd(-M_PI / 2.0, Eigen::Vector3d::UnitZ())) *
                                           Eigen::Quaterniond(Eigen::AngleAxisd(-M_PI / 2.0, Eigen::Vector3d::UnitX()));
            rotation_adjustments_[19] = q_r_other;
            rotation_adjustments_[21] = q_r_other;
            rotation_adjustments_[23] = q_r_other;
        }

        // EMA Filtering Parameters
        this->declare_parameter("alpha_pos", 0.6);
        this->declare_parameter("alpha_quat", 0.6);
        this->declare_parameter("alpha_vel", 0.1);
        this->declare_parameter("alpha_omega", 0.1);

        alpha_pos_ = this->get_parameter("alpha_pos").as_double();
        alpha_quat_ = this->get_parameter("alpha_quat").as_double();
        alpha_vel_ = this->get_parameter("alpha_vel").as_double();
        alpha_omega_ = this->get_parameter("alpha_omega").as_double();

        RCLCPP_INFO(this->get_logger(), "EMA filters initialized: alpha_pos: %.2f, alpha_quat: %.2f, alpha_vel: %.2f, alpha_omega: %.2f",
                    alpha_pos_, alpha_quat_, alpha_vel_, alpha_omega_);
    }

    ~XRoboNode()
    {
        PXREADeinit();
    }

private:
    /**
     * @brief Static callback triggered by the XRobo SDK when device data is available.
     *
     * @param context Pointer to the XRoboNode instance.
     * @param type The type of data received (expected PXREADeviceStateJson).
     * @param status SDK status code.
     * @param userData Data payload (PXREADevStateJson).
     */
    static void on_sdk_callback(void *context, PXREAClientCallbackType type, int status, void *userData)
    {
        (void)status;
        auto *node = static_cast<XRoboNode *>(context);
        if (type == PXREADeviceStateJson)
        {
            auto *devState = static_cast<PXREADevStateJson *>(userData);
            node->handle_json_state(devState->stateJson);
        }
    }

    /**
     * @brief Parses the raw JSON state string from the SDK.
     *
     * Extracts headset, controller, body joint, and motion tracker data.
     *
     * @param json_str The raw JSON string from PXREADeviceStateJson.
     */
    void handle_json_state(const std::string &json_str)
    {
        try
        {
            json root = json::parse(json_str);
            if (!root.contains("value"))
                return;
            json value = json::parse(root["value"].get<std::string>());

            // Create a local message to parse into.
            // This avoids holding the lock during the expensive JSON parsing.
            xrobo_ros2::msg::XRoboState msg;

            uint64_t timestamp_ns = 0;
            if (value.contains("timeStampNs"))
            {
                timestamp_ns = value["timeStampNs"].get<int64_t>();
                msg.header.stamp = rclcpp::Time(timestamp_ns);
            }
            else
            {
                auto now = this->now();
                msg.header.stamp = now;
                timestamp_ns = now.nanoseconds();
            }
            msg.header.frame_id = "world";

            if (value.contains("Head"))
            {
                if (this->parse_pose_to_array(value["Head"]["pose"].get<std::string>(), msg.headset_pose))
                {
                    this->headset_pose_filter_.update(msg.headset_pose, this->alpha_pos_, this->alpha_quat_);
                    auto twist = this->headset_calc_.update(msg.headset_pose, timestamp_ns);
                    this->headset_twist_ema_filter_.update(twist, this->alpha_vel_, this->alpha_omega_);
                    msg.headset_twist = twist;
                }
            }

            if (value.contains("Controller"))
            {
                if (value["Controller"].contains("left"))
                {
                    auto &left = value["Controller"]["left"];
                    this->parse_full_controller_state(left, msg.left_controller_pose,
                                                      msg.left_trigger, msg.left_grip, msg.left_menu_button,
                                                      msg.left_axis_x, msg.left_axis_y, msg.left_axis_click,
                                                      msg.left_primary_button, msg.left_secondary_button);
                    this->left_ctrl_pose_filter_.update(msg.left_controller_pose, this->alpha_pos_, this->alpha_quat_);
                    auto twist = this->left_ctrl_calc_.update(msg.left_controller_pose, timestamp_ns);
                    this->left_ctrl_twist_ema_filter_.update(twist, this->alpha_vel_, this->alpha_omega_);
                    msg.left_controller_twist = twist;
                }
                if (value["Controller"].contains("right"))
                {
                    auto &right = value["Controller"]["right"];
                    this->parse_full_controller_state(right, msg.right_controller_pose,
                                                      msg.right_trigger, msg.right_grip, msg.right_menu_button,
                                                      msg.right_axis_x, msg.right_axis_y, msg.right_axis_click,
                                                      msg.right_primary_button, msg.right_secondary_button);
                    this->right_ctrl_pose_filter_.update(msg.right_controller_pose, this->alpha_pos_, this->alpha_quat_);
                    auto twist = this->right_ctrl_calc_.update(msg.right_controller_pose, timestamp_ns);
                    this->right_ctrl_twist_ema_filter_.update(twist, this->alpha_vel_, this->alpha_omega_);
                    msg.right_controller_twist = twist;
                }
            }

            if (value.contains("Body"))
            {
                auto &body = value["Body"];
                if (body.contains("joints") && body["joints"].is_array())
                {
                    auto joints = body["joints"];
                    int jointCount = std::min(static_cast<int>(joints.size()), 24);
                    for (int i = 0; i < jointCount; i++)
                    {
                        if (joints[i].contains("p"))
                        {
                            std::array<double, 7> pose_arr;
                            if (this->parse_pose_to_array(joints[i]["p"].get<std::string>(), pose_arr))
                            {
                                if (this->adjust_arm_rotations_ && this->rotation_adjustments_.count(i))
                                {
                                    // pose_arr ordering is (x, y, z, qw, qx, qy, qz)
                                    Eigen::Quaterniond q(pose_arr[3], pose_arr[4], pose_arr[5], pose_arr[6]);
                                    Eigen::Quaterniond q_adj = q * this->rotation_adjustments_[i];
                                    pose_arr[3] = q_adj.w();
                                    pose_arr[4] = q_adj.x();
                                    pose_arr[5] = q_adj.y();
                                    pose_arr[6] = q_adj.z();
                                }

                                // Apply EMA filter to joint pose
                                this->body_joint_ema_filters_[i].update(pose_arr, this->alpha_pos_, this->alpha_quat_);

                                // Store into flat_body_poses [i*7 to i*7 + 6]
                                for (int j = 0; j < 7; ++j)
                                {
                                    msg.flat_body_poses[i * 7 + j] = pose_arr[j];
                                }

                                // Calculate and filter twists for specific links
                                std::array<double, 6> twist;
                                if (i == 0)
                                { // Root
                                    twist = this->root_calc_.update(pose_arr, timestamp_ns);
                                    this->root_twist_ema_filter_.update(twist, this->alpha_vel_, this->alpha_omega_);
                                    msg.root_twist = twist;
                                }
                                else if (i == 10)
                                { // Left Foot
                                    twist = this->left_foot_calc_.update(pose_arr, timestamp_ns);
                                    this->left_foot_twist_ema_filter_.update(twist, this->alpha_vel_, this->alpha_omega_);
                                    msg.left_foot_twist = twist;
                                }
                                else if (i == 11)
                                { // Right Foot
                                    twist = this->right_foot_calc_.update(pose_arr, timestamp_ns);
                                    this->right_foot_twist_ema_filter_.update(twist, this->alpha_vel_, this->alpha_omega_);
                                    msg.right_foot_twist = twist;
                                }
                            }
                        }
                    }
                }
            }

            state_pub_->publish(msg);
        }
        catch (const json::exception &e)
        {
            RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                                  "JSON parsing error: %s", e.what());
        }
        catch (const std::exception &e)
        {
            RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                                  "Standard exception in callback: %s", e.what());
        }
        catch (...)
        {
            RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                                  "Unknown exception in SDK callback thread");
        }
    }

    /**
     * @brief Parses a pose string into an array (x, y, z, qw, qx, qy, qz).
     */
    bool parse_pose_to_array(const std::string &s, std::array<double, 7> &out)
    {
        auto vals = string_to_doubles(s, 7);
        if (vals.size() == 7)
        {
            // NOTE FRAME CONVENTION: Convert Unity to standard robotics convention
            out[0] = -vals[2]; // x
            out[1] = -vals[0]; // y
            out[2] = vals[1];  // z
            out[3] = -vals[6]; // qw
            out[4] = vals[5];  // qx
            out[5] = vals[3];  // qy
            out[6] = -vals[4]; // qz
            return true;
        }
        return false;
    }

    /**
     * @brief Extracts controller-specific state directly into XRoboState fields.
     */
    void parse_full_controller_state(const json &item, std::array<double, 7> &pose_state,
                                     double &trigger, double &grip, bool &menu_button,
                                     double &axis_x, double &axis_y, bool &axis_click,
                                     bool &primary_button, bool &secondary_button)
    {
        if (item.contains("pose"))
        {
            parse_pose_to_array(item["pose"].get<std::string>(), pose_state);
        }
        if (item.contains("trigger"))
            trigger = item["trigger"].get<double>();
        if (item.contains("grip"))
            grip = item["grip"].get<double>();
        if (item.contains("menuButton"))
            menu_button = item["menuButton"].get<bool>();
        if (item.contains("axisX"))
            axis_x = item["axisX"].get<double>();
        if (item.contains("axisY"))
            axis_y = item["axisY"].get<double>();
        if (item.contains("axisClick"))
            axis_click = item["axisClick"].get<bool>();
        if (item.contains("primaryButton"))
            primary_button = item["primaryButton"].get<bool>();
        if (item.contains("secondaryButton"))
            secondary_button = item["secondaryButton"].get<bool>();
    }

    /**
     * @brief Helper utility to split a comma-separated string into a vector of doubles.
     */
    std::vector<double> string_to_doubles(const std::string &s, int expected_count)
    {
        std::vector<double> vals;
        std::stringstream ss(s);
        std::string item;
        while (std::getline(ss, item, ',') && static_cast<int>(vals.size()) < expected_count)
        {
            try
            {
                vals.push_back(std::stod(item));
            }
            catch (...)
            {
                break;
            }
        }
        return vals;
    }

    rclcpp::Publisher<xrobo_ros2::msg::XRoboState>::SharedPtr state_pub_;

    bool adjust_arm_rotations_ = false;
    std::map<int, Eigen::Quaterniond> rotation_adjustments_;

    // EMA Filtering
    double alpha_pos_ = 0.8;
    double alpha_quat_ = 0.8;
    double alpha_vel_ = 0.5;
    double alpha_omega_ = 0.5;
    PoseEMAFilter headset_pose_filter_;
    PoseEMAFilter left_ctrl_pose_filter_;
    PoseEMAFilter right_ctrl_pose_filter_;
    std::array<PoseEMAFilter, 24> body_joint_ema_filters_;

    // Manual Velocity Calculation
    VelocityCalculator headset_calc_;
    VelocityCalculator left_ctrl_calc_;
    VelocityCalculator right_ctrl_calc_;
    VelocityCalculator root_calc_;
    VelocityCalculator left_foot_calc_;
    VelocityCalculator right_foot_calc_;

    // Velocity Filtering
    TwistEMAFilter headset_twist_ema_filter_;
    TwistEMAFilter left_ctrl_twist_ema_filter_;
    TwistEMAFilter right_ctrl_twist_ema_filter_;
    TwistEMAFilter root_twist_ema_filter_;
    TwistEMAFilter left_foot_twist_ema_filter_;
    TwistEMAFilter right_foot_twist_ema_filter_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<XRoboNode>();
    RCLCPP_INFO(node->get_logger(), "XRobo ROS2 node started");
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
