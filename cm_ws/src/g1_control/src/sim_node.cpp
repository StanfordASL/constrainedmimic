#include <chrono>
#include <cstdio>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <atomic>

#include "rclcpp/rclcpp.hpp"
#include "unitree_hg/msg/low_cmd.hpp"
#include "unitree_hg/msg/low_state.hpp"
#include "unitree_hg/msg/motor_cmd.hpp"
#include "unitree_hg/msg/motor_state.hpp"
#include "unitree_hg/msg/imu_state.hpp"
#include "std_msgs/msg/empty.hpp"
#include "std_msgs/msg/int32.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "std_msgs/msg/bool.hpp"
#include "g1_control_msgs/msg/frame_state.hpp"

#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>

using namespace std::chrono_literals;

class G1SimNode : public rclcpp::Node
{
public:
    G1SimNode() : Node("sim_node")
    {
        std::string default_model_path = "../cm_control/cm_control/assets/unitree_g1/cm_g1.xml";
        this->declare_parameter("model_path", default_model_path);
        std::string model_path = this->get_parameter("model_path").as_string();
        this->declare_parameter("sim_freq_hz", 1000);
        this->declare_parameter("pub_freq_hz", 500);
        this->declare_parameter("viz_freq_hz", 50);
        // Virtual gantry parameters
        this->declare_parameter("gantry_kp_pos", 2000.0);
        this->declare_parameter("gantry_kd_pos", 200.0);
        this->declare_parameter("gantry_kp_rot", 200.0);
        this->declare_parameter("gantry_kd_rot", 20.0);
        this->declare_parameter("gantry_height_offset", 0.0);
        gantry_kp_pos_ = this->get_parameter("gantry_kp_pos").as_double();
        gantry_kd_pos_ = this->get_parameter("gantry_kd_pos").as_double();
        gantry_kp_rot_ = this->get_parameter("gantry_kp_rot").as_double();
        gantry_kd_rot_ = this->get_parameter("gantry_kd_rot").as_double();
        gantry_height_offset_ = this->get_parameter("gantry_height_offset").as_double();
        int sim_freq = this->get_parameter("sim_freq_hz").as_int();
        int pub_freq = this->get_parameter("pub_freq_hz").as_int();
        int viz_freq = this->get_parameter("viz_freq_hz").as_int();

        if (sim_freq % pub_freq != 0)
        {
            RCLCPP_WARN(this->get_logger(),
                        "Simulation frequency (%d Hz) should be a multiple of publish frequency (%d Hz)",
                        sim_freq, pub_freq);
        }
        sim_dt_ = std::chrono::nanoseconds(static_cast<int64_t>(1e9 / sim_freq));
        vis_dt_ = std::chrono::nanoseconds(static_cast<int64_t>(1e9 / viz_freq));
        publish_divisor_ = sim_freq / pub_freq;

        char error[1000];
        m_ = mj_loadXML(model_path.c_str(), nullptr, error, 1000);
        if (!m_)
        {
            RCLCPP_ERROR(this->get_logger(), "Could not load MuJoCo model: %s", error);
            exit(1);
        }
        m_->opt.timestep = 1.0 / sim_freq;
        d_ = mj_makeData(m_);
        total_mass_ = mj_getTotalmass(m_);

        // Initialize targets
        pd_target_q_.assign(NUM_JOINTS, 0.0);
        pd_target_dq_.assign(NUM_JOINTS, 0.0);
        pd_target_kp_.assign(NUM_JOINTS, 0.0);
        pd_target_kd_.assign(NUM_JOINTS, 0.0);
        pd_target_tau_.assign(NUM_JOINTS, 0.0);

        torque_limits_ = {
            88, 139, 88, 139, 35, 35,
            88, 139, 88, 139, 35, 35,
            88, 35, 35,
            25, 25, 25, 25, 25, 5, 5,
            25, 25, 25, 25, 25, 5, 5};

        // QoS
        auto qos_best_effort_keep_last_volatile_depth_1 = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort().durability_volatile();
        auto qos_reliable_keep_last_volatile_depth_1 = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().durability_volatile();
        // Subscribers
        low_cmd_sub_ = this->create_subscription<unitree_hg::msg::LowCmd>(
            "/lowcmd", qos_best_effort_keep_last_volatile_depth_1, std::bind(&G1SimNode::low_cmd_callback, this, std::placeholders::_1));
        reset_sub_ = this->create_subscription<std_msgs::msg::Empty>(
            "/g1_control/reset", qos_reliable_keep_last_volatile_depth_1, std::bind(&G1SimNode::reset_callback, this, std::placeholders::_1));
        auto qos_reliable_keep_last_transient_depth_1 = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
        gantry_sub_ = this->create_subscription<std_msgs::msg::Bool>(
            "/g1_control/sim_gantry", qos_reliable_keep_last_transient_depth_1, std::bind(&G1SimNode::gantry_callback, this, std::placeholders::_1));
        // Publishers
        low_state_pub_ = this->create_publisher<unitree_hg::msg::LowState>("/lowstate", qos_best_effort_keep_last_volatile_depth_1);
        contact_mode_pub_ = this->create_publisher<std_msgs::msg::Int32>("/g1_control/contact_mode", qos_best_effort_keep_last_volatile_depth_1);
        root_state_pub_ = this->create_publisher<g1_control_msgs::msg::FrameState>("/g1_control/root_state", qos_best_effort_keep_last_volatile_depth_1);

        left_foot_id_ = mj_name2id(m_, mjOBJ_BODY, "left_ankle_roll_link");
        right_foot_id_ = mj_name2id(m_, mjOBJ_BODY, "right_ankle_roll_link");
        torso_id_ = mj_name2id(m_, mjOBJ_BODY, "torso_link");
        if (torso_id_ == -1)
        {
            RCLCPP_ERROR(this->get_logger(), "Could not find body 'torso_link'");
            exit(1);
        }
        gyro_id_ = mj_name2id(m_, mjOBJ_SENSOR, "imu-pelvis-angular-velocity");
        accelerometer_id_ = mj_name2id(m_, mjOBJ_SENSOR, "imu-pelvis-linear-acceleration");
        keyframe_id_ = mj_name2id(m_, mjOBJ_KEY, "home");
        if (left_foot_id_ == -1)
        {
            RCLCPP_ERROR(this->get_logger(), "Could not find body 'left_ankle_roll_link'");
            exit(1);
        }
        if (right_foot_id_ == -1)
        {
            RCLCPP_ERROR(this->get_logger(), "Could not find body 'right_ankle_roll_link'");
            exit(1);
        }
        if (gyro_id_ == -1)
        {
            RCLCPP_ERROR(this->get_logger(), "Could not find sensor 'imu-pelvis-angular-velocity'");
            exit(1);
        }
        if (accelerometer_id_ == -1)
        {
            RCLCPP_ERROR(this->get_logger(), "Could not find sensor 'imu-pelvis-linear-acceleration'");
            exit(1);
        }
        if (keyframe_id_ == -1)
        {
            RCLCPP_ERROR(this->get_logger(), "Could not find keyframe 'home'");
            exit(1);
        }
        gyro_adr_ = m_->sensor_adr[gyro_id_];
        accelerometer_adr_ = m_->sensor_adr[accelerometer_id_];

        // Pre-allocate messages
        ls_msg_ = unitree_hg::msg::LowState();
        rs_msg_ = g1_control_msgs::msg::FrameState();
        cm_msg_ = std_msgs::msg::Int32();

        // Perform an initial reset to the default standing position
        reset_callback(std::make_shared<std_msgs::msg::Empty>());

        running_ = true;
        physics_thread_ = std::thread(&G1SimNode::physics_loop, this);

        RCLCPP_INFO(this->get_logger(), "Simulation Node Started (sim: %d Hz, pub: %d Hz, viz: %d Hz)", sim_freq, pub_freq, viz_freq);
    }

    ~G1SimNode()
    {
        running_ = false;
        if (physics_thread_.joinable())
            physics_thread_.join();
        if (d_)
            mj_deleteData(d_);
        if (m_)
            mj_deleteModel(m_);
    }

    std::mutex &mutex() { return mtx_; }
    mjModel *m() { return m_; }
    mjData *d() { return d_; }
    std::chrono::nanoseconds vis_dt() const { return vis_dt_; }

private:
    void low_cmd_callback(const unitree_hg::msg::LowCmd::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(mtx_);
        for (int i = 0; i < NUM_JOINTS; ++i)
        {
            pd_target_q_[i] = msg->motor_cmd[i].q;
            pd_target_dq_[i] = msg->motor_cmd[i].dq;
            pd_target_kp_[i] = msg->motor_cmd[i].kp;
            pd_target_kd_[i] = msg->motor_cmd[i].kd;
            pd_target_tau_[i] = msg->motor_cmd[i].tau;
        }
        if (!cmd_received_)
        {
            cmd_received_ = true;
            RCLCPP_INFO(this->get_logger(), "First control command received, unpausing physics.");
        }
    }

    void gantry_callback(const std_msgs::msg::Bool::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(mtx_);
        if (msg->data && !gantry_enabled_)
        {
            if (!cmd_received_)
            {
                d_->qpos[2] += gantry_height_offset_;
                mj_forward(m_, d_);
                set_gantry_anchor();
            }
            else
            {
                set_gantry_anchor();
                gantry_anchor_pos_[2] += gantry_height_offset_;
            }
            RCLCPP_INFO(this->get_logger(), "Virtual gantry enabled");
        }
        else if (!msg->data && gantry_enabled_)
        {
            mju_zero(d_->xfrc_applied + 6 * torso_id_, 6);
            RCLCPP_INFO(this->get_logger(), "Virtual gantry released");
        }
        gantry_enabled_ = msg->data;
    }

    // Anchor the virtual gantry at the current torso pose
    void set_gantry_anchor()
    {
        mju_copy3(gantry_anchor_pos_, d_->xpos + 3 * torso_id_);
        mju_copy4(gantry_anchor_quat_, d_->xquat + 4 * torso_id_);
    }

    void apply_gantry_wrench()
    {
        // Torso velocity: [angular; linear] in world frame
        mjtNum vel[6];
        mj_objectVelocity(m_, d_, mjOBJ_BODY, torso_id_, vel, 0);

        mjtNum *wrench = d_->xfrc_applied + 6 * torso_id_;
        // Position spring-damper, with gravity comp
        for (int i = 0; i < 3; ++i)
        {
            wrench[i] = gantry_kp_pos_ * (gantry_anchor_pos_[i] - d_->xpos[3 * torso_id_ + i]) - gantry_kd_pos_ * vel[3 + i];
        }
        wrench[2] -= total_mass_ * m_->opt.gravity[2];
        // Orientation spring-damper
        mjtNum quat[4] = {d_->xquat[4 * torso_id_], d_->xquat[4 * torso_id_ + 1],
                          d_->xquat[4 * torso_id_ + 2], d_->xquat[4 * torso_id_ + 3]};
        mjtNum err_body[3];
        mju_subQuat(err_body, gantry_anchor_quat_, quat);
        mjtNum err_world[3];
        mju_rotVecQuat(err_world, err_body, quat);
        for (int i = 0; i < 3; ++i)
        {
            wrench[3 + i] = gantry_kp_rot_ * err_world[i] - gantry_kd_rot_ * vel[i];
        }
    }

    void reset_callback(const std_msgs::msg::Empty::SharedPtr)
    {
        std::lock_guard<std::mutex> lock(mtx_);
        mj_resetDataKeyframe(m_, d_, keyframe_id_);
        if (gantry_enabled_)
        {
            d_->qpos[2] += gantry_height_offset_;
        }
        mj_forward(m_, d_);
        if (gantry_enabled_)
        {
            set_gantry_anchor();
        }
        // Reset control targets to the current state
        // Note that after the reset and mj_forward, qpos will match "home" keyframe
        for (int i = 0; i < NUM_JOINTS; ++i)
        {
            pd_target_q_[i] = d_->qpos[QPOS_MOTOR_OFFSET + i];
        }
        std::fill(pd_target_dq_.begin(), pd_target_dq_.end(), 0.0);
        std::fill(pd_target_kp_.begin(), pd_target_kp_.end(), 0.0);
        std::fill(pd_target_kd_.begin(), pd_target_kd_.end(), 0.0);
        std::fill(pd_target_tau_.begin(), pd_target_tau_.end(), 0.0);
        RCLCPP_INFO(this->get_logger(), "Resetting simulation");
    }

    void physics_loop()
    {
        auto next_time = std::chrono::steady_clock::now();
        int step_count = 0;

        while (running_ && rclcpp::ok())
        {
            // Default desired publishing rate: 500 Hz
            // Default sim stepping rate: 1000 Hz
            // Will publish if physics steps = multiple of 2
            bool should_publish = (step_count % publish_divisor_ == 0);

            { // Begin lock
                std::lock_guard<std::mutex> lock(mtx_);

                if (cmd_received_)
                {
                    // Step physics once initialized (keep paused until first command)
                    if (gantry_enabled_)
                    {
                        apply_gantry_wrench();
                    }
                    mj_step1(m_, d_);
                    apply_pd_control();
                    mj_step2(m_, d_);
                }

                if (should_publish)
                {
                    capture_state();
                }
            } // Release lock

            if (should_publish)
            {
                broadcast_state();
            }

            step_count++;
            next_time += sim_dt_;
            std::this_thread::sleep_until(next_time);
        }
    }

    void apply_pd_control()
    {
        for (int i = 0; i < NUM_JOINTS; ++i)
        {
            double q = d_->qpos[QPOS_MOTOR_OFFSET + i];
            double dq = d_->qvel[QVEL_MOTOR_OFFSET + i];
            double torque = pd_target_kp_[i] * (pd_target_q_[i] - q) +
                            pd_target_kd_[i] * (pd_target_dq_[i] - dq) +
                            pd_target_tau_[i];

            if (torque > torque_limits_[i])
                torque = torque_limits_[i];
            if (torque < -torque_limits_[i])
                torque = -torque_limits_[i];

            d_->ctrl[i] = torque;
        }
    }

    // Copies data from MuJoCo to ROS messages
    void capture_state()
    {
        // IMU
        ls_msg_.imu_state.quaternion[0] = (float)d_->qpos[3];
        ls_msg_.imu_state.quaternion[1] = (float)d_->qpos[4];
        ls_msg_.imu_state.quaternion[2] = (float)d_->qpos[5];
        ls_msg_.imu_state.quaternion[3] = (float)d_->qpos[6];

        ls_msg_.imu_state.gyroscope[0] = (float)d_->sensordata[gyro_adr_];
        ls_msg_.imu_state.gyroscope[1] = (float)d_->sensordata[gyro_adr_ + 1];
        ls_msg_.imu_state.gyroscope[2] = (float)d_->sensordata[gyro_adr_ + 2];

        ls_msg_.imu_state.accelerometer[0] = (float)d_->sensordata[accelerometer_adr_];
        ls_msg_.imu_state.accelerometer[1] = (float)d_->sensordata[accelerometer_adr_ + 1];
        ls_msg_.imu_state.accelerometer[2] = (float)d_->sensordata[accelerometer_adr_ + 2];

        // Joint States
        for (int i = 0; i < NUM_JOINTS; ++i)
        {
            ls_msg_.motor_state[i].q = (float)d_->qpos[QPOS_MOTOR_OFFSET + i];
            ls_msg_.motor_state[i].dq = (float)d_->qvel[QVEL_MOTOR_OFFSET + i];
            ls_msg_.motor_state[i].tau_est = (float)d_->ctrl[i];
        }

        // Root State
        // NOTE: Mujoco's qvel for angular velocity is in body frame,
        // but we want the root state in world frame.
        mjtNum quat[4] = {d_->qpos[3], d_->qpos[4], d_->qpos[5], d_->qpos[6]};
        mjtNum omega_body[3] = {d_->qvel[3], d_->qvel[4], d_->qvel[5]};
        mjtNum omega_world[3];
        mju_rotVecQuat(omega_world, omega_body, quat);

        for (int i = 0; i < 3; ++i)
        {
            rs_msg_.position[i] = d_->qpos[i];
            rs_msg_.velocity[i] = d_->qvel[i];
            rs_msg_.omega[i] = omega_world[i];
        }
        for (int i = 0; i < 4; ++i)
        {
            rs_msg_.quaternion[i] = d_->qpos[QPOS_QUAT_OFFSET + i];
        }

        // Contact Mode Logic
        bool left_contact = false;
        bool right_contact = false;
        for (int i = 0; i < d_->ncon; ++i)
        {
            mjContact *con = &d_->contact[i];
            int body1 = m_->geom_bodyid[con->geom1];
            int body2 = m_->geom_bodyid[con->geom2];
            if (body1 == left_foot_id_ || body2 == left_foot_id_)
                left_contact = true;
            if (body1 == right_foot_id_ || body2 == right_foot_id_)
                right_contact = true;
        }

        if (!left_contact && !right_contact)
            cm_msg_.data = 0;
        else if (left_contact && !right_contact)
            cm_msg_.data = 1;
        else if (!left_contact && right_contact)
            cm_msg_.data = 2;
        else
            cm_msg_.data = 3;
    }

    // Publishes to ROS 2
    void broadcast_state()
    {
        low_state_pub_->publish(ls_msg_);
        root_state_pub_->publish(rs_msg_);
        contact_mode_pub_->publish(cm_msg_);
    }

    // Pre-allocated Message Members
    unitree_hg::msg::LowState ls_msg_;
    g1_control_msgs::msg::FrameState rs_msg_;
    std_msgs::msg::Int32 cm_msg_;

    static constexpr int NUM_JOINTS = 29;
    static constexpr int QPOS_QUAT_OFFSET = 3;
    static constexpr int QPOS_MOTOR_OFFSET = 7;
    static constexpr int QVEL_MOTOR_OFFSET = 6;

    std::chrono::nanoseconds sim_dt_;
    int publish_divisor_;
    std::chrono::nanoseconds vis_dt_;

    mjModel *m_ = nullptr;
    mjData *d_ = nullptr;
    std::mutex mtx_;
    std::atomic<bool> running_;
    std::atomic<bool> cmd_received_{false};
    std::thread physics_thread_;

    std::vector<double> pd_target_q_, pd_target_dq_, pd_target_kp_, pd_target_kd_, pd_target_tau_;
    std::vector<double> torque_limits_;

    rclcpp::Subscription<unitree_hg::msg::LowCmd>::SharedPtr low_cmd_sub_;
    rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr reset_sub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr gantry_sub_;
    rclcpp::Publisher<unitree_hg::msg::LowState>::SharedPtr low_state_pub_;
    rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr contact_mode_pub_;
    rclcpp::Publisher<g1_control_msgs::msg::FrameState>::SharedPtr root_state_pub_;

    // Virtual gantry parameters
    std::atomic<bool> gantry_enabled_{false};
    mjtNum gantry_anchor_pos_[3];
    mjtNum gantry_anchor_quat_[4];
    double gantry_kp_pos_, gantry_kd_pos_, gantry_kp_rot_, gantry_kd_rot_;
    double gantry_height_offset_;
    double total_mass_;

    int torso_id_ = -1;
    int left_foot_id_ = -1;
    int right_foot_id_ = -1;
    int gyro_id_ = -1;
    int accelerometer_id_ = -1;
    int gyro_adr_ = -1;
    int accelerometer_adr_ = -1;
    int keyframe_id_ = -1;
};

// Global MuJoCo objects for GLFW callbacks
mjModel *m = nullptr;
mjData *d = nullptr;
mjvScene scn;
mjvCamera cam;
mjvOption opt;
mjrContext con;

// mouse state
bool button_left = false;
bool button_middle = false;
bool button_right = false;
double lastx = 0;
double lasty = 0;

void mouse_button(GLFWwindow *window, int button, int act, int mods)
{
    // update button state
    button_left = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);
    button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS);
    button_right = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS);

    // update previous mouse position
    glfwGetCursorPos(window, &lastx, &lasty);
}

void mouse_move(GLFWwindow *window, double xpos, double ypos)
{
    // no buttons pressed: nothing to do
    if (!button_left && !button_middle && !button_right)
    {
        return;
    }

    // compute mouse displacement, save
    double dx = xpos - lastx;
    double dy = ypos - lasty;
    lastx = xpos;
    lasty = ypos;

    // get current window size
    int width, height;
    glfwGetWindowSize(window, &width, &height);

    // get shift key state
    bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                      glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);

    // determine action based on mouse button
    mjtMouse action;
    if (button_right)
    {
        action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
    }
    else if (button_left)
    {
        action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
    }
    else
    {
        action = mjMOUSE_ZOOM;
    }

    // move camera
    mjv_moveCamera(m, action, dx / height, dy / height, &scn, &cam);
}

void scroll(GLFWwindow *window, double xoffset, double yoffset)
{
    // emulate zoom with scroll wheel
    mjv_moveCamera(m, mjMOUSE_ZOOM, 0, -0.05 * yoffset, &scn, &cam);
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<G1SimNode>();
    m = node->m();
    d = node->d();

    if (!glfwInit())
        return 1;
    GLFWwindow *window = glfwCreateWindow(1200, 900, "G1 Simulation", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);
    mjv_defaultScene(&scn);
    mjr_defaultContext(&con);
    mjv_makeScene(m, &scn, 2000);
    mjr_makeContext(m, &con, mjFONTSCALE_150);

    // Track pelvis
    int pelvis_id = mj_name2id(m, mjOBJ_BODY, "pelvis");
    if (pelvis_id != -1)
    {
        cam.type = mjCAMERA_TRACKING;
        cam.trackbodyid = pelvis_id;
        cam.distance = 2.0;
        cam.lookat[0] = 0;
        cam.lookat[1] = 0;
        cam.lookat[2] = 0.8; // Approx pelvis height
    }

    glfwSetCursorPosCallback(window, mouse_move);
    glfwSetMouseButtonCallback(window, mouse_button);
    glfwSetScrollCallback(window, scroll);

    // Simple thread for ROS2 spinner
    std::thread spin_thread([&]()
                            { rclcpp::spin(node); });

    while (!glfwWindowShouldClose(window) && rclcpp::ok())
    {
        {
            std::lock_guard<std::mutex> lock(node->mutex());
            mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
        }

        mjrRect viewport = {0, 0, 0, 0};
        glfwGetFramebufferSize(window, &viewport.width, &viewport.height);
        mjr_render(viewport, &scn, &con);
        glfwSwapBuffers(window);
        glfwPollEvents();
        std::this_thread::sleep_for(node->vis_dt());
    }

    rclcpp::shutdown();
    if (spin_thread.joinable())
        spin_thread.join();
    glfwTerminate();

    return 0;
}
