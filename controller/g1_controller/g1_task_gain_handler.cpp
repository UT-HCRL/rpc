#include "controller/g1_controller/g1_task_gain_handler.hpp"
#include "controller/g1_controller/g1_control_architecture.hpp"
#include "controller/g1_controller/g1_control_architecture_wbic.hpp"
#include "controller/g1_controller/g1_task/g1_com_xy_task.hpp"
#include "controller/g1_controller/g1_task/g1_com_z_task.hpp"
#include "controller/g1_controller/g1_tci_container.hpp"
#include "controller/whole_body_controller/basic_task.hpp"
#include "controller/whole_body_controller/task.hpp"
#include "util/util.hpp"

G1TaskGainHandler::G1TaskGainHandler(G1ControlArchitecture *ctrl_arch)
    : ctrl_arch_(ctrl_arch), ctrl_arch_wbic_(nullptr),
      b_signal_received_(false), b_first_visit_(false), b_com_xy_task_(false),
      count_(0) {

  util::PrettyConstructor(1, "G1TaskGainHandler");
  std::string border = "=";
  for (unsigned int i = 0; i < 79; ++i)
    border += "=";
  util::ColorPrint(color::kBoldRed, border);
}

G1TaskGainHandler::G1TaskGainHandler(
    G1ControlArchitecture_WBIC *ctrl_arch)
    : ctrl_arch_wbic_(ctrl_arch), ctrl_arch_(nullptr),
      b_signal_received_(false), b_first_visit_(false), b_com_xy_task_(false),
      count_(0) {

  util::PrettyConstructor(1, "G1TaskGainHandler WBIC");
  std::string border = "=";
  for (unsigned int i = 0; i < 79; ++i)
    border += "=";
  util::ColorPrint(color::kBoldRed, border);
}

void G1TaskGainHandler::Trigger(const std::string &task_name,
                                   const Eigen::VectorXd &kp,
                                   const Eigen::VectorXd &kd) {
  task_name_ = task_name;
  if (!_TaskExists(task_name_))
    return;

  target_kp_ = kp;
  target_kd_ = kd;
  b_signal_received_ = true;
  b_first_visit_ = true;
}

void G1TaskGainHandler::Trigger(const std::string &task_name,
                                   const Eigen::VectorXd &kp,
                                   const Eigen::VectorXd &kd,
                                   const Eigen::VectorXd &ki) {
  task_name_ = task_name;
  if (!_TaskExists(task_name_))
    return;

  target_kp_ = kp;
  target_kd_ = kd;
  target_ki_ = ki;
  b_signal_received_ = true;
  b_com_xy_task_ = true;
  b_first_visit_ = true;
}

void G1TaskGainHandler::Process() {
  count_ += 1;

  if (ctrl_arch_) {
    // ======================================================
    // IHWBC
    // ======================================================
    if (b_first_visit_) {
      init_kp_ = ctrl_arch_->tci_container_->task_map_[task_name_]->Kp();
      init_kd_ = ctrl_arch_->tci_container_->task_map_[task_name_]->Kd();
      if (b_com_xy_task_)
        init_ki_ = ctrl_arch_->tci_container_->task_map_[task_name_]->Ki();

      b_first_visit_ = false;
    }

    if (count_ <= MAX_COUNT) {
      Eigen::VectorXd current_kp =
          init_kp_ + (target_kp_ - init_kp_) / MAX_COUNT * count_;
      ctrl_arch_->tci_container_->task_map_[task_name_]->SetKp(current_kp);

      Eigen::VectorXd current_kd =
          init_kd_ + (target_kd_ - init_kd_) / MAX_COUNT * count_;
      ctrl_arch_->tci_container_->task_map_[task_name_]->SetKd(current_kd);

      if (b_com_xy_task_) {
        Eigen::VectorXd current_ki =
            init_ki_ + (target_ki_ - init_ki_) / MAX_COUNT * count_;
        ctrl_arch_->tci_container_->task_map_[task_name_]->SetKi(current_ki);
      }
    } else if (ctrl_arch_wbic_) {
      // ======================================================
      // WBIC
      // ======================================================
      if (b_first_visit_) {
        init_kp_ = ctrl_arch_wbic_->tci_container_->task_map_[task_name_]->Kp();
        init_kd_ = ctrl_arch_wbic_->tci_container_->task_map_[task_name_]->Kd();
        if (b_com_xy_task_)
          init_ki_ =
              ctrl_arch_wbic_->tci_container_->task_map_[task_name_]->Ki();

        b_first_visit_ = false;
      }

      if (count_ <= MAX_COUNT) {
        Eigen::VectorXd current_kp =
            init_kp_ + (target_kp_ - init_kp_) / MAX_COUNT * count_;
        ctrl_arch_wbic_->tci_container_->task_map_[task_name_]->SetKp(
            current_kp);

        Eigen::VectorXd current_kd =
            init_kd_ + (target_kd_ - init_kd_) / MAX_COUNT * count_;
        ctrl_arch_wbic_->tci_container_->task_map_[task_name_]->SetKd(
            current_kd);

        if (b_com_xy_task_) {
          Eigen::VectorXd current_ki =
              init_ki_ + (target_ki_ - init_ki_) / MAX_COUNT * count_;
          ctrl_arch_wbic_->tci_container_->task_map_[task_name_]->SetKi(
              current_ki);
        }
      }
    }

    if (count_ == MAX_COUNT)
      _ResetParams();
  }
}

void G1TaskGainHandler::_ResetParams() {
  b_signal_received_ = false;

  init_kp_.setZero();
  init_kd_.setZero();
  init_ki_.setZero();

  task_name_ = "";
  target_kp_.setZero();
  target_kd_.setZero();
  target_ki_.setZero();

  count_ = 0;

  if (b_com_xy_task_)
    b_com_xy_task_ = false;
}

bool G1TaskGainHandler::_TaskExists(const std::string &task_name) {
  // return task_found ? true : false;
  if (ctrl_arch_) {
    auto task_map = ctrl_arch_->tci_container_->task_map_;

    if (task_map.find(task_name) == task_map.end()) {
      std::cout << "================================================"
                << std::endl;
      std::cout << "G1TaskGainHandler: Task not found" << std::endl;
      std::cout << "================================================"
                << std::endl;
      return false;
    }
  } else if (ctrl_arch_wbic_) {
    auto task_map = ctrl_arch_wbic_->tci_container_->task_map_;

    if (task_map.find(task_name) == task_map.end()) {
      std::cout << "================================================"
                << std::endl;
      std::cout << "G1TaskGainHandler: Task not found" << std::endl;
      std::cout << "================================================"
                << std::endl;
      return false;
    }
  }

  return true;
}
