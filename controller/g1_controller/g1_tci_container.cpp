#include "controller/g1_controller/g1_tci_container.hpp"
#include "controller/g1_controller/g1_definition.hpp"
#include "controller/g1_controller/g1_task/g1_cam_task.hpp"
#include "controller/g1_controller/g1_task/g1_com_xy_task.hpp"
#include "controller/g1_controller/g1_task/g1_com_z_task.hpp"
//#include "controller/g1_controller/g1_task/g1_wbo_task.hpp"
#include "controller/whole_body_controller/basic_contact.hpp"
#include "controller/whole_body_controller/basic_task.hpp"
#include "controller/whole_body_controller/force_task.hpp"
#include "controller/whole_body_controller/wbic/wbic.hpp"

#include <utility>

G1TCIContainer::G1TCIContainer(PinocchioRobotSystem *robot,
                                     const YAML::Node &cfg)
    : TCIContainer(robot) {
  util::PrettyConstructor(2, "G1TCIContainer");

  //=============================================================
  // Tasks List
  //=============================================================
  jpos_task_ = new JointTask(robot_);
  com_xy_task_ = new G1CoMXYTask(robot_);
  com_z_task_ = new G1CoMZTask(robot_);
  cam_task_ = new G1CAMTask(robot_);
  torso_ori_task_ = new LinkOriTask(robot_, g1_link::torso_com_link);
  pelvis_ori_task_ = new LinkOriTask(robot_, g1_link::pelvis);
  std::vector<int> upper_body_jidx{
      g1_joint::left_shoulder_pitch_joint, g1_joint::left_shoulder_roll_joint,
      g1_joint::left_shoulder_yaw_joint, g1_joint::left_elbow_joint,
      g1_joint::left_wrist_roll_joint, g1_joint::left_wrist_pitch_joint,
      g1_joint::left_wrist_yaw_joint,
      g1_joint::right_shoulder_pitch_joint, g1_joint::right_shoulder_roll_joint,
      g1_joint::right_shoulder_yaw_joint, g1_joint::right_elbow_joint,
      g1_joint::right_wrist_roll_joint, g1_joint::right_wrist_pitch_joint,
      g1_joint::right_wrist_yaw_joint};
  upper_body_task_ = new SelectedJointTask(robot_, upper_body_jidx);
  lf_pos_task_ = new LinkPosTask(robot_, g1_link::l_foot_contact);
  rf_pos_task_ = new LinkPosTask(robot_, g1_link::r_foot_contact);
  lf_ori_task_ = new LinkOriTask(robot_, g1_link::l_foot_contact);
  rf_ori_task_ = new LinkOriTask(robot_, g1_link::r_foot_contact);
  lh_pos_task_ = new LinkPosTask(robot_, g1_link::left_wrist_yaw_link);   // TODO check
  rh_pos_task_ = new LinkPosTask(robot_, g1_link::right_wrist_yaw_link);  // TODO check
  lh_ori_task_ = new LinkOriTask(robot_, g1_link::left_wrist_yaw_link);   // TODO check
  rh_ori_task_ = new LinkOriTask(robot_, g1_link::right_wrist_yaw_link);  // TODO check
//  wbo_task_ = new G1WBOTask(robot_);

  task_map_.clear();
  task_map_.insert(std::make_pair("joint_task", jpos_task_));
  task_map_.insert(std::make_pair("com_xy_task", com_xy_task_));
  task_map_.insert(std::make_pair("com_z_task", com_z_task_));
  task_map_.insert(std::make_pair("cam_task", cam_task_));
  task_map_.insert(std::make_pair("torso_ori_task", torso_ori_task_));
  task_map_.insert(std::make_pair("pelvis_ori_task", pelvis_ori_task_));
  task_map_.insert(std::make_pair("upper_body_task", upper_body_task_));
  task_map_.insert(std::make_pair("lf_pos_task", lf_pos_task_));
  task_map_.insert(std::make_pair("rf_pos_task", rf_pos_task_));
  task_map_.insert(std::make_pair("lf_ori_task", lf_ori_task_));
  task_map_.insert(std::make_pair("rf_ori_task", rf_ori_task_));
  task_map_.insert(std::make_pair("lh_pos_task", lh_pos_task_));
  task_map_.insert(std::make_pair("rh_pos_task", rh_pos_task_));
  task_map_.insert(std::make_pair("lh_ori_task", lh_ori_task_));
  task_map_.insert(std::make_pair("rh_ori_task", rh_ori_task_));
  // task_map_.insert(std::make_pair("wbo_task", wbo_task_));

  // initialize wbc cost task list
  task_unweighted_cost_map_.clear();
  task_unweighted_cost_map_.insert(std::make_pair("joint_task", NAN));
  task_unweighted_cost_map_.insert(std::make_pair("com_xy_task", NAN));
  task_unweighted_cost_map_.insert(std::make_pair("com_z_task", NAN));
  task_unweighted_cost_map_.insert(std::make_pair("cam_task", NAN));
  task_unweighted_cost_map_.insert(std::make_pair("torso_ori_task", NAN));
  task_unweighted_cost_map_.insert(std::make_pair("pelvis_ori_task", NAN));
  task_unweighted_cost_map_.insert(std::make_pair("upper_body_task", NAN));
  task_unweighted_cost_map_.insert(std::make_pair("lf_pos_task", NAN));
  task_unweighted_cost_map_.insert(std::make_pair("rf_pos_task", NAN));
  task_unweighted_cost_map_.insert(std::make_pair("lf_ori_task", NAN));
  task_unweighted_cost_map_.insert(std::make_pair("rf_ori_task", NAN));
  task_unweighted_cost_map_.insert(std::make_pair("lh_pos_task", NAN));
  task_unweighted_cost_map_.insert(std::make_pair("rh_pos_task", NAN));
  task_unweighted_cost_map_.insert(std::make_pair("lh_ori_task", NAN));
  task_unweighted_cost_map_.insert(std::make_pair("rh_ori_task", NAN));
  task_weighted_cost_map_.clear();
  task_weighted_cost_map_.insert(std::make_pair("joint_task", NAN));
  task_weighted_cost_map_.insert(std::make_pair("com_xy_task", NAN));
  task_weighted_cost_map_.insert(std::make_pair("com_z_task", NAN));
  task_weighted_cost_map_.insert(std::make_pair("cam_task", NAN));
  task_weighted_cost_map_.insert(std::make_pair("torso_ori_task", NAN));
  task_weighted_cost_map_.insert(std::make_pair("pelvis_ori_task", NAN));
  task_weighted_cost_map_.insert(std::make_pair("upper_body_task", NAN));
  task_weighted_cost_map_.insert(std::make_pair("lf_pos_task", NAN));
  task_weighted_cost_map_.insert(std::make_pair("rf_pos_task", NAN));
  task_weighted_cost_map_.insert(std::make_pair("lf_ori_task", NAN));
  task_weighted_cost_map_.insert(std::make_pair("rf_ori_task", NAN));
  task_weighted_cost_map_.insert(std::make_pair("lh_pos_task", NAN));
  task_weighted_cost_map_.insert(std::make_pair("rh_pos_task", NAN));
  task_weighted_cost_map_.insert(std::make_pair("lh_ori_task", NAN));
  task_weighted_cost_map_.insert(std::make_pair("rh_ori_task", NAN));

  // wbc task list for inverse kinematics
  task_vector_.clear();
  task_vector_.push_back(com_z_task_);
  task_vector_.push_back(torso_ori_task_);
  task_vector_.push_back(pelvis_ori_task_);
  task_vector_.push_back(com_xy_task_);
  task_vector_.push_back(upper_body_task_);
  task_vector_.push_back(lf_pos_task_);
  task_vector_.push_back(rf_pos_task_);
  task_vector_.push_back(lf_ori_task_);
  task_vector_.push_back(rf_ori_task_);
  // task_vector_.push_back(wbo_task_);
  // task_vector_.push_back(cam_task_);

  //=============================================================
  // Contacts List
  //=============================================================
  lf_contact_ = new SurfaceContact(robot_, g1_link::l_foot_contact, 0.7,
                                   0.08, 0.025); // params reset later
  rf_contact_ = new SurfaceContact(robot_, g1_link::r_foot_contact, 0.7,
                                   0.08, 0.025); // params reset later

  contact_map_.clear();
  contact_map_.insert(std::make_pair("lf_contact", lf_contact_));
  contact_map_.insert(std::make_pair("rf_contact", rf_contact_));

  contact_vector_.clear();
  contact_vector_.push_back(lf_contact_);
  contact_vector_.push_back(rf_contact_);

  //=============================================================
  // InternalConstraints List
  //=============================================================

  //=============================================================
  // Force Task List
  //=============================================================
  lf_reaction_force_task_ = new ForceTask(robot_, lf_contact_);
  rf_reaction_force_task_ = new ForceTask(robot_, rf_contact_);

  force_task_map_.clear();
  force_task_map_.insert(
      std::make_pair("lf_force_task", lf_reaction_force_task_));
  force_task_map_.insert(
      std::make_pair("rf_force_task", rf_reaction_force_task_));

  force_task_vector_.clear();
  force_task_vector_.push_back(lf_reaction_force_task_);
  force_task_vector_.push_back(rf_reaction_force_task_);

  //=============================================================
  // QP Params
  //=============================================================
  qp_params_ = new QPParams(6, lf_contact_->Dim() + rf_contact_->Dim());

  //=============================================================
  // Tasks, Contacts parameter initialization
  //=============================================================
  this->_InitializeParameters(cfg);
}

G1TCIContainer::~G1TCIContainer() {
  // task
  delete jpos_task_;
  delete com_xy_task_;
  delete com_z_task_;
  delete cam_task_;
  delete torso_ori_task_;
  delete pelvis_ori_task_;
  delete upper_body_task_;
  delete lf_pos_task_;
  delete rf_pos_task_;
  delete lf_ori_task_;
  delete rf_ori_task_;
  delete lh_pos_task_;
  delete rh_pos_task_;
  delete lh_ori_task_;
  delete rh_ori_task_;
  delete wbo_task_;

  // contact
  delete lf_contact_;
  delete rf_contact_;

  // force task
  delete lf_reaction_force_task_;
  delete rf_reaction_force_task_;

  // QP Params
  delete qp_params_;
}

void G1TCIContainer::_InitializeParameters(const YAML::Node &cfg) {
  // determine which WBC
  WBC_TYPE wbc_type;
  std::string wbc_type_string =
      util::ReadParameter<std::string>(cfg, "wbc_type");
  if (wbc_type_string == "ihwbc") {
    wbc_type = WBC_TYPE::IHWBC;
  } else if (wbc_type_string == "wbic") {
    wbc_type = WBC_TYPE::WBIC;
  }

  // task
  com_xy_task_->SetParameters(cfg["wbc"]["task"]["com_xy_task"], wbc_type);
  com_z_task_->SetParameters(cfg["wbc"]["task"]["com_z_task"], wbc_type);
  cam_task_->SetParameters(cfg["wbc"]["task"]["cam_task"], wbc_type);
//  wbo_task_->SetParameters(cfg["wbc"]["task"]["wbo_task"], wbc_type);
  torso_ori_task_->SetParameters(cfg["wbc"]["task"]["torso_ori_task"],
                                 wbc_type);
  pelvis_ori_task_->SetParameters(cfg["wbc"]["task"]["pelvis_ori_task"],
                                 wbc_type);
  upper_body_task_->SetParameters(cfg["wbc"]["task"]["upper_body_task"],
                                  wbc_type);
  lf_pos_task_->SetParameters(cfg["wbc"]["task"]["foot_pos_task"], wbc_type);
  rf_pos_task_->SetParameters(cfg["wbc"]["task"]["foot_pos_task"], wbc_type);
  lf_ori_task_->SetParameters(cfg["wbc"]["task"]["foot_ori_task"], wbc_type);
  rf_ori_task_->SetParameters(cfg["wbc"]["task"]["foot_ori_task"], wbc_type);
  lh_pos_task_->SetParameters(cfg["wbc"]["task"]["hand_pos_task"], wbc_type);
  rh_pos_task_->SetParameters(cfg["wbc"]["task"]["hand_pos_task"], wbc_type);
  lh_ori_task_->SetParameters(cfg["wbc"]["task"]["hand_ori_task"], wbc_type);
  rh_ori_task_->SetParameters(cfg["wbc"]["task"]["hand_ori_task"], wbc_type);

  // contact
  lf_contact_->SetParameters(cfg["wbc"]["contact"]);
  rf_contact_->SetParameters(cfg["wbc"]["contact"]);

  // force task
  lf_reaction_force_task_->SetParameters(cfg["wbc"]["task"]["foot_rf_task"]);
  rf_reaction_force_task_->SetParameters(cfg["wbc"]["task"]["foot_rf_task"]);
}
