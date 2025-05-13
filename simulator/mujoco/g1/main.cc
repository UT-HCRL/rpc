// Copyright 2021 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <thread>

#include "mujoco/mujoco.h"
#include "simulate/array_safety.h"
#include "simulate/glfw_adapter.h"
#include "simulate/simulate.h"

// rpc related headers
#include "configuration.hpp"
#include "controller/g1_controller/g1_interface.hpp"
#include "controller/robot_system/pinocchio_robot_system.hpp"
#include "util/util.hpp"
#include "mujoco_utils.hpp"

#define MUJOCO_PLUGIN_DIR "mujoco_plugin"

extern "C" {
#if defined(_WIN32) || defined(__CYGWIN__)
#include <windows.h>
#else
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif
#include <sys/errno.h>
#include <unistd.h>
#endif
}

namespace {
namespace mj = ::mujoco;
namespace mju = ::mujoco::sample_util;

// constants
const double syncMisalign =
    0.1; // maximum mis-alignment before re-sync (simulation seconds)
const double simRefreshFraction =
    0.7;                       // fraction of refresh available for simulation
const int kErrorLength = 1024; // load error string length

// model and data
mjModel *m = nullptr;
mjData *d = nullptr;

// control noise variables
mjtNum *ctrlnoise = nullptr;

// rpc controller
Interface *g1_interface = nullptr;
G1SensorData *g1_sensor_data = nullptr;
G1Command *g1_command = nullptr;

// pinocchio joint & actuator maps
std::unordered_map<std::string, int> pin_jnt_map_;
std::unordered_map<std::string, int> pin_act_map_;

// mujoco joint & actuator maps
std::unordered_map<std::string, int> mj_qpos_map_;
std::unordered_map<std::string, int> mj_qvel_map_;
std::unordered_map<std::string, int> mj_act_map_;

// imu sensor address (framequat, gyro, accelerometer)
int imu_orientation_adr_;
int imu_ang_vel_adr_;
int imu_lin_acc_adr_;
int imu_lin_vel_adr_;

// for calculating dvel
Eigen::Vector3d prev_imu_lin_vel_in_world_ = Eigen::Vector3d::Zero();

// yaml node
YAML::Node cfg_;

// actuator gains
std::vector<double> kp_;
std::vector<double> kd_;

using Seconds = std::chrono::duration<double>;

//---------------------------------------- plugin handling
//-----------------------------------------

// return the path to the directory containing the current executable
// used to determine the location of auto-loaded plugin libraries
std::string getExecutableDir() {
#if defined(_WIN32) || defined(__CYGWIN__)
  constexpr char kPathSep = '\\';
  std::string realpath = [&]() -> std::string {
    std::unique_ptr<char[]> realpath(nullptr);
    DWORD buf_size = 128;
    bool success = false;
    while (!success) {
      realpath.reset(new (std::nothrow) char[buf_size]);
      if (!realpath) {
        std::cerr << "cannot allocate memory to store executable path\n";
        return "";
      }

      DWORD written = GetModuleFileNameA(nullptr, realpath.get(), buf_size);
      if (written < buf_size) {
        success = true;
      } else if (written == buf_size) {
        // realpath is too small, grow and retry
        buf_size *= 2;
      } else {
        std::cerr << "failed to retrieve executable path: " << GetLastError()
                  << "\n";
        return "";
      }
    }
    return realpath.get();
  }();
#else
  constexpr char kPathSep = '/';
#if defined(__APPLE__)
  std::unique_ptr<char[]> buf(nullptr);
  {
    std::uint32_t buf_size = 0;
    _NSGetExecutablePath(nullptr, &buf_size);
    buf.reset(new char[buf_size]);
    if (!buf) {
      std::cerr << "cannot allocate memory to store executable path\n";
      return "";
    }
    if (_NSGetExecutablePath(buf.get(), &buf_size)) {
      std::cerr << "unexpected error from _NSGetExecutablePath\n";
    }
  }
  const char *path = buf.get();
#else
  const char *path = "/proc/self/exe";
#endif
  std::string realpath = [&]() -> std::string {
    std::unique_ptr<char[]> realpath(nullptr);
    std::uint32_t buf_size = 128;
    bool success = false;
    while (!success) {
      realpath.reset(new (std::nothrow) char[buf_size]);
      if (!realpath) {
        std::cerr << "cannot allocate memory to store executable path\n";
        return "";
      }

      std::size_t written = readlink(path, realpath.get(), buf_size);
      if (written < buf_size) {
        realpath.get()[written] = '\0';
        success = true;
      } else if (written == -1) {
        if (errno == EINVAL) {
          // path is already not a symlink, just use it
          return path;
        }

        std::cerr << "error while resolving executable path: "
                  << strerror(errno) << '\n';
        return "";
      } else {
        // realpath is too small, grow and retry
        buf_size *= 2;
      }
    }
    return realpath.get();
  }();
#endif

  if (realpath.empty()) {
    return "";
  }

  for (std::size_t i = realpath.size() - 1; i > 0; --i) {
    if (realpath.c_str()[i] == kPathSep) {
      return realpath.substr(0, i);
    }
  }

  // don't scan through the entire file system's root
  return "";
}

// scan for libraries in the plugin directory to load additional plugins
void scanPluginLibraries() {
  // check and print plugins that are linked directly into the executable
  int nplugin = mjp_pluginCount();
  if (nplugin) {
    std::printf("Built-in plugins:\n");
    for (int i = 0; i < nplugin; ++i) {
      std::printf("    %s\n", mjp_getPluginAtSlot(i)->name);
    }
  }

  // define platform-specific strings
#if defined(_WIN32) || defined(__CYGWIN__)
  const std::string sep = "\\";
#else
  const std::string sep = "/";
#endif

  // try to open the ${EXECDIR}/MUJOCO_PLUGIN_DIR directory
  // ${EXECDIR} is the directory containing the simulate binary itself
  // MUJOCO_PLUGIN_DIR is the MUJOCO_PLUGIN_DIR preprocessor macro
  const std::string executable_dir = getExecutableDir();
  if (executable_dir.empty()) {
    return;
  }

  const std::string plugin_dir = getExecutableDir() + sep + MUJOCO_PLUGIN_DIR;
  mj_loadAllPluginLibraries(
      plugin_dir.c_str(), +[](const char *filename, int first, int count) {
        std::printf("Plugins registered by library '%s':\n", filename);
        for (int i = first; i < first + count; ++i) {
          std::printf("    %s\n", mjp_getPluginAtSlot(i)->name);
        }
      });
}

//------------------------------------------- simulation
//-------------------------------------------

const char* Diverged(int disableflags, const mjData* d) {
  if (disableflags & mjDSBL_AUTORESET) {
    for (mjtWarning w : {mjWARN_BADQACC, mjWARN_BADQVEL, mjWARN_BADQPOS}) {
      if (d->warning[w].number > 0) {
        return mju_warningText(w, d->warning[w].lastinfo);
      }
    }
  }
  return nullptr;
}

mjModel *LoadModel(const char *file, mj::Simulate &sim) {
  // this copy is needed so that the mju::strlen call below compiles
  char filename[mj::Simulate::kMaxFilenameLength];
  mju::strcpy_arr(filename, file);

  // make sure filename is not empty
  if (!filename[0]) {
    return nullptr;
  }

  // load and compile
  char loadError[kErrorLength] = "";
  mjModel *mnew = 0;
  if (mju::strlen_arr(filename) > 4 &&
      !std::strncmp(filename + mju::strlen_arr(filename) - 4, ".mjb",
                    mju::sizeof_arr(filename) - mju::strlen_arr(filename) +
                        4)) {
    mnew = mj_loadModel(filename, nullptr);
    if (!mnew) {
      mju::strcpy_arr(loadError, "could not load binary model");
    }
  } else {
    mnew = mj_loadXML(filename, nullptr, loadError, kErrorLength);
    // remove trailing newline character from loadError
    if (loadError[0]) {
      int error_length = mju::strlen_arr(loadError);
      if (loadError[error_length - 1] == '\n') {
        loadError[error_length - 1] = '\0';
      }
    }
  }

  mju::strcpy_arr(sim.load_error, loadError);

  if (!mnew) {
    std::printf("%s\n", loadError);
    return nullptr;
  }

  // compiler warning: print and pause
  if (loadError[0]) {
    // mj_forward() below will print the warning message
    std::printf("Model compiled, but simulation warning (paused):\n  %s\n",
                loadError);
    sim.run = 0;
  }

  return mnew;
}

void SetYamlNode(YAML::Node &node) {
  try {
    YAML::Node interface_cfg =
        YAML::LoadFile(THIS_COM "config/g1/INTERFACE.yaml");
    std::string wbc_type = util::ReadParameter<std::string>(
        interface_cfg, "whole_body_controller");
    if (wbc_type == "ihwbc")
      node = YAML::LoadFile(THIS_COM
                            "config/g1/sim/mujoco/ihwbc/mujoco_params.yaml");
    else if (wbc_type == "wbic")
      node = YAML::LoadFile(THIS_COM
                            "config/g1/sim/mujoco/wbic/mujoco_params.yaml");

  } catch (const std::runtime_error &ex) {
    std::cerr << "Error Reading Parameter [" << ex.what() << "] at file: ["
              << __FILE__ << "]" << std::endl;
  }
}

void SetActuatorGains(const std::unordered_map<std::string, int> &mj_act_map) {
  kp_.clear();
  kp_.resize(mj_act_map.size());
  kd_.clear();
  kd_.resize(mj_act_map.size());

  try {
    double tmp;
    for (const auto &[mj_act_name, mj_act_idx] : mj_act_map) {
      util::ReadParameter(cfg_["actuator_gains"][mj_act_name], "kp", tmp);
      kp_[mj_act_idx] = tmp;
      util::ReadParameter(cfg_["actuator_gains"][mj_act_name], "kd", tmp);
      kd_[mj_act_idx] = tmp;
    }
  } catch (const std::runtime_error &ex) {
    std::cerr << "Error Reading Parameter [" << ex.what() << "] at file: ["
              << __FILE__ << "]" << std::endl;
  }
}

void SetInitialConfig(mjModel *m, mjData *d,
                      const std::unordered_map<std::string, int> &mj_qpos_map) {
  try {
    // set floating base pose
    const std::string &base_frame_name =
        g1_interface->GetPinocchioModel()->GetRootFrameName();
    const int base_frame_idx =
        mj_name2id(m, mjOBJ_BODY, base_frame_name.c_str());
    // make sure we have a floating body: it has a single free joint
    if (base_frame_idx >= 0 && m->body_jntnum[base_frame_idx] == 1 &&
        m->jnt_type[m->body_jntadr[base_frame_idx]] == mjJNT_FREE) {
      int qposadr = m->jnt_qposadr[m->body_jntadr[base_frame_idx]];

      const std::vector<std::string> base_pose = {
          "base_pos_x",  "base_pos_y",  "base_pos_z", "base_quat_w",
          "base_quat_x", "base_quat_y", "base_quat_z"};

      for (int i = 0; i < base_pose.size(); ++i) {
        d->qpos[qposadr + i] = util::ReadParameter<double>(
            cfg_["initial_config"], base_pose.at(i));
      }
    }

    // set joint pos
    for (const auto &[joint_name, qpos_adr] : mj_qpos_map) {
      d->qpos[qpos_adr] =
          util::ReadParameter<double>(cfg_["initial_config"], joint_name);
    }
  } catch (const std::runtime_error &ex) {
    std::cerr << "Error Reading Parameter [" << ex.what() << "] at file: ["
              << __FILE__ << "]" << std::endl;
  }

  // optiona for setting the initial config with keyframe tag specified in xml
  // if (m->nkey != 0)
  // mju_copy(d->qpos, m->key_qpos, m->nq);
}

bool CopySensorData() {
  //=================================================
  // joint positions
  //=================================================
  for (const auto &[mj_qpos_name, mj_qpos_idx] : mj_qpos_map_)
    g1_sensor_data->joint_pos_[pin_jnt_map_.at(mj_qpos_name)] =
        d->qpos[mj_qpos_idx];

  //=================================================
  // joint velocities
  //=================================================
  for (const auto &[mj_qvel_name, mj_qvel_idx] : mj_qvel_map_)
    g1_sensor_data->joint_vel_[pin_jnt_map_.at(mj_qvel_name)] =
        d->qvel[mj_qvel_idx];

  //==============================================
  // floating base states for ground truth state estimation
  //==============================================
  const std::string &base_frame_name =
      g1_interface->GetPinocchioModel()->GetRootFrameName();
  const int base_frame_idx = mj_name2id(m, mjOBJ_BODY, base_frame_name.c_str());
  // make sure we have a floating body: it has a single free joint
  if (base_frame_idx >= 0 && m->body_jntnum[base_frame_idx] == 1 &&
      m->jnt_type[m->body_jntadr[base_frame_idx]] == mjJNT_FREE) {
    int qposadr = m->jnt_qposadr[m->body_jntadr[base_frame_idx]];
    int qveladr = m->jnt_dofadr[m->body_jntadr[base_frame_idx]];

    g1_sensor_data->base_joint_pos_[0] = d->qpos[qposadr + 0]; // x
    g1_sensor_data->base_joint_pos_[1] = d->qpos[qposadr + 1]; // y
    g1_sensor_data->base_joint_pos_[2] = d->qpos[qposadr + 2]; // z

    g1_sensor_data->base_joint_quat_[0] = d->qpos[qposadr + 4]; // q.x
    g1_sensor_data->base_joint_quat_[1] = d->qpos[qposadr + 5]; // q.y
    g1_sensor_data->base_joint_quat_[2] = d->qpos[qposadr + 6]; // q.z
    g1_sensor_data->base_joint_quat_[3] = d->qpos[qposadr + 3]; // q.w
    Eigen::Quaterniond world_Q_body(d->qpos[qposadr + 3], d->qpos[qposadr + 4],
                                    d->qpos[qposadr + 5], d->qpos[qposadr + 6]);

    // world linear vel
    g1_sensor_data->base_joint_lin_vel_[0] = d->qvel[qveladr + 0];
    g1_sensor_data->base_joint_lin_vel_[1] = d->qvel[qveladr + 1];
    g1_sensor_data->base_joint_lin_vel_[2] = d->qvel[qveladr + 2];

    // convert to world angular vel
    Eigen::Vector3d body_ang_vel(d->qvel[qveladr + 3], d->qvel[qveladr + 4],
                                 d->qvel[qveladr + 5]);
    g1_sensor_data->base_joint_ang_vel_ =
        world_Q_body.normalized() * body_ang_vel;
  }

  //==============================================
  // process IMU data
  //==============================================
  // imu orienation
  mjtNum *imu_quat = &d->sensordata[imu_orientation_adr_];
  Eigen::Quaterniond world_Q_imu(imu_quat[0], imu_quat[1], imu_quat[2],
                                 imu_quat[3]); // q.w, q.x, q.y, q.z
  g1_sensor_data->imu_frame_quat_ = Eigen::Vector4d(
      world_Q_imu.x(), world_Q_imu.y(), world_Q_imu.z(), world_Q_imu.w());

  // imu angular velocity
  mjtNum *imu_ang_vel = &d->sensordata[imu_ang_vel_adr_];
  Eigen::Vector3d imu_ang_vel_in_imu(imu_ang_vel[0], imu_ang_vel[1],
                                     imu_ang_vel[2]);
  g1_sensor_data->imu_ang_vel_ =
      world_Q_imu.normalized() * imu_ang_vel_in_imu;

  // imu linear acceleration
  const double grav_acc = 9.81;
  mjtNum *imu_lin_acc = &d->sensordata[imu_lin_acc_adr_];
  Eigen::Vector3d imu_lin_acc_in_imu(imu_lin_acc[0], imu_lin_acc[1],
                                     imu_lin_acc[2] + grav_acc);
  world_Q_imu.normalized() * imu_lin_acc_in_imu * m->opt.timestep;
  g1_sensor_data->imu_lin_acc_ =
      world_Q_imu.normalized() * imu_lin_acc_in_imu;
  g1_sensor_data->imu_dvel_ =
      world_Q_imu.normalized() * imu_lin_acc_in_imu * m->opt.timestep;
  // std::cout << "-------------------------------------------------------"
  //<< std::endl;
  // std::cout << "imu_dvel from accelerometer: "
  //<< g1_sensor_data->imu_dvel_.transpose() << std::endl;

  // imu frame lin acc
  mjtNum *imu_lin_vel = &d->sensordata[imu_lin_vel_adr_];
  Eigen::Vector3d imu_lin_vel_in_world(imu_lin_vel[0], imu_lin_vel[1],
                                       imu_lin_vel[2]);
  // g1_sensor_data->imu_dvel_ =
  // imu_lin_vel_in_world - prev_imu_lin_vel_in_world_;
  // prev_imu_lin_vel_in_world_ = imu_lin_vel_in_world;
  // std::cout << "imu_dvel from imu lin vel: "
  //<< g1_sensor_data->imu_dvel_.transpose() << std::endl;
  //==============================================
  // TODO(optional): contact sensor (1. contact normal force, 2. swing foot
  // height)
  //==============================================

  return true;
}

void CopyCommand() {
  // joint impednace control law
  for (const auto &[mj_act_name, mj_act_idx] : mj_act_map_) {
    int pin_act_idx = pin_act_map_.at(mj_act_name);
    int mj_qpos_idx = mj_qpos_map_.at(mj_act_name);
    int mj_qvel_idx = mj_qvel_map_.at(mj_act_name);

    d->ctrl[mj_act_idx] =
        g1_command->joint_trq_cmd_[pin_act_idx] +
        kp_[mj_act_idx] * (g1_command->joint_pos_cmd_[pin_act_idx] -
                           d->qpos[mj_qpos_idx]) +
        kd_[mj_act_idx] *
            (g1_command->joint_vel_cmd_[pin_act_idx] - d->qvel[mj_qvel_idx]);
  }
}

// simulate in background thread (while rendering in main thread)
void PhysicsLoop(mj::Simulate &sim) {
  // cpu-sim syncronization point
  std::chrono::time_point<mj::Simulate::Clock> syncCPU;
  mjtNum syncSim = 0;

  //***************************************************
  // run until asked to exit (main simulation while loop)
  //***************************************************
  while (!sim.exitrequest.load()) {
    if (sim.droploadrequest.load()) {
      sim.LoadMessage(sim.dropfilename);
      mjModel *mnew = LoadModel(sim.dropfilename, sim);
      sim.droploadrequest.store(false);

      std::cout << "sim drop request!" << '\n';

      mjData *dnew = nullptr;
      if (mnew)
        dnew = mj_makeData(mnew);
      if (dnew) {
        sim.Load(mnew, dnew, sim.dropfilename);

        // lock the sim mutex
        const std::unique_lock<std::recursive_mutex> lock(sim.mtx);

        mj_deleteData(d);
        mj_deleteModel(m);

        m = mnew;
        d = dnew;
        mj_forward(m, d);

      } else {
        sim.LoadMessageClear();
      }
    }

    if (sim.uiloadrequest.load()) {
      sim.uiloadrequest.fetch_sub(1);
      sim.LoadMessage(sim.filename);
      mjModel *mnew = LoadModel(sim.filename, sim);
      mjData *dnew = nullptr;

      std::cout << "sim uiload request!" << '\n';

      if (mnew)
        dnew = mj_makeData(mnew);
      if (dnew) {
        sim.Load(mnew, dnew, sim.filename);

        // lock the sim mutex
        const std::unique_lock<std::recursive_mutex> lock(sim.mtx);

        mj_deleteData(d);
        mj_deleteModel(m);

        m = mnew;
        d = dnew;
        mj_forward(m, d);

      } else {
        sim.LoadMessageClear();
      }
    }

    // sleep for 1 ms or yield, to let main thread run
    //  yield results in busy wait - which has better timing but kills battery
    //  life
    if (sim.run && sim.busywait) {
      std::this_thread::yield();
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    {
      // lock the sim mutex
      const std::unique_lock<std::recursive_mutex> lock(sim.mtx);

      //***********************************************************
      // run only if model is present (main simulation loop)
      //***********************************************************
      if (m) {
        // running
        if (sim.run) {
          //*****************************************************
          if (CopySensorData())
            g1_interface->GetCommand(g1_sensor_data, g1_command);
          CopyCommand();
          //*****************************************************

          bool stepped = false;

          // record cpu time at start of iteration
          const auto startCPU = mj::Simulate::Clock::now();

          // elapsed CPU and simulation time since last sync
          const auto elapsedCPU = startCPU - syncCPU;
          double elapsedSim = d->time - syncSim;

          // inject noise
          // if (sim.ctrl_noise_std) {
          // convert rate and scale to discrete time (Ornstein–Uhlenbeck)
          // mjtNum rate = mju_exp(-m->opt.timestep /
          // mju_max(sim.ctrl_noise_rate, mjMINVAL));
          // mjtNum scale = sim.ctrl_noise_std * mju_sqrt(1 - rate * rate);

          // for (int i = 0; i < m->nu; i++) {
          // update noise
          // ctrlnoise[i] =
          // rate * ctrlnoise[i] + scale * mju_standardNormal(nullptr);

          // apply noise
          // d->ctrl[i] = ctrlnoise[i];
          //}
          //}

          // requested slow-down factor
          double slowdown = 100 / sim.percentRealTime[sim.real_time_index];

          // misalignment condition: distance from target sim time is bigger
          // than syncmisalign
          bool misaligned = mju_abs(Seconds(elapsedCPU).count() / slowdown -
                                    elapsedSim) > syncMisalign;

          // out-of-sync (for any reason): reset sync times, step
          if (elapsedSim < 0 || elapsedCPU.count() < 0 ||
              syncCPU.time_since_epoch().count() == 0 || misaligned ||
              sim.speed_changed) {
            // re-sync
            syncCPU = startCPU;
            syncSim = d->time;
            sim.speed_changed = false;

            // run single step, let next iteration deal with timing
            mj_step(m, d);
            const char* message = Diverged(m->opt.disableflags, d);
            if (message) {
              sim.run = 0;
              mju::strcpy_arr(sim.load_error, message);
            } else {
              stepped = true;
            }
          }

          // in-sync: step until ahead of cpu
          else {
            bool measured = false;
            mjtNum prevSim = d->time;

            double refreshTime = simRefreshFraction / sim.refresh_rate;

            // step while sim lags behind cpu and within refreshTime
            while (Seconds((d->time - syncSim) * slowdown) <
                       mj::Simulate::Clock::now() - syncCPU &&
                   mj::Simulate::Clock::now() - startCPU <
                       Seconds(refreshTime)) {
              // measure slowdown before first step
              if (!measured && elapsedSim) {
                sim.measured_slowdown =
                    std::chrono::duration<double>(elapsedCPU).count() /
                    elapsedSim;
                measured = true;
              }

              // inject noise
              // sim.InjectNoise(); //TODO: uncomment if needed @carlos

              // call mj_step
              mj_step(m, d);
              const char* message = Diverged(m->opt.disableflags, d);
              if (message) {
                sim.run = 0;
                mju::strcpy_arr(sim.load_error, message);
              } else {
                stepped = true;
              }

              // break if reset
              if (d->time < prevSim) {
                break;
              }
            }
          }

          // save current state to history buffer
          if (stepped) {
            sim.AddToHistory();
          }
        }

        // paused
        else {
          // run mj_forward, to update rendering and joint sliders
          mj_forward(m, d);
          sim.speed_changed = true;
        }
      }
    } // release std::lock_guard<std::mutex>
  }
}
} // namespace

//-------------------------------------- physics_thread
//--------------------------------------------

void PhysicsThread(mj::Simulate *sim, const char *filename) {
  // request loadmodel if file given (otherwise drag-and-drop)
  if (filename != nullptr) {
    sim->LoadMessage(filename);
    m = LoadModel(filename, *sim);
    if (m) {
      // lock the sim mutex
      const std::unique_lock<std::recursive_mutex> lock(sim->mtx);

      // TEST print model
      // mj_printModel(m, "g1_mjcf_info");

      d = mj_makeData(m);
    }
    if (d) {
      // Construct controller
      //**********************************************
      g1_interface = new G1Interface();
      g1_sensor_data = new G1SensorData();
      g1_command = new G1Command();
      //**********************************************

      // Pass keystroke interrupt
      //**********************************************
      sim->Load(m, d, filename, g1_interface->interrupt_handler_);
      //**********************************************

      // Set up mujoco joint & actuator maps based on the mujoco model
      // ********************************************
      pin_jnt_map_ =
          g1_interface->GetPinocchioModel()->GetJointNameAndIndexMap();
      pin_act_map_ =
          g1_interface->GetPinocchioModel()->GetActuatorNameAndIndexMap();
          
      mjc_utils::SetJointAndActuatorMaps(m, mj_qpos_map_, mj_qvel_map_, mj_act_map_,
                              pin_jnt_map_,
                              pin_act_map_);
      // ********************************************

      // Configure sensor (imu)
      // ********************************************
      mjc_utils::ConfigureSensors(m, imu_orientation_adr_, imu_ang_vel_adr_,
                       imu_lin_acc_adr_, imu_lin_vel_adr_);
      // ********************************************

      // Set YAML Node
      //**********************************************
      SetYamlNode(cfg_);
      //**********************************************

      // Set actuator gains
      //**********************************************
      SetActuatorGains(mj_act_map_);
      //**********************************************

      // Set initial configuration
      //**********************************************
      SetInitialConfig(m, d, mj_qpos_map_);
      //**********************************************

      // lock the sim mutex
      const std::unique_lock<std::recursive_mutex> lock(sim->mtx);

      mj_forward(m, d);

      // allocate ctrlnoise
      // free(ctrlnoise);
      // ctrlnoise = static_cast<mjtNum *>(malloc(sizeof(mjtNum) * m->nu));
      // mju_zero(ctrlnoise, m->nu);
    } else {
      sim->LoadMessageClear();
    }
  }

  PhysicsLoop(*sim);

  // delete everything we allocated
  delete g1_interface;
  delete g1_sensor_data;
  delete g1_command;
  // free(ctrlnoise);
  mj_deleteData(d);
  mj_deleteModel(m);
}

//------------------------------------------ main
//--------------------------------------------------

// machinery for replacing command line error by a macOS dialog box when running
// under Rosetta
#if defined(__APPLE__) && defined(__AVX__)
extern void DisplayErrorDialogBox(const char *title, const char *msg);
static const char *rosetta_error_msg = nullptr;
__attribute__((used, visibility("default"))) extern "C" void
_mj_rosettaError(const char *msg) {
  rosetta_error_msg = msg;
}
#endif

// run event loop
int main(int argc, char **argv) {

  // display an error if running on macOS under Rosetta 2
#if defined(__APPLE__) && defined(__AVX__)
  if (rosetta_error_msg) {
    DisplayErrorDialogBox("Rosetta 2 is not supported", rosetta_error_msg);
    std::exit(1);
  }
#endif

  // print version, check compatibility
  std::printf("MuJoCo version %s\n", mj_versionString());
  if (mjVERSION_HEADER != mj_version()) {
    mju_error("Headers and library have different versions");
  }

  // scan for libraries in the plugin directory to load additional plugins
  scanPluginLibraries();

  mjvCamera cam;
  mjv_defaultCamera(&cam);

  mjvOption opt;
  mjv_defaultOption(&opt);

  mjvPerturb pert;
  mjv_defaultPerturb(&pert);

  // simulate object encapsulates the UI
  auto sim =
      std::make_unique<mj::Simulate>(std::make_unique<mj::GlfwAdapter>(), &cam,
                                     &opt, &pert, /* is_passive = */ false);

  const char *filename = nullptr;

  if (argc > 1) {
    filename = argv[1];
  } else {
    filename = THIS_COM "robot_model/g1/wing_scene.xml";
  }

  // start physics thread
  std::thread physicsthreadhandle(&PhysicsThread, sim.get(), filename);

  // start simulation UI loop (blocking call)
  sim->RenderLoop();
  physicsthreadhandle.join();

  return 0;
}