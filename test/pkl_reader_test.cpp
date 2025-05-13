#include <gtest/gtest.h>
//#include "third_party/sciplot/sciplot.hpp"

#include "util/pkl_reader.hpp"
#include "util/util.hpp"

//using namespace sciplot;
using namespace pkl_utils;

static double err_tol = 1e-5;

class PklReaderTest : public ::testing::Test {
protected:
  PklReaderTest() {
    // Constructor code here
  }

  ~PklReaderTest() override {
    // Destructor code here
  }


  void SetUp() override {
  }

  void TearDown() override {
    // Code to clean up after each test
  }

  std::string file_path;
};


TEST_F(PklReaderTest, readList) {
  file_path = THIS_COM "experiment_data/g1_knee_knocker_sca_on.pkl";

  PickleReader reader(file_path, PickleType::LIST);
  if (!reader.isReady()) {
    std::cerr << "Failed to open the file." << std::endl;
  }

  reader.parse();

  std::vector<Matrix<double, 44, 1>> read_joint_pos = reader.getJointPosDes();
  std::vector<Matrix<double, 43, 1>> read_joint_vel = reader.getJointVelDes();
  std::vector<Matrix<double, 37, 1>> read_joint_tau = reader.getJointTauDes();
  std::vector<double> read_time = reader.getTimeVec();

  std::cout << "[MAIN] - Number of Joint Position Vectors: " << read_joint_pos.size() << std::endl;

  // std::cout << "[MAIN] - Select bezier to print..." << std::endl;
  int selected_composite = 1;
  // std::cin >> selected_composite;
  if (selected_composite < 0 || selected_composite >= read_joint_pos.size()) {
    std::cerr << "[MAIN] - Invalid selection." << std::endl;
  }

}

TEST_F(PklReaderTest, readBezier) {
  file_path = THIS_COM "data_example/g1_test.pkl";

  Matrix<double, 15, 1> test_times;
  test_times << 0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14.;

  PickleReader reader(file_path, PickleType::BEZIER);
  if (!reader.isReady()) {
    std::cerr << "Failed to open the file." << std::endl;
  }

  reader.parse();

  std::vector<CompositeBezierCurve> read_beziers = reader.getCompositeBezierCurves();
  std::vector<std::shared_ptr<CompositeBezierCurve>> bezier_curves_ptrs;

  // Convert using a loop
  for (const auto& curve : read_beziers) {
    bezier_curves_ptrs.push_back(std::make_shared<CompositeBezierCurve>(curve));
  }
  for (unsigned int fr_idx = 0; fr_idx<read_beziers.size(); fr_idx++){
    // get Bezier curves of current frame
    const auto selected_bezier_fr = bezier_curves_ptrs[fr_idx];

    // test a few points for the corresponding frame
    for (const auto& t : test_times) {
      Vector3d pos = pkl_utils::get_frame_des_pos(selected_bezier_fr, t);
      std::cout << "pos[" << fr_idx << "] at t = " << t << " s: " << pos.transpose() << std::endl;
      if (fr_idx == 0) {
        // torso values
        EXPECT_GE(pos(0), -0.15);
        EXPECT_LE(pos(0), 0.45);
        EXPECT_GE(pos(1), -0.1);
        EXPECT_LE(pos(1), 0.1);
        EXPECT_GE(pos(2), 0.5);
        EXPECT_LE(pos(2), 0.7);
      } else if (fr_idx == 1) {
        // LF values
        EXPECT_GE(pos(0), -0.05);
        EXPECT_LE(pos(0), 0.5);
        EXPECT_GE(pos(1), 0.);
        EXPECT_LE(pos(1), 0.3);
        EXPECT_GE(pos(2), 0.);
        EXPECT_LE(pos(2), 0.5);
      } else if (fr_idx == 2) {
        // RF values
        EXPECT_GE(pos(0), -0.05);
        EXPECT_LE(pos(0), 0.5);
        EXPECT_GE(pos(1), -0.3);
        EXPECT_LE(pos(1), 0.);
        EXPECT_GE(pos(2), 0.);
        EXPECT_LE(pos(2), 0.5);
      } else if (fr_idx == 3) {
        // LKnee values
        EXPECT_GE(pos(0), 0.);
        EXPECT_LE(pos(0), 0.65);
        EXPECT_GE(pos(1), 0.);
        EXPECT_LE(pos(1), 0.2);
        EXPECT_GE(pos(2), 0.1);
        EXPECT_LE(pos(2), 0.8);
      } else if (fr_idx == 4) {
        // RKnee values
        EXPECT_GE(pos(0), 0.);
        EXPECT_LE(pos(0), 0.65);
        EXPECT_GE(pos(1), -0.2);
        EXPECT_LE(pos(1), 0.);
        EXPECT_GE(pos(2), 0.1);
        EXPECT_LE(pos(2), 0.8);
      } else {
        // hands values
        EXPECT_GE(pos(0), 0.);
        EXPECT_LE(pos(0), 0.8);
        EXPECT_GE(pos(1), -0.4);
        EXPECT_LE(pos(1), 0.4);
        EXPECT_GE(pos(2), 0.3);
        EXPECT_LE(pos(2), 1.2);
      }
    }
  }
}



int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
