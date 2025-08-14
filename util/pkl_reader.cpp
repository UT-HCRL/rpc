#include "pkl_reader.hpp"

#include <iostream>
#include <fstream>
#include <pybind11/numpy.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace pkl_utils {

class PickleReader::Impl {
public:
    explicit Impl(const std::string& filePath = "", const PickleType& pkl_type = PickleType::BEZIER, const std::vector<std::string>& frame_names = {})
        : guard_(), filename_(filePath), is_ready_(false), pkl_type_(pkl_type), frame_names_(frame_names) 
    {
        try {
            py::module_ sys = py::module_::import("sys");
            sys.attr("path").attr("insert")(0, "../");
            py::exec(R"(
                        import pickle
                        import util.path_parameterization
                        )");
            py::object pickle = py::module_::import("pickle");
            py::object open_func = py::module_::import("builtins").attr("open");
            py::object file = open_func(filePath, "rb");

            if (pkl_type == PickleType::COMPOSITE){
                bool once = true;
                py::list loaded_data;
                while (true) {
                    try {
                        py::object obj = pickle.attr("load")(file);
                        if (py::isinstance<py::dict>(obj) && once) {
                            bez_data_ = obj["bez_path"];
                            once = false;
                        }
                        loaded_data.append(obj);

                    } catch (py::error_already_set& e) {
                        if (e.matches(PyExc_EOFError)) {
                            break;
                        } else if (e.matches(PyExc_ModuleNotFoundError)) {
                            std::cerr << "Missing module during unpickling: " << e.what() << std::endl;
                            break;
                        } else {
                            break;
                        }
                    }
                }
                data_ = loaded_data;
            }
            else if(pkl_type == PickleType::LIST) {
                py::list loaded_data;
                while (true) {
                    try {
                        py::object obj = pickle.attr("load")(file);
                        loaded_data.append(obj);

                    } catch (py::error_already_set& e) {
                        if (e.matches(PyExc_EOFError)) {
                            break;
                        } else if (e.matches(PyExc_ModuleNotFoundError)) {
                            std::cerr << "Missing module during unpickling: " << e.what() << std::endl;
                            break;
                        } else {
                            break;
                        }
                    }
                }
                data_ = loaded_data;
                
            } else if(pkl_type == PickleType::DICT) {
                data_ = pickle.attr("load")(file);
            } else if(pkl_type == PickleType::BEZIER) {
                data_ = pickle.attr("load")(file);
                if (py::isinstance<py::dict>(data_)) {
                    bez_data_ = data_["bez_path"];
                } else {
                    std::cerr << "Unpickled object is not a Bezier Curve\n";
                }
            } else {
                std::cerr << "Unknown pickle type!" << std::endl;
            }

            file.attr("close")();
            is_ready_ = true;

        } catch (const std::exception& e) {
            std::cerr << "[PKL_READER] - Error loading pickle file: " << e.what() << std::endl;
        }
    }

    bool isReady() const {
        return is_ready_;
    }

    void parse(){
        if (!isReady()) {
            std::cerr << "File is not open!" << std::endl;
            return;
        }

        try {
            if(pkl_type_ == PickleType::COMPOSITE){
                parseBezier();
                parseList();
            }
            else if (py::isinstance<py::list>(data_) && pkl_type_ == PickleType::LIST) {
                parseList();
            } else if (py::isinstance<py::dict>(data_) && pkl_type_ == PickleType::DICT) {
                parseDict();
            } else if (py::isinstance<py::list>(bez_data_) && pkl_type_ == PickleType::BEZIER) {
                parseBezier();
            } 
            else {
                std::cerr << "[PKL_READER] - Loaded data is neither a list nor a dictionary!" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
       
    }

    void parseList(){

        if (!isReady()) {
            std::cerr << "File is not open!" << std::endl;
            return;
        }

        try {
            if (py::isinstance<py::list>(data_)) {
                py::list obj_list = data_;
                
                for(size_t i=0; i<py::len(obj_list); i++){
                    py::object obj = obj_list[i];

                    if (py::isinstance<py::dict>(obj)) {
                        py::dict robot_data = obj;
                        for (auto item : robot_data) {
                            std::string key = py::str(item.first);
                            py::object value = py::reinterpret_steal<py::object>(item.second);

                            if (py::isinstance<py::float_>(value)) {
                            } else if (py::isinstance<py::list>(value)) {
                                Vector<double, 34> tmp_pos_vec;
                                Vector<double, 33> tmp_vel_vec;
                                Vector<double, 27> tmp_tau_vec;
                                Vector3d tmp_com_vec;
                                size_t v_idx = 0;
                                py::list pylist = value;

                                for (auto elem : pylist) {
                                    if (py::isinstance<py::float_>(elem)) {
                                        if ( key == "center_of_mass"){
                                            tmp_com_vec(v_idx) = elem.cast<double>();
                                            v_idx++;
                                            if (v_idx == 3){
                                                com_des_.emplace_back(tmp_com_vec);
                                                v_idx = 0;
                                            }
                                        }
                                        else if (key == "time") {
                                            time_vec_.emplace_back(elem.cast<double>());
                                        }
                                    } else {
                                        size_t vec_idx = 0;
                                        if (key == "joint_pos"){
                                            if (py::len(elem) != 34) {
                                                std::cerr << "[PKL_READER] - Unexpected size for joint_pos element: " << py::len(elem) << std::endl;
                                            }
                                            for (auto el : elem) {
                                                tmp_pos_vec(vec_idx) = el.cast<double>();
                                                vec_idx++;
                                            }
                                            // store current vector and clear for next iteration
                                            joint_pos_des_.emplace_back(tmp_pos_vec);
                                            tmp_pos_vec.setZero();
                                        } else if (key == "joint_vel") {
                                            if (py::len(elem) != 33) {
                                                std::cerr << "[PKL_READER] - Unexpected size for joint_vel element: " << py::len(elem) << std::endl;
                                            }
                                            for (auto el : elem) {
                                                tmp_vel_vec(vec_idx) = el.cast<double>();
                                                vec_idx++;
                                            }
                                            // store current vector and clear for next iteration
                                            joint_vel_des_.emplace_back(tmp_vel_vec);
                                            tmp_vel_vec.setZero();
                                        } else if (key == "joint_torque") {
                                            if (py::len(elem) != 27) {
                                                std::cerr << "[PKL_READER] - Unexpected size for joint_torque element: " << py::len(elem) << std::endl;
                                            }
                                            for (auto el : elem) {
                                                tmp_tau_vec(vec_idx) = el.cast<double>();
                                                vec_idx++;
                                            }
                                            // store current vector and clear for next iteration
                                            joint_tau_des_.emplace_back(tmp_tau_vec);
                                            tmp_tau_vec.setZero();
                                        } else if (key == "grf_lfoot") {
                                        } else if (key == "grf_rfoot") {
                                        } else if (key == "grf_lhand") {
                                        } else if (key == "grf_rhand") {
                                        } else if (std::find(frame_names_.begin(), frame_names_.end(), key) != frame_names_.end()){
                                            Vector3d frame_vec;
                                            size_t vec_idx = 0;
                                            py::list coord_list = elem.cast<py::list>();
                                            if (py::len(coord_list) != 3) {
                                                std::cerr << "[PKL_READER] - Unexpected size for " << key << ": " << py::len(coord_list) << std::endl;
                                            }
                                            for (auto coord : coord_list) {
                                                frame_vec(vec_idx++) = coord.cast<double>();
                                            }
                                            frames_des_[key].emplace_back(frame_vec);
                                        } else if (key == "bez_path"){
                                            //pass;
                                        } else if (key == "center_of_mass"){
                                            for (auto el : elem) {
                                                tmp_com_vec(vec_idx) = el.cast<double>();
                                                vec_idx++;
                                            }
                                            com_des_.emplace_back(tmp_com_vec);
                                            tmp_com_vec.setZero();
                                        
                                        }else {
                                            std::cerr << "[PKL_READER] - Unknown format for variable: " << key << std::endl;
                                        }
                                    }
                                }
                            } else if (py::isinstance<py::int_>(value)){
                                // std::cout << value.cast<int>() << std::endl;
                            } else {
                                // std::cout << py::str(value) << " (unhandled type)\n";
                            }
                        }
                    } else {
                        std::cerr << "[PKL_READER] - Object " << i << " is not a dictionary!" << std::endl;
                    }
                }

            }
        } catch (const std::exception& e) {
            std::cerr << "[PKL_READER] -Error: " << e.what() << std::endl;
        }
    }

    void parseDict(){

         try {
            if (py::isinstance<py::dict>(data_)) {
                py::dict robot_data = data_;
                
                // std::cout << "[PKL_READER] - Parsed dictionary:\n";
                for (auto item : robot_data) {
                    std::string key = py::str(item.first);
                    py::object value = py::reinterpret_steal<py::object>(item.second);
                    // std::cout << "[PKL_READER] - " << key << ": ";

                    if (py::isinstance<py::float_>(value)) {
                        // std::cout << value.cast<float>() << std::endl;
                    } else if (py::isinstance<py::list>(value)) {
                        py::list pylist = value;
                        // std::cout << "[ ";
                        for (auto elem : pylist) {
                            if (py::isinstance<py::float_>(elem)) {
                                // std::cout << elem.cast<float>() << " ";
                            } else {
                                // std::cout << py::str(elem) << " ";
                            }
                        }
                        // std::cout << "]\n";
                    } else {
                        // std::cout << py::str(value) << " (unhandled type)\n";
                    }
                }

            } else {
                std::cerr << "[PKL_READER] - Loaded data is not a dictionary!" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }

    }

    void parseBezier(){
        int temp=0;

        py::list composite_bez_list = bez_data_;
        composite_bezier_curves_.reserve(py::len(composite_bez_list));

        for (size_t i = 0; i < composite_bez_list.size(); ++i) {
            py::object obj = composite_bez_list[i];

            // std::cout << "[PKL_READER] - Object " << i << ": "
            //         << std::string(py::str(obj.get_type())) << std::endl;

            // for all the beziers inside the object:
            std::vector<BezierCurve> beziers;
            beziers.reserve(py::len(obj.attr("beziers")));

            for(size_t j = 0; j < py::len(obj.attr("beziers")); j++) {
                py::object bezier = obj.attr("beziers")[py::int_(j)];
                // std::cout << "[PKL_READER] - Bezier " << j << ": " << std::string(py::str(bezier)) << std::endl;
                py::object points = bezier.attr("points");
                // std::cout << "points type: " << std::string(py::str(points.get_type())) << std::endl; <- sono numpy array!

                Matrix<double, 8, 3> points_mat;
                points_mat.setZero();
                points_mat = numpy2eigen(points);
                // for(size_t k = 0; k < points_vec.size(); k++) {
                //     std::cout << "Point " << k << ": " << points_vec[k].transpose() << std::endl;
                // }
                
                py::object h = bezier.attr("h");
                py::object d = bezier.attr("d");
                py::object a = bezier.attr("a");
                py::object b = bezier.attr("b");
                py::object duration = bezier.attr("duration");

                double a_bez = bezier.attr("a").cast<double>();
                double b_bez = bezier.attr("b").cast<double>();
                BezierCurve bezier_curve(points_mat, a_bez, b_bez);

                beziers.emplace_back(bezier_curve);
            
            }

            composite_bezier_curves_.emplace_back(CompositeBezierCurve(beziers));

        }
    }

    const PickleType& getPickleType() const{
        return pkl_type_;
    }

    Matrix<double, 8, 3> numpy2eigen(py::array_t<double> input_array) {
        py::buffer_info buf = input_array.request();
        if (buf.ndim != 2 || buf.shape[1] != 3)
            throw std::runtime_error("Expected shape (N, 3)");

        auto ptr = static_cast<double*>(buf.ptr);
        Matrix<double, 8, 3> result;

        for (ssize_t i = 0; i < buf.shape[1]; ++i) {
            for (ssize_t j = 0; j < buf.shape[0]; ++j) {
                result(j, i) = ptr[i * buf.shape[0] + j];
            }
        }
        return result;
    }
    
    std::vector<CompositeBezierCurve> getCompositeBezierCurves() const {
        return composite_bezier_curves_;
    }

    std::vector<Matrix<double, 34, 1>> getJointPosDes() const {
        return joint_pos_des_;
    }

    std::vector<Matrix<double, 33, 1>> getJointVelDes() const {
        return joint_vel_des_;
    }

    std::vector<Matrix<double, 27, 1>> getJointTauDes() const {
        return joint_tau_des_;
    }

    std::vector<Vector3d> getCoM() const {
        return com_des_;
    }

    std::vector<double> getTimeVec() const {
        return time_vec_;
    }

    std::unordered_map<std::string, std::vector<Vector3d>> getFramesDes() const {
        return frames_des_;
    }

private:
    py::scoped_interpreter guard_;  
    std::string filename_;
    bool is_ready_;
    py::object data_;
    py::object bez_data_;
    PickleType pkl_type_;

    std::vector<std::string> frame_names_;
    std::unordered_map<std::string, std::vector<Vector3d>> frames_des_;

    std::vector<CompositeBezierCurve> composite_bezier_curves_;
    std::vector<Matrix<double, 34, 1>> joint_pos_des_;
    std::vector<Matrix<double, 33, 1>> joint_vel_des_;
    std::vector<Matrix<double, 27, 1>> joint_tau_des_;
    std::vector<Vector3d> com_des_;
    std::vector<double> time_vec_;
    std::function<void(const std::string&)> log_;

};

PickleReader::PickleReader(const std::string& filePath, const PickleType& pkl_type, const std::vector<std::string>& frame_names)
    : impl_(std::make_unique<Impl>(filePath, pkl_type, frame_names)) {}

PickleReader::~PickleReader() = default;

bool PickleReader::isReady() const {
    return impl_->isReady();
}

void PickleReader::parse() {
    impl_->parse();
}

const PickleType& PickleReader::getPickleType() const {
    return impl_->getPickleType();
}

std::vector<CompositeBezierCurve> PickleReader::getCompositeBezierCurves() const {
    return impl_->getCompositeBezierCurves();
}

std::vector<Matrix<double, 34, 1>> PickleReader::getJointPosDes() const {
    return impl_->getJointPosDes();
}

std::vector<Matrix<double, 33, 1>> PickleReader::getJointVelDes() const {
    return impl_->getJointVelDes();
}

std::vector<Matrix<double, 27, 1>> PickleReader::getJointTauDes() const {
    return impl_->getJointTauDes();
}

std::vector<double> PickleReader::getTimeVec() const {
    return impl_->getTimeVec();
}

std::vector<Vector3d> PickleReader::getCoM() const {
    return impl_->getCoM();
}

std::unordered_map<std::string, std::vector<Eigen::Vector3d>> PickleReader::getFramesDes() const {
    return impl_->getFramesDes();
}

} // namespace pkl_utils
