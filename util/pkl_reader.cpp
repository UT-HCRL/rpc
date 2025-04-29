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
    explicit Impl(const std::string& filePath = "", const PickleType& pkl_type = PickleType::BEZIER)
        : guard_(), filename_(filePath), is_ready_(false), pkl_type_(pkl_type) 
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

            // std::cout << "[PKL_READER] - Pickle type is: " << (pkl_type == PickleType::DICT ? "DICT" : (pkl_type == PickleType::LIST ? "LIST" : "BEZIER")) << std::endl;
            
            if(pkl_type == PickleType::LIST) {
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
                    data_ = data_["bez_path"];
                    // std::cout << "[PKL_READER] - Imported bezier path\n";
                } else {
                    std::cerr << "Unpickled object is not a Bezier Curve\n";
                }
            } else {
                std::cerr << "Unknown pickle type!" << std::endl;
            }

            file.attr("close")();
            
            is_ready_ = true;
            // std::cout << "[PKL_READER] - File opened successfully: " << filePath << std::endl;

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
            if (py::isinstance<py::list>(data_) && pkl_type_ == PickleType::LIST) {
                // std::cout << "[PKL_READER] - Parsing list..." << std::endl;
                parseList();
            } else if (py::isinstance<py::dict>(data_) && pkl_type_ == PickleType::DICT) {
                // std::cout << "[PKL_READER] - Parsing dictionary..." << std::endl;
                parseDict();
            } else if (py::isinstance<py::list>(data_) && pkl_type_ == PickleType::BEZIER) {
                // std::cout << "[PKL_READER] - Parsing Bezier curve..." << std::endl;
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
                // std::cout << "[PKL_READER] - Loaded " << py::len(obj_list) << " object(s) from pickle file.\n";
                
                for(size_t i=0; i<py::len(obj_list); i++){
                    py::object obj = obj_list[i];

                    if (py::isinstance<py::dict>(obj)) {
                        py::dict robot_data = obj;
                        for (auto item : robot_data) {
                            std::string key = py::str(item.first);
                            py::object value = py::reinterpret_steal<py::object>(item.second);
                            // std::cout <<"[PKL_READER] - " <<key << ": ";

                            if (py::isinstance<py::float_>(value)) {
                            } else if (py::isinstance<py::list>(value)) {
                                Vector<double, 44> tmp_pos_vec;
                                Vector<double, 43> tmp_vel_vec;
                                Vector<double, 37> tmp_tau_vec;
                                py::list pylist = value;
                                // std::cout << "[ ";
                                for (auto elem : pylist) {
                                    if (py::isinstance<py::float_>(elem)) {
                                        if (key == "time") {
                                            time_vec_.emplace_back(elem.cast<double>());
                                            // std::cout << " LOGGED " << std::endl;
                                        }
                                    } else {
                                        size_t vec_idx = 0;
                                        if (key == "joint_pos"){
                                            for (auto el : elem) {
                                                tmp_pos_vec(vec_idx) = el.cast<double>();
                                                vec_idx++;
                                                // std::cout << " LOGGED " << std::endl;
                                            }
                                            // store current vector and clear for next iteration
                                            joint_pos_des_.emplace_back(tmp_pos_vec);
                                            tmp_pos_vec.setZero();
                                        } else if (key == "joint_vel") {
                                            for (auto el : elem) {
                                                tmp_vel_vec(vec_idx) = el.cast<double>();
                                                vec_idx++;
                                                // std::cout << " LOGGED " << std::endl;
                                            }
                                            // store current vector and clear for next iteration
                                            joint_vel_des_.emplace_back(tmp_vel_vec);
                                            tmp_vel_vec.setZero();
                                        } else if (key == "joint_torque") {
                                            for (auto el : elem) {
                                                tmp_tau_vec(vec_idx) = el.cast<double>();
                                                vec_idx++;
                                                // std::cout << " LOGGED " << std::endl;
                                            }
                                            // store current vector and clear for next iteration
                                            joint_tau_des_.emplace_back(tmp_tau_vec);
                                            tmp_tau_vec.setZero();
                                        } else if (key == "grf_lfoot") {
                                        } else if (key == "grf_rfoot") {
                                        } else if (key == "grf_lhand") {
                                        } else if (key == "grf_rhand") {
                                        } else {
                                            std::cerr << "[PKL_READER] - Unknown format for variable: " << key << std::endl;
                                        }
                                    }
                                }
                                // std::cout << "]\n";
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

        py::list composite_bez_list = data_;
        composite_bezier_curves_.reserve(py::len(composite_bez_list));

        for (size_t i = 0; i < composite_bez_list.size(); ++i) {
            py::object obj = composite_bez_list[i];

            std::cout << "[PKL_READER] - Object " << i << ": "
                    << std::string(py::str(obj.get_type())) << std::endl;

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

    std::vector<Matrix<double, 44, 1>> getJointPosDes() const {
        return joint_pos_des_;
    }

    std::vector<Matrix<double, 43, 1>> getJointVelDes() const {
        return joint_vel_des_;
    }

    std::vector<Matrix<double, 37, 1>> getJointTauDes() const {
        return joint_tau_des_;
    }

    std::vector<double> getTimeVec() const {
        return time_vec_;
    }

private:
    py::scoped_interpreter guard_;  
    std::string filename_;
    bool is_ready_;
    py::object data_;
    PickleType pkl_type_;

    std::vector<CompositeBezierCurve> composite_bezier_curves_;
    std::vector<Matrix<double, 44, 1>> joint_pos_des_;
    std::vector<Matrix<double, 43, 1>> joint_vel_des_;
    std::vector<Matrix<double, 37, 1>> joint_tau_des_;
    std::vector<double> time_vec_;
    std::function<void(const std::string&)> log_;

};

PickleReader::PickleReader(const std::string& filePath, const PickleType& pkl_type)
    : impl_(std::make_unique<Impl>(filePath, pkl_type)) {}

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

std::vector<Matrix<double, 44, 1>> PickleReader::getJointPosDes() const {
    return impl_->getJointPosDes();
}

std::vector<Matrix<double, 43, 1>> PickleReader::getJointVelDes() const {
    return impl_->getJointVelDes();
}

std::vector<Matrix<double, 37, 1>> PickleReader::getJointTauDes() const {
    return impl_->getJointTauDes();
}

std::vector<double> PickleReader::getTimeVec() const {
    return impl_->getTimeVec();
}

} // namespace pkl_utils
