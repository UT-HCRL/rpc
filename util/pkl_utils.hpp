#pragma once

#include <Eigen/Dense>
using namespace Eigen;

namespace pkl_utils {

    enum class PickleType {
        DICT,
        LIST,
        BEZIER
    };

    class BezierCurve {
    public:
        BezierCurve(const Matrix<double, 8, 3>& points, double a = 0, double b = 1)
            : points_(points), a_(a), b_(b), h_(points.rows() - 1), d_(3), duration_(b - a) {
            if (b <= a) {
                throw std::invalid_argument("b must be greater than a");
            }
        }

        const Vector3d eval(const double t) const {
            if (t < a_ || t > b_) {
                throw std::out_of_range("t is out of range");
            }
            Matrix<double, 8, 1> coeffs;
            coeffs.setZero();
            for (size_t i = 0; i <= h_; ++i) {
                coeffs(i) = _bernstein(t, i);
            }
            return coeffs.transpose() * points_;
        }

        const Matrix<double, 8, 3>& getPoints() const { return points_; }
        int getH() const { return h_; }
        int getD() const { return d_; }
        double getA() const { return a_; }
        double getB() const { return b_; }
        double getDuration() const { return duration_; }

    private:
        const double binomial_coefficient(int n, int k) const {
            if (k < 0 || k > n) {
                return 0;
            }
            if (k == 0 || k == n) {
                return 1;
            }
            if (k > n / 2) {
                k = n - k;
            }
            double result = 1;
            for (int i = 1; i <= k; ++i) {
                result = result * (n - i + 1) / i;
            }
            return result;
        }

        const double _bernstein(double t, int n) const {
            double c1 = binomial_coefficient(h_, n);
            double c2 = (t - a_) / duration_;
            double c3 = (b_ - t) / duration_;
            return c1 * std::pow(c2,n) * std::pow(c3, (h_ - n));
        }

        Matrix<double, 8, 3> points_;
        int h_;
        int d_;
        double a_;
        double b_;
        double duration_;
    };

    class CompositeBezierCurve {
    public:
        CompositeBezierCurve(const std::vector<BezierCurve>& beziers) : beziers_(beziers) {
            if (beziers.empty()) {
                throw std::invalid_argument("beziers cannot be empty");
            }

            for (size_t i = 0; i < beziers.size() - 1; ++i) {
                const auto& bez1 = beziers[i];
                const auto& bez2 = beziers[i + 1];
                if (bez1.getB() != bez2.getA()) {
                    throw std::invalid_argument("Consecutive Bezier curves must have matching endpoints");
                }
                if (bez1.getD() != bez2.getD()) {
                    throw std::invalid_argument("Consecutive Bezier curves must have the same dimension");
                }
            }

            N_ = beziers.size();
            d_ = beziers[0].getD();
            a_ = beziers[0].getA();
            b_ = beziers.back().getB();
            duration_ = b_ - a_;

            transition_times_.push_back(a_);
            for (const auto& bez : beziers) {
                transition_times_.push_back(bez.getB());
            }
            
            unsigned int i = 0;
            for(const auto&bez : beziers){
                beziers_[i] = BezierCurve(bez.getPoints(), bez.getA(), bez.getB());
                i++;
            }
        }

        const std::vector<BezierCurve>& getBeziers() const { return beziers_; }
        int getN() const { return N_; }
        int getD() const { return d_; }
        double getA() const { return a_; }
        double getB() const { return b_; }
        double getDuration() const { return duration_; }
        const std::vector<double>& getTransitionTimes() const { return transition_times_; }

    private:
        std::vector<BezierCurve> beziers_;
        int N_;
        int d_;
        double a_;
        double b_;
        double duration_;
        std::vector<double> transition_times_;
    };

    static const Vector3d get_frame_des_pos(const std::shared_ptr<CompositeBezierCurve> bezier, const double& t) {
        unsigned int seg = 0;
        for (const auto& bez : bezier->getBeziers()) {
            if (t >= bez.getA() && t <= bez.getB()) {
                break;
            }
            seg += 1;
        }
        const auto bez_seg = bezier->getBeziers().at(seg);
        return bez_seg.eval(t);
    }

    class BezierCurvesManager {
    public:
        BezierCurvesManager(const std::vector<std::shared_ptr<CompositeBezierCurve>> bezier_curves,
                            const std::vector<std::string>& frame_names) {
            if (bezier_curves.empty()) {
                throw std::invalid_argument("bezier_curves cannot be empty");
            }

            // check if frame_names.size() == bezier_curves.size()
            if (frame_names.size() != bezier_curves.size()) {
                throw std::invalid_argument("frame_names and bezier_curves must have the same size");
            }

            // store Bezier curves in corresponding frame name
            for (unsigned int idx = 0; idx < frame_names.size(); idx++) {
                bezier_curves_map_.emplace(frame_names[idx], bezier_curves.at(idx));
            }
        }

        ~BezierCurvesManager() = default;

        const Vector3d getCurrentDesiredPosition(const std::string& frame_name, const double& t) {
            if (bezier_curves_map_.find(frame_name) == bezier_curves_map_.end()) {
                throw std::invalid_argument("Frame name not found in Bezier curves map");
            }
            std::shared_ptr<CompositeBezierCurve> bezier_curve = bezier_curves_map_[frame_name];
            return get_frame_des_pos(bezier_curve, t);
        }

    private:
        std::unordered_map<std::string, std::shared_ptr<CompositeBezierCurve>> bezier_curves_map_;
    };


} // namespace pkl_utils