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
        BezierCurve(const std::vector<Eigen::Vector3d>& points, double a = 0, double b = 1)
            : points_(points), a_(a), b_(b), h_(points.size() - 1), d_(points.empty() ? 0 : points[0].size()), duration_(b - a) {
            if (b <= a) {
                throw std::invalid_argument("b must be greater than a");
            }
        }

        const std::vector<Eigen::Vector3d>& getPoints() const { return points_; }
        int getH() const { return h_; }
        int getD() const { return d_; }
        double getA() const { return a_; }
        double getB() const { return b_; }
        double getDuration() const { return duration_; }

    private:
        std::vector<Eigen::Vector3d> points_;
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
            
            beziers_.reserve(N_);
            for(const auto&bez : beziers){
                beziers_.emplace_back(bez.getPoints(), bez.getA(), bez.getB());
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

} // namespace pkl_utils