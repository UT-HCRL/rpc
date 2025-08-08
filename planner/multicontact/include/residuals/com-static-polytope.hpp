#ifndef CROCODDYL_RESIDUALS_STATIC_POLYTOPE_HPP_
#define CROCODDYL_RESIDUALS_STATIC_POLYTOPE_HPP_

#include "crocoddyl/core/residual-base.hpp"
#include "crocoddyl/multibody/data/multibody.hpp"
#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"

namespace crocoddyl {

template <typename _Scalar>
struct ResidualDataStaticPolytopeTpl : public ResidualDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualDataAbstractTpl<Scalar> Base;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::Matrix3xs Matrix3xs;

  template <template <typename Scalar> class Model>
  ResidualDataStaticPolytopeTpl(Model<Scalar>* const model,
                             DataCollectorAbstract* const data)
      : Base(model, data) {
    // Check that proper shared data has been passed
    DataCollectorMultibodyTpl<Scalar>* d =
        dynamic_cast<DataCollectorMultibodyTpl<Scalar>*>(shared);
    if (d == NULL) {
      throw_pretty(
          "Invalid argument: the shared data should be derived from "
          "DataCollectorMultibody");
    }

    // Avoids data casting at runtime
    pinocchio = d->pinocchio;
  }
  virtual ~ResidualDataStaticPolytopeTpl() = default;

  pinocchio::DataTpl<Scalar>* pinocchio;  //!< Pinocchio data
  using Base::r;
  using Base::Ru;
  using Base::Rx;
  using Base::shared;
};


template <typename Scalar>
class ResidualModelStaticPolytopeTpl : public ResidualModelAbstractTpl<Scalar> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
  using Base = ResidualModelAbstractTpl<Scalar>;
  using Data = ResidualDataStaticPolytopeTpl<Scalar>;
  using ResidualDataAbstract = ResidualDataAbstractTpl<Scalar>;
  using DataCollectorAbstract = DataCollectorAbstractTpl<Scalar>;
  using StateMultibody = StateMultibodyTpl<Scalar>;
  using DataCollectorMultibody = DataCollectorMultibodyTpl<Scalar>;
  using VectorXs = typename Base::VectorXs;
  using MatrixXs = typename Base::MatrixXs;

  ResidualModelStaticPolytopeTpl(std::shared_ptr<StateMultibody> state,
                                 const MatrixXs& A,
                                 const VectorXs& b,
                                 std::size_t nu = 0)
      : Base(state, A.rows(), nu, true, false, false), A_(A), b_(b) {
    if (A.cols() != 2 || A.rows() != b.rows()) {
      throw std::invalid_argument("A must be Nx2 and b must match A.rows()");
    }
  }

  virtual ~ResidualModelStaticPolytopeTpl() = default;

  virtual void calc(const std::shared_ptr<ResidualDataAbstract>& data,
            const Eigen::Ref<const VectorXs>& x,
            const Eigen::Ref<const VectorXs>& /*u*/) override {
    
    Data* d = static_cast<Data*>(data.get());

    data->r.noalias() = A_ * d->pinocchio->com[0].template head<2>() - b_;
  }

  virtual void calcDiff(const std::shared_ptr<ResidualDataAbstract>& data,
                const Eigen::Ref<const VectorXs>& x,
                const Eigen::Ref<const VectorXs>& /*u*/) override {
    Data* d = static_cast<Data*>(data.get());

    const auto& Jcom = d->pinocchio->Jcom.topRows(2);

    data->Rx.setZero(this->nr_, this->state_->get_ndx());
    data->Rx.leftCols(this->state_->get_nv()).noalias() = A_ * Jcom;

    if (this->nu_ > 0) {
      data->Ru.setZero(this->nr_, this->nu_);
    }
  }

  virtual std::shared_ptr<ResidualDataAbstract> createData(DataCollectorAbstractTpl<Scalar>* const data) override {
    return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
  }

  void setA(const MatrixXs& A) {
    if (A.cols() != 2 || A.rows() != b_.rows()) {
      throw std::invalid_argument("setA: A must be Nx2 and match b");
    }
    A_ = A;
  }

  void setB(const VectorXs& b) {
    if (b.size() != A_.rows()) {
      throw std::invalid_argument("setB: b must match number of rows of A");
    }
    b_ = b;
  }

  void setPolytope(const MatrixXs& A, const VectorXs& b) {
      if (A.cols() != 2) {
          std::cout << "setPolytope: A must have 2 columns (xy projection)";
          throw std::invalid_argument("setPolytope: A must have 2 columns (xy projection)");
      }
      if (A.rows() != b.size()) {
          std::cout << "setPolytope: A.rows must match b.size()";
          throw std::invalid_argument("setPolytope: A.rows must match b.size()");
      }

      A_.resize(A.rows(), A.cols());
      b_.resize(b.size());

      A_ = A;
      b_ = b;
  }

  const MatrixXs& getA() const { return A_; }
  const VectorXs& getB() const { return b_; }

protected:
  MatrixXs A_;
  VectorXs b_;
};

using ResidualModelStaticPolytope = ResidualModelStaticPolytopeTpl<double>;
using ResidualDataStaticPolytope = ResidualDataStaticPolytopeTpl<double>;

}  // namespace crocoddyl

#endif  // CROCODDYL_RESIDUALS_STATIC_POLYTOPE_HPP_
