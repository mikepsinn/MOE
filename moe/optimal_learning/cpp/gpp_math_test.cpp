// gpp_math_test.cpp
/*
  Routines to test the functions in gpp_math.cpp.

  The tests verify GaussianProcess, ExpectedImprovementEvaluator (+OnePotentialSample), and EI optimization from gpp_math.cpp.
  1a) Following gpp_covariance_test.cpp, we define classes (PingGPMean + other GP ping, PingExpectedImprovement) for
     evaluating those functions + their spatial gradients.

     Some Pingable classes for GP functions are less general than their gpp_covariance_test or
     gpp_model_selection_and_hyperparameter_optimization_test counterparts, since GP derivative functions sometimes return sparse
     or incomplete data (e.g., gradient of mean returned as a vector instead of a diagonal matrix; gradient of variance
     only differentiates wrt a single point at a time); hence we need specialized handlers for testing.
  1b) Ping for derivative accuracy (PingGPComponentTest, PingEITest); these unit test the analytic derivatives.
  2) Monte-Carlo EI vs analytic EI validation: the monte-carlo versions are run to "high" accuracy and checked against
     analytic formulae when applicable
  3) Gradient Descent: using polynomials and other simple fucntions with analytically known optima
     to verify that the algorithm(s) underlying EI optimization are performing correctly.
  4) Single-threaded vs multi-threaded EI optimization validation: single and multi-threaded runs are checked to have the same
     output.
  5) End-to-end test of the EI optimization process for the analytic and monte-carlo cases.  These tests use constructed
     data for inputs but otherwise exercise the same code paths used for hyperparameter optimization in production.
*/

// #define OL_VERBOSE_PRINT

#include "gpp_math_test.hpp"

#include <cmath>
#include <cstdio>

#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <vector>

#include <boost/random/uniform_real.hpp>  // NOLINT(build/include_order)
#include <omp.h>  // NOLINT(build/include_order)

#include "gpp_common.hpp"
#include "gpp_covariance.hpp"
#include "gpp_domain.hpp"
#include "gpp_exception.hpp"
#include "gpp_geometry.hpp"
#include "gpp_linear_algebra_test.hpp"
#include "gpp_logging.hpp"
#include "gpp_math.hpp"
#include "gpp_mock_optimization_objective_functions.hpp"
#include "gpp_optimization.hpp"
#include "gpp_optimization_parameters.hpp"
#include "gpp_random.hpp"
#include "gpp_test_utils.hpp"

namespace optimal_learning {

namespace {  // contains classes/routines for ping testing GP/EI quantities and checking EI threaded consistency

/*
  Supports evaluating the GP mean, ComputeMeanOfPoints() and its gradient, ComputeGradMeanOfPoints.

  The gradient is taken wrt points_to_sample[dim][num_to_sample], so this is the "input_matrix" X_{d,i}.
  The other inputs to GP mean are not differentiated against, so they are taken as input and stored by the constructor.

  Also, ComputeGradMeanOfPoints() stores a compact version of the gradient (by skipping known 0s) that *does not* have size
  GetGradientsSize().  EvaluateAndStoreAnalyticGradient and GetAnalyticGradient account for this indexing scheme appropriately.
*/
class PingGPPMean final : public PingableMatrixInputVectorOutputInterface {
 public:
  constexpr static char const * const kName = "GP Mean";

  PingGPPMean(double const * restrict lengths, double const * restrict points_sampled, double const * restrict points_sampled_value, double alpha, int dim, int num_to_sample, int num_sampled) OL_NONNULL_POINTERS
      : dim_(dim),
        num_to_sample_(num_to_sample),
        num_sampled_(num_sampled),
        gradients_already_computed_(false),
        noise_variance_(num_sampled_, 0.0),
        points_sampled_(points_sampled, points_sampled + dim_*num_sampled_),
        points_sampled_value_(points_sampled_value, points_sampled_value + num_sampled_),
        grad_mu_(num_to_sample_*dim_),
        sqexp_covariance_(dim_, alpha, lengths),
        gaussian_process_(sqexp_covariance_, points_sampled_.data(), points_sampled_value_.data(), noise_variance_.data(), dim_, num_sampled_) {
  }

  virtual void GetInputSizes(int * num_rows, int * num_cols) const noexcept override OL_NONNULL_POINTERS {
    *num_rows = dim_;
    *num_cols = num_to_sample_;
  }

  virtual int GetGradientsSize() const noexcept override OL_WARN_UNUSED_RESULT {
    return dim_*num_to_sample_*GetOutputSize();
  }

  virtual int GetOutputSize() const noexcept override OL_WARN_UNUSED_RESULT {
    return num_to_sample_;
  }

  virtual void EvaluateAndStoreAnalyticGradient(double const * restrict points_to_sample, double * restrict gradients) noexcept override OL_NONNULL_POINTERS_LIST(2) {
    if (gradients_already_computed_ == true) {
      OL_WARNING_PRINTF("WARNING: grad_mu alrady set.  Overwriting...\n");
    }
    gradients_already_computed_ = true;

    GaussianProcess::StateType points_to_sample_state(gaussian_process_, points_to_sample, num_to_sample_, true);
    gaussian_process_.ComputeGradMeanOfPoints(points_to_sample_state, grad_mu_.data());

    if (gradients != nullptr) {
      // Since ComputeGradMeanOfPoints does not store known zeros in the gradient, we need to resconstruct the more general
      // tensor structure (including all zeros). This more general tensor is "block" diagonal.

      std::fill(gradients, gradients + dim_*Square(num_to_sample_), 0.0);

      // Loop over just the block diagonal entries and copy over the computed gradients.
      for (int i = 0; i < num_to_sample_; ++i) {
        for (int d = 0; d < dim_; ++d) {
          gradients[i*dim_*num_to_sample_ + i*dim_ + d] = grad_mu_[i*dim_ + d];
        }
      }
    }
  }

  virtual double GetAnalyticGradient(int row_index, int column_index, int output_index) const OL_WARN_UNUSED_RESULT {
    if (gradients_already_computed_ == false) {
      OL_THROW_EXCEPTION(RuntimeException, "PingGPPMean::GetAnalyticGradient() called BEFORE EvaluateAndStoreAnalyticGradient. NO DATA!");
    }

    if (column_index == output_index) {
      return grad_mu_[column_index*dim_ + row_index];
    } else {
      // these entries are analytically known to be 0.0 and thus were not stored
      // in grad_mu_
      return 0.0;
    }
  }

  virtual void EvaluateFunction(double const * restrict points_to_sample, double * restrict function_values) const noexcept override OL_NONNULL_POINTERS {
    GaussianProcess::StateType points_to_sample_state(gaussian_process_, points_to_sample, num_to_sample_, false);
    gaussian_process_.ComputeMeanOfPoints(points_to_sample_state, function_values);
  }

 private:
  int dim_;
  int num_to_sample_;
  int num_sampled_;
  bool gradients_already_computed_;

  std::vector<double> noise_variance_;
  std::vector<double> points_sampled_;
  std::vector<double> points_sampled_value_;
  std::vector<double> grad_mu_;

  SquareExponential sqexp_covariance_;
  GaussianProcess gaussian_process_;

  OL_DISALLOW_DEFAULT_AND_COPY_AND_ASSIGN(PingGPPMean);
};

/*
  Supports evaluating the GP variance, ComputeVarianceOfPoints() and its gradient, ComputeGradVarianceOfPoints.

  The gradient is taken wrt points_to_sample[dim][num_to_sample], so this is the "input_matrix" X_{d,i}.
  The other inputs to GP variance are not differentiated against, so they are taken as input and stored by the constructor.

  The output is a matrix of dimension num_to_sample.  To fit into the PingMatrix...Interface, this is treated as a vector
  of length num_to_sample^2.

  Also, ComputeGradVarianceOfPoints() takes col_index (i.e., 0 .. num_to_sample-1) as input.  Thus it must be called
  num_to_sample times to get all GetGradientsSize() entries of the gradient.

  EvaluateAndStoreAnalyticGradient and GetAnalyticGradient deal with grad_var's API appropriately; all num_to_sample
  calls to ComputeGradVarianceOfPoints() are saved off.
*/
class PingGPPVariance final : public PingableMatrixInputVectorOutputInterface {
 public:
  constexpr static char const * const kName = "GP Variance";

  PingGPPVariance(double const * restrict lengths, double const * restrict points_sampled, double const * restrict OL_UNUSED(points_sampled_value), double alpha, int dim, int num_to_sample, int num_sampled) OL_NONNULL_POINTERS
      : dim_(dim),
        num_to_sample_(num_to_sample),
        num_sampled_(num_sampled),
        gradients_already_computed_(false),
        noise_variance_(num_sampled_, 0.0),
        points_sampled_(points_sampled, points_sampled + dim_*num_sampled_),
        grad_variance_(num_to_sample_),
        sqexp_covariance_(dim_, alpha, lengths),
        gaussian_process_(sqexp_covariance_, points_sampled_.data(), std::vector<double>(num_sampled_, 0.0).data(), noise_variance_.data(), dim_, num_sampled_) {
  }

  virtual void GetInputSizes(int * num_rows, int * num_cols) const noexcept override OL_NONNULL_POINTERS {
    *num_rows = dim_;
    *num_cols = num_to_sample_;
  }

  virtual int GetGradientsSize() const noexcept override OL_WARN_UNUSED_RESULT {
    return dim_*num_to_sample_*GetOutputSize();
  }

  virtual int GetOutputSize() const noexcept override OL_WARN_UNUSED_RESULT {
    return Square(num_to_sample_);
  }

  virtual void EvaluateAndStoreAnalyticGradient(double const * restrict points_to_sample, double * restrict gradients) noexcept override OL_NONNULL_POINTERS_LIST(2) {
    if (gradients_already_computed_ == true) {
      OL_WARNING_PRINTF("WARNING: grad_variance data already set.  Overwriting...\n");
    }
    gradients_already_computed_ = true;

    for (int i = 0; i < num_to_sample_; ++i) {
      grad_variance_[i].resize(dim_*Square(num_to_sample_));

      GaussianProcess::StateType points_to_sample_state(gaussian_process_, points_to_sample, num_to_sample_, true);
      gaussian_process_.ComputeGradVarianceOfPoints(&points_to_sample_state, i, grad_variance_[i].data());
    }

    if (gradients != nullptr) {
      OL_THROW_EXCEPTION(RuntimeException, "PingGPPVariance::EvaluateAndStoreAnalyticGradient() does not support direct gradient output.");
    }
  }

  virtual double GetAnalyticGradient(int row_index, int column_index, int output_index) const override OL_WARN_UNUSED_RESULT {
    if (gradients_already_computed_ == false) {
      OL_THROW_EXCEPTION(RuntimeException, "PingGPPVariance::GetAnalyticGradient() called BEFORE EvaluateAndStoreAnalyticGradient. NO DATA!");
    }

    return grad_variance_[column_index][output_index*dim_ + row_index];
  }

  virtual void EvaluateFunction(double const * restrict points_to_sample, double * restrict function_values) const noexcept override OL_NONNULL_POINTERS {
    GaussianProcess::StateType points_to_sample_state(gaussian_process_, points_to_sample, num_to_sample_, false);
    gaussian_process_.ComputeVarianceOfPoints(&points_to_sample_state, function_values);

    // var_of_points outputs only to the lower triangle.  Copy it into the upper triangle to get a symmetric matrix
    for (int i = 0; i < num_to_sample_; ++i) {
      for (int j = 0; j < i; ++j) {
        function_values[i*num_to_sample_ + j] = function_values[j*num_to_sample_ + i];
      }
    }
  }

 private:
  int dim_;
  int num_to_sample_;
  int num_sampled_;
  bool gradients_already_computed_;

  std::vector<double> noise_variance_;
  std::vector<double> points_sampled_;
  std::vector<std::vector<double> > grad_variance_;

  SquareExponential sqexp_covariance_;
  GaussianProcess gaussian_process_;

  OL_DISALLOW_DEFAULT_AND_COPY_AND_ASSIGN(PingGPPVariance);
};

/*
  Supports evaluating the cholesky factorization of the GP variance, the transpose of the cholesky factorization of: ComputeVarianceOfPoints()
  and its gradient, ComputeGradCholeskyVarianceOfPoints.

  The gradient is taken wrt points_to_sample[dim][num_to_sample], so this is the "input_matrix" X_{d,i}.
  The other inputs to GP variance are not differentiated against, so they are taken as input and stored by the constructor.

  The output is a matrix of dimension num_to_sample.  To fit into the PingMatrix...Interface, this is treated as a vector
  of length num_to_sample^2.

  Also, ComputeGradCholeskyVarianceOfPoints() takes col_index (i.e., 0 .. num_to_sample-1) as input.  Thus it must be called
  num_to_sample times to get all GetGradientsSize() entries of the gradient.

  EvaluateAndStoreAnalyticGradient and GetAnalyticGradient deal with grad_var's API appropriately; all num_to_sample
  calls to ComputeGradCholeskyVarianceOfPoints() are saved off.

  WARNING: this class is NOT THREAD SAFE.
*/
class PingGPPCholeskyVariance final : public PingableMatrixInputVectorOutputInterface {
 public:
  constexpr static char const * const kName = "GP Cholesky Variance";

  PingGPPCholeskyVariance(double const * restrict lengths, double const * restrict points_sampled, double const * restrict OL_UNUSED(points_sampled_value), double alpha, int dim, int num_to_sample, int num_sampled) OL_NONNULL_POINTERS
      : dim_(dim),
        num_to_sample_(num_to_sample),
        num_sampled_(num_sampled),
        gradients_already_computed_(false),
        noise_variance_(num_sampled_, 0.0),
        points_sampled_(points_sampled, points_sampled + dim_*num_sampled_),
        chol_temp_(Square(num_to_sample_)),
        grad_variance_(num_to_sample_),
        sqexp_covariance_(dim_, alpha, lengths),
        gaussian_process_(sqexp_covariance_, points_sampled_.data(), std::vector<double>(num_sampled_, 0.0).data(), noise_variance_.data(), dim_, num_sampled_) {
  }

  virtual void GetInputSizes(int * num_rows, int * num_cols) const noexcept override OL_NONNULL_POINTERS {
    *num_rows = dim_;
    *num_cols = num_to_sample_;
  }

  virtual int GetGradientsSize() const noexcept override OL_WARN_UNUSED_RESULT {
    return dim_*num_to_sample_*GetOutputSize();
  }

  virtual int GetOutputSize() const noexcept override OL_WARN_UNUSED_RESULT {
    return Square(num_to_sample_);
  }

  virtual void EvaluateAndStoreAnalyticGradient(double const * restrict points_to_sample, double * restrict gradients) noexcept override OL_NONNULL_POINTERS_LIST(2) {
    if (gradients_already_computed_ == true) {
      OL_WARNING_PRINTF("WARNING: grad_variance data already set.  Overwriting...\n");
    }
    gradients_already_computed_ = true;

    GaussianProcess::StateType points_to_sample_state(gaussian_process_, points_to_sample, num_to_sample_, true);
    std::vector<double> variance_of_points(Square(num_to_sample_));
    for (int i = 0; i < num_to_sample_; ++i) {
      grad_variance_[i].resize(dim_*Square(num_to_sample_));

      gaussian_process_.ComputeVarianceOfPoints(&points_to_sample_state, variance_of_points.data());
      ComputeCholeskyFactorL(num_to_sample_, variance_of_points.data());
      gaussian_process_.ComputeGradCholeskyVarianceOfPoints(&points_to_sample_state, i, variance_of_points.data(), grad_variance_[i].data());
    }

    if (gradients != nullptr) {
      OL_THROW_EXCEPTION(RuntimeException, "PingGPPCholeskyVariance::EvaluateAndStoreAnalyticGradient() does not support direct gradient output.");
    }
  }

  virtual double GetAnalyticGradient(int row_index, int column_index, int output_index) const override OL_WARN_UNUSED_RESULT {
    if (gradients_already_computed_ == false) {
      OL_THROW_EXCEPTION(RuntimeException, "PingGPPCholeskyVariance::GetAnalyticGradient() called BEFORE EvaluateAndStoreAnalyticGradient. NO DATA!");
    }

    return grad_variance_[column_index][output_index*dim_ + row_index];
  }

  OL_NONNULL_POINTERS void EvaluateFunction(double const * restrict points_to_sample, double * restrict function_values) const noexcept override {
    GaussianProcess::StateType points_to_sample_state(gaussian_process_, points_to_sample, num_to_sample_, false);
    gaussian_process_.ComputeVarianceOfPoints(&points_to_sample_state, chol_temp_.data());
    ComputeCholeskyFactorL(num_to_sample_, chol_temp_.data());
    ZeroUpperTriangle(num_to_sample_, chol_temp_.data());
    MatrixTranspose(chol_temp_.data(), num_to_sample_, num_to_sample_, function_values);
  }

 private:
  int dim_;
  int num_to_sample_;
  int num_sampled_;
  bool gradients_already_computed_;

  std::vector<double> noise_variance_;
  std::vector<double> points_sampled_;
  mutable std::vector<double> chol_temp_;  // temporary storage used by the class
  std::vector<std::vector<double> > grad_variance_;

  SquareExponential sqexp_covariance_;
  GaussianProcess gaussian_process_;

  OL_DISALLOW_DEFAULT_AND_COPY_AND_ASSIGN(PingGPPCholeskyVariance);
};

/*
  Supports evaluating the expected improvement, get_expected_EI(), and its gradient, get_expected_grad_EI().

  The gradient is taken wrt current_point[dim], so this is the "input_matrix" X_{d,i} (with i always indexing 0).
  The other inputs to EI are not differentiated against, so they are taken as input and stored by the constructor.

  The output of EI is a scalar.

  WARNING: this class is NOT THREAD SAFE.
*/
class PingExpectedImprovement final : public PingableMatrixInputVectorOutputInterface {
 public:
  constexpr static char const * const kName = "EI with MC integration";

  PingExpectedImprovement(double const * restrict lengths, double const * restrict points_being_sampled, double const * restrict points_sampled, double const * restrict points_sampled_value, double alpha, double best_so_far, int dim, int num_being_sampled, int num_sampled, int num_mc_iter) OL_NONNULL_POINTERS
      : dim_(dim),
        num_being_sampled_(num_being_sampled),
        num_sampled_(num_sampled),
        gradients_already_computed_(false),
        noise_variance_(num_sampled_, 0.0),
        union_of_points(dim_*(num_being_sampled_+1)),
        points_sampled_(points_sampled, points_sampled + dim_*num_sampled_),
        points_sampled_value_(points_sampled_value, points_sampled_value + num_sampled_),
        grad_EI_(dim_),
        sqexp_covariance_(dim_, alpha, lengths),
        gaussian_process_(sqexp_covariance_, points_sampled_.data(), points_sampled_value_.data(), noise_variance_.data(), dim_, num_sampled_), ei_evaluator_(gaussian_process_, num_mc_iter, best_so_far) {
    std::copy(points_being_sampled, points_being_sampled + dim_*num_being_sampled_, union_of_points.begin() + dim_);
  }

  virtual void GetInputSizes(int * num_rows, int * num_cols) const noexcept override OL_NONNULL_POINTERS {
    *num_rows = dim_;
    *num_cols = 1;
  }

  virtual int GetGradientsSize() const noexcept override OL_WARN_UNUSED_RESULT {
    return dim_*GetOutputSize();
  }

  virtual int GetOutputSize() const noexcept override OL_WARN_UNUSED_RESULT {
    return 1;
  }

  virtual void EvaluateAndStoreAnalyticGradient(double const * restrict current_point, double * restrict gradients) noexcept override OL_NONNULL_POINTERS_LIST(2) {
    if (gradients_already_computed_ == true) {
      OL_WARNING_PRINTF("WARNING: grad_EI data already set.  Overwriting...\n");
    }
    gradients_already_computed_ = true;

    NormalRNG normal_rng(3141);
    std::copy(current_point, current_point + dim_, union_of_points.data());
    ExpectedImprovementEvaluator::StateType ei_state(ei_evaluator_, union_of_points.data(), num_being_sampled_+1, true, &normal_rng);
    ei_evaluator_.ComputeGradExpectedImprovement(&ei_state, grad_EI_.data());

    if (gradients != nullptr) {
      std::copy(grad_EI_.begin(), grad_EI_.end(), gradients);
    }
  }

  virtual double GetAnalyticGradient(int row_index, int OL_UNUSED(column_index), int OL_UNUSED(output_index)) const override OL_WARN_UNUSED_RESULT {
    if (gradients_already_computed_ == false) {
      OL_THROW_EXCEPTION(RuntimeException, "PingExpectedImprovement::GetAnalyticGradient() called BEFORE EvaluateAndStoreAnalyticGradient. NO DATA!");
    }

    return grad_EI_[row_index];
  }

  virtual void EvaluateFunction(double const * restrict current_point, double * restrict function_values) const noexcept override OL_NONNULL_POINTERS {
    std::copy(current_point, current_point + dim_, union_of_points.data());
    NormalRNG normal_rng(3141);
    ExpectedImprovementEvaluator::StateType ei_state(ei_evaluator_, union_of_points.data(), num_being_sampled_+1, false, &normal_rng);
    *function_values = ei_evaluator_.ComputeExpectedImprovement(&ei_state);
  }

 private:
  int dim_;
  int num_being_sampled_;
  int num_sampled_;
  bool gradients_already_computed_;

  std::vector<double> noise_variance_;
  mutable std::vector<double> union_of_points;
  std::vector<double> points_sampled_;
  std::vector<double> points_sampled_value_;
  std::vector<double> grad_EI_;

  SquareExponential sqexp_covariance_;
  GaussianProcess gaussian_process_;
  ExpectedImprovementEvaluator ei_evaluator_;

  OL_DISALLOW_DEFAULT_AND_COPY_AND_ASSIGN(PingExpectedImprovement);
};

/*
  Supports evaluating an analytic special case of expected improvement via OnePotentialSampleExpectedImprovementEvaluator.

  The gradient is taken wrt current_point[dim], so this is the "input_matrix" X_{d,i} (with i always indexing 0).
  The other inputs to EI are not differentiated against, so they are taken as input and stored by the constructor.

  The output of EI is a scalar.
*/
class PingOnePotentialSampleExpectedImprovement final : public PingableMatrixInputVectorOutputInterface {
 public:
  constexpr static char const * const kName = "EI ONE potential sample analytic";

  PingOnePotentialSampleExpectedImprovement(double const * restrict lengths, double const * restrict OL_UNUSED(points_being_sampled), double const * restrict points_sampled, double const * restrict points_sampled_value, double alpha, double best_so_far, int dim, int num_being_sampled, int num_sampled, int OL_UNUSED(num_mc_iter)) OL_NONNULL_POINTERS
      : dim_(dim),
        num_sampled_(num_sampled),
        gradients_already_computed_(false),
        noise_variance_(num_sampled_, 0.0),
        points_sampled_(points_sampled, points_sampled + dim_*num_sampled_),
        points_sampled_value_(points_sampled_value, points_sampled_value + num_sampled_),
        grad_EI_(dim_),
        sqexp_covariance_(dim_, alpha, lengths),
        gaussian_process_(sqexp_covariance_, points_sampled_.data(), points_sampled_value_.data(), noise_variance_.data(), dim_, num_sampled_), ei_evaluator_(gaussian_process_, best_so_far) {
    if (num_being_sampled != 0) {
      OL_THROW_EXCEPTION(InvalidValueException<int>, "PingOnePotentialSample: num_being_sampled MUST be 0!", num_being_sampled, 0);
    }
  }

  virtual void GetInputSizes(int * num_rows, int * num_cols) const noexcept override OL_NONNULL_POINTERS {
    *num_rows = dim_;
    *num_cols = 1;
  }

  virtual int GetGradientsSize() const noexcept override OL_WARN_UNUSED_RESULT {
    return dim_*GetOutputSize();
  }

  virtual int GetOutputSize() const noexcept override OL_WARN_UNUSED_RESULT {
    return 1;
  }

  virtual void EvaluateAndStoreAnalyticGradient(double const * restrict current_point, double * restrict gradients) noexcept override OL_NONNULL_POINTERS_LIST(2) {
    if (gradients_already_computed_ == true) {
      OL_WARNING_PRINTF("WARNING: grad_EI data already set.  Overwriting...\n");
    }
    gradients_already_computed_ = true;

    OnePotentialSampleExpectedImprovementEvaluator::StateType ei_state(ei_evaluator_, current_point, num_being_sampled_+1, true, nullptr);
    ei_evaluator_.ComputeGradExpectedImprovement(&ei_state, grad_EI_.data());

    if (gradients != nullptr) {
      std::copy(grad_EI_.begin(), grad_EI_.end(), gradients);
    }
  }

  virtual double GetAnalyticGradient(int row_index, int OL_UNUSED(column_index), int OL_UNUSED(output_index)) const override OL_WARN_UNUSED_RESULT {
    if (gradients_already_computed_ == false) {
      OL_THROW_EXCEPTION(RuntimeException, "PingOnePotentialSampleExpectedImprovement::GetAnalyticGradient() called BEFORE EvaluateAndStoreAnalyticGradient. NO DATA!");
    }

    return grad_EI_[row_index];
  }

  virtual void EvaluateFunction(double const * restrict current_point, double * restrict function_values) const noexcept override OL_NONNULL_POINTERS {
    OnePotentialSampleExpectedImprovementEvaluator::StateType ei_state(ei_evaluator_, current_point, num_being_sampled_+1, false, nullptr);
    *function_values = ei_evaluator_.ComputeExpectedImprovement(&ei_state);
  }

 private:
  int dim_;
  int num_sampled_;
  const int num_being_sampled_ = 0;
  bool gradients_already_computed_;

  std::vector<double> noise_variance_;
  std::vector<double> points_sampled_;
  std::vector<double> points_sampled_value_;
  std::vector<double> grad_EI_;

  SquareExponential sqexp_covariance_;
  GaussianProcess gaussian_process_;
  OnePotentialSampleExpectedImprovementEvaluator ei_evaluator_;

  OL_DISALLOW_DEFAULT_AND_COPY_AND_ASSIGN(PingOnePotentialSampleExpectedImprovement);
};

/*
  Pings gradients (spatial) of GP components (e.g., mean, variance, cholesky of variance) 50 times with randomly generated test cases
*/
template <typename GPComponentEvaluator>
OL_WARN_UNUSED_RESULT int PingGPComponentTest(double epsilon[2], double tolerance_fine, double tolerance_coarse, double input_output_ratio) {
  int total_errors = 0;
  int errors_this_iteration = 0;
  const int dim = 3;

  int num_being_sampled = 0;
  int num_to_sample = 5;
  int num_sampled = 7;

  std::vector<double> lengths(dim);
  double alpha = 2.80723;

  MockExpectedImprovementEnvironment EI_environment;

  UniformRandomGenerator uniform_generator(2718);
  boost::uniform_real<double> uniform_double(0.5, 2.5);

  for (int i = 0; i < 50; ++i) {
    EI_environment.Initialize(dim, num_to_sample, num_being_sampled, num_sampled);

    for (int j = 0; j < dim; ++j) {
      lengths[j] = uniform_double(uniform_generator.engine);
    }

    GPComponentEvaluator gp_component_evaluator(lengths.data(), EI_environment.points_sampled(), EI_environment.points_sampled_value(), alpha, EI_environment.dim, EI_environment.num_to_sample, EI_environment.num_sampled);
    gp_component_evaluator.EvaluateAndStoreAnalyticGradient(EI_environment.points_to_sample(), nullptr);
    errors_this_iteration = PingDerivative(gp_component_evaluator, EI_environment.points_to_sample(), epsilon, tolerance_fine, tolerance_coarse, input_output_ratio);

    if (errors_this_iteration != 0) {
      OL_PARTIAL_FAILURE_PRINTF("on iteration %d\n", i);
    }
    total_errors += errors_this_iteration;
  }

  if (total_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("%s gradient pings failed with %d errors\n", GPComponentEvaluator::kName, total_errors);
  } else {
    OL_PARTIAL_SUCCESS_PRINTF("%s gradient pings passed\n", GPComponentEvaluator::kName);
  }

  return total_errors;
}

}  // end unnamed namespace

/*
  Pings the gradients (spatial) of the GP mean 50 times with randomly generated test cases
*/
int PingGPMeanTest() {
  double epsilon_gp_mean[2] = {5.0e-3, 1.0e-3};
  int total_errors = PingGPComponentTest<PingGPPMean>(epsilon_gp_mean, 2.0e-3, 2.0e-3, 1.0e-18);
  return total_errors;
}

/*
  Pings the gradients (spatial) of the GP variance 50 times with randomly generated test cases
*/
int PingGPVarianceTest() {
  // TODO(eliu): look at improving ping testing so that we can improve tolerances to ~1.0e-3 or better (#43006)
  double epsilon_gp_variance[2] = {5.32879e-3, 0.942478e-3};
  int total_errors = PingGPComponentTest<PingGPPVariance>(epsilon_gp_variance, 2.0e-2, 4.0e-1, 1.0e-18);
  return total_errors;
}

/*
  Wrapper to ping the gradients (spatial) of the cholesky factorization.
*/
int PingGPCholeskyVarianceTest() {
  // TODO(eliu): look at improving ping testing so that we can improve tolerances to ~1.0e-3 or better (#43006)
  double epsilon_gp_variance[2] = {5.5e-3, 0.932e-3};
  int total_errors = PingGPComponentTest<PingGPPCholeskyVariance>(epsilon_gp_variance, 9.0e-3, 3.0e-1, 1.0e-18);
  return total_errors;
}

/*
  Pings the gradients (spatial) of the EI 50 times with randomly generated test cases
  Works with various EI evaluators (e.g., MC, analytic formulae)
*/
template <typename EIEvaluator>
OL_WARN_UNUSED_RESULT int PingEITest(int num_to_sample, int num_being_sampled, double epsilon[2], double tolerance_fine, double tolerance_coarse, double input_output_ratio) {
  int total_errors = 0;
  int errors_this_iteration;
  const int dim = 3;

  int num_sampled = 7;

  std::vector<double> lengths(dim);
  double alpha = 2.80723;
  // set best_so_far to be larger than max(points_sampled_value) (but don't make it huge or stability will be suffer)
  double best_so_far = 10.0;
  const int num_mc_iter = 11;

  MockExpectedImprovementEnvironment EI_environment;

  UniformRandomGenerator uniform_generator(2718);
  boost::uniform_real<double> uniform_double(0.5, 2.5);

  for (int i = 0; i < 50; ++i) {
    EI_environment.Initialize(dim, num_to_sample, num_being_sampled, num_sampled);

    for (int j = 0; j < dim; ++j) {
      lengths[j] = uniform_double(uniform_generator.engine);
    }

    EIEvaluator EI_evaluator(lengths.data(), EI_environment.points_being_sampled(), EI_environment.points_sampled(), EI_environment.points_sampled_value(), alpha, best_so_far, EI_environment.dim, EI_environment.num_being_sampled, EI_environment.num_sampled, num_mc_iter);
    EI_evaluator.EvaluateAndStoreAnalyticGradient(EI_environment.points_to_sample(), nullptr);
    errors_this_iteration = PingDerivative(EI_evaluator, EI_environment.points_to_sample(), epsilon, tolerance_fine, tolerance_coarse, input_output_ratio);

    if (errors_this_iteration != 0) {
      OL_PARTIAL_FAILURE_PRINTF("on iteration %d\n", i);
    }
    total_errors += errors_this_iteration;
  }

  if (total_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("%s gradient pings failed with %d errors\n", EIEvaluator::kName, total_errors);
  } else {
    OL_PARTIAL_SUCCESS_PRINTF("%s gradient pings passed\n", EIEvaluator::kName);
  }

  return total_errors;
}

/*
  Pings the gradients (spatial) of the EI 50 times with randomly generated test cases
*/
int PingEIGeneralTest() {
  double epsilon_EI[2] = {5.0e-3, 1.0e-3};
  int total_errors = PingEITest<PingExpectedImprovement>(1, 5, epsilon_EI, 2.0e-3, 9.0e-2, 1.0e-18);
  return total_errors;
}

/*
  Pings the gradients (spatial) of the EI (one potential sample special case) 50 times with randomly generated test cases
*/
int PingEIOnePotentialSampleTest() {
  double epsilon_EI_one_potential_sample[2] = {5.0e-3, 9.0e-4};
  int total_errors = PingEITest<PingOnePotentialSampleExpectedImprovement>(1, 0, epsilon_EI_one_potential_sample, 2.0e-3, 7.0e-2, 1.0e-18);
  return total_errors;
}

/*
  Generates a set of 50 random test cases for expected improvement with only one potential sample.
  The general EI (which uses MC integration) is evaluated to reasonably high accuracy (while not taking too long to run)
  and compared against the analytic formula version for consistency.  The gradients (spatial) of EI are also checked.
*/
int RunEIConsistencyTests() {
  int total_errors = 0;

  const int num_mc_iter = 1000000;
  const int dim = 3;
  const int num_being_sampled = 0;
  const int num_to_sample = 1;
  const int num_sampled = 7;

  double alpha = 2.80723;
  // set best_so_far to be larger than max(points_sampled_value) (but don't make it huge or stability will be suffer)
  double best_so_far = 10.0;

  int max_num_threads = 4;
  int chunk_size = 5;

#pragma omp parallel num_threads(max_num_threads)
  {
    int tid = omp_get_thread_num();
    UniformRandomGenerator uniform_generator(31278 + tid);
    boost::uniform_real<double> uniform_double(0.5, 2.5);

    MockExpectedImprovementEnvironment EI_environment;

    std::vector<double> lengths(dim);
    std::vector<double> grad_EI_general(dim);
    std::vector<double> grad_EI_one_potential_sample(dim);
    double EI_general;
    double EI_one_potential_sample;

#pragma omp for nowait schedule(static, chunk_size) reduction(+:total_errors)
    for (int i = 0; i < 40; ++i) {
      EI_environment.Initialize(dim, num_to_sample, num_being_sampled, num_sampled, &uniform_generator);

      for (int j = 0; j < dim; ++j) {
        lengths[j] = uniform_double(uniform_generator.engine);
      }

      PingOnePotentialSampleExpectedImprovement EI_one_potential_sample_evaluator(lengths.data(), EI_environment.points_being_sampled(), EI_environment.points_sampled(), EI_environment.points_sampled_value(), alpha, best_so_far, EI_environment.dim, EI_environment.num_being_sampled, EI_environment.num_sampled, num_mc_iter);
      EI_one_potential_sample_evaluator.EvaluateAndStoreAnalyticGradient(EI_environment.points_to_sample(), grad_EI_one_potential_sample.data());
      EI_one_potential_sample_evaluator.EvaluateFunction(EI_environment.points_to_sample(), &EI_one_potential_sample);

      PingExpectedImprovement EI_evaluator(lengths.data(), EI_environment.points_being_sampled(), EI_environment.points_sampled(), EI_environment.points_sampled_value(), alpha, best_so_far, EI_environment.dim, EI_environment.num_being_sampled, EI_environment.num_sampled, num_mc_iter);
      EI_evaluator.EvaluateAndStoreAnalyticGradient(EI_environment.points_to_sample(), grad_EI_general.data());
      EI_evaluator.EvaluateFunction(EI_environment.points_to_sample(), &EI_general);

      int ei_errors_this_iteration = 0;
      if (!CheckDoubleWithinRelative(EI_general, EI_one_potential_sample, 5.0e-4)) {
        ++ei_errors_this_iteration;
      }
      if (ei_errors_this_iteration != 0) {
        OL_PARTIAL_FAILURE_PRINTF("in EI on iteration %d\n", i);
      }
      total_errors += ei_errors_this_iteration;

      int grad_ei_errors_this_iteration = 0;
      for (int j = 0; j < dim; ++j) {
        if (!CheckDoubleWithinRelative(grad_EI_general[j], grad_EI_one_potential_sample[j], 6.5e-3)) {
          ++grad_ei_errors_this_iteration;
        }
      }

      if (grad_ei_errors_this_iteration != 0) {
        OL_PARTIAL_FAILURE_PRINTF("in EI gradients on iteration %d\n", i);
      }
      total_errors += grad_ei_errors_this_iteration;
    }
  }

  if (total_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("comparing MC EI to analytic EI failed with %d total_errors\n\n", total_errors);
  } else {
    OL_PARTIAL_SUCCESS_PRINTF("comparing MC EI to analytic EI passed\n");
  }

  return total_errors;
}

int RunGPPingTests() {
  int total_errors = 0;
  int current_errors = 0;

  {
    current_errors = PingGPMeanTest();
    if (current_errors != 0) {
      OL_PARTIAL_FAILURE_PRINTF("pinging GP mean failed with %d errors\n", current_errors);
    }
    total_errors += current_errors;
  }

  {
    current_errors = PingGPVarianceTest();
    if (current_errors != 0) {
      OL_PARTIAL_FAILURE_PRINTF("pinging GP variance failed with %d errors\n", current_errors);
    }
    total_errors += current_errors;
  }

  {
    current_errors = PingGPCholeskyVarianceTest();
    if (current_errors != 0) {
      OL_PARTIAL_FAILURE_PRINTF("pinging GP cholesky of variance failed with %d errors\n", current_errors);
    }
    total_errors += current_errors;
  }

  {
    current_errors = PingEIGeneralTest();
    if (current_errors != 0) {
      OL_PARTIAL_FAILURE_PRINTF("pinging EI failed with %d errors\n", current_errors);
    }
    total_errors += current_errors;
  }

  {
    current_errors = PingEIOnePotentialSampleTest();
    if (current_errors != 0) {
      OL_PARTIAL_FAILURE_PRINTF("pinging analytic (one potential sample) EI failed with %d errors\n", current_errors);
    }
    total_errors += current_errors;
  }

  if (total_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("Pinging GP functions failed with %d errors\n\n", total_errors);
  } else {
    OL_PARTIAL_SUCCESS_PRINTF("Pinging GP functions passed\n");
  }

  return total_errors;
}

/*
  Tests that single & multithreaded EI optimization produce *the exact same* results.

  We do this by first setting up EI optimization in a single threaded scenario with 2 starting points and 2 random number generators.
  Optimization is run one from starting point 0 with RNG 0, and then again from starting point 1 with RNG 1.

  Then we run the optimization multithreaded (with 2 threads) over both starting points simultaneously.  One of the threads
  will see the winning (point, RNG) pair from the single-threaded won.  Hence one result point will match with the single threaded
  results exactly.

  Then we re-run the multithreaded optimization, swapping the position of the RNGs and starting points.  If thread 0 won in the
  previous test, thread 1 will win here (and vice versa).

  Note that it's tricky to run single-threaded optimization over both starting points simultaneously because we won't know which
  (point, RNG) pair won (which is required to ascertain the 'winner' since we are not computing EI accurately enough to avoid
  error).
*/
int MultithreadedEIOptimizationTest(ExpectedImprovementEvaluationMode ei_mode) {
  using DomainType = TensorProductDomain;
  const int num_sampled = 17;
  static const int kDim = 3;
  int num_being_sampled = 6;
  if (ei_mode == ExpectedImprovementEvaluationMode::kAnalytic) {
    num_being_sampled = 0;
  }
  std::vector<double> points_being_sampled(kDim*num_being_sampled);

  // gradient descent parameters
  const double gamma = 0.9;
  const double pre_mult = 1.0;
  const double max_relative_change = 1.0;
  const double tolerance = 1.0e-7;

  const int max_gradient_descent_steps = 300;
  const int max_num_restarts = 10;
  GradientDescentParameters gd_params(0, max_gradient_descent_steps, max_num_restarts, gamma, pre_mult, max_relative_change, tolerance);

  int max_mc_iterations = 1000;

  int total_errors = 0;

  // seed randoms
  UniformRandomGenerator uniform_generator(314);
  boost::uniform_real<double> uniform_double_hyperparameter(1.0, 2.5);
  boost::uniform_real<double> uniform_double_lower_bound(-2.0, 0.5);
  boost::uniform_real<double> uniform_double_upper_bound(2.5, 5.5);

  std::vector<double> noise_variance(num_sampled, 0.0003);
  MockGaussianProcessPriorData<DomainType> mock_gp_data(SquareExponential(kDim, 1.0, 1.0), noise_variance, kDim, num_sampled, uniform_double_lower_bound, uniform_double_upper_bound, uniform_double_hyperparameter, &uniform_generator);

  for (int j = 0; j < num_being_sampled; ++j) {
    mock_gp_data.domain_ptr->GeneratePointInDomain(&uniform_generator, points_being_sampled.data() + j*kDim);
  }

  const int pi_array[] = {314, 3141, 31415, 314159};
  static const int kMaxNumThreads = 2;
  std::vector<NormalRNG> normal_rng_vec(kMaxNumThreads);
  for (int j = 0; j < kMaxNumThreads; ++j) {
    normal_rng_vec[j].SetExplicitSeed(pi_array[j]);
  }

  std::vector<double> starting_points(kDim*kMaxNumThreads);
  for (int j = 0; j < kMaxNumThreads; ++j) {
    mock_gp_data.domain_ptr->GeneratePointInDomain(&uniform_generator, starting_points.data() + j*kDim);
  }

  // we will optimize over the expanded region
  std::vector<ClosedInterval> domain_bounds(mock_gp_data.domain_bounds);
  ExpandDomainBounds(3.2, &domain_bounds);
  DomainType domain(domain_bounds.data(), kDim);

  // build truth data by using single threads
  bool found_flag = false;
  double best_next_point_single_thread[kDim*kMaxNumThreads*kMaxNumThreads];
  int num_threads = 1;
  for (int j = 0; j < kMaxNumThreads; ++j) {
    NormalRNG normal_rng(pi_array[j]);
    ComputeOptimalPointToSampleViaMultistartGradientDescent(*mock_gp_data.gaussian_process_ptr, gd_params, domain, starting_points.data() + j*kDim, points_being_sampled.data(), 1, num_being_sampled, mock_gp_data.best_so_far, max_mc_iterations, num_threads, &normal_rng, &found_flag, best_next_point_single_thread + j*kDim);
    if (!found_flag) {
      ++total_errors;
    }

    normal_rng.SetExplicitSeed(pi_array[kMaxNumThreads - j - 1]);
    ComputeOptimalPointToSampleViaMultistartGradientDescent(*mock_gp_data.gaussian_process_ptr, gd_params, domain, starting_points.data() + j*kDim, points_being_sampled.data(), 1, num_being_sampled, mock_gp_data.best_so_far, max_mc_iterations, num_threads, &normal_rng, &found_flag, best_next_point_single_thread + j*kDim + kDim*kMaxNumThreads);
    if (!found_flag) {
      ++total_errors;
    }
  }

  // now multithreaded to generate test data
  double best_next_point_multithread[kDim];
  num_threads = 2;
  found_flag = false;
  ComputeOptimalPointToSampleViaMultistartGradientDescent(*mock_gp_data.gaussian_process_ptr, gd_params, domain, starting_points.data(), points_being_sampled.data(), kMaxNumThreads, num_being_sampled, mock_gp_data.best_so_far, max_mc_iterations, num_threads, normal_rng_vec.data(), &found_flag, best_next_point_multithread);
  if (!found_flag) {
    ++total_errors;
  }

  // best_next_point_multithread must be PRECISELY one of the points determined by single threaded runs
  double error[kMaxNumThreads*kMaxNumThreads];
  for (int i = 0; i < kMaxNumThreads; ++i) {
    for (int j = 0; j < kMaxNumThreads; ++j) {
      error[i*kMaxNumThreads + j] = 0.0;
      for (int k = 0; k < kDim; ++k) {
        error[i*kMaxNumThreads + j] += std::fabs(best_next_point_single_thread[i*kDim*kMaxNumThreads + j*kDim + k] - best_next_point_multithread[k]);
      }
    }
  }
  // normally double precision checks like this are bad
  // but here, we want to ensure that the multithreaded & singlethreaded paths executed THE EXACT SAME CODE IN THE SAME ORDER
  // and hence their results must be identical
  bool pass = false;
  for (int i = 0; i < kMaxNumThreads*kMaxNumThreads; ++i) {
    if (error[i] == 0.0) {
      pass = true;
      break;
    }
  }
  if (pass == false) {
    OL_PARTIAL_FAILURE_PRINTF("multi & single threaded results differ 1: ");
    PrintMatrix(error, 1, Square(kMaxNumThreads));
    ++total_errors;
  }

  // reset random state & flip the points & generators so that if thread 0 won before, thread 1 wins now (or vice versa)
  for (int j = 0; j < kMaxNumThreads; ++j) {
    normal_rng_vec[kMaxNumThreads-j-1].SetExplicitSeed(pi_array[j]);
  }

  std::vector<double> starting_points_flip(kDim*kMaxNumThreads);
  for (int j = 0; j < kMaxNumThreads; ++j) {
    for (int k = 0; k < kDim; ++k) {
      starting_points_flip[(kMaxNumThreads-j-1)*kDim + k] = starting_points[j*kDim + k];
    }
  }

  // check multithreaded results again
  found_flag = false;
  ComputeOptimalPointToSampleViaMultistartGradientDescent(*mock_gp_data.gaussian_process_ptr, gd_params, domain, starting_points_flip.data(), points_being_sampled.data(), kMaxNumThreads, num_being_sampled, mock_gp_data.best_so_far, max_mc_iterations, num_threads, normal_rng_vec.data(), &found_flag, best_next_point_multithread);
  if (!found_flag) {
    ++total_errors;
  }

  for (int i = 0; i < kMaxNumThreads; ++i) {
    for (int j = 0; j < kMaxNumThreads; ++j) {
      error[i*kMaxNumThreads + j] = 0.0;
      for (int k = 0; k < kDim; ++k) {
        error[i*kMaxNumThreads + j] += std::fabs(best_next_point_single_thread[i*kDim*kMaxNumThreads + j*kDim + k] - best_next_point_multithread[k]);
      }
    }
  }
  // normally double precision checks like this are bad
  // but here, we want to ensure that the multithreaded & singlethreaded paths executed THE EXACT SAME CODE IN THE SAME ORDER
  // and hence their results must be identical
  pass = false;
  for (int i = 0; i < kMaxNumThreads*kMaxNumThreads; ++i) {
    if (error[i] == 0.0) {
      pass = true;
      break;
    }
  }
  if (pass == false) {
    OL_PARTIAL_FAILURE_PRINTF("multi & single threaded results differ 2: ");
    PrintMatrix(error, 1, Square(kMaxNumThreads));
    ++total_errors;
  }

  if (total_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("Single/Multithreaded EI Optimization Consistency Check failed with %d errors\n", total_errors);
  } else {
    OL_PARTIAL_SUCCESS_PRINTF("Single/Multithreaded EI Optimization Consistency Check succeeded\n");
  }

  return total_errors;
}

namespace {  // contains tests of EI optimization

OL_WARN_UNUSED_RESULT int ExpectedImprovementOptimizationTestCore(ExpectedImprovementEvaluationMode ei_mode) {
  using DomainType = TensorProductDomain;
  const int dim = 3;

  int total_errors = 0;
  int current_errors = 0;

  // gradient descent parameters
  const double gamma = 0.9;
  const double pre_mult = 1.0;
  const double max_relative_change = 1.0;
  const double tolerance = 1.0e-7;
  const int max_gradient_descent_steps = 1000;
  const int max_num_restarts = 10;
  const int num_multistarts = 20;
  GradientDescentParameters gd_params(num_multistarts, max_gradient_descent_steps, max_num_restarts, gamma, pre_mult, max_relative_change, tolerance);

  // grid search parameters
  int num_grid_search_points = 100000;

  // EI computation parameters
  int num_being_sampled = 0;
  int max_int_steps = 1000;

  // random number generators
  UniformRandomGenerator uniform_generator(314);
  boost::uniform_real<double> uniform_double_hyperparameter(0.4, 1.3);
  boost::uniform_real<double> uniform_double_lower_bound(-2.0, 0.5);
  boost::uniform_real<double> uniform_double_upper_bound(2.0, 3.5);

  const int64_t pi_array[] = {314, 3141, 31415, 314159, 3141592, 31415926, 314159265, 3141592653, 31415926535, 314159265359};
  static const int kMaxNumThreads = 4;
  std::vector<NormalRNG> normal_rng_vec(kMaxNumThreads);
  for (int j = 0; j < kMaxNumThreads; ++j) {
    normal_rng_vec[j].SetExplicitSeed(pi_array[j]);
  }

  int num_sampled;
  if (ei_mode == ExpectedImprovementEvaluationMode::kAnalytic) {
    num_sampled = 20;  // need to keep this similar to the number of multistarts
  } else {
    num_sampled = 80;  // matters less here b/c we end up starting one multistart from the LHC-search optima
  }

  std::vector<double> noise_variance(num_sampled, 0.002);
  MockGaussianProcessPriorData<DomainType> mock_gp_data(SquareExponential(dim, 1.0, 1.0), noise_variance, dim, num_sampled, uniform_double_lower_bound, uniform_double_upper_bound, uniform_double_hyperparameter, &uniform_generator);

  // we will optimize over the expanded region
  std::vector<ClosedInterval> domain_bounds(mock_gp_data.domain_bounds);
  ExpandDomainBounds(2.2, &domain_bounds);
  DomainType domain(domain_bounds.data(), dim);

  // set up parallel experiments, if any
  if (ei_mode == ExpectedImprovementEvaluationMode::kAnalytic) {
    num_being_sampled = 0;
  } else {
    // using MC integration
    num_being_sampled = 2;
    max_int_steps = 1000;

    gd_params.max_num_steps = 200;
    gd_params.tolerance = 1.0e-5;
  }
  std::vector<double> points_being_sampled(dim*num_being_sampled);

  if (ei_mode == ExpectedImprovementEvaluationMode::kMonteCarlo) {
    // generate two non-trivial parallel samples
    // picking these randomly could place them in regions where EI is 0, which means errors in the computation would
    // likely be masked (making for a bad test)
    bool found_flag = false;
    for (int j = 0; j < num_being_sampled; ++j) {
      ComputeOptimalPointToSampleWithRandomStarts(*mock_gp_data.gaussian_process_ptr, gd_params, domain, points_being_sampled.data(), j, mock_gp_data.best_so_far, max_int_steps, kMaxNumThreads, &found_flag, &uniform_generator, normal_rng_vec.data(), points_being_sampled.data() + j*dim);
    }
    printf("setup complete, points_being_sampled:\n");
    PrintMatrixTrans(points_being_sampled.data(), num_being_sampled, dim);
  }

  std::vector<double> next_point(dim);

  // optimize EI
  bool found_flag = false;
  std::vector<double> grid_search_best_point(dim);
  ComputeOptimalPointToSampleViaLatinHypercubeSearch(*mock_gp_data.gaussian_process_ptr, domain, points_being_sampled.data(), num_grid_search_points, num_being_sampled, mock_gp_data.best_so_far, max_int_steps, kMaxNumThreads, &found_flag, &uniform_generator, normal_rng_vec.data(), grid_search_best_point.data());
  if (!found_flag) {
    ++total_errors;
  }

  if (ei_mode == ExpectedImprovementEvaluationMode::kAnalytic) {
    found_flag = false;
    ComputeOptimalPointToSampleWithRandomStarts(*mock_gp_data.gaussian_process_ptr, gd_params, domain, points_being_sampled.data(), num_being_sampled, mock_gp_data.best_so_far, max_int_steps, kMaxNumThreads, &found_flag, &uniform_generator, normal_rng_vec.data(), next_point.data());
    if (!found_flag) {
      ++total_errors;
    }
  } else {
    int num_multistarts_mc = 8;
    gd_params.num_multistarts = num_multistarts_mc;
    gd_params.max_num_steps = 300;
    max_int_steps = 6000;
    found_flag = false;
    std::vector<double> initial_guesses(num_multistarts_mc*dim);
    domain.GenerateUniformPointsInDomain(num_multistarts_mc - 1, &uniform_generator, initial_guesses.data() + dim);
    std::copy(grid_search_best_point.begin(), grid_search_best_point.end(), initial_guesses.begin());

    ComputeOptimalPointToSampleViaMultistartGradientDescent(*mock_gp_data.gaussian_process_ptr, gd_params, domain, initial_guesses.data(), points_being_sampled.data(), num_multistarts_mc, num_being_sampled, mock_gp_data.best_so_far, max_int_steps, kMaxNumThreads, normal_rng_vec.data(), &found_flag, next_point.data());
    if (!found_flag) {
      ++total_errors;
    }
  }

  printf("next best point  : "); PrintMatrix(next_point.data(), 1, dim);
  printf("grid search point: "); PrintMatrix(grid_search_best_point.data(), 1, dim);

  // results
  double ei_optimized, ei_grid_search;
  std::vector<double> grad_ei(dim);

  // set up evaluators and state to check results
  std::vector<double> union_of_points((num_being_sampled+1)*dim);
  std::copy(next_point.begin(), next_point.end(), union_of_points.begin());
  std::copy(points_being_sampled.begin(), points_being_sampled.end(), union_of_points.begin() + dim);

  std::vector<double> union_of_points_grid_search((num_being_sampled+1)*dim);
  std::copy(grid_search_best_point.begin(), grid_search_best_point.end(), union_of_points_grid_search.begin());
  std::copy(points_being_sampled.begin(), points_being_sampled.end(), union_of_points_grid_search.begin() + dim);

  double tolerance_result = tolerance;
  if (ei_mode == ExpectedImprovementEvaluationMode::kAnalytic) {
    OnePotentialSampleExpectedImprovementEvaluator ei_evaluator(*mock_gp_data.gaussian_process_ptr, mock_gp_data.best_so_far);
    OnePotentialSampleExpectedImprovementEvaluator::StateType ei_state(ei_evaluator, union_of_points.data(), num_being_sampled + 1, true, nullptr);

    ei_optimized = ei_evaluator.ComputeExpectedImprovement(&ei_state);
    ei_evaluator.ComputeGradExpectedImprovement(&ei_state, grad_ei.data());

    ei_state.UpdateCurrentPoint(ei_evaluator, union_of_points_grid_search.data());
    ei_grid_search = ei_evaluator.ComputeExpectedImprovement(&ei_state);
  } else {
    max_int_steps = 10000000;
    tolerance_result = 3.0e-4;  // reduce b/c we cannot achieve full accuracy in the monte-carlo case
    // while still having this test run in a reasonable amt of time
    ExpectedImprovementEvaluator ei_evaluator(*mock_gp_data.gaussian_process_ptr, max_int_steps, mock_gp_data.best_so_far);
    ExpectedImprovementEvaluator::StateType ei_state(ei_evaluator, union_of_points.data(), num_being_sampled + 1, true, normal_rng_vec.data());

    ei_optimized = ei_evaluator.ComputeExpectedImprovement(&ei_state);
    ei_evaluator.ComputeGradExpectedImprovement(&ei_state, grad_ei.data());

    ei_state.UpdateCurrentPoint(ei_evaluator, union_of_points_grid_search.data());
    ei_grid_search = ei_evaluator.ComputeExpectedImprovement(&ei_state);
  }

  printf("optimized EI: %.18E, grid_search_EI: %.18E\n", ei_optimized, ei_grid_search);
  printf("grad_EI: "); PrintMatrix(grad_ei.data(), 1, dim);

  if (ei_optimized < ei_grid_search) {
    ++total_errors;
  }

  current_errors = 0;
  for (const auto& entry : grad_ei) {
    if (!CheckDoubleWithinRelative(entry, 0.0, tolerance_result)) {
      ++current_errors;
    }
  }
  total_errors += current_errors;

  return total_errors;
}

OL_WARN_UNUSED_RESULT int ExpectedImprovementOptimizationSimplexTestCore(ExpectedImprovementEvaluationMode ei_mode) {
  using DomainType = SimplexIntersectTensorProductDomain;
  const int dim = 3;

  int total_errors = 0;
  int current_errors = 0;

  // gradient descent parameters
  const double gamma = 0.9;
  const double pre_mult = 0.02;
  const double max_relative_change = 0.99;
  const double tolerance = 1.0e-7;
  const int max_gradient_descent_steps = 1000;
  const int max_num_restarts = 10;
  const int num_multistarts = 20;
  GradientDescentParameters gd_params(num_multistarts, max_gradient_descent_steps, max_num_restarts, gamma, pre_mult, max_relative_change, tolerance);

  // grid search parameters
  int num_grid_search_points = 200000;

  // EI computation parameters
  int num_being_sampled = 0;
  int max_int_steps = 1000;

  // random number generators
  UniformRandomGenerator uniform_generator(314);
  boost::uniform_real<double> uniform_double_hyperparameter(0.05, 0.1);
  boost::uniform_real<double> uniform_double_lower_bound(0.11, 0.15);
  boost::uniform_real<double> uniform_double_upper_bound(0.3, 0.35);

  const int64_t pi_array[] = {314, 3141, 31415, 314159, 3141592, 31415926, 314159265, 3141592653, 31415926535, 314159265359};
  static const int kMaxNumThreads = 4;
  std::vector<NormalRNG> normal_rng_vec(kMaxNumThreads);
  for (int j = 0; j < kMaxNumThreads; ++j) {
    normal_rng_vec[j].SetExplicitSeed(pi_array[j]);
  }

  int num_sampled;
  if (ei_mode == ExpectedImprovementEvaluationMode::kAnalytic) {
    num_sampled = 20;  // need to keep this similar to the number of multistarts
  } else {
    num_sampled = 80;  // matters less here b/c we end up starting one multistart from the LHC-search optima
  }

  std::vector<double> noise_variance(num_sampled, 0.002);
  MockGaussianProcessPriorData<DomainType> mock_gp_data(SquareExponential(dim, 1.0, 1.0), noise_variance, dim, num_sampled);
  mock_gp_data.InitializeHyperparameters(uniform_double_hyperparameter, &uniform_generator);
  mock_gp_data.hyperparameters[0] = 0.8;
  mock_gp_data.InitializeDomain(uniform_double_lower_bound, uniform_double_upper_bound, &uniform_generator);
  mock_gp_data.InitializeGaussianProcess(&uniform_generator);

  // we will optimize over the expanded region
  std::vector<ClosedInterval> domain_bounds(mock_gp_data.domain_bounds);
  ExpandDomainBounds(2.2, &domain_bounds);
  // intersect domain with bounding box of unit simplex
  for (auto& interval : domain_bounds) {
    interval.min = std::fmax(interval.min, 0.0);
    interval.max = std::fmin(interval.max, 1.0);
  }
  DomainType domain(domain_bounds.data(), dim);

  // set up parallel experiments, if any
  if (ei_mode == ExpectedImprovementEvaluationMode::kAnalytic) {
    num_being_sampled = 0;
  } else {
    // using MC integration
    num_being_sampled = 2;
    max_int_steps = 1000;

    gd_params.max_num_steps = 600;
    gd_params.tolerance = 1.0e-5;
  }
  std::vector<double> points_being_sampled(dim*num_being_sampled);

  if (ei_mode == ExpectedImprovementEvaluationMode::kMonteCarlo) {
    // generate two non-trivial parallel samples
    // picking these randomly could place them in regions where EI is 0, which means errors in the computation would
    // likely be masked (making for a bad test)
    bool found_flag = false;
    for (int j = 0; j < num_being_sampled; ++j) {
      ComputeOptimalPointToSampleWithRandomStarts(*mock_gp_data.gaussian_process_ptr, gd_params, domain, points_being_sampled.data(), j, mock_gp_data.best_so_far, max_int_steps, kMaxNumThreads, &found_flag, &uniform_generator, normal_rng_vec.data(), points_being_sampled.data() + j*dim);
    }
    printf("setup complete, points_being_sampled:\n");
    PrintMatrixTrans(points_being_sampled.data(), num_being_sampled, dim);
  }

  std::vector<double> next_point(dim);

  // optimize EI
  bool found_flag = false;
  std::vector<double> grid_search_best_point(dim);
  ComputeOptimalPointToSampleViaLatinHypercubeSearch(*mock_gp_data.gaussian_process_ptr, domain, points_being_sampled.data(), num_grid_search_points, num_being_sampled, mock_gp_data.best_so_far, max_int_steps, kMaxNumThreads, &found_flag, &uniform_generator, normal_rng_vec.data(), grid_search_best_point.data());
  if (!found_flag) {
    ++total_errors;
  }

  if (ei_mode == ExpectedImprovementEvaluationMode::kAnalytic) {
    found_flag = false;
    ComputeOptimalPointToSampleWithRandomStarts(*mock_gp_data.gaussian_process_ptr, gd_params, domain, points_being_sampled.data(), num_being_sampled, mock_gp_data.best_so_far, max_int_steps, kMaxNumThreads, &found_flag, &uniform_generator, normal_rng_vec.data(), next_point.data());
    if (!found_flag) {
      ++total_errors;
    }
  } else {
    int num_multistarts_mc = 1;
    gd_params.num_multistarts = num_multistarts_mc;
    gd_params.max_num_steps = 1000;
    max_int_steps = 10000;
    found_flag = false;
    std::vector<double> initial_guesses(num_multistarts_mc*dim);
    std::copy(grid_search_best_point.begin(), grid_search_best_point.end(), initial_guesses.begin());

    ComputeOptimalPointToSampleViaMultistartGradientDescent(*mock_gp_data.gaussian_process_ptr, gd_params, domain, initial_guesses.data(), points_being_sampled.data(), num_multistarts_mc, num_being_sampled, mock_gp_data.best_so_far, max_int_steps, kMaxNumThreads, normal_rng_vec.data(), &found_flag, next_point.data());
    if (!found_flag) {
      ++total_errors;
    }
  }

  printf("next best point  : "); PrintMatrix(next_point.data(), 1, dim);
  printf("grid search point: "); PrintMatrix(grid_search_best_point.data(), 1, dim);

  // results
  double ei_optimized, ei_grid_search;
  std::vector<double> grad_ei(dim);

  // set up evaluators and state to check results
  std::vector<double> union_of_points((num_being_sampled+1)*dim);
  std::copy(next_point.begin(), next_point.end(), union_of_points.begin());
  std::copy(points_being_sampled.begin(), points_being_sampled.end(), union_of_points.begin() + dim);

  std::vector<double> union_of_points_grid_search((num_being_sampled+1)*dim);
  std::copy(grid_search_best_point.begin(), grid_search_best_point.end(), union_of_points_grid_search.begin());
  std::copy(points_being_sampled.begin(), points_being_sampled.end(), union_of_points_grid_search.begin() + dim);

  double tolerance_result = tolerance;
  if (ei_mode == ExpectedImprovementEvaluationMode::kAnalytic) {
    OnePotentialSampleExpectedImprovementEvaluator ei_evaluator(*mock_gp_data.gaussian_process_ptr, mock_gp_data.best_so_far);
    OnePotentialSampleExpectedImprovementEvaluator::StateType ei_state(ei_evaluator, union_of_points.data(), num_being_sampled + 1, true, nullptr);

    ei_optimized = ei_evaluator.ComputeExpectedImprovement(&ei_state);
    ei_evaluator.ComputeGradExpectedImprovement(&ei_state, grad_ei.data());

    ei_state.UpdateCurrentPoint(ei_evaluator, union_of_points_grid_search.data());
    ei_grid_search = ei_evaluator.ComputeExpectedImprovement(&ei_state);
  } else {
    max_int_steps = 10000000;
    tolerance_result = 3.0e-4;  // reduce b/c we cannot achieve full accuracy in the monte-carlo case
    // while still having this test run in a reasonable amt of time
    ExpectedImprovementEvaluator ei_evaluator(*mock_gp_data.gaussian_process_ptr, max_int_steps, mock_gp_data.best_so_far);
    ExpectedImprovementEvaluator::StateType ei_state(ei_evaluator, union_of_points.data(), num_being_sampled + 1, true, normal_rng_vec.data());

    ei_optimized = ei_evaluator.ComputeExpectedImprovement(&ei_state);
    ei_evaluator.ComputeGradExpectedImprovement(&ei_state, grad_ei.data());

    ei_state.UpdateCurrentPoint(ei_evaluator, union_of_points_grid_search.data());
    ei_grid_search = ei_evaluator.ComputeExpectedImprovement(&ei_state);
  }

  printf("optimized EI: %.18E, grid_search_EI: %.18E\n", ei_optimized, ei_grid_search);
  printf("grad_EI: "); PrintMatrix(grad_ei.data(), 1, dim);

  if (ei_optimized < ei_grid_search) {
    ++total_errors;
  }

  current_errors = 0;
  for (const auto& entry : grad_ei) {
    if (!CheckDoubleWithinRelative(entry, 0.0, tolerance_result)) {
      ++current_errors;
    }
  }
  total_errors += current_errors;

  return total_errors;
}

}  // end unnamed namespace

/*
  At the moment, this test is very bare-bones.  It checks:
  1) method succeeds
  2) points returned are all inside the specified domain
  3) points returned are not within epsilon of each other (i.e., distinct)
  4) result of gradient-descent optimization is *no worse* than result of a random search
  5) final grad EI is sufficiently small

  The test sets up a toy problem by repeatedly drawing from a GP with made-up hyperparameters.
  Then it runs EI optimization, attempting to sample 4 points simultaneously.
*/
int ExpectedImprovementOptimizationMultipleSamplesTest() {
  using DomainType = TensorProductDomain;
  const int dim = 3;

  int total_errors = 0;
  int current_errors = 0;

  // gradient descent parameters
  const double gamma = 0.9;
  const double pre_mult = 1.0;
  const double max_relative_change = 1.0;
  const double tolerance = 1.0e-5;
  const int max_gradient_descent_steps = 300;
  const int max_num_restarts = 10;
  const int num_multistarts = 20;
  GradientDescentParameters gd_params(num_multistarts, max_gradient_descent_steps, max_num_restarts, gamma, pre_mult, max_relative_change, tolerance);

  // number of simultaneous samples
  const int num_samples_to_generate = 4;

  // grid search parameters
  int num_grid_search_points = 10000;

  // EI computation parameters
  const int num_being_sampled = 0;
  std::vector<double> points_being_sampled(dim*num_being_sampled);
  int max_int_steps = 1000;

  // random number generators
  UniformRandomGenerator uniform_generator(314);
  boost::uniform_real<double> uniform_double_hyperparameter(0.4, 1.3);
  boost::uniform_real<double> uniform_double_lower_bound(-2.0, 0.5);
  boost::uniform_real<double> uniform_double_upper_bound(2.0, 3.5);

  const int64_t pi_array[] = {314, 3141, 31415, 314159, 3141592, 31415926, 314159265, 3141592653, 31415926535, 314159265359};
  static const int kMaxNumThreads = 4;
  std::vector<NormalRNG> normal_rng_vec(kMaxNumThreads);
  for (int j = 0; j < kMaxNumThreads; ++j) {
    normal_rng_vec[j].SetExplicitSeed(pi_array[j]);
  }

  const int num_sampled = 20;
  std::vector<double> noise_variance(num_sampled, 0.002);
  MockGaussianProcessPriorData<DomainType> mock_gp_data(SquareExponential(dim, 1.0, 1.0), noise_variance, dim, num_sampled, uniform_double_lower_bound, uniform_double_upper_bound, uniform_double_hyperparameter, &uniform_generator);

  // we will optimize over the expanded region
  std::vector<ClosedInterval> domain_bounds(mock_gp_data.domain_bounds);
  ExpandDomainBounds(1.5, &domain_bounds);
  DomainType domain(domain_bounds.data(), dim);

  std::vector<double> best_points_to_sample(dim*num_samples_to_generate);

  // optimize EI using grid search to set the baseline
  bool found_flag = false;
  std::vector<double> grid_search_best_point_set(dim*num_samples_to_generate);
  ComputeOptimalSetOfPointsToSample(*mock_gp_data.gaussian_process_ptr, gd_params, domain, points_being_sampled.data(), num_being_sampled, mock_gp_data.best_so_far, max_int_steps, kMaxNumThreads, true, num_grid_search_points, num_samples_to_generate, &found_flag, &uniform_generator, normal_rng_vec.data(), grid_search_best_point_set.data());
  if (!found_flag) {
    ++total_errors;
  }

  // optimize EI using gradient descent
  found_flag = false;
  ComputeOptimalSetOfPointsToSample(*mock_gp_data.gaussian_process_ptr, gd_params, domain, points_being_sampled.data(), num_being_sampled, mock_gp_data.best_so_far, max_int_steps, kMaxNumThreads, false, num_grid_search_points, num_samples_to_generate, &found_flag, &uniform_generator, normal_rng_vec.data(), best_points_to_sample.data());
  if (!found_flag) {
    ++total_errors;
  }

  // check points are in domain
  current_errors = CheckPointsInDomain(domain, best_points_to_sample.data(), num_samples_to_generate);
#ifdef OL_ERROR_PRINT
  if (current_errors != 0) {
    OL_ERROR_PRINTF("ERROR: points were not in domain!  points:\n");
    PrintMatrixTrans(best_points_to_sample.data(), num_samples_to_generate, dim);
    OL_ERROR_PRINTF("domain:\n");
    PrintDomainBounds(domain_bounds.data(), dim);
  }
#endif
  total_errors += current_errors;

  // check points are distinct; points within tolerance are considered non-distinct
  const double distinct_point_tolerance = 1.0e-5;
  current_errors = CheckPointsAreDistinct(best_points_to_sample.data(), num_samples_to_generate, dim, distinct_point_tolerance);
#ifdef OL_ERROR_PRINT
  if (current_errors != 0) {
    OL_ERROR_PRINTF("ERROR: points were not distinct!  points:\n");
    PrintMatrixTrans(best_points_to_sample.data(), num_samples_to_generate, dim);
  }
#endif
  total_errors += current_errors;

  // results
  double ei_optimized, ei_grid_search;
  std::vector<double> grad_ei(dim);

  // set up evaluators and state to check results
  std::vector<double> union_of_points((num_being_sampled+num_samples_to_generate)*dim);
  std::copy(best_points_to_sample.begin(), best_points_to_sample.end(), union_of_points.begin());
  std::copy(points_being_sampled.begin(), points_being_sampled.end(), union_of_points.begin() + dim*num_samples_to_generate);

  std::vector<double> union_of_points_grid_search((num_being_sampled+num_samples_to_generate)*dim);
  std::copy(grid_search_best_point_set.begin(), grid_search_best_point_set.end(), union_of_points_grid_search.begin());
  std::copy(points_being_sampled.begin(), points_being_sampled.end(), union_of_points_grid_search.begin() + dim*num_samples_to_generate);

  double tolerance_result = tolerance;
  {
    max_int_steps = 10000000;  // evaluate the final results with high accuracy
    tolerance_result = 3.0e-4;  // reduce b/c we cannot achieve full accuracy in the monte-carlo case
    // while still having this test run in a reasonable amt of time
    ExpectedImprovementEvaluator ei_evaluator(*mock_gp_data.gaussian_process_ptr, max_int_steps, mock_gp_data.best_so_far);
    ExpectedImprovementEvaluator::StateType ei_state(ei_evaluator, union_of_points.data(), num_being_sampled + 1, true, normal_rng_vec.data());

    ei_optimized = ei_evaluator.ComputeExpectedImprovement(&ei_state);
    ei_evaluator.ComputeGradExpectedImprovement(&ei_state, grad_ei.data());

    ei_state.UpdateCurrentPoint(ei_evaluator, union_of_points_grid_search.data());
    ei_grid_search = ei_evaluator.ComputeExpectedImprovement(&ei_state);
  }

  printf("optimized EI: %.18E, grid_search_EI: %.18E\n", ei_optimized, ei_grid_search);
  printf("grad_EI: "); PrintMatrix(grad_ei.data(), 1, dim);

  if (ei_optimized < ei_grid_search) {
    ++total_errors;
  }

  current_errors = 0;
  for (const auto& entry : grad_ei) {
    if (!CheckDoubleWithinRelative(entry, 0.0, tolerance_result)) {
      ++current_errors;
    }
  }
  total_errors += current_errors;

  return total_errors;
}

int EvaluateEIAtPointListTest() {
  using DomainType = TensorProductDomain;
  const int dim = 3;

  int total_errors = 0;

  // grid search parameters
  int num_grid_search_points = 100000;

  // EI computation parameters
  int num_being_sampled = 0;
  int max_int_steps = 1000;

  // random number generators
  UniformRandomGenerator uniform_generator(314);
  boost::uniform_real<double> uniform_double_hyperparameter(0.4, 1.3);
  boost::uniform_real<double> uniform_double_lower_bound(-2.0, 0.5);
  boost::uniform_real<double> uniform_double_upper_bound(2.0, 3.5);

  const int64_t pi_array[] = {314, 3141, 31415, 314159, 3141592, 31415926, 314159265, 3141592653, 31415926535, 314159265359};
  static const int kMaxNumThreads = 4;
  std::vector<NormalRNG> normal_rng_vec(kMaxNumThreads);
  for (int j = 0; j < kMaxNumThreads; ++j) {
    normal_rng_vec[j].SetExplicitSeed(pi_array[j]);
  }

  int num_sampled = 20;  // arbitrary
  std::vector<double> noise_variance(num_sampled, 0.002);
  MockGaussianProcessPriorData<DomainType> mock_gp_data(SquareExponential(dim, 1.0, 1.0), noise_variance, dim, num_sampled, uniform_double_lower_bound, uniform_double_upper_bound, uniform_double_hyperparameter, &uniform_generator);

  // no parallel experiments
  num_being_sampled = 0;
  std::vector<double> points_being_sampled(dim*num_being_sampled);

  std::vector<double> next_point(dim);
  bool found_flag = false;
  std::vector<double> grid_search_best_point(dim);
  std::vector<double> function_values(num_grid_search_points);
  std::vector<double> initial_guesses(dim*num_grid_search_points);
  num_grid_search_points = mock_gp_data.domain_ptr->GenerateUniformPointsInDomain(num_grid_search_points, &uniform_generator, initial_guesses.data());

  EvaluateEIAtPointList(*mock_gp_data.gaussian_process_ptr, *mock_gp_data.domain_ptr, initial_guesses.data(), points_being_sampled.data(), num_grid_search_points, num_being_sampled, mock_gp_data.best_so_far, max_int_steps, kMaxNumThreads, &found_flag, normal_rng_vec.data(), function_values.data(), grid_search_best_point.data());
  if (!found_flag) {
    ++total_errors;
  }

  // find the max function_value and the index at which it occurs
  auto max_value_ptr = std::max_element(function_values.begin(), function_values.end());
  auto max_index = std::distance(function_values.begin(), max_value_ptr);

  // check that EvaluateEIAtPointList found the right point
  for (int i = 0; i < dim; ++i) {
    if (!CheckDoubleWithin(grid_search_best_point[i], initial_guesses[max_index*dim + i], 0.0)) {
      ++total_errors;
    }
  }

  // now check multi-threaded & single threaded give the same result
  {
    std::vector<double> grid_search_best_point_single_thread(dim);
    std::vector<double> function_values_single_thread(num_grid_search_points);
    int single_thread = 1;
    found_flag = false;
    EvaluateEIAtPointList(*mock_gp_data.gaussian_process_ptr, *mock_gp_data.domain_ptr, initial_guesses.data(), points_being_sampled.data(), num_grid_search_points, num_being_sampled, mock_gp_data.best_so_far, max_int_steps, single_thread, &found_flag, normal_rng_vec.data(), function_values_single_thread.data(), grid_search_best_point_single_thread.data());

    // check against multi-threaded result matches single
    for (int i = 0; i < dim; ++i) {
      if (!CheckDoubleWithin(grid_search_best_point[i], grid_search_best_point_single_thread[i], 0.0)) {
        ++total_errors;
      }
    }

    // check all function values match too
    for (int i = 0; i < num_grid_search_points; ++i) {
      if (!CheckDoubleWithin(function_values[i], function_values_single_thread[i], 0.0)) {
        ++total_errors;
      }
    }
  }

  return total_errors;
}

int ExpectedImprovementOptimizationTest(DomainTypes domain_type, ExpectedImprovementEvaluationMode ei_mode) {
  switch (domain_type) {
    case DomainTypes::kTensorProduct: {
      switch (ei_mode) {
        case ExpectedImprovementEvaluationMode::kAnalytic:
        case ExpectedImprovementEvaluationMode::kMonteCarlo: {
          return ExpectedImprovementOptimizationTestCore(ei_mode);
        }
        default: {
          OL_ERROR_PRINTF("%s: INVALID ei_mode choice: %d\n", OL_CURRENT_FUNCTION_NAME, ei_mode);
          return 1;
        }
      }  // end switch over ei_mode
    }  // end case kTensorProduct
    case DomainTypes::kSimplex: {
      switch (ei_mode) {
        case ExpectedImprovementEvaluationMode::kAnalytic:
        case ExpectedImprovementEvaluationMode::kMonteCarlo: {
          return ExpectedImprovementOptimizationSimplexTestCore(ei_mode);
        }
        default: {
          OL_ERROR_PRINTF("%s: INVALID ei_mode choice: %d\n", OL_CURRENT_FUNCTION_NAME, ei_mode);
          return 1;
        }
      }  // end switch over ei_mode
    }  // end case kSimplex
    default: {
      OL_ERROR_PRINTF("%s: INVALID domain_type choice: %d\n", OL_CURRENT_FUNCTION_NAME, domain_type);
      return 1;
    }
  }  // end switch over domain_type
}

}  // end namespace optimal_learning
