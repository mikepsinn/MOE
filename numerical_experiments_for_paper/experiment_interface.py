import numpy
import time

from test_functions import Branin

from moe.optimal_learning.python.data_containers import HistoricalData
from moe.optimal_learning.python.geometry_utils import ClosedInterval

from moe.optimal_learning.python.python_version.domain import TensorProductDomain as pythonTensorProductDomain
from moe.optimal_learning.python.python_version.gaussian_process import GaussianProcess as pythonGaussianProcess
from moe.optimal_learning.python.python_version.expected_improvement import ExpectedImprovement as pythonExpectedImprovement
from moe.optimal_learning.python.python_version.covariance import SquareExponential as pythonSquareExponential

from moe.optimal_learning.python.cpp_wrappers.domain import TensorProductDomain as cppTensorProductDomain
from moe.optimal_learning.python.cpp_wrappers.gaussian_process import GaussianProcess as cppGaussianProcess
from moe.optimal_learning.python.cpp_wrappers.log_likelihood import GaussianProcessLogLikelihood as cppGaussianProcessLogLikelihood
from moe.optimal_learning.python.cpp_wrappers.covariance import SquareExponential as cppSquareExponential
from moe.optimal_learning.python.cpp_wrappers.optimization import NewtonOptimizer as cppNewtonOptimizer
from moe.optimal_learning.python.cpp_wrappers.optimization import GradientDescentOptimizer as cppGradientDescentOptimizer
from moe.optimal_learning.python.cpp_wrappers.optimization import NewtonParameters as cppNewtonParameters
from moe.optimal_learning.python.cpp_wrappers.optimization import GradientDescentParameters as cppGradientDescentParameters
from moe.optimal_learning.python.cpp_wrappers.log_likelihood import multistart_hyperparameter_optimization
from moe.optimal_learning.python.cpp_wrappers.expected_improvement import ExpectedImprovement as cppExpectedImprovement
from moe.optimal_learning.python.cpp_wrappers.expected_improvement import multistart_expected_improvement_optimization
from moe.optimal_learning.python.cpp_wrappers.expected_improvement import constant_liar_expected_improvement_optimization
from moe.optimal_learning.python.cpp_wrappers.expected_improvement import kriging_believer_expected_improvement_optimization

def optimization_method(experiment, method_name, num_to_sample, num_threads, num_mc_itr, dum_search_itr=100000, lie_value=0):
    """Perform a full round of generating points_to_sample, given the optimization method to use

    :param experiment: object of class NumericalExperiment
    :type experiment: NumericalExperiment
    :param method_name: name of the optimization method to use
    :type method_name: string
    :param num_to_sample: number of points generated to sample
    :type num_to_sample: int > 0
    :param num_threads: number of threads for multi-threading
    :type num_threads: int > 0
    :param num_mc_itr: number of iterations for MC simulation
    :type num_mc_itr: int > 0
    :param dum_search_itr: number of points generated for dummy search
    :type dum_search_itr: int > 0
    :param lie_value: used for constant lier method
    :type lie_value: double
    """
    if (method_name == "exact_qEI"):
        python_gp = pythonGaussianProcess(experiment._python_cov, experiment._historical_data)
        points_to_sample = []   #TODO wait till dennitz finish BFGS
        return points_to_sample
    else:
        cpp_gp = cppGaussianProcess(experiment._cpp_cov, experiment._historical_data)
        ei_evaluator = cppExpectedImprovement(gaussian_process=cpp_gp, num_mc_iterations=num_mc_itr)
        optimizer = cppGradientDescentOptimizer(experiment._cpp_search_domain, ei_evaluator, experiment._gd_opt_parameters, dum_search_itr)
        if (method_name == "epi_cpu"):
            return multistart_expected_improvement_optimization(optimizer, 0, num_to_sample, max_num_threads=num_threads)
        elif (method_name == "CL"):
            return constant_liar_expected_improvement_optimization(optimizer, 0, num_to_sample, lie_value, max_num_threads=num_threads)
        elif (method_name == "KB"):
            return kriging_believer_expected_improvement_optimization(optimizer, 0, num_to_sample, max_num_threads=num_threads)
        else:
            raise NotImplementedError("Not a valid optimization method")

class NumericalExperiment():

    r"""Class of setting up numerical experiment

    """

    def __init__(self, dim, search_domain_bounds, hyper_domain_bounds, num_init_points, objective_function):
        self._dim = dim
        self._sample_var = 0.01
        self._objective_function = objective_function

        # domain
        self._python_search_domain = pythonTensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in search_domain_bounds])
        self._cpp_search_domain = cppTensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in search_domain_bounds])
        self._cpp_hyper_domain = cppTensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in hyper_domain_bounds])
        # optimization parameters
        self._newton_opt_parameters = cppNewtonParameters(num_multistarts=100, max_num_steps=200, gamma=1.01, time_factor=1.0e-3, max_relative_change=1.0, tolerance=1.0e-10)
        self._gd_opt_parameters = cppGradientDescentParameters(num_multistarts=100, max_num_steps=200, max_num_restarts=4, num_steps_averaged=20, gamma=0.7, pre_mult=1.0, max_relative_change=0.7, tolerance=1.0e-7)
        # initial sampled points
        self._init_points = self._python_search_domain.generate_uniform_random_points_in_domain(num_init_points)
        self._reset_state()

    def _reset_state(self):
        self._historical_data = HistoricalData(self._dim)
        for point in self._init_points:
            self._historical_data.append_sample_points([[point, self._objective_function(point), self._sample_var],])
        self._python_cov = pythonSquareExponential(numpy.ones(self._dim + 1))
        self._cpp_cov = cppSquareExponential(numpy.ones(self._dim + 1))

    def _update_state(self, update_hyper, max_num_thread, points_to_sample = None):
        if (points_to_sample is not None):
            for point in points_to_sample:
                self._historical_data.append_sample_points([[point, self._objective_function(point), self._sample_var],])
        if (update_hyper == True):
            cpp_gp_loglikelihood = cppGaussianProcessLogLikelihood(self._cpp_cov, self._historical_data)
            newton_optimizer = cppNewtonOptimizer(self._cpp_hyper_domain, cpp_gp_loglikelihood, self._newton_opt_parameters)
            best_param = multistart_hyperparameter_optimization(newton_optimizer, 0, max_num_threads=max_num_threads)
            self._cpp_cov.set_hyperparameters(best_param)
            self._python_cov.set_hyperparameters(best_param)

    def compare_best_so_far(self, method_name, hyper_skip_turns, num_itr, num_to_sample, num_threads, num_mc_itr=1000000, dum_search_itr=100000, lie_value=0.0):
        best_so_far = []
        num_func_eval = []
        self._reset_state()
        self._update_state(True, num_threads)
        for itr in range(num_itr):
            print "{0}th iteration\n".format(itr)
            points_to_sample = optimization_method(self, method_name, num_to_sample, num_threads, num_mc_itr, dum_search_itr, lie_value)
            self._update_state((itr+1)%hyper_skip_turns==0, num_threads, points_to_sample=points_to_sample)
            # record num_func_eval & best_so_far
            num_func_eval.append(itr * num_to_sample)
            best_so_far.append(numpy.amin(self._historical_data.points_sampled_value))
        return num_func_eval, best_so_far

    def timing_analytic_vs_gpu_qp_ei(self, num_to_sample_table, which_gpu):
        tol = 1e-15
        python_gp = pythonGaussianProcess(self._python_cov, self._historical_data)
        cpp_gp = cppGaussianProcess(self._cpp_cov, self._historical_data)

        analytic_time_table = numpy.zeros(len(num_to_sample_table))
        gpu_time_table = numpy.zeros(len(num_to_sample_table))
        ei_diff_table = numpy.zeros(len(num_to_sample_table))
        analytic_ei_table = numpy.zeros(len(num_to_sample_table))
        gpu_ei_table = numpy.zeros(len(num_to_sample_table))
        for i, num_to_sample in enumerate(num_to_sample_table):
            points_to_sample = self._python_search_domain.generate_uniform_random_points_in_domain(num_to_sample)
            # for analytic
            ei_evaluator = pythonExpectedImprovement(python_gp, points_to_sample)
            mu_star = python_gp.compute_mean_of_points(points_to_sample)
            var_star = python_gp.compute_variance_of_points(points_to_sample)
            start_time = time.time()
            analytic_ei_table[i] = ei_evaluator._compute_expected_improvement_qd_analytic(mu_star, var_star)
            end_time = time.time()
            analytic_time_table[i] = end_time - start_time
            # for gpu
            gpu_ei_evaluator = cppExpectedImprovement(cpp_gp, points_to_sample, num_mc_iterations=10000000)
            ei_and_time = gpu_ei_evaluator.time_expected_improvement(use_gpu=True, which_gpu=which_gpu, num_repeat=5)
            gpu_time_table[i] = ei_and_time[1]
            gpu_ei_table[i] = ei_and_time[0]
            if (gpu_ei_table[i] < tol):
                ei_diff_table[i] = abs(analytic_ei_table[i] - gpu_ei_table[i])
            else:
                ei_diff_table[i] = abs(analytic_ei_table[i] - gpu_ei_table[i]) / gpu_ei_table[i]

        return gpu_time_table, analytic_time_table, analytic_ei_table, gpu_ei_table, ei_diff_table

if __name__ == "__main__":
    num_to_sample_table = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    experiment = NumericalExperiment(dim = 2, search_domain_bounds = [[0,10],[0,10]], hyper_domain_bounds = [[1,10],[1,10],[1,10]], num_init_points = 20, objective_function = Branin)
    gpu_time, analytic_time, analytic_ei, gpu_ei, ei_diff = experiment.timing_analytic_vs_gpu_qp_ei(num_to_sample_table, 0)
    print gpu_time
    print analytic_time
    print ei_diff


