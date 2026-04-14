import numpy as np
from typing import Tuple

from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator

################################### Problem 1-1 ###################################

def solve_primal_opt(
    X: np.ndarray,
    y: np.ndarray
    ) -> np.ndarray:
    ################# YOUR CODE COMES HERE ######################

    #############################################################

    P = csc_matrix(P) # Convert dense numpy array to sparse CSC matrix for efficient computation
    G = csc_matrix(G) # Convert dense numpy array to sparse CSC matrix for efficient computation

    sol = solve_qp(P, q, G, h, A=None, b=None, solver="osqp")
    return sol

def calculate_weights_from_primal_solution(
    solution: np.ndarray
    ) -> Tuple[np.ndarray, float]:
    ################# YOUR CODE COMES HERE ######################
    # w: coefficient of the model to input features,
    # b: bias of the model

    #############################################################
    return w, b

################################### Problem 1-2 ###################################

def solve_dual_opt(
    X: np.ndarray,
    y: np.ndarray
    ) -> np.ndarray:
    ################# YOUR CODE COMES HERE ######################

    #############################################################

    P = csc_matrix(P) # Convert dense numpy array to sparse CSC matrix for efficient computation
    G = csc_matrix(G) # Convert dense numpy array to sparse CSC matrix for efficient computation
    A = csc_matrix(A) # Convert dense numpy array to sparse CSC matrix for efficient computation

    sol = solve_qp(P, q, G, h, A, b, solver="osqp")
    return sol

def calculate_weights_from_dual_solution(
    solution: np.ndarray,
    X: np.ndarray,
    y: np.ndarray
    ) -> Tuple[np.ndarray, float]:
    ################# YOUR CODE COMES HERE ######################
    # w: coefficient of the model to input features,
    # b: bias of the model

    #############################################################
    return w, b

################################### Problem 2 ###################################

def learn_svm_clf(
    X: np.ndarray,
    y: np.ndarray,
    C: float
    ) -> LinearSVC:
    #################### YOUR CODE COMES HERE ####################

    ##############################################################
    return model

################################### Problem 3-1 ###################################

def learn_poly_kernel_svm_clf(
    X: np.ndarray,
    y: np.ndarray,
    C: float,
    degree: int,
    coef0: float
    ) -> SVC:
    #################### YOUR CODE COMES HERE ####################

    ##############################################################
    return model

################################### Problem 3-2 ###################################

def learn_rbf_kernel_svm_clf(
    X: np.ndarray,
    y: np.ndarray,
    C: float,
    gamma: float
    ) -> SVC:
    #################### YOUR CODE COMES HERE ####################

    ##############################################################
    return model

################################### Problem 3-3 ###################################

def learn_kernel_svm_clf_best(
    X: np.ndarray,
    y: np.ndarray
    ) -> BaseEstimator:
    #################### YOUR CODE COMES HERE ####################

    ##############################################################
    return model