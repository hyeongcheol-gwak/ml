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
    # Convert labels from {0, 1} to {-1, +1}
    t = y * 2 - 1
    n_samples, n_features = X.shape

    # Decision variable: z = [w (n_features), b (1)]
    # Objective: minimize 0.5 * ||w||^2 = 0.5 * z^T P z + q^T z
    # P is (n_features+1) x (n_features+1), only the w part has identity
    P = np.zeros((n_features + 1, n_features + 1))
    P[:n_features, :n_features] = np.eye(n_features)
    q = np.zeros(n_features + 1)

    # Constraints: t_i * (w^T x_i + b) >= 1  =>  -t_i * (w^T x_i + b) <= -1
    # G z <= h  where G_i = -t_i * [x_i, 1], h_i = -1
    G = np.zeros((n_samples, n_features + 1))
    for i in range(n_samples):
        G[i, :n_features] = -t[i] * X[i]
        G[i, n_features] = -t[i]
    h = -np.ones(n_samples)
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
    w = solution[:-1]
    b = solution[-1]
    #############################################################
    return w, b

################################### Problem 1-2 ###################################

def solve_dual_opt(
    X: np.ndarray,
    y: np.ndarray
    ) -> np.ndarray:
    ################# YOUR CODE COMES HERE ######################
    # Convert labels from {0, 1} to {-1, +1}
    t = y * 2 - 1
    n_samples = X.shape[0]

    # Dual QP: maximize sum(alpha) - 0.5 * alpha^T H alpha
    # Equivalent to minimize 0.5 * alpha^T H alpha - 1^T alpha
    # where H_ij = t_i * t_j * x_i^T x_j
    H = np.outer(t, t) * (X @ X.T)
    P = H.astype(np.float64)
    q = -np.ones(n_samples)

    # Inequality constraints: alpha_i >= 0  =>  -alpha_i <= 0
    G = -np.eye(n_samples)
    h = np.zeros(n_samples)

    # Equality constraint: sum(alpha_i * t_i) = 0
    A = t.reshape(1, -1).astype(np.float64)
    b = np.zeros(1)
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
    t = y * 2 - 1
    alpha = solution
    # w = sum(alpha_i * t_i * x_i)
    w = (alpha * t) @ X
    # Find support vectors (alpha > threshold)
    sv_idx = alpha > 1e-5
    # b = t_s - w^T x_s for any support vector s
    b = np.mean(t[sv_idx] - X[sv_idx] @ w)
    #############################################################
    return w, b

################################### Problem 2 ###################################

def learn_svm_clf(
    X: np.ndarray,
    y: np.ndarray,
    C: float
    ) -> LinearSVC:
    #################### YOUR CODE COMES HERE ####################
    model = LinearSVC(C=C, loss="hinge", max_iter=10000)
    model.fit(X, y)
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
    model = SVC(kernel="poly", C=C, degree=degree, coef0=coef0)
    model.fit(X, y)
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
    model = SVC(kernel="rbf", C=C, gamma=gamma)
    model.fit(X, y)
    ##############################################################
    return model

################################### Problem 3-3 ###################################

def learn_kernel_svm_clf_best(
    X: np.ndarray,
    y: np.ndarray
    ) -> BaseEstimator:
    #################### YOUR CODE COMES HERE ####################
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10, gamma=1))
    ])
    model.fit(X, y)
    ##############################################################
    return model