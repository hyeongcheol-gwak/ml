# Problem 1-1

**Used LLM** :  claude Opus 4.5

**Code** :  
```python
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
```


**Code Description** :  
Hard-margin SVM의 primal 문제를 QP(Quadratic Programming) 형식으로 변환하여 풀이하는 코드이다. 먼저 레이블을 {0, 1}에서 {-1, +1}로 변환한다. 결정 변수를 $z = [w, b]$로 정의하고, 목적 함수 $\min \frac{1}{2}\|w\|^2$를 QP 형태 $\frac{1}{2}z^T P z + q^T z$로 변환한다. 이를 위해 행렬 $P$는 $(n_{features}+1) \times (n_{features}+1)$ 크기로 생성하되 $w$ 부분에만 단위행렬을, $b$ 부분은 0으로 설정하여 $b$가 목적 함수에 영향을 주지 않도록 한다. 선형 항 $q$는 영벡터이다. 제약 조건 $t_i(w^T x_i + b) \ge 1$은 $Gz \le h$ 형식으로 변환하며, 각 행에 $-t_i \cdot [x_i, 1]$을 넣고 $h = -\mathbf{1}$로 설정한다. 제약 행렬 $G$를 구성할 때 for 루프를 사용하여 각 샘플별로 행을 설정한다. `calculate_weights_from_primal_solution`에서는 solution 벡터의 마지막 원소 이전까지를 $w$, 마지막 원소를 $b$로 단순 분리한다.


---

**Used LLM** :  Gemini 3 Pro

**Code** :  
```python
def solve_primal_opt(
    X: np.ndarray,
    y: np.ndarray
    ) -> np.ndarray:
    ################# YOUR CODE COMES HERE ######################
    # Convert labels: 0 -> -1, 1 -> +1
    t = 2 * y - 1
    n, d = X.shape

    # QP objective: minimize 0.5 * z^T P z + q^T z  where z = [w; b]
    P = np.block([
        [np.eye(d), np.zeros((d, 1))],
        [np.zeros((1, d)), np.zeros((1, 1))]
    ])
    q = np.zeros(d + 1)

    # Inequality constraint: -t_i * (x_i^T w + b) <= -1 for all i
    G = -np.diag(t) @ np.hstack([X, np.ones((n, 1))])
    h = -np.ones(n)
    #############################################################

    P = csc_matrix(P)
    G = csc_matrix(G)

    sol = solve_qp(P, q, G, h, A=None, b=None, solver="osqp")
    return sol

def calculate_weights_from_primal_solution(
    solution: np.ndarray
    ) -> Tuple[np.ndarray, float]:
    ################# YOUR CODE COMES HERE ######################
    # w: coefficient of the model to input features,
    # b: bias of the model
    w = solution[:len(solution) - 1]
    b = float(solution[-1])
    #############################################################
    return w, b
```


**Code Description** :  
Gemini 버전은 Claude와 동일한 수학적 정식화를 따르지만, 제약 행렬 $G$의 구성 방식에 차이가 있다. Claude가 for 루프를 사용하여 각 샘플별로 $G$의 행을 설정한 반면, Gemini는 벡터화 연산을 활용한다. 먼저 `np.hstack([X, np.ones((n, 1))])`로 데이터에 1 열을 붙여 augmented 행렬을 만든 뒤, `np.diag(t)`를 곱하여 한 번에 모든 행을 계산한다. 이 방식은 Python 레벨의 반복문을 제거하여 대규모 데이터에서 더 효율적이다. 또한 $P$ 행렬 구성에 `np.block`을 사용하여 블록 행렬 구조를 더 명시적으로 표현하였다. `calculate_weights_from_primal_solution`에서는 `solution[:len(solution) - 1]` 형태로 슬라이싱하고, $b$를 `float()`로 명시적 스칼라 변환하는 점이 다르나, 최종 결과는 동일하다.


---------------------
# Problem 1-2

**Used LLM** :  claude Opus 4.5

**Code** :  
```python
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
```


**Code Description** :  
Hard-margin SVM의 dual 문제를 QP 형식으로 풀이하는 코드이다. Dual 문제는 $\max_\alpha \sum \alpha_i - \frac{1}{2}\alpha^T H \alpha$ (단, $H_{ij} = t_i t_j x_i^T x_j$)이며, 이를 최소화 문제 $\min \frac{1}{2}\alpha^T H \alpha - \mathbf{1}^T \alpha$로 변환한다. 커널 행렬 $H$는 `np.outer(t, t) * (X @ X.T)`로 한 줄에 계산한다. 부등식 제약 조건 $\alpha_i \ge 0$은 $-I\alpha \le 0$으로, 등식 제약 조건 $\sum \alpha_i t_i = 0$은 $A = t^T$, $b = 0$으로 설정한다. `calculate_weights_from_dual_solution`에서 $w = \sum \alpha_i t_i x_i$를 벡터 연산 `(alpha * t) @ X`으로 계산하고, support vector는 $\alpha > 10^{-5}$ 임계값으로 판별한다. $b$는 모든 support vector에 대해 $t_s - w^T x_s$의 평균을 취하여 수치적 안정성을 확보한다.


---

**Used LLM** :  gemini 3 pro

**Code** :  
```python
def solve_dual_opt(
    X: np.ndarray,
    y: np.ndarray
    ) -> np.ndarray:
    ################# YOUR CODE COMES HERE ######################
    # Map labels to {-1, +1}
    t = 2 * y - 1
    n = len(y)

    # Kernel matrix H: H_ij = t_i * t_j * <x_i, x_j>
    K = X.dot(X.T)
    H = np.outer(t, t) * K
    P = H.astype(np.float64)
    q = -np.ones(n)

    # Constraint: alpha_i >= 0  =>  -alpha_i <= 0
    G = -np.eye(n)
    h = np.zeros(n)

    # Equality: t^T alpha = 0
    A = t.astype(np.float64).reshape(1, n)
    b = np.array([0.0])
    #############################################################

    P = csc_matrix(P)
    G = csc_matrix(G)
    A = csc_matrix(A)

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
    t = 2 * y - 1
    alpha = solution
    # Compute weight vector
    w = X.T @ (alpha * t)
    # Identify support vectors using threshold
    sv_mask = alpha > 1e-4
    # Compute bias from support vectors
    b = np.mean(t[sv_mask] - X[sv_mask] @ w)
    #############################################################
    return w, b
```


**Code Description** :  
Gemini 버전은 Claude와 동일한 dual QP 정식화를 따르지만, 몇 가지 구현 방식에 차이가 있다. 첫째, 커널 행렬 계산 시 Claude는 `X @ X.T`를 사용한 반면, Gemini는 `X.dot(X.T)`를 사용하여 `.dot()` 메서드로 행렬 곱을 수행한다(결과는 동일). 둘째, $w$ 계산에서 Claude는 `(alpha * t) @ X` (행 벡터 × 행렬)을 사용한 반면, Gemini는 `X.T @ (alpha * t)` (행렬의 전치 × 열 벡터) 형태로 계산한다. 수학적으로 $w = X^T(\alpha \circ t)$와 $w = (\alpha \circ t)^T X$는 동일한 결과를 준다. 셋째, support vector 판별 임계값이 Claude는 $10^{-5}$인 반면 Gemini는 $10^{-4}$로 약간 더 큰 값을 사용하여 수치 오차에 더 관대하다. 또한 등식 제약의 $b$ 값을 `np.array([0.0])`으로 명시적 배열로 선언하는 점이 다르다.


---------------------
# Problem 2

**Plot** :
![problem 2](outputs/problem_2.png)

**Analysis** :
4가지 서로 다른 C 값(0.01, 0.1, 1, 100)에 따른 soft-margin SVM의 decision boundary 변화를 분석한다. 분홍색 원으로 표시된 점들은 margin 내부에 위치하거나 margin을 위반하는 샘플들(support vector 후보)이다.

- **C = 0.01**: C가 매우 작아 margin 위반에 대한 패널티가 약하다. 그 결과 margin이 매우 넓어지며, 대부분의 데이터 포인트가 margin 내부에 위치한다(분홍색 원이 많음). 분류 오류를 어느 정도 허용하면서 넓은 margin을 확보하는 경향이 강하다. Decision boundary의 기울기도 다른 C 값들과 상이하며, 일부 오분류가 발생할 수 있다.

- **C = 0.1**: C가 약간 증가하여 margin 위반에 대한 패널티가 강해졌다. Margin이 C=0.01보다 좁아졌으며, margin 내부의 support vector 수도 줄어들었다. 하지만 여전히 상당수의 샘플이 margin 내부에 위치한다.

- **C = 1**: 적절한 수준의 C 값으로, margin 폭과 분류 정확도 사이의 균형이 잡힌다. Margin 내부의 support vector가 더 줄어들었으며, decision boundary가 두 클래스를 잘 분리한다.

- **C = 100**: C가 매우 커서 margin 위반을 거의 허용하지 않는다. Margin이 좁아지며, support vector 수가 최소화된다. Hard-margin SVM에 근접한 결과를 보이며, 일부 경계 부근의 핵심 샘플들만 support vector로 남는다. 그러나 이 경우 과적합(overfitting)의 위험이 있으며, noise에 민감해질 수 있다.

결론적으로, C가 증가할수록 margin은 좁아지고 margin 위반의 허용도가 낮아진다. C 값이 작으면 더 일반화된(regularized) 모델을, C 값이 크면 학습 데이터에 더 정확히 맞추는 모델을 얻는다.


---------------------
# Problem 3-1

**Plot** :
![problem 3-1](outputs/problem_3_1.png)

**Analysis** :
Polynomial kernel SVM의 세 가지 hyperparameter(degree $d$, $C$, $coef0$)의 조합에 따른 비선형 decision boundary의 변화를 분석한다. 총 8개의 조합($d \in \{3, 5\}$, $C \in \{0.1, 10\}$, $coef0 \in \{0, 1\}$)을 비교한다.

- **degree의 영향**: $d=3$에서 $d=5$로 증가하면 decision boundary가 더 복잡한 형태를 띈다. $d=3$일 때는 비교적 부드러운 곡선으로 클래스를 분리하는 반면, $d=5$일 때는 경계가 더 구불구불해지며 데이터의 세부적인 분포에 더 민감하게 반응한다. 특히 $d=5$, $C=10$인 경우 경계가 매우 복잡해져 과적합의 징후를 보인다.

- **C의 영향**: $C=0.1$에서 $C=10$으로 증가하면 margin 위반에 대한 패널티가 강해진다. $C=0.1$일 때는 비교적 단순하고 일반화된 경계를 형성하지만, $C=10$일 때는 학습 데이터의 분포를 더 정밀하게 따라가는 복잡한 경계를 형성한다.

- **coef0의 영향**: Polynomial kernel $K(x, x') = (\gamma x^T x' + coef0)^d$에서 $coef0$는 커널 함수의 독립항(independent term)을 결정한다. $coef0=0$일 때는 homogeneous polynomial kernel이 되고, $coef0=1$일 때는 inhomogeneous polynomial kernel이 된다. 두 경우를 비교하면, $coef0=1$일 때 decision boundary가 약간 더 변화하고 대칭성이 달라지는 것을 확인할 수 있으며, 특히 고차원($d=5$)에서 그 차이가 더 두드러진다. $coef0=1$은 저차 다항식 항도 커널에 포함시키므로, 경계의 유연성이 약간 달라진다.

- **종합**: $d$가 커지고 $C$가 커질수록 모델의 복잡도와 학습 데이터에 대한 적합도가 높아지지만, 동시에 과적합의 위험도 증가한다. 실제 일반화 성능을 위해서는 적절한 $d$와 $C$ 값의 조합을 교차 검증 등을 통해 선택해야 한다.


---------------------
# Problem 3-2

**Plot** :
![problem 3-2](outputs/problem_3_2.png)

**Analysis** :
RBF(Gaussian Radial Basis Function) kernel SVM의 두 가지 hyperparameter($\gamma$, $C$)의 조합에 따른 비선형 decision boundary 변화를 분석한다. 총 4개의 조합($\gamma \in \{0.1, 10\}$, $C \in \{0.1, 100\}$)을 비교한다.

- **gamma = 0.1, C = 0.1** (좌상단): $\gamma$와 $C$ 모두 작은 경우이다. $\gamma$가 작으면 RBF 커널의 영향 반경이 넓어져 각 데이터 포인트가 넓은 범위에 영향을 미치고, $C$가 작으면 margin 위반을 많이 허용한다. 그 결과 decision boundary가 매우 부드럽고 단순한 형태를 보이며, 두 클래스를 대략적으로만 분리한다. 가장 과소적합(underfitting)에 가까운 형태이다.

- **gamma = 10, C = 0.1** (우상단): $\gamma$가 크면 각 데이터 포인트의 영향 반경이 좁아져 가까운 이웃만을 고려한다. 그러나 $C$가 작아 패널티가 약하므로 복잡한 경계를 만들지 못한다. 결과적으로 전체 영역이 거의 한 클래스로 분류되는 현상이 나타나며, 모델이 데이터를 제대로 분리하지 못하는 독특한 양상을 보인다.

- **gamma = 0.1, C = 100** (좌하단): $\gamma$가 작아 부드러운 경계를 형성하되, $C$가 커서 대부분의 데이터를 올바르게 분류하려 한다. 비교적 부드러우면서도 데이터의 분포를 잘 반영하는 경계가 형성되어, 과적합과 과소적합 사이의 적절한 균형을 보인다.

- **gamma = 10, C = 100** (우하단): $\gamma$와 $C$ 모두 큰 경우이다. 각 데이터 포인트의 영향 반경이 좁고 패널티도 강하므로, 각 학습 데이터 포인트 주변에 좁은 "섬(island)" 형태의 decision region이 형성된다. 이는 전형적인 과적합(overfitting)의 양상으로, 학습 데이터에는 거의 완벽히 맞지만 새로운 데이터에 대한 일반화 성능은 떨어질 수 있다.

- **종합**: $\gamma$는 모델의 복잡도를, $C$는 오분류 허용도를 조절한다. $\gamma$가 커지면 경계가 복잡해지고(각 데이터 포인트에 민감), $C$가 커지면 학습 데이터에 더 정확히 맞추려 한다. 최적의 일반화 성능을 위해서는 두 하이퍼파라미터의 적절한 균형이 필요하며, 좌하단($\gamma=0.1$, $C=100$)의 경우가 이 데이터셋에서 비교적 좋은 균형을 보인다.


---------------------