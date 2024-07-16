
# Scikit-learn

## 1. Scikit-learn 소개

    - Scikit-learn은 파이썬에서 사용되는 머신러닝 라이브러리로, 다양한 분류, 회귀, 클러스터링 알고리즘을 제공

## 2. 설치 방법

```bash
pip install scikit-learn
```

## 3. 기본 사용법

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 데이터셋 로드
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 데이터셋을 학습용과 테스트용으로 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 모델 생성 및 학습
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# 예측 및 정확도 평가
y_pred = knn.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
```

## 4. 주요 알고리즘

    - Scikit-learn의 다양한 알고리즘 소개

### 4.1. 분류 (Classification)

- 데이터를 미리 정의된 클래스 레이블 중 하나로 예측하는 작업:

- [K-최근접 (KNeighbors)](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier)
    - 입력한 데이터에 가장 가까운 이웃을 찾아 거리와 이웃 샘플의 인덱스를 반환

```python
from sklearn.neighbors import KNeighborsClassifier
# 기본 특성값: (n_neighbors=5, *, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
```

- [로지스틱 회귀 (Logistic Regression)](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression)
    - 선형 분류 알고리즘으로 선을 통해 클래스를 분류하는 방식으로 클래스 확률을 출력
    - 이진 분류일때 시그모이드 함수, 다중 분류일때 소프트맥스 함수 사용

```python
from sklearn.linear_model import LinearRegression
# 기본 특성값: (penalty='l2', *, dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='deprecated', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)
```

- [확률적 경사 하강법 (Stochastic Gradient Descent)](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier)
    - 훈련 세트에서 샘플 하나씩 꺼내 손실 함수의 경사를 따라 최적의 모델을 찾는 알고리즘
    - 2개의 매개변수를 지정하여 loss 를 통해 손실함수를 지정하고, max_iter 를 통해 에포크 횟수를 지정

```python
from sklearn.linear_model import SGDClassifier
# 예시 SGDClassifier(loss='log_loss', max_iter=10, random_state=42)
# 기본 특성값: (loss='hinge', *, penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=1000, tol=0.001, shuffle=True, verbose=0, epsilon=0.1, n_jobs=None, random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, class_weight=None, warm_start=False, average=False)
```

- [결정 트리 (Decision Trees)](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier)
    - 예/아니오에 대한 질문을 이어나가면서 정답을 찾아 학습하는 알고리즘

```python
from sklearn.tree import DecisionTreeClassifier
# 기본 특성값: (*, criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, class_weight=None, ccp_alpha=0.0, monotonic_cst=None)
```

- [랜덤 포레스트 (Random Forests)](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
    - 부트스트랩 샘플을 사용하고 랜덤하게 일부 특성을 선택하여 트리를 만드는 알고리즘

```python
from sklearn.ensemble import RandomForestClassifier
# 기본 특성값: (n_estimators=100, *, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None, monotonic_cst=None)
```

- [엑스트라 트리 (Extra Trees)](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html)
    - 부트스트랩 샘플을 사용하지 않고 각 결정 트리를 만들 때 전체 훈련 세트 사용하는 알고리즘

```python
from sklearn.ensemble import ExtraTreesClassifier
# 기본 특성값: (n_estimators=100, *, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=False, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None, monotonic_cst=None)
```

- [그레이디언트 부스팅 (Gradient Boosting)](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
    - 깊이가 얕은 결정 트리를 사용하여 이전 트리의 오차를 보완하는 방식의 알고리즘

```python
from sklearn.ensemble import GradientBoostingClassifier
# 기본 특성값: (*, loss='log_loss', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0)
```

- [히스토그램 기반 그레이디언트 부스팅(HistGradient Boosting)](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)
    - 정형 데이터를 다루는 ML알고리즘 중에 가장 인기가 높은 알고리즘

```python
from sklearn.ensemble import HistGradientBoostingClassifier
# 기본 특성값: (loss='log_loss', *, learning_rate=0.1, max_iter=100, max_leaf_nodes=31, max_depth=None, min_samples_leaf=20, l2_regularization=0.0, max_features=1.0, max_bins=255, categorical_features='warn', monotonic_cst=None, interaction_cst=None, warm_start=False, early_stopping='auto', scoring='loss', validation_fraction=0.1, n_iter_no_change=10, tol=1e-07, verbose=0, random_state=None, class_weight=None)
```

### 4.2. 회귀 (Regression)

회귀는 연속적인 값을 예측하는 작업입니다. 주요 회귀 알고리즘에는 다음이 포함됩니다:

- 선형 회귀 (Linear Regression)
- 릿지 회귀 (Ridge Regression)
- 라쏘 회귀 (Lasso Regression)

### 4.3. 클러스터링 (Clustering)

클러스터링은 데이터 포인트를 유사한 그룹으로 나누는 작업입니다. 주요 클러스터링 알고리즘에는 다음이 포함됩니다:

- k-평균 (k-Means)
- DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
- 계층적 클러스터링 (Hierarchical Clustering)

## 5. 참고 자료

Scikit-learn에 대한 더 많은 정보와 예제는 다음 링크를 참고하세요:

- [Scikit-learn 공식 문서](https://scikit-learn.org)
- [Scikit-learn GitHub 저장소](https://github.com/scikit-learn/scikit-learn)