import numpy as np
from sklearn.utils import check_array, check_X_y
from joblib import Parallel, delayed
from .regression_tree import RegressionTree
import copy


class StochasticThresholdModelTrees():
    """
    Class of the Stochastic Threshold Model Trees.
    - Extended ensemble method based on tree-based regressors.
    """
    def __init__(
        self,
        n_estimators=100,
        criterion=None,
        regressor=None,
        threshold_selector=None,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='auto',
        f_select=True,
        ensemble_pred='mean',
        scaling=False,
        bootstrap=True,
        random_state=None,
        split_continue=False,
        verbose=0
    ):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.regressor = regressor
        self.threshold_selector = threshold_selector
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.f_select = f_select
        self.ensemble_pred = ensemble_pred
        self.scaling = scaling
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.split_continue = split_continue
        self.verbose = verbose

    def fit(self, X, y):
        """Build a forest of trees from the training set (X, y)."""
        X, y = check_X_y(X, y, ['csr', 'csc'])

        random_state = self.check_random_state(self.random_state)
        seeds = random_state.randint(
            np.iinfo(np.int32).max, size=self.n_estimators)

        self.forest = Parallel(n_jobs=-1, verbose=self.verbose)(
            delayed(self._build_trees)(X, y, seeds[i])
            for i in range(self.n_estimators))

    def predict(self, X, return_std=False):
        """Predict regression target for X.

        The predicted regression target of an input sample is computed as the
        mean or median predicted regression targets of the trees in the forest.
        """
        X = check_array(X, accept_sparse='csr')

        pred = np.array([tree.predict(X).tolist() for tree in self.forest])
        if return_std:
            if self.ensemble_pred == 'mean':
                return np.mean(pred, axis=0), np.std(pred, axis=0)
            elif self.ensemble_pred == 'median':
                return np.median(pred, axis=0), np.std(pred, axis=0)
        else:
            if self.ensemble_pred == 'mean':
                return np.mean(pred, axis=0)
            elif self.ensemble_pred == 'median':
                return np.median(pred, axis=0)
            else:
                return pred

    def _build_trees(self, X, y, seed):
        tree = RegressionTree(
            criterion=self.criterion,
            regressor=self.regressor,
            threshold_selector=self.threshold_selector,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            f_select=self.f_select,
            scaling=self.scaling,
            random_state=seed,
            split_continue=self.split_continue
        )
        if self.bootstrap:
            X_bootstrap, y_bootstrap = self._bootstrap(seed, X, y)
            tree.fit(X_bootstrap, y_bootstrap)
        else:
            tree.fit(X, y)
        return tree

    def count_selected_feature(self):
        """Count the number of features used to divide the tree."""
        return np.array(
            [tree.count_feature() for tree in self.forest])

    def _bootstrap(self, seed, X, y):
        n_samples, n_features = X.shape
        random_state = np.random.RandomState(seed)
        boot_index = random_state.randint(0, n_samples, n_samples)

        if isinstance(self.max_features, int):
            if not 1 <= self.max_features:
                print('The number of features must be one or more.')
            boot_features = self.max_features

        elif isinstance(self.max_features, float):
            if not 1. >= self.max_features:
                print('The fraction of features is must be less than 1.0.')
            elif not 0 < self.max_features:
                print('The fraction of features is must be more than 0.')
            boot_features = int(n_features * self.max_features)

        else:
            if self.max_features == 'auto':
                boot_features = n_features
            elif self.max_features == 'sqrt':
                boot_features = int(np.sqrt(n_features))
            elif self.max_features == 'log2':
                boot_features = int(np.log2(n_features))

        boot_feature_index = random_state.permutation(
            n_features)[0:boot_features]
        remove_feature_index = list(set(range(
            n_features)) - set(boot_feature_index.tolist()))
        boot_X = X[boot_index, :].copy()
        boot_X[:, remove_feature_index] = 0.0
        return boot_X, y[boot_index]

    def check_random_state(self, seed):
        if seed is None or seed is np.random:
            return np.random.mtrand._rand
        if isinstance(seed, int):
            return np.random.RandomState(seed)
        if isinstance(seed, np.random.RandomState):
            return seed

    def get_params(self, deep=True):
        return {
            'n_estimators': self.n_estimators,
            'criterion': self.criterion,
            'regressor': self.regressor,
            'threshold_selector': self.threshold_selector,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_features': self.max_features,
            'f_select': self.f_select,
            'ensemble_pred': self.ensemble_pred,
            'scaling': self.scaling,
            'bootstrap': self.bootstrap,
            'random_state': self.random_state,
            'split_continue': self.split_continue
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self

    def score(self, X, y, sample_weight=None):
        """
        ref: 
        from RegerssorMixin that is the parent class of Random Forest Regressor
        https://github.com/scikit-learn/scikit-learn/blob/c72ace8cd652309963ecdd6c01e33fa3c58c9161/sklearn/base.py#L673
        
        Return the coefficient of determination of the prediction.
        The coefficient of determination :math:`R^2` is defined as
        :math:`(1 - \\frac{u}{v})`, where :math:`u` is the residual
        sum of squares ``((y_true - y_pred)** 2).sum()`` and :math:`v`
        is the total sum of squares ``((y_true - y_true.mean()) ** 2).sum()``.
        The best possible score is 1.0 and it can be negative (because the
        model can be arbitrarily worse). A constant model that always predicts
        the expected value of `y`, disregarding the input features, would get
        a :math:`R^2` score of 0.0.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples. For some estimators this may be a precomputed
            kernel matrix or a list of generic objects instead with shape
            ``(n_samples, n_samples_fitted)``, where ``n_samples_fitted``
            is the number of samples used in the fitting for the estimator.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for `X`.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
        Returns
        -------
        score : float
            :math:`R^2` of ``self.predict(X)`` wrt. `y`.
        Notes
        -----
        The :math:`R^2` score used when calling ``score`` on a regressor uses
        ``multioutput='uniform_average'`` from version 0.23 to keep consistent
        with default value of :func:`~sklearn.metrics.r2_score`.
        This influences the ``score`` method of all the multioutput
        regressors (except for
        :class:`~sklearn.multioutput.MultiOutputRegressor`).
        """

        from sklearn.metrics import r2_score

        y_pred = self.predict(X)
        return r2_score(y, y_pred, sample_weight=sample_weight)



