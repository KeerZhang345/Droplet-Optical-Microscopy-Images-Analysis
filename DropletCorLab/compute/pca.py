import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.decomposition import PCA
from DropletCorLab.scalers import Data_scaling
from typing import Any, Tuple
import pandas as pd


class PCA_analysis:
    def __init__(self, scaling_method='standard'):
        self.scaling_method = scaling_method

    def data_scaling(self, ori_data):
        scaled, scaler = Data_scaling.data_scaling(ori_data, self.scaling_method)
        return scaler, scaled

    def CEV(self, ori_data):
        _, data_scaled = self.data_scaling(ori_data)
        pca = PCA()
        pca.fit(data_scaled)
        return pca.explained_variance_ratio_.cumsum()

    def pca_analysis_without_dimension_reduction(self, ori_data):
        _, data_scaled = self.data_scaling(ori_data)
        pca = PCA(n_components=data_scaled.shape[-1])
        pca.fit(data_scaled)
        return data_scaled, pca

    @staticmethod
    def generate_random_data(n_samples, n_features):
        return norm.rvs(size=(n_samples, n_features))

    def parallel_analysis(self, ori_data, n_iter=100):
        scaler, data_scaled = self.data_scaling(ori_data)

        n_components = data_scaled.shape[-1]
        pca = PCA(n_components=n_components)
        pca.fit(data_scaled)

        real_eigenvalues = pca.explained_variance_
        random_eigenvalues = np.zeros((n_iter, n_components))

        for i in range(n_iter):
            rand = self.generate_random_data(data_scaled.shape[0], n_components)
            rand_scaled = Data_scaling.fit_scaler(rand, scaler, self.scaling_method)
            pca.fit(rand_scaled)
            random_eigenvalues[i, :] = pca.explained_variance_

        return real_eigenvalues, np.mean(random_eigenvalues, axis=0)

    def scree(self, ori_data):
        _, data_scaled = self.data_scaling(ori_data)
        pca = PCA(n_components=None)
        pca.fit(data_scaled)
        return pca.explained_variance_




def pca_visulization(ori_data: Any, scaling_method: str) -> None:
    """
    Visualize PCA result with cumulative explained variance and scree plot.

    :param
    ----------
    ori_data : Any
        array-like of shape (n_samples, n_features).
    scaling_method : str
        data scaling method before PCA. Choose from 'standard', 'robust', 'minmax', 'power', 'L1', 'L2', 'maxabs', 'log'.

    :return
    -------
    None.

    """

    pca_tool = PCA_analysis(scaling_method=scaling_method)

    cev = pca_tool.CEV(ori_data)
    eg_val = pca_tool.scree(ori_data)

    fig, ax = plt.subplots(figsize=(6, 4))
    cev_plot = ax.plot(np.arange(1, len(cev) + 1), cev,
                       marker='o', label='CEV', color='blue')[0]
    ax2 = ax.twinx()
    eg_val_plot = ax2.plot(np.arange(1, len(eg_val) + 1), eg_val,
                           marker='x', label='Scree', color='orange')[0]

    ax.set_xlabel('Number of Components')
    ax.set_xticks([int(len(cev) / 10) * i for i in range(len(cev) // (int(len(cev) / 10)) + 1)])
    ax.set_ylabel('Cumulative Explained Variance')
    ax2.set_ylabel('Eigenvalue')
    ax.grid(True, color='black', linestyle='--')
    ax2.grid(False)

    plots = [cev_plot, eg_val_plot]
    labels = [p.get_label() for p in plots]
    ax.legend(plots, labels, loc='center right')

    plt.show()
    plt.close()


def pca_given_n(ori_data: Any, n: int, scaling_method: str) -> Tuple[Any, object, np.ndarray, str]:
    """
    PCA fit-transform on given data with specified number of components.

    :param
    ----------
    ori_data : Any
        array-like of shape (n_samples, n_features).
    n: int
        number of principal components.
    scaling_method : str
        data scaling method before PCA. Choose from 'standard', 'robust', 'minmax', 'power', 'L1', 'L2', 'maxabs', 'log'.

    :return
    -------
    reduced: Any
        projection of ori_data in the first n principal components
    pca: object
        fitted PCA object
    normalized_weights: np.ndarray
        array of variance explained by each PC, normalized to achieve sum = 1.
    scaling_method: str
        method for scaling ori_data.
    """

    pca_tool = PCA_analysis(scaling_method=scaling_method)

    _, data_scaled = pca_tool.data_scaling(ori_data)
    pca = PCA(n_components=n)
    pca.fit(data_scaled)
    reduced = pca.transform(data_scaled)
    explained_variance_ratio = pca.explained_variance_ratio_
    normalized_weights = explained_variance_ratio / np.sum(explained_variance_ratio)

    if not isinstance(reduced, pd.DataFrame):
        reduced = pd.DataFrame(reduced)

    return reduced, pca, normalized_weights, scaling_method


def pca_dynamic_n(ori_data: Any, scaling_method: str, variance_threshold: float = 0.90) -> Tuple[Any, object, np.ndarray, str]:
    """
    PCA fit-transform on given data with given variance threshold, such that the cumulative
    variance explained by the retained principal components is greater than the threshold.

    :param
    ----------
    ori_data : Any
        array-like of shape (n_samples, n_features).
    scaling_method : str
        data scaling method before PCA. Choose from 'standard', 'robust', 'minmax', 'power', 'L1', 'L2', 'maxabs', 'log'.
    variance_threshold: float
        threshold that the cumulative variance explained by the retained principal components should be greater than.

    :return
    -------
    reduced: Any
        projection of ori_data in the first n principal components
    pca: object
        fitted PCA object
    normalized_weights: np.ndarray
        array of variance explained by each PC, normalized to achieve sum = 1.
    scaling_method: str
        method for scaling ori_data.
    """

    pca_tool = PCA_analysis(scaling_method=scaling_method)

    _, data_scaled = pca_tool.data_scaling(ori_data)
    pca = PCA(n_components=variance_threshold)
    pca.fit(data_scaled)
    reduced = pca.transform(data_scaled)
    explained_variance_ratio = pca.explained_variance_ratio_
    normalized_weights = explained_variance_ratio / np.sum(explained_variance_ratio)

    if not isinstance(reduced, pd.DataFrame):
        reduced = pd.DataFrame(reduced)

    return reduced, pca, normalized_weights, scaling_method


def pca_transform(ori_data: Any, scaling_method: str, pca_obj: PCA) -> pd.DataFrame:
    """
    PCA transformation on passed data using pre-fitted PCA object.

    :param
    ----------
    ori_data : Any
        array-like of shape (n_samples, n_features).
    scaling_method : str
        data scaling method before PCA. Choose from 'standard', 'robust', 'minmax', 'power', 'L1', 'L2', 'maxabs', 'log'.
    pca_obj: PCA
        pre-fitted PCA object.

    :return
    -------
    reduced: pd.DataFrame
        projection of ori_data in the first n principal components transformed from passed PCA scaler.
    """

    pca_tool = PCA_analysis(scaling_method=scaling_method)
    _, data_scaled = pca_tool.data_scaling(ori_data)
    reduced = pca_obj.transform(data_scaled)
    if not isinstance(reduced, pd.DataFrame):
        reduced = pd.DataFrame(reduced)

    return reduced
