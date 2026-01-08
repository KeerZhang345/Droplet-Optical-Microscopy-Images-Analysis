import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import Birch
from DropletCorLab import Data_scaling
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, silhouette_samples
from typing import Any, List, Dict, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# extracted raw features --> scaling --> PCA --> scaling --> clustering

class Clustering_methods:
    def __init__(self):
        pass

    @staticmethod
    def kmeans_m(data, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, init='random', random_state=42).fit(data)  # Replace with the appropriate number of clusters
        labels = kmeans.labels_
        cluster_centers = kmeans.cluster_centers_

        return kmeans, cluster_centers, labels

    @staticmethod
    def kmeans_m_plus(data, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42).fit(data)  # Replace with the appropriate number of clusters
        labels = kmeans.labels_
        cluster_centers = kmeans.cluster_centers_

        return kmeans, cluster_centers, labels

    @staticmethod
    def birch_m(data, n_clusters):
        birch = Birch(n_clusters=n_clusters, threshold=0.3, branching_factor=20)
        labels = birch.fit_predict(data)
        cluster_centers = birch.subcluster_centers_

        return birch, cluster_centers, labels

    @staticmethod
    def kmedoids_m(data, n_clusters):
        try:
            from sklearn_extra.cluster import KMedoids
            kmedoids = KMedoids(n_clusters=n_clusters, init= 'k-medoids++', random_state=42)
            kmedoids.fit(data)
            labels = kmedoids.labels_
            medoids = kmedoids.cluster_centers_

            return kmedoids, medoids, labels
        except Exception:
            print("Fail to import KMedoids, try another clustering method.")

    @staticmethod
    def som_m(data, grid_y, grid_x=2):
        try:
            from minisom import MiniSom
            som = MiniSom(x=grid_x, y=grid_y, input_len=data.shape[1], sigma=1.0, learning_rate=0.5)
            som.random_weights_init(data)
            som.train_random(data=data, num_iteration=100)
            prototypes = np.array([som.get_weights()[i][j] for i in range(grid_x) for j in range(grid_y)])

            labels = []
            for sample in data:
                bmu = som.winner(sample)  # Find BMU for the sample
                cluster_label = bmu[0]*grid_y+bmu[1]  # Use both row and column indices as the label #bmu[1]  # Use the column index (y-axis) for cluster label
                labels.append(cluster_label)

            return som, prototypes, labels
        except Exception:
            print("Fail to import SOM, try another clustering method.")

    @staticmethod
    def perform_clustering(data, clustering_method, n_clusters):
        if clustering_method =='kmeans':
            trained_classifier, prob_intermediate, labels = Clustering_methods.kmeans_m(data, n_clusters)
        elif clustering_method =='kmeans++':
            trained_classifier, prob_intermediate, labels = Clustering_methods.kmeans_m_plus(data, n_clusters)
        elif clustering_method == 'kmedo':
            trained_classifier, prob_intermediate, labels = Clustering_methods.kmedoids_m(data, n_clusters)
        elif clustering_method == 'birch':
            trained_classifier, prob_intermediate, labels = Clustering_methods.birch_m(data, n_clusters)
        elif clustering_method == 'som':
            trained_classifier, prob_intermediate, labels = Clustering_methods.som_m(data, n_clusters)
        else:
            print('Clustering method not supported.')
            raise NameError

        return trained_classifier, prob_intermediate, labels

class Single_datapoint_prob:  # for gmm, used scaled data, for all tne rest, use original data
    def __init__(self):
        pass

    @staticmethod
    def calculate_weighted_distance(sample, center, weights):
        return np.sqrt(np.sum(weights * (sample - center) ** 2))

    @staticmethod
    def softmax(distances):
        distances = np.maximum(distances, 1e-12)
        exp_neg_distances = np.exp(-distances)
        return exp_neg_distances / np.sum(exp_neg_distances)

    @staticmethod
    def inverse_prob(distances):
        distances = np.maximum(distances, 1e-12)
        inverse_dist = 1 / distances
        return inverse_dist / np.sum(inverse_dist)

    @staticmethod
    def kmeans_pred(data, cluster_centers, weights, distance_method):
        distances = np.array([
            Single_datapoint_prob.calculate_weighted_distance(data, center, weights)
            for center in cluster_centers
        ])
        # raw distance used
        if distance_method == 'inverse':
            probs = Single_datapoint_prob.inverse_prob(distances)
        elif distance_method == 'softmax':
            probs = Single_datapoint_prob.softmax(distances)
        else:
            raise NameError

        return probs

    @staticmethod
    def kmeans_plus_pred(data, cluster_centers, weights, distance_method):
        distances = np.array([
            Single_datapoint_prob.calculate_weighted_distance(data, center, weights)
            for center in cluster_centers
        ])
        if distance_method == 'inverse':
            probs = Single_datapoint_prob.inverse_prob(distances)
        elif distance_method == 'softmax':
            probs = Single_datapoint_prob.softmax(distances)
        else:
            raise NameError
        return probs

    @staticmethod
    def kmedo_pred(data, medoids, weights, distance_method):
        distances = np.array([
            Single_datapoint_prob.calculate_weighted_distance(data, center, weights)
            for center in medoids
        ])
        if distance_method == 'inverse':
            probs = Single_datapoint_prob.inverse_prob(distances)
        elif distance_method == 'softmax':
            probs = Single_datapoint_prob.softmax(distances)
        else:
            raise NameError

        return probs

    @staticmethod
    def birch_pred(data, cluster_centers, weights, distance_method):
        distances = np.array([
            Single_datapoint_prob.calculate_weighted_distance(data, center, weights)
            for center in cluster_centers
        ])
        if distance_method == 'inverse':
            probs = Single_datapoint_prob.inverse_prob(distances)
        elif distance_method == 'softmax':
            probs = Single_datapoint_prob.softmax(distances)
        else:
            raise NameError
        return probs

    @staticmethod
    def som_pred(data, prototypes, weights, distance_method):
        distances = np.array([
            Single_datapoint_prob.calculate_weighted_distance(data, center, weights)
            for center in prototypes
        ])
        if distance_method == 'inverse':
            probs = Single_datapoint_prob.inverse_prob(distances)
        elif distance_method == 'softmax':
            probs = Single_datapoint_prob.softmax(distances)
        else:
            raise NameError
        return probs

    @staticmethod
    def predict_prob(data, weights, trained_classifier, clustering_method, prob_intermediate, pca_scaler,
                     scale_pca_reduced, distance_method):
        if clustering_method == 'kmeans':
            # convert center back to original scale
            prob_intermediate = Data_scaling.reverse_scaling(prob_intermediate, scale_pca_reduced, pca_scaler)
            probs = Single_datapoint_prob.kmeans_pred(data, prob_intermediate, weights, distance_method)
        elif clustering_method == 'kmeans++':
            prob_intermediate = Data_scaling.reverse_scaling(prob_intermediate, scale_pca_reduced, pca_scaler)
            probs = Single_datapoint_prob.kmeans_plus_pred(data, prob_intermediate, weights, distance_method)
        elif clustering_method == 'kmedo':
            prob_intermediate = Data_scaling.reverse_scaling(prob_intermediate, scale_pca_reduced, pca_scaler)
            probs = Single_datapoint_prob.kmedo_pred(data, prob_intermediate, weights, distance_method)
        elif clustering_method == 'birch':
            prob_intermediate = Data_scaling.reverse_scaling(prob_intermediate, scale_pca_reduced, pca_scaler)
            probs = Single_datapoint_prob.birch_pred(data, prob_intermediate, weights, distance_method)
        elif clustering_method == 'som':
            prob_intermediate = Data_scaling.reverse_scaling(prob_intermediate, scale_pca_reduced, pca_scaler)
            probs = Single_datapoint_prob.som_pred(data, prob_intermediate, weights, distance_method)
        else:
            print('Clustering method not supported.')
            raise NameError

        return probs

class Clusterability_evaluation:
    def __init__(self):
        pass

    @staticmethod
    def hopkins_statistic_single(sample, data, random_data):
        nbrs = NearestNeighbors(n_neighbors=2).fit(data)
        _, u_distances = nbrs.kneighbors(sample, n_neighbors=2)
        u_distances = u_distances[:, 1]  # Distance to the nearest neighbor

        _, w_distances = nbrs.kneighbors(random_data, n_neighbors=1)
        w_distances = w_distances[:, 0]  # Distance to the nearest neighbor in the random sample

        return np.sum(u_distances) / (np.sum(u_distances) + np.sum(w_distances))

    @staticmethod
    def hopkins_statistics(data_scaled, n_iter=20, seed=None):

        sample_size = data_scaled.shape[0]

        np.random.seed(seed)

        samples = [data_scaled[np.random.choice(data_scaled.shape[0], sample_size, replace=False)] for _ in range(n_iter)]
        random_points = [np.random.uniform(data_scaled.min(axis=0), data_scaled.max(axis=0), size=(sample_size, data_scaled.shape[1])) for _ in range(n_iter)]

        # avoid resources overlap
        results = []
        for sample, random_point in zip(samples, random_points):
            result = Clusterability_evaluation.hopkins_statistic_single(sample, data_scaled, random_point)
            results.append(result)

        return np.mean(results)

    @staticmethod
    def cal_silhouette_score(pca_reduced_data_scaled, n_clusters, clustering_method):
        _, _, labels = Clustering_methods.perform_clustering(data=pca_reduced_data_scaled, clustering_method=clustering_method, n_clusters=n_clusters)

        return silhouette_score(pca_reduced_data_scaled, labels) # returns the mean Silhouette Coefficient over all samples


def calculate_clustering_metrics(pc_data: np.ndarray, num_clusters: List[int], scaling_method: str, clustering_method: str) -> Dict:
    """
    Compute Hopkins statistics and silhouette score across multiple number of clusters.

    :param
    ----------
    pc_data: Any
        array-like of shape (n_samples, n_pcs)
    num_clusters: List[int]
        a list of number of clusters.
    scaling_method: str
        method to scale principal components before clustering.
        choose from 'standard', 'robust', 'minmax', 'power', 'L1', 'L2', 'maxabs', 'log'.
    clustering_method: str
        method for clustering. choose from 'kmeans', 'kmeans++' 'birch', 'kmedo', 'som'
        when using 'som', number of grid on x axis is always 2.

    :return
    -------
    results_: Dict
        clustering metrics

    """
    clus_eva_obj = Clusterability_evaluation()

    pc_data, _ = Data_scaling.data_scaling(pc_data, scaling_method)

    results_ = {}
    hop_score = clus_eva_obj.hopkins_statistics(pc_data)
    results_['Hopkins'] = hop_score

    for num_c in num_clusters:
        sil_score = clus_eva_obj.cal_silhouette_score(
        pc_data,
        n_clusters=num_c,
        clustering_method=clustering_method
        )
        results_[f'Silhouette_{num_c}'] = sil_score


    return results_


def update_cluster_center(new_data: np.ndarray,
                          cluster_centers: np.ndarray,
                          dro_cluster_idx: int,
                          labels: Any) -> Tuple:

    """
    Update the cluster center for a specific cluster by incorporating new data.

    :param
    ----------
    new_data: np.ndarray
        new data points to incorporate into the cluster center.
    cluster_centers:
        cluster centers obtained from primary clustering.
    cluster_idx: int,
        index of the cluster to update.
    labels:
        cluster labels for all data points.

    :return
    ----------
    updated_cluster_centers: np.ndarray,
        the modified cluster centers.

    """

    # Get the existing cluster center for the specified cluster
    current_center = cluster_centers[dro_cluster_idx]

    # Calculate the number of points in the existing cluster and the new data
    num_old_points = len(labels == dro_cluster_idx)
    num_new_points = len(new_data)

    # Calculate the new cluster center using a weighted mean
    updated_center = (
            (current_center * num_old_points + new_data.mean(axis=0) * num_new_points)
            / (num_old_points + num_new_points)
    )

    return updated_center


def perform_clustering(pc_df: pd.DataFrame,
                num_clusters: int,
                clustering_method: str,
                scaling_method: str,
                pc_columns: List,
                show_center: bool = False,
                update_centroid_with_initial_frame: bool = False,
                pc_df_initial: pd.DataFrame = None,
                dro_idx: int =  0,
                merge_initial_cluster_color: bool = False
                ) -> Tuple[pd.DataFrame, float, List, np.ndarray]:
    """
    Perform clustering on end of stage pc features (pc_features_df[features_columns]),
    with optional cluster center updating using initial stage pc features (pc_features_df_initial[features_columns])

    :param
    ----------
    pc_features_df: pd.DataFrame
        principal component dataframe.
    num_clusters: int
        number of clusters.
    clustering_method: str
        choose from 'kmeans', 'kmeans++' 'birch', 'kmedo', 'som'
        when using 'som', number of grid on x axis is always 2.
    scaling_method: str
        choose from 'standard', 'robust', 'minmax', 'power', 'L1', 'L2', 'maxabs', 'log'.
    pc_columns: List
        list of pc_column name.
    show_center: bool
        whether to show cluster center or not.
    update_centroid_with_initial_frame: bool
        whether to update cluster center with initial frame(s).

    Optional parameters (only relevant when update_centroid_with_initial_frame is True)
    ----------
    pc_features_df_initial: pd.DataFrame
        principal component dataframe of initial frame(s) for updating cluster centers.
    dro_idx: int
        label index of non-corroding (no significant color change from initial frame) cluster.
    merge_initial_cluster_color: bool
        whether to use same color to visualize the cluster identified by 'dro_idx' from primary cluster, and
        the new added cluster (pc_features_df_initial[features_columns])

    :return
    -------
    pca_features_scaled_df: pd.DataFrame
        scaled end of stage pc features (pc_features_df[features_columns]), with newly generated cluster information.
    avg_silhouette: float
        silhouette score using current clustering method on end of stage pc features (pc_features_df[features_columns]).
    trained_classifier: object
        classifier trained on end of stage pc features.
    clustering_method: str
        passed clustering method.
    prob_intermediate: np.ndarray
        computed cluster center, optionally updated by initial frame(s) pc.
    pca_scaler: object
        scaler object fitted on end of stage pc features
    scaling_method: str
        selected scaling method to fit pca_scaler.
    counts: np.ndarray
        number of instances in each cluster.

    """

    pca_features_df = pc_df[pc_columns]
    pca_features = np.asarray(pca_features_df)

    pca_features_scaled, pca_scaler = Data_scaling.data_scaling(pca_features, scaling_method)

    pca_features_scaled_df = pd.DataFrame(pca_features_scaled, index=pca_features_df.index,
                                          columns=pca_features_df.columns)
    pca_features_scaled_df['index'] = pc_df['index']

    trained_classifier, prob_intermediate, labels = Clustering_methods.perform_clustering(pca_features_scaled,
                                                                                          clustering_method=clustering_method,
                                                                                          n_clusters=num_clusters,
                                                                                          )
    pca_features_scaled_df['cluster'] = labels

    silhouette_vals = silhouette_samples(pca_features_scaled_df[pc_columns],
                                         pca_features_scaled_df['cluster'])
    pca_features_scaled_df['silhouette_score'] = silhouette_vals
    avg_silhouette = silhouette_score(pca_features_scaled_df[pc_columns], pca_features_scaled_df['cluster'])

    if update_centroid_with_initial_frame:

        pca_features_df_initial = pc_df_initial[pc_columns]
        pca_features_initial = np.asarray(pca_features_df_initial)
        pca_features_initial_scaled = Data_scaling.fit_scaler(pca_features_initial, pca_scaler, scaling_method)

        prob_intermediate[dro_idx] = update_cluster_center(pca_features_initial_scaled,
                                                                        prob_intermediate, dro_idx, labels)

        pca_features_scaled = np.vstack((pca_features_scaled, pca_features_initial_scaled))
        labels = np.hstack((labels, np.ones(len(pca_features_initial_scaled),
                                            dtype=int) * dro_idx)) if merge_initial_cluster_color else np.hstack(
            (labels, np.ones(len(pca_features_initial_scaled), dtype=int) * 2))

    print(labels.shape)

    if pca_features_scaled.shape[1] >= 2:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=pca_features_scaled[:, 0], y=pca_features_scaled[:, 1], hue=labels, palette='viridis', edgecolor='none',)
        plt.grid(False)
        plt.axis(False)
    if show_center:
        plt.scatter(prob_intermediate[0][0], prob_intermediate[0][1], color='red')
        plt.scatter(prob_intermediate[1][0], prob_intermediate[1][1], color='orange')

    # Get length of each cluster
    counts = np.bincount(labels)

    return pca_features_scaled_df, avg_silhouette, [trained_classifier, clustering_method, prob_intermediate,
                                                    pca_scaler, scaling_method], counts