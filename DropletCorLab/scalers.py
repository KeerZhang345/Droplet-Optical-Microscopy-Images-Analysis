import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, Normalizer, MaxAbsScaler
from typing import Any, Tuple

class Data_scaling:
    def __init__(self):
        pass
    @staticmethod
    def data_scaling(data: Any, method: str) -> Tuple[Any, object]:
        """
        Initialize a new scaler and scale data, using given scaling method.

        :param
        ----------
        data : Any
            array-like of shape (n_samples, n_features)
        method: str
            scaling method, choose from 'standard', 'robust', 'minmax', 'power', 'L1', 'L2', 'maxabs', 'log'.
            otherwise, original data returned.

        :return
        -------
        data: Any
            scaled data, array-like of shape (n_samples, n_features)
        scaler:
            fitted scaler.
            if scaling method not supported, return None
        """

        if method=='standard':
            scaler = StandardScaler()
        elif method=='robust':
            scaler = RobustScaler()
        elif method=='minmax':
            scaler = MinMaxScaler()
        elif method=='power':
            scaler = PowerTransformer()
        elif method=='L2':
            scaler = Normalizer(norm='l2')
        elif method=='L1':
            scaler = Normalizer(norm='l1')
        elif method=='maxabs':
            scaler = MaxAbsScaler()
        elif method=='log':
            return np.log(data + 20), None
        else:
            scaler = None
            print("Scaling not supported, original value returned")

        if scaler is not None:
            return scaler.fit_transform(data), scaler
        else:
            return data, None

    @staticmethod
    def fit_scaler(data: Any, scaler: object, method: str) -> Any:
        """
        Scale data with pre-fitted scaler, using given scaling method.

        Parameters
        ----------
        data : Any
            array-like of shape (n_samples, n_features)
        scaler: object
            fitted scaler.
        method: str
            method of the passed scaler.
        Returns
        -------
        data: Any
            scaled data, array-like of shape (n_samples, n_features)
        """

        if method=='log':
            return np.log(data + 20)
        elif method in ['standard', 'robust', 'minmax', 'power', 'L2', 'L1', 'maxabs']:
            return scaler.transform(data)
        else:
            print("Scaling not supported, original value returned")
            return data

    @staticmethod
    def reverse_scaling(scaled_data, method, scaler) -> Any:
        """
        Reverse the scaler-scaled data back to its original value

        Parameters
        ----------
        scaled_data : Any
            scaled data, array-like of shape (n_samples, n_features)
        scaler: object
            scaler that connects original and scaled data.
        method: str
            method of the passed scaler.
        Returns
        -------
        data: Any
            original data converted back from the scaled data space, array-like of shape (n_samples, n_features)
        """

        if method == 'standard':
            return scaled_data * scaler.scale_ + scaler.mean_
        elif method == 'robust':
            return scaled_data * scaler.scale_ + scaler.center_
        elif method == 'minmax':
            return scaled_data * (scaler.data_max_ - scaler.data_min_) + scaler.data_min_
        elif method == 'power':
            return scaler.inverse_transform(scaled_data)
        elif method in ['L2', 'L1']:
            print("Cannot reverse L1/L2 Normalization without the original norms.")
            return None
        elif method == 'maxabs':
            return scaled_data * scaler.max_abs_
        elif method == 'log':
            return np.exp(scaled_data) - 20
        else:
            print("Scaling not supported, original value returned.")
            return scaled_data