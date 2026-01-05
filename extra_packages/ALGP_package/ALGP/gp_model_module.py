

from ALGP.get_one_gp_model_module import get_gp_model
import numpy as np

class GPModel:
    def __init__(
        self,
        x_training,
        y_training,
        gp_config_path=None,
    ):
        assert x_training.ndim == 2, "x_train must be a 2D array"
        if y_training.ndim == 1:
            y_training = y_training.reshape(-1, 1)
        assert y_training.ndim == 2, "y_train must be a 2D array"
        assert x_training.shape[0] == y_training.shape[0], "x_train and y_train must have the same number of samples"
        
        self.x = x_training
        self.y = y_training
        self.len_x_feature = x_training.shape[-1]
        self.len_y_feature = y_training.shape[-1]
        self.gp_model = [get_gp_model(config_path=gp_config_path) for _ in range(self.len_y_feature)]
        self.fit(self.x, self.y)

    def fit(self, x_train, y_train):
        for i in range(self.len_y_feature):
            self.gp_model[i].fit(x_train, y_train[:, i])

    def predict(self, x_test, return_std=False):
        y_mean_list = []
        y_std_list = []
        for i in range(self.len_y_feature):
            y_mean, y_std = self.gp_model[i].predict(x_test, return_std=True)
            y_mean_list.append(y_mean)
            y_std_list.append(y_std)
        
        if return_std:
            return np.stack(y_mean_list, axis=-1), np.stack(y_std_list, axis=-1)
        else:
            return np.stack(y_mean_list, axis=-1)