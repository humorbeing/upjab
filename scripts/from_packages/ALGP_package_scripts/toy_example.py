from ALGP.ALGP_module import ALGP
from ALGP import AP
import numpy as np

# Prepare your data
x_train = np.random.rand(1000, 5)  # 1000 samples, 5 features
y_train = np.random.rand(1000, 4)  # 1000 samples, 4 outputs

# Train with active learning
model = ALGP(
    x_train=x_train,
    y_train=y_train,
    ActiveLearning_Target_Index=0,  # Use first output for sample selection in active learning
    Budget_Training_Sample=50,  # Use only 50 samples
    NUM_INIT_SAMPLE=10,        # Start with 10 random samples
    TOP_K=1,                   # Select 1 sample per iteration
    gp_config_path=AP('configs/gp-l_bfgs_v01.yaml')
)

# Make predictions
x_test = np.random.rand(2, 5)
y_pred, y_std = model.predict(x_test, return_std=True)

print('Predictions:', y_pred)
print('Uncertainties:', y_std)

print('End of ALGP toy example.')