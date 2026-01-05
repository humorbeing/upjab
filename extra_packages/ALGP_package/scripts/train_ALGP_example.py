from ALGP.ALGP_module import ALGP
from ALGP.eval_module import evaluate_result
from ALGP import AP

from load_data import load_data

dataset = load_data()
x_train = dataset['x_train']  # x_train is standardized
y_train = dataset['y_train']  # y_train is standardized
x_test = dataset['x_test']  # x_test is standardized
y_true = dataset['y_true']  # y_true is in original scale
reverse_y_fn = dataset['reverse_y_fn']  # function to reverse y_pred standardization

model = ALGP(
    x_train=x_train,
    y_train=y_train,    
    ActiveLearning_Target_Index = 3,  # Use 3rd output for sample selection in active learning
    Budget_Training_Sample = 50,  # Use only 50 samples
    NUM_INIT_SAMPLE=10,  # Start with 10 random samples
    TOP_K = 1,  # Select 1 sample per iteration
    gp_config_path = AP('configs/gp-l_bfgs_v01.yaml'),
)

y_pred, y_std = model.predict(x_test, return_std=True)
y_pred = reverse_y_fn(y_pred)
logs = evaluate_result(y_true=y_true, y_pred=y_pred)


print('End of ALGP example training script.')