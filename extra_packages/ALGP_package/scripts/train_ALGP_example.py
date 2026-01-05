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

MAPE = float(logs[-6].split()[-1])*100

print(f'''
Dataset: x has 3 features, y has 4 outputs. 4th output (y[3]) is used for active learning sample selection.
Dataset has 1000 samples. 900 samples are used for training, 100 samples are used for testing.

      0. Randomly select 10 samples as initial training set (x and y[3]).

      1. Start with training a GP model with training set.
      2. Apply the GP model on the remaining training samples for predicting 4th output.
      3. Select ONE sample with highest predicted uncertainty and add it to the training set.

Repeat steps 2-3 until reaching the budget of 50 training samples.

      4. Use the selected 50 training samples to train the final model and predict on the test set.

Results:
After using 50 training samples from 900 training samples to train the final model, the model achieves Mean Absolute Percentage Error (MAPE) of {MAPE:.2f} % on the test set of y[3].
''')

print('End of ALGP example training script.')