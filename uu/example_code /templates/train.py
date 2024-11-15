



from model import ModelName
from utilities.get_one_batch import get_one_batch
from utilities.train_one_batch import train_one_batch


criterion = 1
optimizer = 1

device = 1
args = 1


trainsetloader = 1
testsetloader = 1


model = ModelName()


STEP = 1
for step in range(STEP):
    one_batch = get_one_batch(
        train_loader=trainsetloader,
        step=step,
        transform_data=model.transform_train_data,
        transform_label=model.transform_train_label
    )
    results = train_one_batch(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        one_batch=one_batch,
        # clip_grad_norm,
        # scaler,
        # NUM_ACCUMULATION_STEPS,
    )
