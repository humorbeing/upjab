# import sys  # flash this. fix import error
# sys.path.append(".")  # flash this. fix import error

from experiment_configurations.v0001.args_setup import args_setup
from upjab.tool.args_setup_and_logger import args_setup_and_logger
args = args_setup("video_classification_temp_example")
logging = args_setup_and_logger(args)


from datasets.dataloader import DataLoader


D = DataLoader(batch_size=args.batch_size)


from models.image.frame16_112at15.C3D_model import C3D
model = C3D(2)

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = torch.nn.CrossEntropyLoss()

# optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=10,
    gamma=0.1)  # the scheduler divides the lr by 10 every 10 epochs
scheduler = None
scaler = torch.cuda.amp.GradScaler()


from utilities.video_classification.train_one_batch import train_one_batch
from utilities.video_classification.evaluate import evaluate


best_eval_metric = 0
best_eval_loss = 9999

for step in range(20):
    train_data = D.get_one_batch(step)
    train_results = train_one_batch(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        one_batch=train_data,
        scaler=scaler
    )
    if scheduler is not None:
        scheduler.step()
        print(f'{scheduler.get_last_lr()=}')
    log1 = f'{train_results["loss"]=}.'
    print(log1)
    logging.info(log1)
    
    if step % 1 == 0:
        eval_results = evaluate(model=model, data_loader=D.test_loader, criterion=criterion)
        log2 = f'{eval_results["loss"]=}. {eval_results["accuracy"]=}'
        
        logging.info(log2)
        eval_metric = eval_results["accuracy"]
        
        if eval_metric > best_eval_metric:
            best_eval_metric = eval_metric

            print(f'saving: {eval_metric=}')
            torch.save(model.state_dict(), args.checkpoint_save_path)
        elif eval_metric == best_eval_metric:
            eval_loss = eval_results["loss"]
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                print(f'saving: {eval_metric=}, {eval_loss=}')
                torch.save(model.state_dict(), args.checkpoint_save_path)
print('done')