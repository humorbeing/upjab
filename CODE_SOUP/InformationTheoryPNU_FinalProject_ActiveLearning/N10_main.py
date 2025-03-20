import torch.optim as optim
import torch
from N04_resnet import ResNet18
from N04_model import Net
from N04_getDataset import get_dataset_loader


# from losses.video_anomaly_detection.RTFM.loss import loss_function



from N04_cross_entropy_loss import CrossEntropyLoss
from torch.nn.functional import softmax
from sklearn.metrics import auc as auc_rename_avoid_duplication
from sklearn.metrics import roc_curve
import os
import gc

def train_eval(
    labeled_path,
    batch_size = 5000,
    num_steps = 200,
    NUM_ACCUMULATION_STEPS = 5,
):
    
    criterion = CrossEntropyLoss(
        num_classes=2,
        use_gpu=True,
        label_smooth=True
    )    
    
    dataset_loaders = get_dataset_loader(batch_size, labeled_path=labeled_path)

    normal_loader = dataset_loaders['train_normal_loader']
    abnormal_loader = dataset_loaders['train_abnormal_loader']
    test_loader = dataset_loaders['test_loader']

    # model = Net()
    model = ResNet18()
    num_features = model.linear.in_features #the number of nodes in the last layer for the ResNet model
    num_classes = 2 #whatever number of classes u have.
    model.fc = torch.nn.Linear(num_features, num_classes)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    model = model.to(device)

    optimizer = optim.Adadelta(model.parameters(), lr=1.0)
    
    optimizer.zero_grad()    

    STEP_TRACKER = 0
    loss = 0
    for step in range(num_steps):   

        if (step) % len(normal_loader) == 0:
            normal_iter = iter(normal_loader)

        if (step) % len(abnormal_loader) == 0:
            abnormal_iter = iter(abnormal_loader)

        
        model.train()
        ninput, nlabel = next(normal_iter)
        ainput, alabel = next(abnormal_iter)
        input_features = torch.cat((ninput, ainput), 0).to(device)
        labels = torch.cat((nlabel, alabel), 0).to(device)

        outputs = model(input_features)        
        loss1 = criterion(outputs, labels)
        print(f'step: {step}, loss: {loss1.item()}')
        
        loss += loss1 / NUM_ACCUMULATION_STEPS
        
        STEP_TRACKER = STEP_TRACKER + 1
        if (STEP_TRACKER % NUM_ACCUMULATION_STEPS == 0):
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss = 0
        
        
    model.eval()
    pred_list1 = torch.zeros(0).to(device)
    pred_list2 = torch.zeros(0).to(device)
    labels_list = torch.zeros(0)
    for (images, labels) in test_loader:
        images = images.to(device)
        with torch.no_grad():
            temp12 = model(images)
            
            sf = softmax(temp12, dim=1)
            prediction1 = sf[:,0]
            prediction2 = sf[:,1]

            labels_list = torch.cat((labels_list, labels))            
            pred_list1 = torch.cat((pred_list1, prediction1))
            pred_list2 = torch.cat((pred_list2, prediction2))
        

    pred_np1 = pred_list1.cpu().detach().numpy()
    pred_np2 = pred_list2.cpu().detach().numpy()
    gt_np = labels_list.cpu().detach().numpy()


    fpr, tpr, threshold = roc_curve(gt_np, pred_np1)
    auc_roc1 = auc_rename_avoid_duplication(fpr, tpr)

    fpr, tpr, threshold = roc_curve(gt_np, pred_np2)
    auc_roc2 = auc_rename_avoid_duplication(fpr, tpr)

    print(f'1: {auc_roc1}, 2: {auc_roc2}')
        
    save_folder = os.path.dirname(__file__) + '/model_weights'
    os.makedirs(save_folder, exist_ok=True)

    sig = labeled_path.split('/')[-1]
    torch.save(model.state_dict(), save_folder+f'/{sig}_{auc_roc2:.03f}_model.pkl')        
    
    
    del model
    del optimizer
    # Invoke garbage collector
    gc.collect()

    # Clear GPU cache
    torch.cuda.empty_cache()




root = os.path.dirname(__file__)
root1 = f'{root}/data/color/06_labeled'
# merge_labeled(target_folder, first_labeled)


for i in os.scandir(root1):
    folder_name = i.name
    folder_path = os.path.join(root1, folder_name)
    
    
    train_eval(folder_path)

print('end')

