import torch
import torch.nn as nn

from upjab_ActGP.models.gp_design.deep_NN_v01 import MLPRegressor
from upjab_ActGP.train_val_test.deeplearning_v01 import train_one_step

from copy import deepcopy

def train_model(
    x_train,
    x_val,
    y_train,
    y_val,
    device = None,
    EPOCH = 100,
    nn_structure = (16, 32, 64, 64, 32, 16),
    lr=1e-3,
    dropout=0.1,
    validation_interval=5
    ):
    
    in_dim = x_train.shape[1]
    if y_train.ndim == 1:
        out_dim = 1
    else:
        out_dim = y_train.shape[1]
    model = MLPRegressor(in_dim=in_dim, hidden=nn_structure, out_dim=out_dim, dropout=dropout)
    
    # Loss & Optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # (optional) scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    train_data = {
        'x': torch.tensor(x_train, dtype=torch.float32),
        'y': torch.tensor(y_train, dtype=torch.float32)
    }

    val_data = {
        'x': torch.tensor(x_val, dtype=torch.float32),
        'y': torch.tensor(y_val, dtype=torch.float32)
    }

    best_val_loss = float('inf')
    for epoch in range(1, EPOCH+1):
        
        train_loss = train_one_step(train_data, model, criterion, optimizer)    
        val_loss = train_one_step(val_data, model, criterion, optimizer=None)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = deepcopy(model)

        if epoch % validation_interval == 0 or epoch == 1:
            
            print(f"Epoch {epoch}/{EPOCH}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
    

    return best_model
