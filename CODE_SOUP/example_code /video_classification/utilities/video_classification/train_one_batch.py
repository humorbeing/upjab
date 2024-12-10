
import torch
import torch.nn as nn


STEP_TRACKER = 0

def train_one_batch(
    model,
    criterion,
    optimizer,
    one_batch,    
    clip_grad_norm=None,
    scaler=None,
    NUM_ACCUMULATION_STEPS=None):

    device = next(model.parameters()).device
    model.train()

    ninput = one_batch['normal_input']
    nlabel = one_batch['normal_label']
    ainput = one_batch['abnormal_input']
    alabel = one_batch['abnormal_label']  
    image = torch.cat((ninput, ainput), 0).to(device)
    target = torch.cat((nlabel, alabel), 0).to(device)

    # image, target = image.to(device), target.to(device)
    with torch.cuda.amp.autocast(enabled=scaler is not None):
        output = model(image)
        loss = criterion(output, target)

    
    if scaler is not None:
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        if clip_grad_norm is not None:
            # we should unscale the gradients of optimizer's assigned params if do gradient clipping
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        scaler.step(optimizer)
        scaler.update()

    elif NUM_ACCUMULATION_STEPS is None:
        optimizer.zero_grad()
        loss.backward()
        if clip_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()
    
    else:
        global STEP_TRACKER
        loss = loss / NUM_ACCUMULATION_STEPS
        loss.backward()
        STEP_TRACKER = STEP_TRACKER + 1
        if (STEP_TRACKER % NUM_ACCUMULATION_STEPS == 0):
            if clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
    
    return {
        'loss': loss.item(),
        'output': output,
        'target': target
    }



