import torch
import torch.nn as nn


STEP_TRACKER = 0

def train_one_batch(
    model,
    criterion,
    optimizer,
    one_batch,    
    clip_grad_norm,
    scaler,
    NUM_ACCUMULATION_STEPS):

    device = next(model.parameters()).device
    model.train()

    image = one_batch['image']
    target = one_batch['target']    

    image, target = image.to(device), target.to(device)
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
