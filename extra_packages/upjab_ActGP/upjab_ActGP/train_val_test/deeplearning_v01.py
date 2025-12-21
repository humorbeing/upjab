



def train_one_step(data, model, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    device = next(model.parameters()).device
    total_loss, n = 0.0, 0
    x = data['x']
    y = data['y']

    x, y = x.to(device), y.to(device)
    pred = model(x)
    loss = criterion(pred, y)

    if is_train:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    bs = x.size(0)
    total_loss += loss.item() * bs
    n += bs

    return total_loss / n