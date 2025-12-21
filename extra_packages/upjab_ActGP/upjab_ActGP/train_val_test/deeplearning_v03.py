import torch


from upjab_ActGP.metric.MAPE import MAPE

def evaluate_model(
    model,
    x_test,
    y_test,
    device = None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        model.eval()
        x_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
        x_pred = model(x_tensor).cpu().numpy().reshape(-1)

    mape = MAPE(x_pred, y_test)

    return mape