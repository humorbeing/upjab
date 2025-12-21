import torch


def one_gpytorch_inference(
    model,
    likelihood,
    x_test,
    device=None,
):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    likelihood.eval()

    with torch.no_grad():

        x_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
        preds = model(x_tensor)
        y_pred = preds.mean.cpu().numpy()

    return y_pred
