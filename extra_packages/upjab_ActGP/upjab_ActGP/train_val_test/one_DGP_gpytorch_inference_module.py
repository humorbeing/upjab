import torch


def one_DGP_gpytorch_inference(
    model,    
    x_test,
    device=None,
):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    # likelihood.eval()

    with torch.no_grad():

        x_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
        temp11 = model(x_tensor)
        temp12 = model.likelihood(temp11)
        temp13 = temp12.mean
        y_pred = temp13.mean(0).cpu().numpy()
        # y_pred = preds.mean.cpu().numpy()

    return y_pred
