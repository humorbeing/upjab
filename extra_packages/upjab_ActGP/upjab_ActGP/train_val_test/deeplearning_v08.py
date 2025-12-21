import torch


from upjab_ActGP.metric.MAPE import MAPE

def evaluate_model(
    model,
    likelihood,
    x_test,
    y_test,
    device = None,
    transformer_fns=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if transformer_fns is not None:
            x_test = transformer_fns['input_transformer'].transform(x_test)

    model.eval()
    likelihood.eval()
    
    with torch.no_grad():        

        x_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
        preds = model(x_tensor)
        y_pred = preds.mean.cpu().numpy()
        # y_pred = model(x_tensor).cpu().numpy()
    if transformer_fns is not None:
        y_pred = transformer_fns['output_transformer'].inverse_transform(y_pred)
    
    assert y_pred.shape == y_test.shape, f"y_pred shape {y_pred.shape} does not match y_test shape {y_test.shape}"
    # if y_test.ndim == 2:
    #     nf = y_test.shape[1]
    #     mapes = []
    #     for i in range(nf):
    #         mape = MAPE(y_pred[:, i], y_test[:, i])
    #         mapes.append(mape)
    mape = MAPE(y_pred, y_test)

    return mape, y_pred