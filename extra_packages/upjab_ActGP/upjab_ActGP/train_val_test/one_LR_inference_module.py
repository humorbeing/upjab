


def one_LR_inference(
    model,    
    x_test,    
):    
    y_pred = model.predict(x_test)
    
    return y_pred
