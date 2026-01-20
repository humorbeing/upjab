try:
    from .load_data_module import LoadDATA
except:
    from load_data_module import LoadDATA
    
def load_data():  

    data = LoadDATA()

    x_train = data.x_trainval
    y_train = data.y_trainval

    x_test = data.x_test
    y_true = data.original_y_test
    return {
        'x_train': x_train,
        'y_train': y_train,
        'x_test': x_test,
        'y_true': y_true,
        'reverse_y_fn': data.reverse_y,
    }