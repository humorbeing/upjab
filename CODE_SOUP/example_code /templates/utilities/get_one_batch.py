


def get_one_batch(train_loader, step=None, transform_data=None, transform_label=None):
    # iter next
    transformed_data = 1
    transformed_label = 1

    if transform_data is not None:
        transformed_data = transform_data(transformed_data)
    if transform_label is not None:
        transformed_label = transform_label(transformed_label)

    one_batch = {
        'data': transformed_data,
        'label': transformed_label
    }
    return one_batch