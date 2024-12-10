from .train_test_loader import get_dataset_loader


class DataLoader:
    def __init__(self, batch_size):
        datasetloader = get_dataset_loader(batch_size=batch_size)
        train_normal_loader = datasetloader['train_normal_loader']
        train_abnormal_loader = datasetloader['train_abnormal_loader']
        test_loader = datasetloader['test_loader']
        train_batch_size = datasetloader['train_batch_size']
        test_batch_size = datasetloader['test_batch_size']

        self.normal_loader = train_normal_loader
        self.abnormal_loader = train_abnormal_loader
        self.test_loader = test_loader

    def get_one_batch(self, step):
        if (step) % len(self.normal_loader) == 0:
            self.normal_iter = iter(self.normal_loader)
            # print('normal_iter reset')

        if (step) % len(self.abnormal_loader) == 0:
            self.abnormal_iter = iter(self.abnormal_loader)
            # print('abnormal_iter reset---------')

        ninput, nlabel = next(self.normal_iter)
        abinput, ablabel = next(self.abnormal_iter)

        train_data = {
            'normal_input': ninput,
            'abnormal_input': abinput,
            'normal_label': nlabel,
            'abnormal_label': ablabel,
        }
        return train_data