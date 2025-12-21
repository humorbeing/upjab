import numpy as np


class NormalizeData:
    def __init__(self, data):
        self.original_data = data
        self.list_1d_to_2d()
        self.normalize_data()



    def list_1d_to_2d(self):
        if self.original_data.ndim == 1:
            self.original_data = self.original_data[:, np.newaxis]
        

    def normalize_data(self):
        self.min = np.min(self.original_data, axis=0)
        self.max = np.max(self.original_data, axis=0)
        self.transformed_data = (self.original_data - self.min) / (self.max - self.min)
        

    def transform(self, data):
        return (data - self.min) / (self.max - self.min)

    def inverse_transform(self, normalized_data):
        return normalized_data * (self.max - self.min) + self.min
    


if __name__ == "__main__":
    data = np.array([[1, 2], [3, 4], [5, 6]])
    normalizer = NormalizeData(data)
    print("Original Data:\n", data)
    print("Normalized Data:\n", normalizer.transformed_data)
    new_data = np.array([[2, 3], [4, 5]])
    transformed_data = normalizer.transform(new_data)
    print("Transformed New Data:\n", transformed_data)
    inversed_data = normalizer.inverse_transform(transformed_data)
    print("Inversed Transformed Data:\n", inversed_data)

    data = np.array([1, 2, 3, 4, 5])
    normalizer_1d = NormalizeData(data)
    print("Original 1D Data:\n", data)
    print("Normalized 1D Data:\n", normalizer_1d.transformed_data)
    new_data_1d = np.array([2, 3, 4])
    transformed_data_1d = normalizer_1d.transform(new_data_1d)
    print("Transformed New 1D Data:\n", transformed_data_1d)
    inversed_data_1d = normalizer_1d.inverse_transform(transformed_data_1d)
    print("Inversed Transformed 1D Data:\n", inversed_data_1d)