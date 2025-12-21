
from upjab_ActGP.transforms.utils.standardize02 import StandardizeData
from upjab_ActGP.transforms.utils.normalize02 import NormalizeData

class DataTransformer:
    def __init__(self, data, transform_type='standardize'):
        self.original_data = data
        self.transform_type = transform_type
        if transform_type == 'standardize':
            self.transformer = StandardizeData(data)
        elif transform_type == 'normalize':
            self.transformer = NormalizeData(data)
        else:
            raise ValueError("transform_type must be either 'standardize' or 'normalize'")
        # self.normalizer = NormalizeData(data)
        # self.normalized_data = self.normalizer.transformed_data
        # self.standardizer = StandardizeData(data)
        # self.standardized_data = self.standardizer.transformed_data
    
    def normalize(self, new_data):
        return self.normalizer.transform(new_data)
    
    def inverse_normalize(self, normalized_data):
        return self.normalizer.inverse_transform(normalized_data)
    
    def standardize(self, new_data):
        return self.standardizer.transform(new_data)    
    
    def inverse_standardize(self, standardized_data):
        return self.standardizer.inverse_transform(standardized_data)


if __name__ == "__main__":
    import numpy as np

    data = np.array([[1, 2], [4, 5], [7, 8]])
    transformer = TransformData(data)

    print("Original Data:\n", transformer.original_data)
    print("Normalized Data:\n", transformer.normalized_data)
    print("Standardized Data:\n", transformer.standardized_data)

    new_data = np.array([[2, 3], [5, 6]])
    normalized_new_data = transformer.normalize(new_data)
    standardized_new_data = transformer.standardize(new_data)

    print("New Data:\n", new_data)
    print("Normalized New Data:\n", normalized_new_data)
    print("Standardized New Data:\n", standardized_new_data)

    recovered_data_from_norm = transformer.inverse_normalize(normalized_new_data)
    recovered_data_from_std = transformer.inverse_standardize(standardized_new_data)

    print("Recovered Data from Normalization:\n", recovered_data_from_norm)
    print("Recovered Data from Standardization:\n", recovered_data_from_std)