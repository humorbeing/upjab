


import numpy as np

from ALGP import AP


class StandardizeData:
    def __init__(self, data):
        self.original_data = data
        self.list_1d_to_2d()
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        self.transformed_data = (data - self.mean) / self.std
    
    def list_1d_to_2d(self):
        if self.original_data.ndim == 1:
            self.original_data = self.original_data[:, np.newaxis]


    def transform(self, new_data):
        return (new_data - self.mean) / self.std
    
    def inverse_transform(self, standardized_data):
        return standardized_data * self.std + self.mean
        

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


def retrieve_data(values, keys, key, dtype=np.float32):
    return np.array(values[:, keys.index(key)], dtype=dtype)

def load_DOE_csv(path):
    with open(path) as f:
        lines = [line for line in f.readlines() if line and not '#' in line]
    keys = [t.strip() for t in lines[0].strip().split(',')]
    values = [[float(v) for v in line.strip().split(',')] for line in lines[1:]]
    return keys, np.array(values, dtype=np.float32)


def load_design_dataset(    
    DOE_csv_path = AP('scripts/data.csv') 
):

    metadata_keys, metadata_values = load_DOE_csv(DOE_csv_path)

    Pt_in_Pa = retrieve_data(metadata_values, metadata_keys, 'Pt_in [Pa]')
    absolute_Pt_in_Pa = np.abs(Pt_in_Pa)  # Ensure Pt_in values are positive
    Pt_out_Pa = retrieve_data(metadata_values, metadata_keys, 'Pt_out [Pa]')
    absolute_Pt_out_Pa = np.abs(Pt_out_Pa)  # Ensure Pt_out values are positive

    del_Pt_Pa = retrieve_data(metadata_values, metadata_keys, 'del_Pt [Pa]')
    abs_del_Pt_Pa = np.abs(del_Pt_Pa)  # Ensure del_Pt values are positive

    headm = retrieve_data(metadata_values, metadata_keys, 'Head[m]')
    absolute_headm = np.abs(headm)  # Ensure head values are positive

    volume_flow_rate = retrieve_data(metadata_values, metadata_keys, 'VolumeFlowRate')
    absolute_volume_flow_rate = np.abs(volume_flow_rate)  # Ensure flow rate values are positive

    rpm = retrieve_data(metadata_values, metadata_keys, 'RPM')
    absolute_rpm = np.abs(rpm)  # Ensure RPM values are positive
    

    torque_values = retrieve_data(metadata_values, metadata_keys, 'Torque [N m]')
    absolute_torque_values = np.abs(torque_values)  # Ensure torque values are positive

    eff_values = retrieve_data(metadata_values, metadata_keys, 'Eff [%]')  
    absolute_eff_values = np.abs(eff_values)  # Ensure efficiency values are positive      

    

    return {        
        'Pt_in_Pa': Pt_in_Pa,
        'absolute_Pt_in_Pa': absolute_Pt_in_Pa,
        'Pt_out_Pa': Pt_out_Pa,
        'absolute_Pt_out_Pa': absolute_Pt_out_Pa,
        'del_Pt_Pa': del_Pt_Pa,
        'abs_del_Pt_Pa': abs_del_Pt_Pa,
        'headm': headm,
        'absolute_headm': absolute_headm,
        'volume_flow_rate': volume_flow_rate,
        'absolute_volume_flow_rate': absolute_volume_flow_rate,
        'rpm': rpm,
        'absolute_rpm': absolute_rpm,
        'torque_values': torque_values,
        'absolute_torque_values': absolute_torque_values,
        'eff_values': eff_values,
        'absolute_eff_values': absolute_eff_values,
    }

    # return design_variable, Pt_in_Pa, Pt_out_Pa, absolute_torque_values, eff_values



def load_dataset(
    num_trainval = 900,
    DOE_csv_path = AP('data/Pump_data/DOE_data.csv'),
    is_random = False,
    train_val_ratio = 0.8,
    num_test = 100
):
    data = load_design_dataset(DOE_csv_path)

    metadata_keys, metadata_values = load_DOE_csv(DOE_csv_path)
    Pt_in_Pa = retrieve_data(metadata_values, metadata_keys, 'Pt_in [Pa]')
    absolute_Pt_in_Pa = np.abs(Pt_in_Pa)  # Ensure Pt_in values are positive
    Pt_out_Pa = retrieve_data(metadata_values, metadata_keys, 'Pt_out [Pa]')
    absolute_Pt_out_Pa = np.abs(Pt_out_Pa)  # Ensure Pt_out values are positive

    del_Pt_Pa = retrieve_data(metadata_values, metadata_keys, 'del_Pt [Pa]')
    abs_del_Pt_Pa = np.abs(del_Pt_Pa)  # Ensure del_Pt values are positive

    headm = retrieve_data(metadata_values, metadata_keys, 'Head[m]')
    absolute_headm = np.abs(headm)  # Ensure head values are positive

    volume_flow_rate = retrieve_data(metadata_values, metadata_keys, 'VolumeFlowRate')
    absolute_volume_flow_rate = np.abs(volume_flow_rate)  # Ensure flow rate values are positive

    rpm = retrieve_data(metadata_values, metadata_keys, 'RPM')
    absolute_rpm = np.abs(rpm)  # Ensure RPM values are positive
    

    torque_values = retrieve_data(metadata_values, metadata_keys, 'Torque [N m]')
    absolute_torque_values = np.abs(torque_values)  # Ensure torque values are positive

    eff_values = retrieve_data(metadata_values, metadata_keys, 'Eff [%]')  
    absolute_eff_values = np.abs(eff_values)  # Ensure efficiency values are positive      



    x_dataset = np.stack([headm, volume_flow_rate, rpm], axis=-1)

    y_dataset = np.stack(
        [
            absolute_Pt_in_Pa,
            Pt_out_Pa,
            # abs_del_Pt_Pa,
            absolute_torque_values,
            eff_values
        ],axis=-1)
    

    

    x_test = x_dataset[-num_test:]
    y_test = y_dataset[-num_test:]

    if is_random:
        indices = np.arange(len(x_dataset) - num_test)
        np.random.shuffle(indices)
        selected_indices = indices[:num_trainval]
        x_trainval = x_dataset[selected_indices]
        y_trainval = y_dataset[selected_indices]
    else:
        x_trainval = x_dataset[:num_trainval]
        y_trainval = y_dataset[:num_trainval]
    
    num_train = int(train_val_ratio * num_trainval)
    num_val = num_trainval - num_train


    x_train = x_trainval[:num_train]
    y_train = y_trainval[:num_train]
    x_val = x_trainval[num_train:num_train+num_val]
    y_val = y_trainval[num_train:num_train+num_val]
    

    return {
        'x_train': x_train,
        'y_train': y_train,
        'x_val': x_val,
        'y_val': y_val,
        'x_test': x_test,
        'y_test': y_test,
        'x_trainval': x_trainval,
        'y_trainval': y_trainval,
        # 'data': data
    }
    # return train_design_variable, test_design_variable, train_torque_values, test_torque_values, train_eff_values, test_eff_values




class LoadDATA:
    def __init__(self,        
        num_trainval = 900,
        is_random_sample = False,
        which_FeatureAugmentation = 0,
        FeatureAugmentation_Runs = 0,
        x_transformer='standardize',
        y_transformer='standardize',
        DOE_csv_path = AP('scripts/data.csv'),
        # USE_ABSOLUTE_ROOT_PATH = True,
    ):
        
        

        # if USE_ABSOLUTE_ROOT_PATH:
        #     from ALGP import AP
        #     DOE_csv_path = AP(DOE_csv_path)

        dataset = load_dataset(
            num_trainval = num_trainval,
            DOE_csv_path = DOE_csv_path,
            is_random = is_random_sample,
            train_val_ratio = 0.8,
            num_test = 100)


        x_trainval = dataset['x_trainval']
        y_trainval = dataset['y_trainval']
        x_test = dataset['x_test']
        y_test = dataset['y_test']
        x_train = dataset['x_train']
        y_train = dataset['y_train']
        x_val = dataset['x_val']
        y_val = dataset['y_val']    
           
        x_train_transformer = DataTransformer(x_train, transform_type=x_transformer)
        y_train_transformer = DataTransformer(y_train, transform_type=y_transformer)

        x_train_norm = x_train_transformer.transformer.transform(x_train)
        x_val_norm = x_train_transformer.transformer.transform(x_val)
        x_test_norm = x_train_transformer.transformer.transform(x_test)
        x_trainval_norm = x_train_transformer.transformer.transform(x_trainval) 


        y_train_norm = y_train_transformer.transformer.transform(y_train)
        y_val_norm = y_train_transformer.transformer.transform(y_val)
        y_test_norm = y_train_transformer.transformer.transform(y_test)
        y_trainval_norm = y_train_transformer.transformer.transform(y_trainval)


        self.x_trainval = x_trainval_norm
        self.y_trainval = y_trainval_norm
        self.x_train = x_train_norm
        self.y_train = y_train_norm
        self.x_val = x_val_norm
        self.y_val = y_val_norm
        self.x_test = x_test_norm
        self.y_test = y_test_norm
        self.x_train_transformer = x_train_transformer
        self.y_train_transformer = y_train_transformer

        self.original_x_train = x_train
        self.original_y_train = y_train
        self.original_x_val = x_val
        self.original_y_val = y_val
        self.original_x_test = x_test
        self.original_y_test = y_test
        self.original_x_trainval = x_trainval
        self.original_y_trainval = y_trainval
        self.reverse_x = x_train_transformer.transformer.inverse_transform
        self.reverse_y = y_train_transformer.transformer.inverse_transform





