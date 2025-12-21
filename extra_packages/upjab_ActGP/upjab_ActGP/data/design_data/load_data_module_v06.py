
from upjab_ActGP.data.design_data.load_data_module_v05 import load_dataset
from upjab_ActGP.transforms.design_data.data_transform02 import DataTransformer

class LoadDATA:
    def __init__(self,        
        num_trainval = 100,
        is_random_sample = False,
        which_FeatureAugmentation = 3,
        FeatureAugmentation_Runs = 1,
        x_transformer='standardize',
        y_transformer='standardize',
        DOE_csv_path = 'data/Pump_data/DOE_data.csv',
        USE_ABSOLUTE_ROOT_PATH = True,
    ):
        
        

        if USE_ABSOLUTE_ROOT_PATH:
            from upjab_ActGP import AP
            DOE_csv_path = AP(DOE_csv_path)

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
        
        if which_FeatureAugmentation == 4:
            from upjab_ActGP.transforms.design_data.FA01 import FA04
            FeatureAugmentation = FA04
            x_trainval = FeatureAugmentation(x_trainval, FAR=FeatureAugmentation_Runs)
            x_train = FeatureAugmentation(x_train, FAR=FeatureAugmentation_Runs)
            x_val = FeatureAugmentation(x_val, FAR=FeatureAugmentation_Runs)
            x_test = FeatureAugmentation(x_test, FAR=FeatureAugmentation_Runs)
        else:
            if which_FeatureAugmentation == 0:
                FeatureAugmentation = lambda x: x
            elif which_FeatureAugmentation == 1:
                from upjab_ActGP.transforms.design_data.FA01 import FA01
                FeatureAugmentation = FA01
            elif which_FeatureAugmentation == 2:
                from upjab_ActGP.transforms.design_data.FA01 import FA02
                FeatureAugmentation = FA02
            elif which_FeatureAugmentation == 3:
                from upjab_ActGP.transforms.design_data.FA01 import FA03
                FeatureAugmentation = FA03
            else:
                raise ValueError(f'Unknown FeatureAugmentation: {which_FeatureAugmentation}')

            
            for fa_run in range(FeatureAugmentation_Runs):
                # print(f'FeatureAugmentation Run {fa_run+1}/{FeatureAugmentation_Runs}')
                x_trainval = FeatureAugmentation(x_trainval)
                x_train = FeatureAugmentation(x_train)
                x_val = FeatureAugmentation(x_val)
                x_test = FeatureAugmentation(x_test)    

        
        
            
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


