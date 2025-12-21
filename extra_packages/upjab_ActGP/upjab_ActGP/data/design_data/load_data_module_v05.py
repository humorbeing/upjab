


import numpy as np
from upjab_ActGP.data.load_DOE_module import load_DOE_csv
from upjab_ActGP.transforms.mesh.utils import retrieve_data
from upjab_ActGP.data.load_design_module import load_design_csv

from upjab_ActGP import AP



def load_design_dataset(    
    DOE_csv_path = AP('data/Pump_data/DOE_data.csv') 
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
            abs_del_Pt_Pa,
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


if __name__ == "__main__":
    # data = load_design_dataset()
    dataset = load_dataset(
        num_trainval=900,
        num_test=100  
        )
    
    # print('data keys:', data.keys())
    print('dataset keys:', dataset.keys())
    print('done')

