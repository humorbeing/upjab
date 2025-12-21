


import numpy as np
from upjab_ActGP.data.load_DOE_module import load_DOE_csv
from upjab_ActGP.transforms.mesh.utils import retrieve_data
from upjab_ActGP.data.load_design_module import load_design_csv

from upjab_ActGP import AP



def load_design_dataset(    
    DOE_csv_path = AP('data/Pump_data/DOE_data.csv'),
    design_csv_path = AP('data/Pump_data/design_variable.csv')
):

    metadata_keys, metadata_values = load_DOE_csv(DOE_csv_path)

    Pt_in_Pa = retrieve_data(metadata_values, metadata_keys, 'Pt_in [Pa]')
    absolute_Pt_in_Pa = np.abs(Pt_in_Pa)  # Ensure Pt_in values are positive
    Pt_out_Pa = retrieve_data(metadata_values, metadata_keys, 'Pt_out [Pa]')
    absolute_Pt_out_Pa = np.abs(Pt_out_Pa)  # Ensure Pt_out values are positive



    torque_values = retrieve_data(metadata_values, metadata_keys, 'Torque [N m]')
    absolute_torque_values = np.abs(torque_values)  # Ensure torque values are positive

    eff_values = retrieve_data(metadata_values, metadata_keys, 'Eff [%]')        

    design_keys, design_variable = load_design_csv(design_csv_path)

    design_variable = design_variable[:, 1:]

    return {
        'design_variable': design_variable,
        'Pt_in_Pa': Pt_in_Pa,
        'Pt_out_Pa': Pt_out_Pa,
        'torque_values': absolute_torque_values,
        'eff_values': eff_values
    }

    # return design_variable, Pt_in_Pa, Pt_out_Pa, absolute_torque_values, eff_values



def load_front_train_val_data(
    num_train = 900,
    num_test = 100
):
    data = load_design_dataset()
    design_variable = data['design_variable']
    Pt_in_Pa = data['Pt_in_Pa']
    Pt_out_Pa = data['Pt_out_Pa']
    torque_values = data['torque_values']
    eff_values = data['eff_values']
    # design_variable, Pt_in_Pa, Pt_out_Pa, torque_values, eff_values = load_design_dataset()

    train_design_variable = design_variable[:num_train]
    test_design_variable = design_variable[-num_test:]

    train_Pt_in_Pa = Pt_in_Pa[:num_train]
    test_Pt_in_Pa = Pt_in_Pa[-num_test:]

    train_Pt_out_Pa = Pt_out_Pa[:num_train]
    test_Pt_out_Pa = Pt_out_Pa[-num_test:]

    train_torque_values = torque_values[:num_train]
    test_torque_values = torque_values[-num_test:]

    train_eff_values = eff_values[:num_train]
    test_eff_values = eff_values[-num_test:]
    

    return {
        'train_design_variable': train_design_variable,
        'test_design_variable': test_design_variable,
        'train_Pt_in_Pa': train_Pt_in_Pa,
        'test_Pt_in_Pa': test_Pt_in_Pa,
        'train_Pt_out_Pa': train_Pt_out_Pa,
        'test_Pt_out_Pa': test_Pt_out_Pa,
        'train_torque_values': train_torque_values,
        'test_torque_values': test_torque_values,
        'train_eff_values': train_eff_values,
        'test_eff_values': test_eff_values
    }
    # return train_design_variable, test_design_variable, train_torque_values, test_torque_values, train_eff_values, test_eff_values


if __name__ == "__main__":
    data = load_design_dataset()
    dataset = load_front_train_val_data(
        num_train=900,
        num_test=100  
        )
    
    print('data keys:', data.keys())
    print('dataset keys:', dataset.keys())
    print('done')

