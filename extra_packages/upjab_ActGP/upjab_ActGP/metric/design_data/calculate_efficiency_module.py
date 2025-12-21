
import math

def calculate_efficiency(x, y):
    # pi = 3.141592653589793115997963468544185161590576171875
    Q = x[..., 1] / 3600
    RPM = x[..., 2] * 2 * math.pi / 60
    Pt_in = y[..., 0]
    Pt_out = y[..., 1]
    del_Pt = y[..., 2]
    Torque = y[..., 3]
    e1 = ((Pt_out - Pt_in)*Q) / (Torque * RPM) * 100
    e2 = (Pt_out*Q) / (Torque * RPM) * 100
    e3 = (del_Pt*Q) / (Torque * RPM) * 100
    return e1, e2, e3