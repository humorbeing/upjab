
try:
    from .frame16_112at15.C3D_model import C3D
    from .frame16_112at15.R2Plus1D_model import R2Plus1DClassifier
    from .frame16_112at15.R3D_model import R3DClassifier
    # print("Using relative import")
except ImportError:
    from frame16_112at15.C3D_model import C3D
    from frame16_112at15.R2Plus1D_model import R2Plus1DClassifier
    from frame16_112at15.R3D_model import R3DClassifier
    # print("Using absoloute import")


models = {
    'C3D': C3D,
    'R2Plus1D': R2Plus1DClassifier,
    'R3D': R3DClassifier,
}