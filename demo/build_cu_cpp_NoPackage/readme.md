
## Install
### `.so` file saved to upjab choosen folder
```bash
python demo/build_cu_cpp_NoPackage/setup.py build_ext --build-lib demo/build_cu_cpp_NoPackage --build-temp demo/build_cu_cpp_NoPackage/build
```

### `.so` file saved to current location folder
```bash
python setup.py build_ext --inplace
```