import torch
import my_add_NoPackage_ext

def main():
    assert torch.cuda.is_available(), "CUDA not available"

    a = torch.tensor([1.0, 2.0, 3.0], device="cuda", dtype=torch.float32)
    b = torch.tensor([10.0, 20.0, 30.0], device="cuda", dtype=torch.float32)

    out = my_add_NoPackage_ext.add(a, b)
    print("a:", a)
    print("b:", b)
    print("out:", out)
    print("out (cpu):", out.cpu())

if __name__ == "__main__":
    main()
