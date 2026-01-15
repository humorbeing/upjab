import torch
from build_cu_cpp import my_add_ext

from build_cu_cpp import my_add_forward_backward_ext


class AddFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        # Save nothing needed for add; just return forward result
        return my_add_forward_backward_ext.forward(a.contiguous(), b.contiguous())

    @staticmethod
    def backward(ctx, grad_out):
        grad_out = grad_out.contiguous()
        grad_a, grad_b = my_add_forward_backward_ext.backward(grad_out)
        return grad_a, grad_b

def my_add(a, b):
    return AddFn.apply(a, b)




def main():
    assert torch.cuda.is_available(), "CUDA not available"

    a = torch.tensor([1.0, 2.0, 3.0], device="cuda", dtype=torch.float32, requires_grad=True)
    b = torch.tensor([10.0, 20.0, 30.0], device="cuda", dtype=torch.float32, requires_grad=True)

    out = my_add_ext.add(a, b)
    print("a:", a)
    print("b:", b)
    print("out:", out)
    print("out (cpu):", out.cpu())

    out = my_add(a, b)
    loss = out.sum()          # simple scalar so backward is easy
    loss.backward()

    print("out:", out)
    print("loss:", loss.item())
    print("a.grad:", a.grad)
    print("b.grad:", b.grad)

    
if __name__ == "__main__":
    main()
