# https://github.com/gupta-abhay/pytorch-frn/edit/master/frn.py
# https://arxiv.org/pdf/1911.09737.pdf
import torch
import torch.nn as nn

# FilterResponseNormalization
class FRN(nn.Module):
    def __init__(self, num_features, eps=1e-6, learnable_eps=False):
        """
        Input Variables:
        ----------------
            num_features: A integer indicating the number of input feature dimensions.
            eps: A scalar constant or learnable variable.
            learnable_eps: A bool value indicating whether the eps is learnable.
        """
        super(FRN, self).__init__()
        self.eps = nn.Parameter(torch.Tensor([eps]))
        self.learnable_eps = learnable_eps
        if not self.learnable_eps:
            self.eps.requires_grad_(False)
        self.gamma = nn.Parameter(torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.beta = nn.Parameter(torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.tau = nn.Parameter(torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.reset_parameters()
    
    def forward(self, x):
        """
        Input Variables:
        ----------------
            x: Input tensor of shape [NxCxHxW]
        """
        nu2 = torch.pow(x, 2).mean(dim=(2, 3), keepdim=True)
        x = x * torch.rsqrt(nu2 + torch.abs(self.eps))
        return torch.max(self.gamma * x + self.beta, self.tau)

    def reset_parameters(self):
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)
        nn.init.zeros_(self.tau)

# def convert(module, flag_name):
#     mod = module
#     before_ch = None
#     for name, child in module.named_children():
#         if hasattr(child, flag_name) and getattr(child, flag_name):
#             if isinstance(child, BatchNorm2d):
#                 before_ch = child.num_features
#                 mod.add_module(name, FRN(num_features=child.num_features))
#             # TODO bn is no good...
#             if isinstance(child, (ReLU, LeakyReLU)):
#                 mod.add_module(name, TLU(num_features=before_ch))
#         else:
#             mod.add_module(name, convert(child, flag_name))
#     return mod


# def remove_flags(module, flag_name):
#     mod = module
#     for name, child in module.named_children():
#         if hasattr(child, 'is_convert_frn'):
#             delattr(child, flag_name)
#             mod.add_module(name, remove_flags(child, flag_name))
#         else:
#             mod.add_module(name, remove_flags(child, flag_name))
#     return mod


# def bnrelu_to_frn2(model, input_size=(3, 128, 128), batch_size=2, flag_name='is_convert_frn'):
#     forard_hooks = list()
#     backward_hooks = list()

#     is_before_bn = [False]

#     def register_forward_hook(module):
#         def hook(self, input, output):
#             if isinstance(module, (nn.Sequential, nn.ModuleList)) or (module == model):
#                 is_before_bn.append(False)
#                 return

#             # input and output is required in hook def
#             is_converted = is_before_bn[-1] and isinstance(self, (ReLU, LeakyReLU))
#             if is_converted:
#                 setattr(self, flag_name, True)
#             is_before_bn.append(isinstance(self, BatchNorm2d))
#         forard_hooks.append(module.register_forward_hook(hook))

#     is_before_relu = [False]
