# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Bad gradients in PyTorch graph

# %%

import torch
from draugr.torch_utilities import register_bad_grad_hooks

x = torch.randn(10, 10, requires_grad=True)
y = torch.randn(10, 10, requires_grad=True)

z = x / (y * 0)
z = z.sum() * 2
get_dot = register_bad_grad_hooks(z)
z.backward()
dot = get_dot()
# dot.save('tmp.dot') # to get .dot
# dot.render('tmp') # to get SVG
dot  # in Jupyter, you can just render the variable
