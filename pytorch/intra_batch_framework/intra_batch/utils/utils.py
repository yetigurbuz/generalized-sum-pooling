import os
import os.path as osp
import torch


def make_dir(dir):
    if not osp.isdir(dir):
        os.makedirs(dir, exist_ok=True)

def freeze_model(model: torch.nn.Module,
                 exclude=[]):

    model.requires_grad_(False)

    for layer_name in exclude:

        parent = model
        can_update = True
        for attr in layer_name.split('.'):
            if hasattr(parent, attr):
                parent = getattr(parent, attr)
            else:
                print('WARNING::MODEL::FREEZE: Model has no layer named {}, thus skipping...\n'.format(layer_name))
                can_update = False

        if can_update:
            if hasattr(parent, 'parameters'):

                for p in parent.parameters():
                    if p.dtype == torch.float32:
                        p.requires_grad = True

            elif isinstance(parent, torch.nn.Parameter):
                parent.requires_grad = True
            else:
                pass


def unfreeze_model(model: torch.nn.Module,
                   exclude=[]):

    model.requires_grad_(True)

    for layer_name in exclude:

        parent = model
        for attr in layer_name.split('.'):
            parent = getattr(parent, attr)

        parent.requires_grad_(False)






