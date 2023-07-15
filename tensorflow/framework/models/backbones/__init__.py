#!/usr/bin/env python
# coding: utf-8
__all__ = ["factory"]

import os
import importlib

class Backbones:
    def list_functions(self):
        for k in self.__dict__.keys():
            print(k)

factory = Backbones()

def register_backbone(name):

    def register_backbone_fn(cls):
        if hasattr(factory, name):
            raise ValueError("%s layer is already registered!" % name)

        setattr(factory, name, cls)
        return cls
    return register_backbone_fn


for file in os.listdir(os.path.dirname(__file__)):
    new_module_name = None
    if os.path.isdir(os.path.join(os.path.dirname(__file__), file)):
        if '__init__.py' in os.listdir(os.path.join(os.path.dirname(__file__), file)):
            new_module_name = file
    elif file.endswith('.py') and (not file.startswith('_')):
        new_module_name = file[:file.find('.py')]

    if new_module_name:
        module = importlib.import_module('.' + new_module_name, __name__)
