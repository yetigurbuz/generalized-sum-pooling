#!/usr/bin/env python
# coding: utf-8
__all__ = ["factory"]

import tensorflow as tf
import os
import importlib

class LossFactory:
    def list_functions(self):
        for k in self.__dict__.keys():
            print(k)

factory = LossFactory()

def register_loss(name):

    def register_loss_fn(cls):
        if hasattr(cls, name):
            raise ValueError("%s loss is already registered!" % name)
        if not issubclass(cls, tf.keras.losses.Loss):
            raise ValueError("Class %s is not a subclass of %s" % (cls, tf.keras.losses.Loss))

        setattr(factory, name, cls)
        return cls
    return register_loss_fn

for file in os.listdir(os.path.dirname(__file__)):
    new_module_name = None
    if os.path.isdir(os.path.join(os.path.dirname(__file__), file)):
        if '__init__.py' in os.listdir(os.path.join(os.path.dirname(__file__), file)):
            new_module_name = file
    elif file.endswith('.py') and (not file.startswith('_')):
        new_module_name = file[:file.find('.py')]

    if new_module_name:
        module = importlib.import_module('.' + new_module_name, __name__)