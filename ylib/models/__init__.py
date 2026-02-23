# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .dhpdetr import build


def build_model(args, num_classes, num_ext_classes):
    return build(args, num_classes, num_ext_classes)
