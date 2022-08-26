from .common import BasicAugmentation

import numpy as np
import torch
import torch.utils.data as datautil
import torchvision.transforms as tf
from torch import nn
from PIL import Image

from abc import ABC, abstractmethod
from typing import Tuple, Callable, Optional, Union
from collections import OrderedDict, namedtuple

from ..classifiers import build_classifier
from ..utils import is_notebook
from ..evaluation import predict_class_scores, balanced_accuracy_from_predictions

if is_notebook():
    from tqdm.notebook import tqdm, trange
else:
    from tqdm import tqdm, trange


class CrossEntropyClassifier(BasicAugmentation):
    """ Standard cross-entropy classification as baseline.

    See `BasicAugmentation` for a documentation of the available hyper-parameters.
    """

    def get_loss_function(self) -> Callable:

        return nn.CrossEntropyLoss(reduction='mean')

    def get_optimizer(self, model: nn.Module, max_epochs: int, max_iter: int) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        """ Instantiates an optimizer and learning rate schedule.

        Parameters
        ----------
        model : nn.Module
            The model to be trained.
        max_epochs : int
            The total number of epochs.
        max_iter : int
            The total number of iterations (epochs * batches_per_epoch).
        
        Returns
        -------
        optimizer : torch.optim.Optimizer
        lr_schedule : torch.optim.lr_scheduler._LRScheduler
        """
        if (self.hparams['optimizer'].lower() == "adam"):
          optimizer = torch.optim.Adam(model.parameters(), lr=self.hparams['lr'], weight_decay=self.hparams['weight_decay'])
        elif (self.hparams['optimizer'].lower() == "sgd"):
          optimizer = torch.optim.SGD(model.parameters(), lr=self.hparams['lr'], momentum=self.hparams["momentum"], weight_decay=self.hparams['weight_decay'])
        else:
          raise ValueError("Insert a valid value of the optimizer")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iter)
        return optimizer, scheduler

    @staticmethod
    def default_hparams() -> dict:
        """ Returns a dictionary specifying the default values for all hyper-parameters supported by this method. """
        return { 
            **super(BasicAugmentation, BasicAugmentation).default_hparams(),
            'normalize' : True,
            'recompute_statistics' : False,
            'target_size' : None,
            'min_scale' : 1.0,
            'max_scale' : 1.0,
            'rand_shift' : 0,
            'hflip' : True,
            'vflip' : False,
            'lr' : 0.01, 
            'weight_decay' : 0.001 , 
            'optimizer': "adam",
            'momentum': 0.9,
        }
