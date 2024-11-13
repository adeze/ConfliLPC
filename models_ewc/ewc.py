"""
Implementation of Elastic weight consolidation object
"""

import torch
from torch import nn
from torch.autograd import Variable
import copy
from models_ewc.masked_cross_entropy import *
from copy import deepcopy
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss

USE_CUDA = False

def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)


class EWC(object):
    def __init__(self, model: nn.Module, dataset: list, num_labels: int, loss_fct):

        self.model = model
        self.dataset = dataset
        self.num_labels = num_labels
        self.loss_fct = loss_fct

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()

        for n, p in deepcopy(self.params).items():
            self._means[n] = variable(p.data)

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = variable(p.data)

        self.device = torch.device("cuda")

        self.model.eval()
        if isinstance(self.dataset[0], dict) or isinstance(self.dataset[0].data, dict):
            inputs = {
                key: value.squeeze(1).to(self.device) for key, value in self.dataset[0].items()
            }
            inputs["labels"] = self.dataset[1].to(self.device)
        else:
            batch = tuple(t.to(self.device) for t in self.dataset)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[3],
            }
        outputs = self.model(**inputs)
        loss = outputs[0]
        logits = outputs[1]
        labels = inputs['labels']

        self.model.zero_grad()
        if self.loss_fct:
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))  # Directly use cross_entropy which combines log_softmax and nll_loss

        loss.backward()

        for n, p in self.model.named_parameters():
            precision_matrices[n].data += p.grad.data ** 2 / len(self.dataset)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices


    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss
