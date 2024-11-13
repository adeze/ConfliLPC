# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""LPC optimizer"""
from __future__ import print_function

import logging
import math
import numpy as np

import torch
from torch.optim import Optimizer
import torch.optim as optim


logger = logging.getLogger(__name__)


def anneal_function(function, step, k, t0, weight):
    if function == 'sigmoid':
        return float(1 / (1 + np.exp(-k * (step - t0)))) * weight
    elif function == 'linear':
        return min(1, step / t0) * weight
    elif function == 'constant':
        return weight
    else:
        ValueError


class LPC(Optimizer):
    """ Implementation of LPC local sgd optimizer, a variant of LPC optimizer.

    Parameters:
        reg_lambda: hyperparameter for the regularizer. Default: 1.0
        lr (float): learning rate. Default 1e-3.
        betas (tuple of 2 floats): Adams beta parameters (b1, b2). Default: (0.9, 0.999)
        eps (float): Adams epsilon. Default: 1e-6
        weight_decay (float): Weight decay. Default: 0.0
        correct_bias (bool): can be set to False to avoid correcting bias in Adam (e.g. like in Bert TF repository). Default True.
        anneal_fun (str): a hyperparam for the anneal function, decide the function of the curve. Default 'sigmoid'.
        anneal_k (float): a hyperparam for the anneal function, decide the slop of the curve. Choice: [0.05, 0.1, 0.2, 0.5, 1]
        anneal_t0 (float): a hyperparam for the anneal function, decide the middle point of the curve. Choice: [100, 250, 500, 1000]
        anneal_w (float): a hyperparam for the anneal function, decide the scale of the curve. Default 1.0.
        pretrain_cof (float): the coefficient of the quadratic penalty. Default 5000.0.
        pretrain_params (list of tensors): the corresponding group of params in the pretrained model.
    """

    def __init__(self, params, reg_lambda, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.0, correct_bias=True,
                 anneal_fun='sigmoid', anneal_k=0, anneal_t0=0, anneal_w=1.0, pretrain_cof=5000.0, pretrain_params=None):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(reg_lambda=reg_lambda, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias,
                        anneal_fun=anneal_fun, anneal_k=anneal_k, anneal_t0=anneal_t0, anneal_w=anneal_w,
                        pretrain_cof=pretrain_cof, pretrain_params=pretrain_params)
        super(LPC, self).__init__(params, defaults)
        self.reg_lambda = reg_lambda

    def __setstate__(self, state):
        super(LPC, self).__setstate__(state)

    def step(self, reg_params, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p, pp in zip(group["params"], group["pretrain_params"]):
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                param_dict = reg_params[p]
                omega = param_dict['omega']
                omega = omega.to(p.device)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(1.0 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1.0 - beta2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                # With LPC method, the optimization objective is
                # Loss = lambda(t)*Loss_N + (1-lambda(t))*Loss_B
                # Loss = lambda(t)*Loss_N + (1-lambda(t))*\delta\gamma\Omega*\sum((\theta_i-\theta_i^*)^2)
                if group['anneal_w'] > 0.0:
                    # We calculate the lambda as the annealing function
                    anneal_lambda = anneal_function(group['anneal_fun'], state["step"], group['anneal_k'],
                                                    group['anneal_t0'], group['anneal_w'])
                    assert anneal_lambda <= group['anneal_w']
                    # The loss of the target task is multiplied by lambda(t)
                    p.data.addcdiv_(-step_size * anneal_lambda, exp_avg, denom)
                    # Add the quadratic penalty to simulate the pretraining tasks
                    p.data.add_(-group["lr"] * (group['anneal_w'] - anneal_lambda) * group["pretrain_cof"], torch.mul(2 * self.reg_lambda * omega, p.data - pp.data.to(p.device)))
                else:
                    p.data.addcdiv_(-step_size, exp_avg, denom)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    p.data.add_(-group["lr"] * group["weight_decay"], p.data)

        return loss


class LPC_omega_update(optim.SGD):
    # update omega

    def __init__(self, params, lr=0.001):
        super(LPC_omega_update, self).__init__(params, lr)

    def __setstate__(self, state):
        super(LPC_omega_update, self).__setstate__(state)

    def step(self, args, reg_params, batch_index, batch_size, closure=None):
        loss = None

        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue

                # The absolute value of the grad_data that is to be added to omega
                grad_data_copy = p.grad.data.clone()
                grad_data_copy = grad_data_copy.abs()

                param_dict = reg_params[p]

                omega = param_dict['omega']
                omega = omega.to(args.device)

                param_dict['prev_omega'] = omega

                current_size = (batch_index + 1) * batch_size
                step_size = 1 / float(current_size)

                # Incremental update for the omega
                omega = omega + step_size * (grad_data_copy - batch_size * (omega))

                param_dict['omega'] = omega

                reg_params[p] = param_dict

        return loss


def consolidate_reg_params(model):
    """
    Input:
    1) model: A reference to the model that is being trained
    2) reg_params: A dictionary containing importance weights (omega), init_val (keep a reference
    to the initial values of the parameters) for all trainable parameters

    Output:
    1) reg_params: A dictionary containing importance weights (omega), init_val (keep a reference
    to the initial values of the parameters) for all trainable parameters


    Function: This function updates the value (adds the value) of omega across the tasks that the model is
    exposed to

    """
    # Get the reg_params for the model
    reg_params = model.reg_params

    for name, param in model.named_parameters():
        param_dict = reg_params[param]

        # Store the previous values of omega
        prev_omega = param_dict['prev_omega']
        new_omega = param_dict['omega']

        new_omega = torch.add(prev_omega, new_omega)
        del param_dict['prev_omega']

        param_dict['omega'] = new_omega

        # the key for this dictionary is the name of the layer
        reg_params[param] = param_dict

    model.reg_params = reg_params

    return model


def compute_omega_grads_norm(args, model, dataloader, optimizer):
    """
    Inputs:
    1) model: A reference to the model for which omega is to be calculated
    2) reg_params: A dictionary containing importance weights (omega), init_val (keep a reference
    to the initial values of the parameters) for all trainable parameters
    3) dataloader: A dataloader to feed the data to the model
    4) optimizer: An instance of the "omega_update" class
    5) use_gpu: Flag is set to True if the model is to be trained on the GPU

    Outputs:
    1) model: An updated reference to the model is returned

    Function: Global version for computing the l2 norm of the function (neural network's) outputs. In
    addition to this, the function also accumulates the values of omega across the items of a task

    """
    model.eval()

    index = 0
    for data in dataloader:
        data = tuple(t.to(args.device) for t in data)

        # get the inputs and labels
        if args.task != 'qa':
            inputs = {"input_ids": data[0], "attention_mask": data[1], "labels": data[3]}
        else:
            inputs = {"input_ids": data[0], "attention_mask": data[1]}

        # Zero the parameter gradients
        optimizer.zero_grad()

        # get the function outputs
        outputs = model(**inputs)[1]
        if args.multi_label:
            outputs = outputs.sigmoid()

        # compute the sqaured l2 norm of the function outputs
        if args.task == 'ner':
            active_outputs = outputs.view(-1, args.num_labels)
            l2_norm = torch.norm(active_outputs, 2, dim=1)
            del active_outputs
        else:
            l2_norm = torch.norm(outputs, 2, dim=1)
        del outputs

        squared_l2_norm = l2_norm ** 2
        del l2_norm

        sum_norm = torch.sum(squared_l2_norm)
        del squared_l2_norm

        # compute gradients for these parameters
        sum_norm.backward()

        # optimizer.step computes the omega values for the new batches of data
        if args.task == 'ner':
            batch_size = len(inputs['labels'].view(-1))
        else:
            batch_size = len(inputs['input_ids'])
        optimizer.step(args, model.reg_params, index, batch_size)
        del inputs

        index = index + 1

    return model
