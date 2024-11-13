from torch.nn.modules.module import Module
from torch.nn import functional as F
from torch.nn import _reduction as _Reduction

from torch import Tensor
from typing import Optional

import warnings

import torch
from torch.nn import grad  # noqa: F401


class _Loss(Module):
    reduction: str

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction


class _WeightedLoss(_Loss):
    def __init__(self, weight: Optional[Tensor] = None, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(_WeightedLoss, self).__init__(size_average, reduce, reduction)
        self.register_buffer('weight', weight)


def log_softmax_lc(input_n: Tensor, input_b: Tensor, logits_calibraion_degree: float, dim: Optional[int] = None, _stacklevel: int = 3, dtype: Optional[int] = None) -> Tensor:
    r"""Applies a softmax followed by a logarithm.

    While mathematically equivalent to log(softmax(x)), doing these two
    operations separately is slower, and numerically unstable. This function
    uses an alternative formulation to compute the output and gradient correctly.

    See :class:`~torch.nn.LogSoftmax` for more details.

    Args:
        input_n (Tensor): input_n
        input_b (Tensor): input_b
        logits_calibraion_degree (float): logits_calibraion_degree
        dim (int): A dimension along which log_softmax will be computed.
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
          If specified, the input tensor is casted to :attr:`dtype` before the operation
          is performed. This is useful for preventing data type overflows. Default: None.
    """
    if dtype is None:
        ret = torch.sub(torch.add(torch.sub(input_n, torch.mul(torch.ones(input_n.size()).cuda(), torch.max(input_n))), torch.mul(logits_calibraion_degree, torch.sub(torch.sub(input_n, torch.mul(torch.ones(input_n.size()).cuda(), torch.max(input_n))), torch.sub(input_b, torch.mul(torch.ones(input_b.size()).cuda(), torch.max(input_b)))))),
                        torch.log(torch.sum(torch.exp(torch.add(torch.sub(input_n, torch.mul(torch.ones(input_n.size()).cuda(), torch.max(input_n))), torch.mul(logits_calibraion_degree, torch.sub(torch.sub(input_n, torch.mul(torch.ones(input_n.size()).cuda(), torch.max(input_n))), torch.sub(input_b, torch.mul(torch.ones(input_b.size()).cuda(), torch.max(input_b))))))))))
    else:
        ret = torch.sub(torch.add(torch.sub(input_n, torch.mul(torch.ones(input_n.size()).cuda(), torch.max(input_n))), torch.mul(logits_calibraion_degree, torch.sub(torch.sub(input_n, torch.mul(torch.ones(input_n.size()).cuda(), torch.max(input_n))), torch.sub(input_b, torch.mul(torch.ones(input_b.size()).cuda(), torch.max(input_b)))))),
                        torch.log(torch.sum(torch.exp(torch.add(torch.sub(input_n, torch.mul(torch.ones(input_n.size()).cuda(), torch.max(input_n))), torch.mul(logits_calibraion_degree, torch.sub(torch.sub(input_n, torch.mul(torch.ones(input_n.size()).cuda(), torch.max(input_n))), torch.sub(input_b, torch.mul(torch.ones(input_b.size()).cuda(), torch.max(input_b))))))))))

    return ret


def mselc_loss(
    input_n: Tensor,
    input_b: Tensor,
    target: Tensor,
    logits_calibraion_degree: float,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> Tensor:
    r"""mselc_loss(input_n, input_b, target, logits_calibraion_degree, size_average=None, reduce=None, reduction='mean') -> Tensor

    Measures the element-wise mean squared error with logits calibration.

    See :class:`~torch.nn.MSELoss` for details.
    """
    if not (target.size() == input_n.size() and target.size() == input_b.size()):
        warnings.warn(
            "Using a target size ({}) that is different to the input_n size ({}), input_b size ({})."
            "This will likely lead to incorrect results due to broadcasting. "
            "Please ensure they have the same size.".format(target.size(), input_n.size(), input_b.size()),
            stacklevel=2,
        )
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)

    expanded_input_n, expanded_input_b, expanded_target = torch.broadcast_tensors(input_n, input_b, target)

    reduction_enum = _Reduction.get_enum(reduction)
    if reduction_enum == 0:
        ret = torch.add(torch.pow(torch.sub(expanded_input_n, expanded_target), 2), torch.mul(logits_calibraion_degree, torch.pow(torch.sub(expanded_input_n, expanded_input_b), 2)))
    elif reduction_enum == 1:
        ret = torch.mean(torch.add(torch.pow(torch.sub(expanded_input_n, expanded_target), 2), torch.mul(logits_calibraion_degree, torch.pow(torch.sub(expanded_input_n, expanded_input_b), 2))))
    elif reduction_enum == 2:
        ret = torch.sum(torch.add(torch.pow(torch.sub(expanded_input_n, expanded_target), 2), torch.mul(logits_calibraion_degree, torch.pow(torch.sub(expanded_input_n, expanded_input_b), 2))))
    else:
        ret = -1  # TODO: remove once JIT exceptions support control flow
        raise ValueError("{} is not a valid value for reduction".format(reduction))

    return ret


def celc_loss(
    input_n: Tensor,
    input_b: Tensor,
    target: Tensor,
    logits_calibraion_degree: float,
    weight: Optional[Tensor] = None,
    size_average: Optional[bool] = None,
    ignore_index: int = -100,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> Tensor:
    r"""This criterion combines `log_softmax` and `nll_loss` in a single
    function.

    See :class:`~torch.nn.CrossEntropyLoss` for details.

    Args:
        input_n (Tensor) : :math:`(N, C)` where `C = number of classes` or :math:`(N, C, H, W)`
            in case of 2D Loss, or :math:`(N, C, d_1, d_2, ..., d_K)` where :math:`K \geq 1`
            in the case of K-dimensional loss.
        input_b (Tensor) : :math:`(N, C)` where `C = number of classes` or :math:`(N, C, H, W)`
            in case of 2D Loss, or :math:`(N, C, d_1, d_2, ..., d_K)` where :math:`K \geq 1`
            in the case of K-dimensional loss.
        target (Tensor) : :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`,
            or :math:`(N, d_1, d_2, ..., d_K)` where :math:`K \geq 1` for
            K-dimensional loss.
        logits_calibraion_degree (float): logits_calibraion_degree.
        weight (Tensor, optional): a manual rescaling weight given to each
            class. If given, has to be a Tensor of size `C`
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. When :attr:`size_average` is
            ``True``, the loss is averaged over non-ignored targets. Default: -100
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Examples::

        >>> import loss_lc
        >>> input_n = torch.randn(3, 5, requires_grad=True)
        >>> input_b = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.randint(5, (3,), dtype=torch.int64)
        >>> logits_calibraion_degree = torch.rand(1)
        >>> loss = loss_lc.celc_loss(input_n, input_b, target, logits_calibraion_degree)
        >>> loss.backward()
    """
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)
    return F.nll_loss(log_softmax_lc(input_n, input_b, logits_calibraion_degree, 1), target, weight, None, ignore_index, None, reduction)


class MSELCLoss(_Loss):
    r"""Creates a criterion that measures the mean squared error (squared L2 norm) between
    each element in the input :math:`x` and target :math:`y`.

    The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = \left( x_n - y_n \right)^2,

    where :math:`N` is the batch size. If :attr:`reduction` is not ``'none'``
    (default ``'mean'``), then:

    .. math::
        \ell(x, y) =
        \begin{cases}
            \operatorname{mean}(L), &  \text{if reduction} = \text{`mean';}\\
            \operatorname{sum}(L),  &  \text{if reduction} = \text{`sum'.}
        \end{cases}

    :math:`x` and :math:`y` are tensors of arbitrary shapes with a total
    of :math:`n` elements each.

    The mean operation still operates over all the elements, and divides by :math:`n`.

    The division by :math:`n` can be avoided if one sets ``reduction = 'sum'``.

    Args:
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when :attr:`reduce` is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Shape:
        - Input_n: :math:`(N, *)` where :math:`*` means, any number of additional
          dimensions
        - Input_b: :math:`(N, *)` where :math:`*` means, any number of additional
          dimensions
        - Target: :math:`(N, *)`, same shape as the input
        - logits_calibraion_degree (float): logits_calibraion_degree


    Examples::

        >>> import loss_lc
        >>> loss = loss_lc.MSELCLoss()
        >>> input_n = torch.randn(3, 5, requires_grad=True)
        >>> input_b = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.randn(3, 5)
        >>> logits_calibraion_degree = torch.rand(1)
        >>> output = loss(input_n, input_b, target, logits_calibraion_degree)
        >>> output.backward()
    """
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(MSELCLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input_n: Tensor, input_b: Tensor, target: Tensor, logits_calibraion_degree: float) -> Tensor:
        return mselc_loss(input_n, input_b, target, logits_calibraion_degree, reduction=self.reduction)


class CELCLoss(_WeightedLoss):
    r"""This criterion combines :class:`~torch.nn.LogSoftmax` and :class:`~torch.nn.NLLLoss` in one single class.

    It is useful when training a classification problem with `C` classes.
    If provided, the optional argument :attr:`weight` should be a 1D `Tensor`
    assigning weight to each of the classes.
    This is particularly useful when you have an unbalanced training set.

    The `input` is expected to contain raw, unnormalized scores for each class.

    `input` has to be a Tensor of size either :math:`(minibatch, C)` or
    :math:`(minibatch, C, d_1, d_2, ..., d_K)`
    with :math:`K \geq 1` for the `K`-dimensional case (described later).

    This criterion expects a class index in the range :math:`[0, C-1]` as the
    `target` for each value of a 1D tensor of size `minibatch`; if `ignore_index`
    is specified, this criterion also accepts this class index (this index may not
    necessarily be in the class range).

    The loss can be described as:

    .. math::
        \text{loss}(x, class) = -\log\left(\frac{\exp(x[class])}{\sum_j \exp(x[j])}\right)
                       = -x[class] + \log\left(\sum_j \exp(x[j])\right)

    or in the case of the :attr:`weight` argument being specified:

    .. math::
        \text{loss}(x, class) = weight[class] \left(-x[class] + \log\left(\sum_j \exp(x[j])\right)\right)

    The losses are averaged across observations for each minibatch. If the
    :attr:`weight` argument is specified then this is a weighted average:

    .. math::
        \text{loss} = \frac{\sum^{N}_{i=1} loss(i, class[i])}{\sum^{N}_{i=1} weight[class[i]]}

    Can also be used for higher dimension inputs, such as 2D images, by providing
    an input of size :math:`(minibatch, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1`,
    where :math:`K` is the number of dimensions, and a target of appropriate shape
    (see below).


    Args:
        weight (Tensor, optional): a manual rescaling weight given to each class.
            If given, has to be a Tensor of size `C`
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when :attr:`reduce` is ``False``. Default: ``True``
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. When :attr:`size_average` is
            ``True``, the loss is averaged over non-ignored targets.
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will
            be applied, ``'mean'``: the weighted mean of the output is taken,
            ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in
            the meantime, specifying either of those two args will override
            :attr:`reduction`. Default: ``'mean'``

    Shape:
        - Input_n: :math:`(N, C)` where `C = number of classes`, or
          :math:`(N, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1`
          in the case of `K`-dimensional loss.
        - Input_b: :math:`(N, C)` where `C = number of classes`, or
          :math:`(N, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1`
          in the case of `K`-dimensional loss.
        - Target: :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`, or
          :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 1` in the case of
          K-dimensional loss.
        - logits_calibraion_degree (float): logits_calibraion_degree
        - Output: scalar.
          If :attr:`reduction` is ``'none'``, then the same size as the target:
          :math:`(N)`, or
          :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 1` in the case
          of K-dimensional loss.

    Examples::

        >>> import loss_lc
        >>> loss = loss_lc.CELCLoss()
        >>> input_n = torch.randn(3, 5, requires_grad=True)
        >>> input_b = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.empty(3, dtype=torch.long).random_(5)
        >>> logits_calibraion_degree = torch.rand(1)
        >>> output = loss(input_n, input_b, target, logits_calibraion_degree)
        >>> output.backward()
    """
    __constants__ = ['ignore_index', 'reduction']
    ignore_index: int

    def __init__(self, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean') -> None:
        super(CELCLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, input_n: Tensor, input_b: Tensor, target: Tensor, logits_calibraion_degree: float) -> Tensor:
        assert self.weight is None or isinstance(self.weight, Tensor)
        return celc_loss(input_n, input_b, target, logits_calibraion_degree, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)
