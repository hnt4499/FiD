# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import inspect
import logging
from typing import List, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor as T
from torch.utils.checkpoint import checkpoint

import numpy as np
import transformers
from transformers.models.t5.modeling_t5 import (
    T5ForConditionalGeneration,
    T5Model,
    T5Config,
    T5Stack,
    T5Block,
    T5LayerNorm,
    T5LayerFF,
    T5LayerSelfAttention,
    T5LayerCrossAttention,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
)


logger = logging.getLogger(__name__)


class DotDict:
    """
    DotDict but without inheriting from dict itself. Helper to convert
    dict-like output types (e.g., Seq2SeqLMOutput) to non-dict-like outputs
    with exactly the same functionalities.
    """
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __contains__(self, key):
        return hasattr(self, key)

    def __getitem__(self, key):
        if not hasattr(self, key):
            raise KeyError(key)
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)


class CustomT5Block(T5Block):
    """
    Custom T5 block that allows the use of pre-initialized T5 layers.
    """
    def __init__(
        self,
        config,
        has_relative_attention_bias: bool = False,
        layers: nn.ModuleList = None,
    ):
        super(T5Block, self).__init__()  # note that call stack
        self.config = config
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias

        if layers is None:
            self.layer = nn.ModuleList()
            self.layer.append(T5LayerSelfAttention(
                config,
                has_relative_attention_bias=has_relative_attention_bias
            ))
            if self.is_decoder:
                self.layer.append(T5LayerCrossAttention(
                    config,
                ))

            self.layer.append(T5LayerFF(config))
        else:
            self.layer = layers

    def to_encoder(self):
        """
        Return a new object that shares parameters with the current object,
        and with the cross attention layer removed. This function should
        only be called from a T5 block of a decoder (NOT encoder).
        """
        assert self.is_decoder, (
            "This function must be called from a block of a decoder"
        )

        layers = nn.ModuleList([
            self.layer[0],
            self.layer[2],
        ])  # remove the second layer
        config = copy.deepcopy(self.config)
        config.is_decoder = False

        return CustomT5Block(
            config,
            self.has_relative_attention_bias,
            layers=layers,
        )


class CustomT5Stack(T5Stack):
    """
    Custom T5 stack that allows the use of pre-initialized T5Blocks.
    """
    def __init__(
        self,
        config,
        embed_tokens: nn.Embedding = None,
        block: nn.ModuleList = None,
    ):
        super(T5Stack, self).__init__(config)  # note the call stack

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        if block is None:
            self.block = nn.ModuleList(
                [CustomT5Block(config, has_relative_attention_bias=bool(i == 0))
                 for i in range(config.num_layers)]
            )
        else:
            self.block = block

        self.final_layer_norm = T5LayerNorm(
            config.d_model,
            eps=config.layer_norm_epsilon,
        )  # we don't need to share this layer
        self.dropout = nn.Dropout(config.dropout_rate)

        self.init_weights()
        # Model parallel
        self.model_parallel = False
        self.device_map = None


class FiDT5(T5ForConditionalGeneration):
    def __init__(
        self,
        config,
        num_passages: int,
        device: str,
        gradient_checkpointing: bool = True,
        share_encoder_decoder: bool = False,
    ):
        # Note the call stack
        super(T5ForConditionalGeneration, self).__init__(config)

        self.model_dim = config.d_model
        self.num_passages = num_passages
        self._device = device
        self.gradient_checkpointing = gradient_checkpointing
        self.share_encoder_decoder = share_encoder_decoder

        # Embedding look up
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        # Initialize the decoder first as it has more layers
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        self.decoder = CustomT5Stack(decoder_config, self.shared, block=None)

        # Initialize the encoder, with the option of parameter sharing
        if share_encoder_decoder:
            logger.info(
                f"[{self.__class__.__name__}] Sharing the weights of encoder "
                f"and decoder architectures"
            )
            block = self.decoder.block

            # Need to convert this block to be compatible with encoder module
            new_block = nn.ModuleList()
            for block_i in block:
                block_i: CustomT5Block
                block_i = block_i.to_encoder()
                new_block.append(block_i)
            block = new_block

        else:
            logger.info(
                f"[{self.__class__.__name__}] Weights of the encoder and "
                f"decoder are not shared!"
            )
            block = None

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = CustomT5Stack(encoder_config, self.shared, block=block)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    @classmethod
    def init_model(
        cls,
        cfg_name: str,
        num_passages: int,
        device: str,
        dropout: float = 0.1,
        pretrained: bool = True,
        gradient_checkpointing: bool = True,
        share_encoder_decoder: bool = False,
        **kwargs,
    ) -> T5Model:
        """
        Main interface to initialize a FiD model.

        Parameters
        ----------
        num_passages : int
            Number of top-k passages for the reader to consider (i.e., read) for
            each example.
        gradient_checkpointing : bool
            Whether to enable gradient checkpointing.
        share_encoder_decoder : bool
            Whether to share the weights of the encoder and decoder components
            of the underlying T5 model.
        """
        cfg = T5Config.from_pretrained(cfg_name, **kwargs)
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout

        if pretrained:
            model = cls.from_pretrained(
                cfg_name,
                config=cfg,
                num_passages=num_passages,
                device=device,
                gradient_checkpointing=gradient_checkpointing,
                share_encoder_decoder=share_encoder_decoder,
            )
        else:
            model = cls(
                cfg,
                num_passages=num_passages,
                device=device,
                gradient_checkpointing=gradient_checkpointing,
                share_encoder_decoder=share_encoder_decoder,
            )

        # Wrap after initialized
        model.lazy_wrap()
        return model

    def set_num_passages(self, num_passages: int):
        self.num_passages = num_passages
        self.encoder.set_num_passages(num_passages)

    @property
    def is_wrapped(self):
        return isinstance(self.encoder, FiDT5Encoder)

    def lazy_wrap(self):
        """
        Wraps only the encoder such that:
        1. Input tensors are properly handled specific for FiD model (i.e., each
        top-k passage is processed independently).
        2. Gradient checkpointing is integrated into appropriate modules, since
        HF doesn't provide this functionality for T5 yet.
        """
        if self.is_wrapped:
            raise RuntimeError("Lazy wrapping should only be called once.")

        # Wrap encoder with gradient checkpointing and custom forward pass logic
        self.encoder = FiDT5Encoder(
            self.encoder,
            num_passages=self.num_passages,
            device=self._device,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        # Wrap decoder with gradient checkpointing
        self.decoder = wrap_gradient_checkpointing(
            self.decoder,
            device=self._device,
            stack_level=False,
            gradient_checkpointing=self.gradient_checkpointing,
        )

    def forward(self, input_ids: T = None, attention_mask: T = None, **kwargs):
        """
        We need to resize the input ids and attention mask as (B, N * L) instead
        of (B * N, L) here because the T5 forward method uses the input tensors'
        shape to infer dimensions used in the decoder.
        Note that in EncoderWrapper, the input ids and attention mask are
        re-resized as (B * N, L).
        """
        if input_ids is not None and input_ids.ndim == 3:
            assert input_ids.shape[1] == self.num_passages
            assert attention_mask.shape[1] == self.num_passages

        if input_ids is not None:
            input_ids = input_ids.view(input_ids.size(0), -1)  # (B, N * L)
        if attention_mask is not None:
            attention_mask = attention_mask.view(attention_mask.size(0), -1)

        # Somehow torch distributed causes OrderedDict turned into dict
        # Without this, the process will fail at
        # https://github.com/huggingface/transformers/blob/96d1cfb13db094d1468883060ec2d4471f63fd01/src/transformers/models/t5/modeling_t5.py#L1569
        # where we try to index an OrderedDict-like object, which have already
        # gotten turned into a plain python dictionary.
        if (not self.training) and isinstance(kwargs["encoder_outputs"], dict) \
                and "encoder_outputs" in kwargs:
            kwargs["encoder_outputs"] = \
                BaseModelOutputWithPastAndCrossAttentions(
                    **kwargs["encoder_outputs"])

        outputs = super(FiDT5, self).forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

        # Somehow torch distributed causes OrderedDict turned into dict
        # Without this, the process will fail at
        # https://github.com/huggingface/transformers/blob/96d1cfb13db094d1468883060ec2d4471f63fd01/src/transformers/generation_utils.py#L1303
        # where we try to get an attribute from an OrderedDict-like object, which have already
        # gotten turned into a plain python dictionary.
        if (not self.training) and isinstance(outputs, dict) and \
                "logits" in outputs:
            outputs = DotDict(**outputs)
        return outputs

    def generate(self, input_ids: T, attention_mask: T, max_length: int):
        assert input_ids.ndim == 3 and input_ids.shape[1] == self.num_passages
        assert attention_mask.ndim == 3 and attention_mask.shape[1] \
            == self.num_passages

        # We need to resize the inputs here, as the generate method expect 2D
        # tensors
        return super(FiDT5, self).generate(
            input_ids=input_ids.view(input_ids.size(0), -1),
            attention_mask=attention_mask.view(attention_mask.size(0), -1),
            max_length=max_length,
        )


class FiDT5Encoder(nn.Module):
    """
    Wrapper for T5 encoder to obtain a Fusion-in-Decoder model.

    Wraps only the encoder such that:
    1. Input tensors are properly handled specific for FiD model (i.e., each
    top-k passage is processed independently).
    2. Gradient checkpointing is integrated into appropriate modules, since
    HF doesn't provide this functionality for T5 yet.

    Parameters
    ----------
    num_passages : int
        Number of top-k passages for the reader to consider (i.e., read) for
        each example.
    gradient_checkpointing : bool
        Whether to enable gradient checkpointing.
    """
    def __init__(
        self,
        encoder: CustomT5Stack,
        num_passages: int,
        device: str,
        gradient_checkpointing: bool = True,
    ):
        super(FiDT5Encoder, self).__init__()
        self.num_passages = num_passages
        self.encoder = encoder

        # Wrap each block in the T5 encoder with gradient checkpointing
        wrap_gradient_checkpointing(
            encoder,
            device=device,
            stack_level=False,
            gradient_checkpointing=gradient_checkpointing,
        )

    def set_num_passages(self, num_passages: int):
        self.num_passages = num_passages

    def forward(self, input_ids: T, attention_mask: T, **kwargs):
        """
        Forward pass logic, which takes care of processing input passages
        independently.
        Because of the forward pass of `FiDT5`, the inputs was resized to
        (B, N * L) (to comply with T5). We hence need to reshape it the other
        way round (i.e., (B * N, L)), so that each passage is processed
        independently.
        """
        # Sanity check
        batch_size, total_length = input_ids.shape
        assert (total_length % self.num_passages) == 0
        passage_length = total_length // self.num_passages

        input_ids = input_ids.view(-1, passage_length)
        attention_mask = attention_mask.view(-1, passage_length)

        outputs: BaseModelOutputWithPastAndCrossAttentions = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        outputs.last_hidden_state = outputs.last_hidden_state.view(
            batch_size, self.num_passages * passage_length, -1,
        )
        return outputs


def _flatten(
    inputs,
    flattened_inputs: list,
    structure: list,
    only_tensor: bool,
):
    """
    Helper for the `flatten` function.
    """
    if isinstance(inputs, (list, tuple)):
        sub_structure = []
        for input in inputs:
            _flatten(input, flattened_inputs, sub_structure, only_tensor)
        structure.append(tuple(sub_structure))

    elif (only_tensor and (inputs is None or isinstance(inputs, T))) \
            or (not only_tensor):
        flattened_inputs.append(inputs)
        structure.append(-1)

    else:
        raise TypeError(f"Unrecognized input type: {type(inputs)}")


def flatten(inputs, only_tensor: bool):
    """"
    Recursively traverse the inputs and flatten it into a list of tensors.
    Also returns a list of structured objects that helps to recover the
    original inputs.

    Parameters
    ----------
    inputs
        Input to flatten. Recursively speaking, sub-object of this `inputs`
        (including `inputs` itself) should be either a None, a tensor or a
        tuple/list of these types. See `only_tensor`.
    only_tensor : bool
        If True, recursively check if every "leaf" object is either None or a
        torch tensor. Raise an error if an object of another type is found.
        This is useful for checking for outputs of `forward`
        functions.

    Returns
    -------
        A tuple of (flattened_inputs, structure), where the `structure` object
        could be used to recover the original structured inputs.
    """
    flattened_inputs = []
    structure = []
    _flatten(inputs, flattened_inputs, structure, only_tensor)
    return tuple(flattened_inputs), structure[0]


def _unflatten(inputs, unflattened_inputs: list, structure, index: int = 0):
    """
    Helper for the `unflatten` function.
    """
    if isinstance(structure, int):
        unflattened_inputs.append(inputs[index])
        return index + 1
    elif isinstance(structure, (list, tuple)):
        unflattened_input = []

        for sub_structure in structure:
            index = _unflatten(inputs, unflattened_input, sub_structure, index)
        unflattened_inputs.append(tuple(unflattened_input))
        return index
    else:
        raise TypeError(f"Unrecognized length object type: {type(structure)}")


def unflatten(inputs, structure: list):
    """
    Recover the flatten inputs obtained from `flatten` function.
    """
    unflatten_inputs = []
    _unflatten(inputs, unflatten_inputs, structure)
    return unflatten_inputs[0]


def convert_kwargs_to_args(func, args, kwargs, exclude_self: bool):
    """
    Inspect the input function `func` and convert positional arguments
    (`args`) and keyword arguments (`kwargs`) to a tuple of positional arguments
    (only `args`), while filling missing keyword arguments with their default
    values.

    Example:
    >>> def f(x, y=None, z=1, t=False):
    >>>     pass
    >>> convert_kwargs_to_args(f)  # raise TypeError, since `x` is required
    >>> convert_kwargs_to_args(f, 1)  # (1, None, 1, False)
    >>> convert_kwargs_to_args(f, 1, t=True, y=[1, 2, 3])  # (1, [1, 2, 3], 1, True)

    Parameters
    ----------
    exclude_self : bool
        Whether to exclude `self` from the list of arguments. Useful for methods
        (not functions).
    """
    # Get function specs
    func_spec = inspect.getfullargspec(func)
    func_args = func_spec.args[1:] if exclude_self else func_spec.args
    func_defaults = func_spec.defaults  # kwargs' defaults
    func_defaults = [] if func_defaults is None else list(func_defaults)

    # Sanity check
    if len(args) < len(func_args) - len(func_defaults):
        num_missing = len(func_args) - len(func_defaults) - len(args)
        args_missing = func_args[len(args):len(args) + num_missing]
        args_missing = [
            arg_missing for arg_missing in args_missing
            if arg_missing not in kwargs
        ]

        if len(args_missing) > 0:
            raise TypeError(
                f"Missing {num_missing} required positional argument: "
                f"{args_missing}"
            )

    # Process args and kwargs, merge them into args
    all_args = list(args).copy()
    # Collect all positional arguments that have been fed as keyword arguments
    while len(all_args) < len(func_args) - len(func_defaults):
        arg_name = func_args[len(all_args)]
        assert arg_name in kwargs  # this has been checked in the previous code
        value = kwargs.pop(arg_name)
        all_args.append(value)

    # Collect all keyword arguments in sequential order, filling missing values
    # with default values
    num_remain_args = len(func_args) - len(all_args)
    remain_args = func_args[-num_remain_args:]
    remain_defaults = func_defaults[-num_remain_args:]

    for remain_arg, remain_default in zip(remain_args, remain_defaults):
        if remain_arg in kwargs:
            value = kwargs.pop(remain_arg)
            all_args.append(value)
        else:
            all_args.append(remain_default)

    assert len(kwargs) == 0
    return tuple(all_args)


def is_tensor(object):
    """
    Recursively check whether the input object is a tensor or a tuple / list of
    tensor. Note that if one of the (sub-)objects is a tuple / list containing
    some tensors and some non-tensors, this function will return False,
    considering that object as a non-tensor object.
    """
    if isinstance(object, T):
        return True
    elif isinstance(object, (list, tuple)):
        is_tensors = [is_tensor(o) for o in object]
        return all(is_tensors)
    else:
        return False


def wrap_function(func, args, exclude_self: bool, _func=None):
    """
    This function pre-fills all non-tensor arguments with their provided values
    and return a new function (wrapped over the original function) that only
    accepts tensor arguments.

    Parameters
    ----------
    args : tuple
        Tuple of processed positional arguments returned by
        `convert_kwargs_to_args`.
    exclude_self : bool
        Whether to exclude `self` from the list of arguments. Useful for
        methods (not functions).
    _func
        Function to be wrapped over. Normal usage is that: `func` is the forward
        function and `_func` is the `__call__` function. If not provided, use
        `func` instead.

    Returns
    -------
    A tuple of (wrapped function, args), where args contains only tensor
    objects. Calling `func(args)` would return the desired results.
    """
    # Get function specs
    func_spec = inspect.getfullargspec(func)
    func_args = func_spec.args[1:] if exclude_self else func_spec.args
    if len(func_args) != len(args):
        raise ValueError(
            f"Arguments mismatch. Got {len(args)} arguments, while the "
            f"function requires {len(func_args)} arguments. Make sure to check "
            f"`exclude_self` option and that `args` is processed using the "
            f"`convert_kwargs_to_args` function."
        )

    # Process args and kwargs
    processed_arg_values = []  # tensor container
    processed_arg_names = []  # contains argument name of `processed_args`
    processed_kwargs = {}

    for arg_value, arg_name in zip(args, func_args):
        if is_tensor(arg_value):
            processed_arg_values.append(arg_value)
            processed_arg_names.append(arg_name)
        else:
            processed_kwargs[arg_name] = arg_value

    # Wrap function
    _func = _func if _func is not None else func
    def wrapper(*args):
        kwargs = processed_kwargs.copy()
        args_with_names = dict(zip(processed_arg_names, args))
        kwargs.update(args_with_names)
        return _func(**kwargs)

    return wrapper, processed_arg_values


class CheckpointWrapper(nn.Module):
    """
    Wrapper replacing None outputs by empty tensors, which allows the use of
    checkpointing. This is needed since checkpointing requires output to be
    a tensor or a tuple of tensors.
    """
    def __init__(
        self,
        module: CustomT5Block,
        device: str,
        gradient_checkpointing: bool = True,
    ):
        super(CheckpointWrapper, self).__init__()
        self.module = module
        self._device = device
        self.gradient_checkpointing = gradient_checkpointing

    def forward(self, *args, **kwargs):
        """
        Forward pass with gradient checkpointing.
        """
        if self.gradient_checkpointing and self.training:
            # Convert all arguments into positional arguments; here we assume
            # that the function of interest is `self.module.forward`
            args = convert_kwargs_to_args(
                self.module.forward,
                args=args,
                kwargs=kwargs,
                exclude_self=True,
            )
            # Pre-fill the function with arguments that are not tensors
            wrapped_func, args = wrap_function(
                self.module.forward,
                args=args,
                exclude_self=True,
                # Wrap over this function instead of `forward`
                _func=self.module.__call__,
            )

            # Flatten the arguments so that all Tensors are present at the root
            # level
            args, input_structure = flatten(args, only_tensor=False)
            # Workaround to get output structure from within `custom_forward`
            output_structure = []

            def get_empty_tensor():
                """Helper function to get empty tensor with gradients"""
                empty = torch.tensor(
                    [],
                    dtype=torch.float32,
                    device=self._device,
                    requires_grad=True,
                )
                return empty

            def custom_forward(dummy, *_args):
                # Unflatten the inputs
                _args = unflatten(_args, input_structure)

                # Forward
                output: List[Union[T, Tuple[T]]] = wrapped_func(*_args)

                # Flatten output
                output, output_structure_ = flatten(output, only_tensor=True)
                output_structure.append(output_structure_)

                # Replace None with a tensor that requires grad
                output = [
                    output_i if output_i is not None else get_empty_tensor()
                    for output_i in output
                ]
                return tuple(output)

            output: List[T] = checkpoint(
                custom_forward,
                get_empty_tensor(),  # need at least one tensor requiring grad
                *args,
            )

            # Unwrap outputs
            output = [
                output_i if output_i.numel() > 0 else None
                for output_i in list(output)
            ]
            output = unflatten(output, output_structure[0])

        else:
            output = self.module(*args, **kwargs)
        return output


def wrap_gradient_checkpointing(
    module: T5Stack,
    device: str,
    stack_level: bool,
    gradient_checkpointing: bool = True,
):
    """
    Wrap a T5 block with gradient checkpointing. This function works with both
    T5 encoder and decoder.

    Parameters
    ----------
    stack_level : bool
        Whether to checkpoint at stack level or block level. If true, this
        function returns wrapped version of the stack. Otherwise, the
        wrapping is done in-place.
    """
    if stack_level:
        wrapped_module = CheckpointWrapper(
            module=module,
            device=device,
            gradient_checkpointing=gradient_checkpointing,
        )
        return wrapped_module
    else:
        wrapped_block = []
        for sub_module in module.block:
            wrapped_mod = CheckpointWrapper(
                module=sub_module,
                device=device,
                gradient_checkpointing=gradient_checkpointing
            )
            wrapped_block.append(wrapped_mod)
        module.block = nn.ModuleList(wrapped_block)
        return module


def cross_attention_forward(
        self,
        input,
        mask=None,
        kv=None,
        position_bias=None,
        past_key_value_state=None,
        head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
    """
    This only works for computing cross attention over the input
    """
    assert(kv != None)
    assert(head_mask == None)
    assert(position_bias != None or self.has_relative_attention_bias)

    bsz, qlen, dim = input.size()
    n_heads, d_heads = self.n_heads, self.d_kv
    klen = kv.size(1)

    q = self.q(input).view(bsz, -1, n_heads, d_heads).transpose(1, 2)
    if past_key_value_state == None:
        k = self.k(kv).view(bsz, -1, n_heads, d_heads).transpose(1, 2)
        v = self.v(kv).view(bsz, -1, n_heads, d_heads).transpose(1, 2)
    else:
        k, v = past_key_value_state

    scores = torch.einsum("bnqd,bnkd->bnqk", q, k)

    if mask is not None:
       scores += mask

    if position_bias is None:
        position_bias = self.compute_bias(qlen, klen)
    scores += position_bias

    if self.score_storage is None:
        self.score_storage = scores

    attn = F.softmax(scores.float(), dim=-1).type_as(scores)
    attn = F.dropout(attn, p=self.dropout, training=self.training)

    output = torch.matmul(attn, v)
    output = output.transpose(1, 2).contiguous().view(bsz, -1, self.inner_dim)
    output = self.o(output)

    if use_cache:
        output = (output,) + ((k, v),)
    else:
        output = (output,) + (None,)

    if output_attentions:
        output = output + (attn,)

    if self.has_relative_attention_bias:
        output = output + (position_bias,)

    return output

class RetrieverConfig(transformers.BertConfig):

    def __init__(self,
                 indexing_dimension=768,
                 apply_question_mask=False,
                 apply_passage_mask=False,
                 extract_cls=False,
                 passage_maxlength=200,
                 question_maxlength=40,
                 projection=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.indexing_dimension = indexing_dimension
        self.apply_question_mask = apply_question_mask
        self.apply_passage_mask = apply_passage_mask
        self.extract_cls=extract_cls
        self.passage_maxlength = passage_maxlength
        self.question_maxlength = question_maxlength
        self.projection = projection

class Retriever(transformers.PreTrainedModel):

    config_class = RetrieverConfig
    base_model_prefix = "retriever"

    def __init__(self, config, initialize_wBERT=False):
        super().__init__(config)
        assert config.projection or config.indexing_dimension == 768, \
            'If no projection then indexing dimension must be equal to 768'
        self.config = config
        if initialize_wBERT:
            self.model = transformers.BertModel.from_pretrained('bert-base-uncased')
        else:
            self.model = transformers.BertModel(config)
        if self.config.projection:
            self.proj = nn.Linear(
                self.model.config.hidden_size,
                self.config.indexing_dimension
            )
            self.norm = nn.LayerNorm(self.config.indexing_dimension)
        self.loss_fct = torch.nn.KLDivLoss()

    def forward(self,
                question_ids,
                question_mask,
                passage_ids,
                passage_mask,
                gold_score=None):
        question_output = self.embed_text(
            text_ids=question_ids,
            text_mask=question_mask,
            apply_mask=self.config.apply_question_mask,
            extract_cls=self.config.extract_cls,
        )
        bsz, n_passages, plen = passage_ids.size()
        passage_ids = passage_ids.view(bsz * n_passages, plen)
        passage_mask = passage_mask.view(bsz * n_passages, plen)
        passage_output = self.embed_text(
            text_ids=passage_ids,
            text_mask=passage_mask,
            apply_mask=self.config.apply_passage_mask,
            extract_cls=self.config.extract_cls,
        )

        score = torch.einsum(
            'bd,bid->bi',
            question_output,
            passage_output.view(bsz, n_passages, -1)
        )
        score = score / np.sqrt(question_output.size(-1))
        if gold_score is not None:
            loss = self.kldivloss(score, gold_score)
        else:
            loss = None

        return question_output, passage_output, score, loss

    def embed_text(self, text_ids, text_mask, apply_mask=False, extract_cls=False):
        text_output = self.model(
            input_ids=text_ids,
            attention_mask=text_mask if apply_mask else None
        )
        if type(text_output) is not tuple:
            text_output.to_tuple()
        text_output = text_output[0]
        if self.config.projection:
            text_output = self.proj(text_output)
            text_output = self.norm(text_output)

        if extract_cls:
            text_output = text_output[:, 0]
        else:
            if apply_mask:
                text_output = text_output.masked_fill(~text_mask[:, :, None], 0.)
                text_output = torch.sum(text_output, dim=1) / torch.sum(text_mask, dim=1)[:, None]
            else:
                text_output = torch.mean(text_output, dim=1)
        return text_output

    def kldivloss(self, score, gold_score):
        gold_score = torch.softmax(gold_score, dim=-1)
        score = torch.nn.functional.log_softmax(score, dim=-1)
        return self.loss_fct(score, gold_score)
