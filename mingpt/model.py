"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""
import jax
import jax.numpy as jnp
import haiku as hk
from functools import partial
import math

class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)

class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768

normal_init = hk.initializers.RandomNormal(stddev=0.02, mean=0.0)
Linear = partial(hk.Linear, w_init=normal_init, b_init=hk.initializers.Constant(0.0))
LayerNorm = partial(hk.LayerNorm, axis=-1, create_scale=True, create_offset=True)

def Dropout(is_training):
    def dropout(pdrop, x):
        return hk.dropout(hk.next_rng_key(), pdrop, x) if is_training and pdrop>0.0 else x
    return dropout

def causal_self_attention(x, config, dropout):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use hk.MultiHeadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """
    assert config.n_embd % config.n_head == 0
    #TODO


    return y

def block(x, config, dropout):
    """ an unassuming Transformer block """
    #TODO

def gpt(x, config, is_training):
    """  the full GPT language model, with a context size of block_size """
    dropout = Dropout(is_training)
    # input embedding stem

        
    # decoder head


    # forward the GPT model

    # each index maps to a (learnable) vector

    # each position maps to a (learnable) vector

    # transformer


def cross_entropy(logits, targets):
    one_hot = jax.nn.one_hot(targets, logits.shape[-1])
    loss = -jax.nn.log_softmax(logits) * one_hot 
    loss = loss.sum() / one_hot.sum() 
    return loss 

def loss_fn(idx, targets, config, is_training):
    return cross_entropy(jax.vmap(gpt, in_axes=[0, None, None])(idx, config, is_training), targets)
