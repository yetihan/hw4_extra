from typing import TYPE_CHECKING, cast, List
from needle.autograd import Tensor
import needle.backend_ndarray.ndarray as ndarray
from needle import ops
import needle.init as init
import numpy as np
from .nn_sequence import Embedding
from .nn_basic import (
    Parameter, 
    Module, 
    ReLU,
    Dropout,
    LayerNorm1d,
    Linear,
    Sequential
)


class MultiHeadAttention(Module):
    """
    The multi-head self attention module.
    """
    def __init__(
        self,
        *,
        dropout = 0.,
        causal = False,
        device = None,
        dtype = "float32",
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype

        self.causal = causal
        self.dropout = Dropout(dropout)

    def create_causal_mask(self, i, j, device):
        """
        return a triangular causal mask.
        Input: i, j: the shape of the mask to be created
        """
        mask = -np.finfo(np.float32).max * np.triu(
            np.ones((1, 1, i, j), dtype=np.float32), j - i + 1)

        return ndarray.array(
            mask, device=device)

    def matmul(self, a: Tensor, b: Tensor):
        """
        batched matrix multiplication;
        a:           (..., M, K)     ──reshape──► (..., M, 1, K) ──broadcast──► (..., M, N, K)
        b            (..., K, N) 
        b_transpose: (..., N, K)     ──reshape──► (..., 1, N, K) ──broadcast──► (..., M, N, K)
                                                                                │
                                                                        逐元素乘
                                                                                │      
                                                                        sum(axis=-1)
                                                                                │
                                                                        (..., M, N)

        """
        a_shape = (*a.shape[:-1], 1, *a.shape[-1:])
        a = a.reshape(a_shape)

        b_transpose = b.transpose((-1,-2))
        b_transpose_shape = (*b_transpose.shape[:-2], 1, *b_transpose.shape[-2:])
        b_transpose = b_transpose.reshape(b_transpose_shape)

        broadcast_shape = list(a_shape)
        broadcast_shape[-2] = b_transpose_shape[-2]
        a = a.broadcast_to(broadcast_shape)

        broadcast_shape = list(b_transpose_shape)
        broadcast_shape[-3] = a_shape[-3]
        b_transpose = b_transpose.broadcast_to(broadcast_shape)

        return (a * b_transpose).sum(len(a.shape) - 1)

    def softmax(self, logit):
        """
        The softmax function; 
        """
        max_val = Tensor(
            logit.realize_cached_data().max(axis=3),
            device=logit.device,
            dtype=logit.dtype,
            requires_grad=False
        )

        max_val = max_val.reshape((*logit.shape[:-1], 1))
        max_val = max_val.broadcast_to(logit.shape)

        probs = ops.exp(logit - max_val)

        denom = probs.sum(axes=3)
        denom = denom.reshape((*logit.shape[:-1], 1))
        denom = denom.broadcast_to(logit.shape)

        return probs / denom

    def forward(
        self,
        q: Tensor, k: Tensor, v: Tensor,
    ):
        """
        The forward function of the MultiHeadAttention activation function.
        Input: three states q, k, v, with shape (batch_size, num_head, seq_len, dim_head)
        Output: the activation output `result` and attention softmax probability `probs` (with dropout applied)
        """
        batch_size, num_head, queries_len, q_dim = q.shape #  B,H,T,D
        _, _, keys_values_len, k_dim = k.shape
        _, _, _, v_dim = v.shape

        assert q_dim == k_dim == v_dim

        result = None
        probs = None

        ### BEGIN YOUR SOLUTION
        
        # softmax(q@k.T/d**0.5+mask)v
        
        s = self.matmul(q, k.T)/np.sqrt(q_dim)
        if self.causal:
            mask = self.create_causal_mask(keys_values_len, queries_len, device=self.device)
            mask = mask.broadcast_to(s.shape)
            s = s + mask
        probs = self.dropout(self.softmax(s))
        result =  self.matmul(probs, v)        
        ### END YOUR SOLUTION

        return result, probs


class AttentionLayer(Module):

    def __init__(
        self,
        q_features: int,
        num_head: int,
        dim_head: int,
        *,
        k_features: int = None,
        v_features: int = None,
        out_features: int = None,
        dropout = 0.,
        causal = True,
        device = None,
        dtype = "float32",
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype

        if k_features is None:
            k_features = q_features
        if v_features is None:
            v_features = q_features
        if out_features is None:
            out_features = q_features

        self.q_features = q_features
        self.k_features = k_features
        self.v_features = v_features
        self.out_features = out_features

        self.num_head = num_head
        self.dim_head = dim_head

        self.prenorm_q = LayerNorm1d(
            q_features, device=device, dtype=dtype)
        self.prenorm_k = LayerNorm1d(
            k_features, device=device, dtype=dtype)
        self.prenorm_v = LayerNorm1d(
            v_features, device=device, dtype=dtype)

        inner_dim = num_head * dim_head
        
        self.q_projection = Linear(
            q_features, inner_dim, bias=False,
            device=device, dtype=dtype)
        self.k_projection = Linear(
            k_features, inner_dim, bias=False,
            device=device, dtype=dtype)
        self.v_projection = Linear(
            v_features, inner_dim, bias=False,
            device=device, dtype=dtype)

        self.attn = MultiHeadAttention(
            dropout=dropout, causal=causal,
            device=device, dtype=dtype)

        self.out_projection = Linear(
            inner_dim, out_features, bias=False,
            device=device, dtype=dtype)

    def forward(
        self,
        q, k=None, v=None,
    ):
        """
        The forward function of the self-attention layer.
        Input: `q` with shape (batch_size, q_len, q_dim)
               `k` (if not None) with shape (batch_size, kv_len, k_dim)
               `v` (if not None) with shape (batch_size, kv_len, v_dim)
        Output: the output `result` with shape (batch_size, kv_len, out_features)
        """

        if k is None:
            k = q
        if v is None:
            v = q

        B, q_len, q_dim = q.shape
        _, kv_len, k_dim = k.shape
        _, _, v_dim = v.shape

        result = None

        ### BEGIN YOUR SOLUTION
        H = self.num_head
        D = self.dim_head
        
        Q = self.q_projection(self.prenorm_q(q.reshape((B*q_len, q_dim )))).reshape((B, q_len, H, D))
        K = self.k_projection(self.prenorm_k(k.reshape((B*kv_len, k_dim )))).reshape((B, kv_len,  H, D))
        V = self.v_projection(self.prenorm_v(v.reshape((B*kv_len, v_dim )))).reshape((B, kv_len,  H, D))
        # B,T,H,D -> B,H,T,D
        Q = Q.transpose((1,2))
        K = K.transpose((1,2))
        V = V.transpose((1,2))
        
        res, self.probs = self.attn(Q,K,V)
        if TYPE_CHECKING:
            res = cast(Tensor, res)
        T = q_len 
        # B,H,T,D -> B,T,H,D-> B*T,H*D
        res = res.transpose((1,2)).reshape((B*T,H*D))
        
        result = self.out_projection(res).reshape((B, T, self.out_features))


        ### END YOUR SOLUTION

        return result


class TransformerLayer(Module):

    def __init__(
        self,
        q_features: int,
        num_head: int,
        dim_head: int,
        hidden_size: int,
        *,
        dropout = 0.,
        causal = True,
        device = None,
        dtype = "float32",
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype

        ### BEGIN YOUR SOLUTION
        self.mha = AttentionLayer(q_features, num_head, dim_head,
                                dropout=dropout, causal=causal,device=device,dtype=dtype)
        self.dropout1 = Dropout(p=dropout)
        
        self.dropout2 = Dropout(p=dropout)
        
        linear1 = Linear(in_features=q_features, out_features=hidden_size, bias=True, device=device, dtype=dtype)
        linear2 = Linear(in_features=hidden_size, out_features=q_features, bias=True, device=device, dtype=dtype)
        self.ffn = Sequential(linear1, ReLU(), Dropout(p=dropout), linear2)
        self.layer_norm = LayerNorm1d(
            q_features, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(
        self,
        x
    ):
        """
        The forward function of a Transformer Layer.
        Input: the hidden states from previous layers `x` with shape (batch_size, seq_len, x_dim)
        Ouput: the hidden states after the Transformer Layer `x` with shape (batch_size, seq_len, x_dim)
        """

        B, T, x_dim = x.shape
        ### BEGIN YOUR SOLUTION
        x = x+self.dropout1(self.mha(x))
        ffn_input = x.reshape((B*T, x_dim))
        x = x + self.dropout2(self.ffn(self.layer_norm(ffn_input)).reshape((B, T, x_dim)))
        ### END YOUR SOLUTION
        return x


class Transformer(Module):

    def __init__(
        self,
        embedding_size: int,
        hidden_size: int,
        num_layers: int, 
        *,
        num_head: int = 8,
        dim_head: int = 32,
        dropout = 0.,
        causal = True,
        device = None,
        dtype = "float32",
        batch_first = False,
        sequence_len = 2048
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype
        self.batch_first = batch_first

        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(
        self,
        x, h=None
    ):

        if not self.batch_first:
            x = ops.transpose(x, axes=(0, 1))

        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

        if not self.batch_first:
            x = ops.transpose(x, axes=(0, 1))

        return x, init.zeros_like(x)