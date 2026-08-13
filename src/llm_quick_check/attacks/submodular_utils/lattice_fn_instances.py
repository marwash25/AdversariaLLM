"""
Lattice function instances.
"""

from typing import Optional, Tuple
import torch
from torch import Tensor
import logging
from .lattice_functions import LatticeFunction, SequentialLatticeFunction, LinearCombinationLatticeFn, make_zero_lattice_fn
from .setfn_reductions import SetToLatticeMap


class QuadraticFn(SequentialLatticeFunction):
    """Quadratic lattice function F(x) = 0.5 * x^T Q x with symmetric Q.

    Pass a 2D tensor Q for a general quadratic, or a 0-dim tensor together with n 
    for an n-by-n constant matrix Q = c * 11^T, i.e. F(x) = 0.5 * c * (sum_i x_i)^2. 
    """

    def __init__(self, Q: Tensor, k: int, n: Optional[int] = None):
        if Q.dim() == 0:
            assert n is not None and n > 0, "n is required when Q is a scalar (constant matrix c * 11^T)"
            self._sum_x = None
        elif Q.dim() == 2 and Q.shape[0] == Q.shape[1]:
            assert n is None or n == Q.shape[0], "n must match Q.shape[0] when both are given"
            assert torch.equal(Q, Q.T), "Q must be symmetric" # switch to allclose if we want to allow small numerical errors
            n = Q.shape[0]
        else:
            raise ValueError("Q must be a square matrix or a scalar tensor (constant c) with n set")
        super().__init__(k, n)
        self.Q = Q

    def _eval_batch(self, x: Tensor) -> Tuple[Tensor, int]:
        assert x.device == self.Q.device, "x and Q must be on the same device"
        if self.Q.dim() == 0: 
            sum_x = x.sum(dim=1)
            Fvalues = 0.5 * self.Q * sum_x**2
        else:
            xq = x.to(dtype=self.Q.dtype) @ self.Q  
            Fvalues = 0.5 * (x * xq).sum(dim=1)
        return Fvalues, 0

    def set_state(self, x: Tensor, F_val: Tensor, *, sum_x: Optional[Tensor] = None) -> None:
        """For scalar Q, pass sum_x to set _sum_x in O(1) when it is already known."""
        super().set_state(x, F_val)
        if self.Q.dim() == 0:
            self._sum_x = sum_x if sum_x is not None else self.current_x.sum()


    def add(self, i: int, weight: Tensor) -> Tuple[Tensor, Tensor, int]:
        assert weight.device == self.Q.device == self.current_x.device, "weight, current_x and Q must be on the same device"
        new_x = self.current_x.clone()
        new_x[i] += weight
        if self.Q.dim() == 0:
            assert self._sum_x is not None
            new_sum = self._sum_x + weight
            new_val = 0.5 * self.Q * new_sum**2
        else:
            new_val = self.current_val + weight * (self.Q[i, :] * self.current_x).sum() + 0.5 * weight**2 * self.Q[i, i]
        return new_val, new_x, 0

    # new_sum already computed in add/remove, so it's a bit inefficient to recompute it in add_update/remove_update, 
    # but want to keep return of add/remove consistent with base class. 
    # TODO: if we refactor SequentialLatticeFunction to maintain a state object this can be avoided
    def add_update(self, i: int, weight: Tensor) -> Tuple[Tensor, Tensor, int]:
        new_val, new_x, flops = self.add(i, weight)
        new_sum = self._sum_x + weight if self.Q.dim() == 0 else None 
        self.set_state(new_x, new_val, sum_x=new_sum)
        return new_val, new_x, flops


class EmbeddingQuadraticFn(QuadraticFn):
    r"""Lattice function F(x) = 0.5 * p_x^T Q p_x with symmetric Q.
    and p_x = embedding_projections[x] if normalize is False.
    Otherwise, F(x) = 0.5 * (p_x - p_0)^T Q (p_x - p_0). 
    I'm normalizing this way instead of simply subtracting F(0) to make F(x) non-increasing in x
    so a bound on the Lipschitz constant of its Lovasz extension can be easily computed as -F((k-1) 1_n)
    """
    def __init__(
        self,
        Q: Tensor,
        embedding_projections: Tensor,
        k: int,
        n: Optional[int] = None,
        normalize: bool = True,
    ):
        assert embedding_projections.shape == (k,), "embedding_projections must have shape (k,)"
        assert Q.device == embedding_projections.device, "Q and embedding_projections must be on the same device"
        if Q.dim() == 0:
            assert n is not None and n > 0, "n is required when Q is a scalar (constant matrix c * 11^T)"
            Q = Q * torch.ones((n, n), dtype=Q.dtype, device=Q.device)
        super().__init__(Q, k, n)
        self.embedding_projections = embedding_projections
        zero_x = torch.zeros(1, self.n, dtype=torch.long, device=self.Q.device)
        self.p_0 = self._projections(zero_x) if normalize else zero_x
        # if normalize:
        #     self.F_0, _ = super()._eval_batch(self._projections(zero_x))
        # else:
        #     self.F_0 = torch.zeros(1, device=self.Q.device)

    def _projections(self, x: Tensor) -> Tensor:
        return self.embedding_projections[x] 

    def _eval_batch(self, x: Tensor) -> Tuple[Tensor, int]:  
        return super()._eval_batch(self._projections(x) - self.p_0)   
        # Fvalues, flops = super()._eval_batch(self._projections(x))      
        # return Fvalues - self.F_0, flops

    def add(self, i: int, weight: Tensor) -> Tuple[Tensor, Tensor, int]:
        assert weight.device == self.Q.device == self.current_x.device, "weight, current_x and Q must be on the same device"
        new_x = self.current_x.clone()
        new_x[i] += weight
        delta_p = self.embedding_projections[new_x[i]] - self.embedding_projections[self.current_x[i]]
        current_p = self._projections(self.current_x)
        new_val = self.current_val + delta_p * (self.Q[i, :] * (current_p - self.p_0)).sum() + 0.5 * delta_p**2 * self.Q[i, i]
        return new_val, new_x, 0
        


class ModularFn(SequentialLatticeFunction): # TODO: not used anywhere yet, remove if not needed
    """Modular lattice function F(x) = w^T x."""

    def __init__(self, w: Tensor, k: int):
        super().__init__(k, w.shape[0])
        self.w = w

    def _eval_batch(self, x: Tensor) -> Tuple[Tensor, int]:
        assert x.device == self.w.device, "x and w must be on the same device"
        return (x * self.w).sum(dim=1), 0

    def add(self, i: int, weight: Tensor) -> Tuple[Tensor, Tensor, int]:
        assert weight.device == self.w.device == self.current_x.device, "weight, current_x and w must be on the same device"
        new_x = self.current_x.clone()
        new_x[i] += weight
        new_val = self.current_val + weight * self.w[i]
        return new_val, new_x, 0


class LatticeFnWithModReduction(LatticeFunction):
    """Lattice function F: V^n -> R whose set function reduction F_set: 2^([n] x [b]) -> R is modular, i.e.,
    F_set(S) = sum_{(i, j) in S} W[i, j] for some weight matrix W of shape (n, b).

    F_set is given by F_set(S) = F(M(S)) where M: 2^([n] x [b]) -> V^n is [M(S)]_i = sum_{j in [b], (i, j) in S} weights[j].
    Conversely, F is given by F(x) = F_set(M^{-1}(x)) where M^{-1}: V^n -> 2^([n] x [b]) is the inverse map of M.
    """
    # This doesn't have a simple closed form that doesn't require going through M^{-1}. 
    # This function is needed in DCA for H_lowerbd which is combined with G and their set function reduction is minimized by the inner solver
    # It's inefficient to go through this lattice function when we already have the form of the set function reduction.
    # Evaluating corresponding SetFnReduction.set_fn will map from sets to ints and back to sets in eval_batch. But we currently only use 
    # this method in singleton_L_bound which is not used for this function.
    # We override _eval_chain to avoid unecessary map to ints and back.
    # TODO: refactor code to have set fn class and linear combination of set fns that can be both from reductions or not.

    def __init__(self, map: SetToLatticeMap, W: Tensor):
        assert W.shape[0] == map.n and W.shape[1] == map.b, "W must have shape (map.n, map.b)"
        super().__init__(map.k, map.n)
        self.W = W
        self.map = map

    def _eval_batch(self, x: Tensor) -> Tuple[Tensor, int]:
        assert x.device == self.W.device == self.map.device, "x, W, and map must be on the same device"
        X = self.map.ints2binary(x)
        return (X * self.W).sum(dim=(1, 2)), 0

    def _eval_chain(
        self, rows: Tensor, cols: Tensor, weights: Tensor, x_chain: Tensor
    ) -> Tuple[Tensor, int]:
        """Evaluate F(x^i) for the chain of inputs x^i = x^{i-1} + weights[cols[i-1]] * e_{rows[i-1]}.
        by directly evaluating F_set(S^i) for the corresponding sets S^i = S^{i-1} + {(rows[i-1], cols[i-1])}.
        """
        device = rows.device
        m = rows.shape[0]

        Fvalues = torch.zeros((m,), dtype=self.W.dtype, device=device)
        Fvalues[0] = self.W[rows[0], cols[0]]
        for i in range(1, m):
            Fvalues[i] = Fvalues[i-1] + self.W[rows[i], cols[i]]

        return Fvalues, 0


def DR_submodular_decomposition(
    F_batch: LatticeFunction,
    hessian_upperbd: Tensor,
    embedding_projections: Tensor | None = None,
) -> Tuple[LatticeFunction, LatticeFunction]:
    r"""Decompose a lattice function F: V^n -> R into the difference of two DR-submodular lattice functions G and H: 
    F = G - H, with G = F + H and 
    If embedding_matrix is not None:
        H(x) = 0.5 * (p_x - p_0)^T Q (p_x - p_0), where p_x = embedding_projections[x],  
    Otherwise:
        H(x) = 0.5 * x^T Q x 
    
    where Q = -max(hessian_upperbd, 0) if hessian_upperbd is a matrix or Q = -max(hessian_upperbd, 0) * 11^T if it is a scalar, 
    and
        ((F(x + a_i1 e_i1 + a_i2 e_i2) - F(x + a_i2 e_i2)) - (F(x + a_i1 e_i1) - F(x))) <=  a_i1 a_i2 hessian_upperbd[i1, i2]
    """
    if hessian_upperbd.max().item() <= 0: # alpha == 0 is useful to test if dca correctly reduces to its submin inner solver in this case
        logging.info("hessian_upperbd <= 0 implies F is already DR-submodular, returning F as G and zero lattice function as H")
        H_batch = make_zero_lattice_fn(F_batch.k, F_batch.n)
        return F_batch, H_batch

    alpha = -torch.clamp(hessian_upperbd, min=0)    
    if embedding_projections is None:
        H_batch = QuadraticFn(alpha, F_batch.k, F_batch.n)
    else:
        H_batch = EmbeddingQuadraticFn(alpha, embedding_projections, F_batch.k, F_batch.n)
    G_batch = LinearCombinationLatticeFn([F_batch, H_batch], [1.0, 1.0])
    return G_batch, H_batch
   
