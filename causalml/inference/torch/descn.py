"""
This module implements DESCN [1], Deep Entire Space Cross Networks, a multi-task neural
network for estimating the Conditional Average Treatment Effect (CATE) on **binary**
outcomes. Instead of fitting the treated and control response functions separately in
their individual sample spaces — which suffers from treatment bias and sample imbalance
when the two groups are biased, unequal-sized samples of the population — DESCN models
the treatment propensity and both responses jointly over the *entire* sample space,
ESMM-style, through a cross network:

- a shared representation trunk feeding four towers;
- a propensity tower, treated- and control-response towers, and a pseudo-treatment-effect
  tower that relieves sample imbalance by cross-fitting the responses through it.

This estimator is a port of the official reference implementation at
https://github.com/kailiang-zhong/DESCN (BSD-2 licensed), reimplemented in CausalML
style. It is scoped to binary outcomes, the method's published form: the response towers
are sigmoid heads trained with binary cross-entropy, and the CATE readout is the
difference of the two response probabilities. The reference's optional MMD/Wasserstein
representation-imbalance penalty (disabled there by default) and its randomized-data mask
input are omitted.

**References**

[1] K. Zhong, F. Xiao, Y. Ren, Y. Liang, W. Yao, X. Yang, L. Cen (2022).
    | DESCN: Deep Entire Space Cross Networks for Individual Treatment Effect Estimation.
    | Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining.
    | https://doi.org/10.1145/3534678.3539198
    | https://arxiv.org/abs/2207.09920
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from causalml.inference._arg_order import shim_arg_order
from causalml.inference.meta.utils import convert_pd_to_np

# The propensity is clipped into [clip, 1 - clip] so that the entire-space cross
# probabilities p_estr = p_prpsy * p_h1 and p_escr = (1 - p_prpsy) * p_h0 stay in the
# open (0, 1) interval required by binary_cross_entropy.
_PROPENSITY_CLIP = 1e-3

# The treated fraction p_t drives the inverse-propensity reweighting
# (1 / (2 * p_t) and 1 / (2 * (1 - p_t))); clamp it away from 0 and 1 so a
# single-group batch cannot divide by zero.
_BALANCE_EPS = 1e-6


def _init_weights(m):
    """Initialize Linear layers as in the reference: N(0, 1 / sqrt(fan_in)) weights, zero bias."""
    if isinstance(m, nn.Linear):
        stdv = 1.0 / math.sqrt(m.weight.size(1))
        nn.init.normal_(m.weight, mean=0.0, std=stdv)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def _safe_sqrt(x):
    """Numerically safe sqrt, mirroring the reference: clip to [1e-9, 1e9] before the root."""
    return torch.sqrt(torch.clamp(x, 1e-9, 1e9))


class ShareNetwork(nn.Module):
    """Shared representation trunk feeding all four DESCN towers.

    An optional input ``BatchNorm1d`` followed by two ``Linear -> ELU -> Dropout``
    blocks and a final ``Linear(input_dim -> base_dim) -> ELU`` projection. With
    ``normalization="divide"`` the output rows are L2-normalized, as in TARnet/CFR.
    """

    def __init__(
        self,
        input_dim,
        share_dim,
        base_dim,
        dropout=0.0,
        normalization="divide",
        batch_norm=False,
    ):
        """
        Args:
            input_dim (int): number of input features
            share_dim (int): width of the shared hidden layers
            base_dim (int): width of the shared output representation
            dropout (float): dropout probability applied after each hidden layer
            normalization (str): "divide" to L2-normalize the representation rows,
                any other value to skip the normalization
            batch_norm (bool): apply BatchNorm1d to the input features (requires
                training batches of at least 2 samples)
        """
        super().__init__()
        layers = []
        if batch_norm:
            layers.append(nn.BatchNorm1d(input_dim))
        layers += [
            nn.Linear(input_dim, share_dim),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(share_dim, share_dim),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(share_dim, base_dim),
            nn.ELU(),
        ]
        if batch_norm:
            layers.append(nn.Dropout(p=dropout))
        self.DNN = nn.Sequential(*layers)
        self.DNN.apply(_init_weights)
        self.normalization = normalization

    def forward(self, x):
        h_rep = self.DNN(x)
        if self.normalization == "divide":
            h_rep = h_rep / _safe_sqrt(
                torch.sum(torch.square(h_rep), dim=1, keepdim=True)
            )
        return h_rep


class TowerNetwork(nn.Module):
    """A DESCN tower: a three-layer base model stack plus a scalar logit head.

    The reference instantiates this structure four times as the propensity
    (PrpsyNetwork), treated-response (Mu1Network), control-response (Mu0Network)
    and pseudo-effect (TauNetwork) towers; the towers share this layout and
    differ only in their role.
    """

    def __init__(self, base_dim, dropout=0.0):
        """
        Args:
            base_dim (int): width of the hidden layers and size of the input
            dropout (float): dropout probability applied after each hidden layer
        """
        super().__init__()
        layers = []
        for _ in range(3):
            layers += [nn.Linear(base_dim, base_dim), nn.ELU(), nn.Dropout(p=dropout)]
        self.base_model = nn.Sequential(*layers)
        self.logit_layer = nn.Linear(base_dim, 1)
        self.base_model.apply(_init_weights)
        self.logit_layer.apply(_init_weights)

    def forward(self, inputs):
        return self.logit_layer(self.base_model(inputs))


class ESX(nn.Module):
    """The Entire Space Cross (ESX) network of DESCN.

    A shared trunk (:class:`ShareNetwork`) feeds a propensity tower, treated- and
    control-response towers and a pseudo-effect tower. The responses are crossed
    with the propensity over the entire sample space, ESMM-style:

    .. math::
        p_{estr} = e(x) \\cdot h_1(x), \\qquad p_{escr} = (1 - e(x)) \\cdot h_0(x)

    where :math:`h_1` and :math:`h_0` are the treated and control conditional
    response probabilities.
    """

    def __init__(
        self,
        input_dim,
        share_dim=256,
        base_dim=128,
        dropout=0.0,
        normalization="divide",
        batch_norm=False,
    ):
        """
        Args:
            input_dim (int): number of input features
            share_dim (int): width of the shared hidden layers
            base_dim (int): width of the shared output representation and towers
            dropout (float): dropout probability applied after each hidden layer
            normalization (str): "divide" to L2-normalize the shared representation,
                any other value to skip the normalization
            batch_norm (bool): apply BatchNorm1d to the input features
        """
        super().__init__()
        self.share_network = ShareNetwork(
            input_dim, share_dim, base_dim, dropout, normalization, batch_norm
        )
        self.prspy_network = TowerNetwork(base_dim, dropout)
        self.mu1_network = TowerNetwork(base_dim, dropout)
        self.mu0_network = TowerNetwork(base_dim, dropout)
        self.tau_network = TowerNetwork(base_dim, dropout)

    def forward(self, x):
        shared_h = self.share_network(x)

        p_prpsy_logit = self.prspy_network(shared_h)
        mu1_logit = self.mu1_network(shared_h)
        mu0_logit = self.mu0_network(shared_h)
        tau_logit = self.tau_network(shared_h)

        p_prpsy = torch.clamp(
            torch.sigmoid(p_prpsy_logit), _PROPENSITY_CLIP, 1 - _PROPENSITY_CLIP
        )
        p_h1 = torch.sigmoid(mu1_logit)
        p_h0 = torch.sigmoid(mu0_logit)

        p_estr = torch.mul(p_prpsy, p_h1)
        p_escr = torch.mul(1 - p_prpsy, p_h0)

        return {
            "p_prpsy_logit": p_prpsy_logit,
            "p_prpsy": p_prpsy,
            "p_estr": p_estr,
            "p_escr": p_escr,
            "mu1_logit": mu1_logit,
            "mu0_logit": mu0_logit,
            "tau_logit": tau_logit,
            "p_h1": p_h1,
            "p_h0": p_h0,
            "shared_h": shared_h,
        }


def _masked_bce(pred, target):
    """Binary cross-entropy over a masked selection, 0 when the selection is empty.

    The mean-reduced BCE of an empty selection is nan (0/0), so a single-group
    batch must not contribute through the term of the other group.

    Args:
        pred (torch.Tensor): predicted probabilities in (0, 1)
        target (torch.Tensor): binary targets of the same shape

    Returns:
        (torch.Tensor): scalar loss
    """
    if pred.numel() == 0:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    return F.binary_cross_entropy(pred, target)


def descn_loss(
    outputs,
    treatment,
    y,
    prspy_w=0.5,
    escvr1_w=1.0,
    escvr0_w=1.0,
    h1_w=0.0,
    h0_w=0.0,
    mu1hat_w=0.8,
    mu0hat_w=0.3,
):
    """Multi-task loss of DESCN: seven binary cross-entropy terms over the entire space.

    The samples are inverse-propensity reweighted with the treated fraction of the
    batch, ``p_t = mean(treatment)`` (clamped away from 0 and 1 for single-group
    batches): ``sample_weight = t / (2 * p_t) + (1 - t) / (2 * (1 - p_t))``, and the
    propensity term carries a ``pos_weight`` of ``1 / (2 * p_t)``.

    Args:
        outputs (dict): outputs of :meth:`ESX.forward`
        treatment (torch.Tensor): treatment vector of shape (n, 1) or (n,)
        y (torch.Tensor): binary outcome vector of shape (n, 1) or (n,)
        prspy_w (float): weight of the propensity term
        escvr1_w (float): weight of the entire-space treated-response term
        escvr0_w (float): weight of the entire-space control-response term
        h1_w (float): weight of the factual treated-response term
        h0_w (float): weight of the factual control-response term
        mu1hat_w (float): weight of the cross-fitting treated term
        mu0hat_w (float): weight of the cross-fitting control term

    Returns:
        (torch.Tensor): scalar total loss
    """
    t = treatment.reshape(-1, 1)
    y = y.reshape(-1, 1).to(t.dtype)

    p_t = torch.clamp(t.mean(), _BALANCE_EPS, 1 - _BALANCE_EPS)
    sample_weight = t / (2.0 * p_t) + (1.0 - t) / (2.0 * (1.0 - p_t))

    treated = t.bool()
    control = ~treated

    prspy_loss = prspy_w * F.binary_cross_entropy_with_logits(
        outputs["p_prpsy_logit"], t, pos_weight=1.0 / (2.0 * p_t)
    )
    estr_loss = escvr1_w * F.binary_cross_entropy(
        outputs["p_estr"], y * t, weight=sample_weight
    )
    escr_loss = escvr0_w * F.binary_cross_entropy(
        outputs["p_escr"], y * (1.0 - t), weight=sample_weight
    )

    h1_loss = h1_w * _masked_bce(outputs["p_h1"][treated], y[treated])
    h0_loss = h0_w * _masked_bce(outputs["p_h0"][control], y[control])

    cross_tr_loss = mu1hat_w * _masked_bce(
        torch.sigmoid(outputs["mu0_logit"] + outputs["tau_logit"])[treated], y[treated]
    )
    cross_cr_loss = mu0hat_w * _masked_bce(
        torch.sigmoid(outputs["mu1_logit"] - outputs["tau_logit"])[control], y[control]
    )

    return (
        prspy_loss
        + estr_loss
        + escr_loss
        + h1_loss
        + h0_loss
        + cross_tr_loss
        + cross_cr_loss
    )


@shim_arg_order
class DESCN:
    """Deep Entire Space Cross Networks (DESCN) for CATE estimation on binary outcomes.

    DESCN learns the treatment propensity, the treated and control response functions
    and a pseudo treatment effect jointly over the entire sample space, mitigating the
    treatment bias and sample imbalance of learning each response in its own sample
    space. The estimator is a port of the official BSD-2 reference implementation,
    scoped to binary outcomes: the responses are probabilities and the loss is the
    seven-term multi-task binary cross-entropy of the paper.

    Example::

        from causalml.inference.torch import DESCN

        descn = DESCN(share_dim=32, base_dim=16, epochs=50)
        tau = descn.fit_predict(X, treatment, y)  # (n, 1) CATE estimates
    """

    def __init__(
        self,
        share_dim=256,
        base_dim=128,
        prspy_w=0.5,
        escvr1_w=1.0,
        escvr0_w=1.0,
        h1_w=0.0,
        h0_w=0.0,
        mu1hat_w=0.8,
        mu0hat_w=0.3,
        dropout=0.0,
        normalization="divide",
        batch_norm=False,
        batch_size=4096,
        lr=1e-3,
        l2=1e-2,
        optimizer="adam",
        epochs=15,
        device="cpu",
    ):
        """
        Initializes a DESCN estimator.

        Args:
            share_dim (int): width of the shared trunk's hidden layers
            base_dim (int): width of the shared representation and of the towers
            prspy_w (float): weight of the propensity loss term
            escvr1_w (float): weight of the entire-space treated-response loss term
            escvr0_w (float): weight of the entire-space control-response loss term
            h1_w (float): weight of the factual treated-response loss term
            h0_w (float): weight of the factual control-response loss term
            mu1hat_w (float): weight of the cross-fitting treated loss term
            mu0hat_w (float): weight of the cross-fitting control loss term
            dropout (float): dropout probability in the trunk and the towers
            normalization (str): "divide" to L2-normalize the shared representation,
                any other value to skip the normalization
            batch_norm (bool): apply BatchNorm1d to the input features (requires
                training batches of at least 2 samples)
            batch_size (int): batch size of the training loop
            lr (float): learning rate of the optimizer
            l2 (float): weight decay (L2 penalty) of the optimizer
            optimizer (str): "adam" or "sgd"
            epochs (int): number of training epochs
            device (str): torch device to train and predict on

        The loss weights default to the reference implementation's configuration for
        the ACIC synthetic dataset; they are dataset-tuned, so expose them all here.
        """
        self.share_dim = share_dim
        self.base_dim = base_dim
        self.prspy_w = prspy_w
        self.escvr1_w = escvr1_w
        self.escvr0_w = escvr0_w
        self.h1_w = h1_w
        self.h0_w = h0_w
        self.mu1hat_w = mu1hat_w
        self.mu0hat_w = mu0hat_w
        self.dropout = dropout
        self.normalization = normalization
        self.batch_norm = batch_norm
        self.batch_size = batch_size
        self.lr = lr
        self.l2 = l2
        self.optimizer = optimizer
        self.epochs = epochs
        self.device = device

    def fit(self, X, treatment, y, p=None):
        """
        Fits the DESCN model.

        Args:
            X (np.matrix or np.array or pd.Dataframe): a feature matrix
            treatment (np.array or pd.Series): a binary treatment vector
            y (np.array or pd.Series): a binary outcome vector
        """
        X, treatment, y = convert_pd_to_np(X, treatment, y)

        self.descn = ESX(
            input_dim=X.shape[1],
            share_dim=self.share_dim,
            base_dim=self.base_dim,
            dropout=self.dropout,
            normalization=self.normalization,
            batch_norm=self.batch_norm,
        ).to(self.device)

        if self.optimizer.lower() == "sgd":
            optimizer = torch.optim.SGD(
                self.descn.parameters(), lr=self.lr, weight_decay=self.l2
            )
        else:
            optimizer = torch.optim.Adam(
                self.descn.parameters(), lr=self.lr, weight_decay=self.l2
            )

        X_tensor = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        t_tensor = torch.as_tensor(
            treatment, dtype=torch.float32, device=self.device
        ).reshape(-1, 1)
        y_tensor = torch.as_tensor(y, dtype=torch.float32, device=self.device).reshape(
            -1, 1
        )

        loader = DataLoader(
            TensorDataset(X_tensor, t_tensor, y_tensor),
            batch_size=self.batch_size,
            shuffle=True,
        )

        self.descn.train()
        for _ in range(self.epochs):
            for X_batch, t_batch, y_batch in loader:
                optimizer.zero_grad()
                loss = descn_loss(
                    self.descn(X_batch),
                    t_batch,
                    y_batch,
                    prspy_w=self.prspy_w,
                    escvr1_w=self.escvr1_w,
                    escvr0_w=self.escvr0_w,
                    h1_w=self.h1_w,
                    h0_w=self.h0_w,
                    mu1hat_w=self.mu1hat_w,
                    mu0hat_w=self.mu0hat_w,
                )
                loss.backward()
                optimizer.step()

    def predict(self, X, treatment=None, y=None, p=None):
        """
        Calls predict on the fitted DESCN model.

        Args:
            X (np.matrix or np.array or pd.Dataframe): a feature matrix

        Returns:
            (np.array): a 2D array with shape (X.shape[0], 3), where each row takes
                the form of (outcome do(t=0), outcome do(t=1), propensity)
        """
        X_tensor = torch.as_tensor(np.asarray(X, dtype=np.float32), device=self.device)

        self.descn.eval()
        with torch.no_grad():
            outputs = self.descn(X_tensor)

        preds = torch.cat([outputs["p_h0"], outputs["p_h1"], outputs["p_prpsy"]], dim=1)
        return preds.cpu().numpy()

    def predict_propensity(self, X):
        """
        Predicts the individual propensity scores.

        Args:
            X (np.matrix or np.array or pd.Dataframe): a feature matrix

        Returns:
            (np.array): propensity score vector
        """
        preds = self.predict(X)
        return preds[:, 2]

    def predict_tau(self, X):
        """
        Predicts the individual treatment effect (tau / "ITE").

        Args:
            X (np.matrix or np.array or pd.Dataframe): a feature matrix

        Returns:
            (np.array): treatment effect vector
        """
        preds = self.predict(X)
        return (preds[:, 1] - preds[:, 0]).reshape(-1, 1)

    def fit_predict(self, X, treatment, y, p=None, return_components=False):
        """
        Fits the DESCN model and then predicts.

        Args:
            X (np.matrix or np.array or pd.Dataframe): a feature matrix
            treatment (np.array or pd.Series): a binary treatment vector
            y (np.array or pd.Series): a binary outcome vector
            return_components (bool, optional): whether to return

        Returns:
            (np.array): predictions based on return_components flag
                if return_components=False (default), each row is treatment effect
                if return_components=True, each row is (outcome do(t=0), outcome do(t=1), propensity)
        """
        self.fit(X=X, treatment=treatment, y=y)
        if return_components:
            return self.predict(X=X)
        return self.predict_tau(X=X)
