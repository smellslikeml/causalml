"""Tests for the DESCN estimator.

Mirrors the torch-test idiom of ``tests/test_cevae.py`` / ``tests/test_dragonnet.py``:
the torch import is wrapped in try/except and every test carries the ``torch`` marker,
so the suite skips them unless ``--runtorch`` is passed.

The primary test recovers the average treatment effect of a synthetic binary-outcome
uplift dataset with a known generating uplift, which is DESCN's published setting.
"""

import numpy as np
import pytest

try:
    import torch

    from causalml.inference.torch import DESCN
    from causalml.inference.torch.descn import ESX, descn_loss
except ImportError:
    pass

from causalml.dataset import make_uplift_classification

from .const import RANDOM_SEED


@pytest.mark.torch
def test_descn_ate_recovery_on_uplift_classification():
    """DESCN recovers the known uplift of a binary synthetic dataset.

    make_uplift_classification adds delta_uplift_increase to the treated conversion
    probability (with positive_class_proportion=0.5), so the generating ATE is
    delta_uplift_increase. The tolerance is loose: this is a sanity gate, not a
    benchmark-parity run.
    """
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    delta = 0.2
    df, x_names = make_uplift_classification(
        n_samples=1000,
        treatment_name=["control", "treatment1"],
        y_name="conversion",
        n_classification_features=10,
        n_classification_informative=5,
        n_uplift_increase_dict={"treatment1": 2},
        delta_uplift_increase_dict={"treatment1": delta},
        n_uplift_increase_mix_informative_dict={"treatment1": 1},
        positive_class_proportion=0.5,
        random_seed=RANDOM_SEED,
    )
    X = df[x_names].values
    treatment = (df["treatment_group_key"] == "treatment1").astype(int).values
    y = df["conversion"].values

    descn = DESCN(share_dim=32, base_dim=16, batch_size=256, epochs=50)
    tau = descn.fit_predict(X=X, treatment=treatment, y=y)

    assert tau.shape == (len(y), 1)
    assert np.isfinite(tau).all()
    assert abs(tau.mean() - delta) < 0.1


@pytest.mark.torch
def test_descn_module_forward_backward():
    """The ESX module and multi-task loss produce finite outputs and gradients."""
    torch.manual_seed(RANDOM_SEED)

    model = ESX(input_dim=6, share_dim=16, base_dim=8)
    x = torch.randn(8, 6)
    t = torch.tensor([[1.0], [0.0], [1.0], [1.0], [0.0], [1.0], [0.0], [0.0]])
    y = torch.tensor([[1.0], [0.0], [1.0], [0.0], [1.0], [1.0], [0.0], [0.0]])

    outputs = model(x)

    assert outputs["p_prpsy"].shape == (8, 1)
    assert outputs["p_estr"].shape == (8, 1)
    assert outputs["shared_h"].shape == (8, 8)
    for key in ("p_prpsy", "p_estr", "p_escr", "p_h1", "p_h0"):
        assert torch.isfinite(outputs[key]).all()
        assert (outputs[key] > 0).all() and (outputs[key] < 1).all()

    loss = descn_loss(outputs, t, y)
    assert torch.isfinite(loss)

    loss.backward()
    for tower_name in ("prspy_network", "mu1_network", "mu0_network", "tau_network"):
        for param in getattr(model, tower_name).parameters():
            assert param.grad is not None
            assert torch.isfinite(param.grad).all()

    # the optional BatchNorm1d trunk is wired the same way
    model_bn = ESX(input_dim=6, share_dim=16, base_dim=8, batch_norm=True)
    outputs_bn = model_bn(x)
    assert torch.isfinite(outputs_bn["p_prpsy"]).all()


@pytest.mark.torch
def test_descn_degenerate_batches():
    """Single-group batches and constant outcomes do not NaN the loss or the fit.

    The inverse-propensity reweighting divides by the batch's treated fraction,
    and the missing group's loss terms select an empty batch, whose mean-reduced
    BCE would otherwise be nan.
    """
    torch.manual_seed(RANDOM_SEED)

    model = ESX(input_dim=6, share_dim=16, base_dim=8)
    x = torch.randn(8, 6)
    y = torch.randint(0, 2, (8, 1)).float()

    for t_value in (1.0, 0.0):
        treatment = torch.full((8, 1), t_value)

        loss = descn_loss(model(x), treatment, y)
        assert torch.isfinite(loss)

        model.zero_grad()
        loss.backward()
        for param in model.parameters():
            assert param.grad is None or torch.isfinite(param.grad).all()

    # a constant outcome carries no treatment-effect signal
    np.random.seed(RANDOM_SEED)
    X = np.random.normal(size=(200, 6))
    treatment = (np.random.rand(200) < 0.5).astype(int)
    y_constant = np.zeros(200)

    descn = DESCN(share_dim=16, base_dim=8, batch_size=64, epochs=10)
    tau = descn.fit_predict(X=X, treatment=treatment, y=y_constant)

    assert np.isfinite(tau).all()
    assert abs(tau.mean()) < 0.05


@pytest.mark.torch
def test_descn_fit_predict_smoke():
    """fit_predict / predict / predict_propensity return the documented shapes."""
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    df, x_names = make_uplift_classification(
        n_samples=200,
        treatment_name=["control", "treatment1"],
        y_name="conversion",
        n_uplift_increase_dict={"treatment1": 2},
        delta_uplift_increase_dict={"treatment1": 0.2},
        random_seed=RANDOM_SEED,
    )
    X = df[x_names].values
    treatment = (df["treatment_group_key"] == "treatment1").astype(int).values
    y = df["conversion"].values

    descn = DESCN(share_dim=16, base_dim=8, batch_size=128, epochs=5)
    tau = descn.fit_predict(X=X, treatment=treatment, y=y)
    assert tau.shape == (len(y), 1)
    assert np.isfinite(tau).all()

    components = descn.predict(X=X)
    assert components.shape == (len(y), 3)
    assert (components >= 0).all() and (components <= 1).all()

    propensity = descn.predict_propensity(X=X)
    assert propensity.shape == (len(y),)
    assert (propensity > 0).all() and (propensity < 1).all()
