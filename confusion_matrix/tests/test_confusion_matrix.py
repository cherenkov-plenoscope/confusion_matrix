import confusion_matrix
import numpy as np
import pytest


def test_apply_unity_confusion_matrix():
    x = np.array([1, 2, 3,])
    cm = np.eye(3)
    y = confusion_matrix.apply(x, cm)
    assert y[0] == x[0]
    assert y[1] == x[1]
    assert y[2] == x[2]


def test_apply_confusion_matrix():
    x = np.array([1, 2, 3,])
    cm = np.array([[0.9, 0.1, 0.0], [0.2, 0.7, 0.1], [0.1, 0.1, 0.8],])
    y = confusion_matrix.apply(x, cm)
    assert y[0] == x[0] * 0.9 + x[1] * 0.2 + x[2] * 0.1
    assert y[1] == x[0] * 0.1 + x[1] * 0.7 + x[2] * 0.1
    assert y[2] == x[0] * 0.0 + x[1] * 0.1 + x[2] * 0.8


def test_confusion_matrix_bad_dimensions():
    with pytest.raises(AssertionError) as err:
        x = np.array([1, 2])
        cm = np.array([[1,]])
        y = confusion_matrix.apply(x, cm)


def test_confusion_matrix_not_normalized():
    x = np.array([1, 2, 3,])
    cm = np.array([[0.9, 0.1, 0.0], [0.1, 0.7, 0.1], [0.0, 0.2, 0.9],])
    with pytest.raises(AssertionError) as err:
        y = confusion_matrix.apply(x, cm)
    y = confusion_matrix.apply(x, cm.T)


def test_init():
    prng = np.random.Generator(np.random.MT19937(seed=42))

    K = 1000 * 1000
    ax0n = 23
    ax1n = 25

    ax0_val = np.linspace(0, 1, K) + prng.normal(size=K, scale=0.1)
    ax1_val = np.linspace(0, 1, K) + prng.normal(size=K, scale=0.1)

    ax0_edges = np.linspace(0, 1, ax0n + 1)
    ax1_edges = np.linspace(0, 1, ax1n + 1)

    min_exposure_ax0 = 55

    cm = confusion_matrix.init(
        ax0_key="ax0",
        ax0_values=ax0_val,
        ax0_bin_edges=ax0_edges,
        ax1_key="ax1",
        ax1_values=ax1_val,
        ax1_bin_edges=ax1_edges,
        weights=np.ones(K),
        min_exposure_ax0=min_exposure_ax0,
        default_low_exposure=np.nan,
    )

    assert cm["ax0_key"] == "ax0"
    assert cm["ax1_key"] == "ax1"

    np.testing.assert_array_almost_equal(cm["ax0_bin_edges"], ax0_edges)
    np.testing.assert_array_almost_equal(cm["ax1_bin_edges"], ax1_edges)

    assert cm["counts"].shape == (ax0n, ax1n)
    assert np.all(cm["counts"] >= 0)
    assert np.sum(cm["counts"]) <= K
    assert cm["counts_ru"].shape == (ax0n, ax1n)
    assert np.all(cm["counts_ru"] >= 0)
    assert cm["counts_au"].shape == (ax0n, ax1n)
    assert np.all(cm["counts_au"] >= 0)
    assert cm["exposure_ax0"].shape == (ax0n, )
    assert cm["counts_ax0"].shape == (ax0n, )
    assert np.sum(cm["counts_ax0"]) == np.sum(cm["counts"])
    assert cm["min_exposure_ax0"] == min_exposure_ax0
