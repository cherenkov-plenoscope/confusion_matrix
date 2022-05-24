import numpy as np


def init(
    ax0_key,
    ax0_values,
    ax0_bin_edges,
    ax1_key,
    ax1_values,
    ax1_bin_edges,
    ax0_weights=None,
    min_exposure_ax0=100,
    default_low_exposure=np.nan,
):
    assert len(ax0_values) == len(ax1_values)
    if ax0_weights is not None:
        assert len(ax0_values) == len(ax0_weights)

    num_bins_ax0 = len(ax0_bin_edges) - 1
    assert num_bins_ax0 >= 1

    num_bins_ax1 = len(ax1_bin_edges) - 1
    assert num_bins_ax1 >= 1

    counts = np.histogram2d(
        ax0_values,
        ax1_values,
        weights=ax0_weights,
        bins=[ax0_bin_edges, ax1_bin_edges],
    )[0]

    cb_rel_unc, cb_abs_unc = estimate_rel_abs_uncertainty_in_counts(
        counts=counts
    )

    exposure_no_weights = np.histogram2d(
        ax0_values, ax1_values, bins=[ax0_bin_edges, ax1_bin_edges],
    )[0]

    counts_normalized_on_ax0 = counts.copy()
    cbn_abs_unc = cb_abs_unc.copy()
    for i0 in range(num_bins_ax0):
        if np.sum(exposure_no_weights[i0, :]) >= min_exposure_ax0:
            axsum = np.sum(counts[i0, :])
            counts_normalized_on_ax0[i0, :] /= axsum
            cbn_abs_unc[i0, :] /= axsum
        else:
            counts_normalized_on_ax0[i0, :] = (
                np.ones(num_bins_ax1) * default_low_exposure
            )

    return {
        "ax0_key": ax0_key,
        "ax1_key": ax1_key,
        "ax0_bin_edges": ax0_bin_edges,
        "ax1_bin_edges": ax1_bin_edges,
        "counts": counts,
        "counts_rel_unc": cb_rel_unc,
        "counts_abs_unc": cb_abs_unc,
        "counts_normalized_on_ax0": counts_normalized_on_ax0,
        "counts_normalized_on_ax0_abs_unc": cbn_abs_unc,
        "exposure_ax0_no_weights": np.sum(exposure_no_weights, axis=1),
        "exposure_ax0": np.sum(counts, axis=1),
        "min_exposure_ax0": min_exposure_ax0,
    }


def estimate_rel_abs_uncertainty_in_counts(counts):
    cb = counts
    assert np.all(cb >= 0)
    shape = cb.shape

    rel_unc = np.nan * np.ones(shape=shape)
    abs_unc = np.nan * np.ones(shape=shape)

    has_expo = cb > 0
    no_expo = cb == 0

    # frequency regime
    # ----------------
    rel_unc[has_expo] = 1.0 / np.sqrt(cb[has_expo])
    abs_unc[has_expo] = cb[has_expo] * rel_unc[has_expo]

    # no frequency regime, have to approximate
    # ----------------------------------------
    _num_bins_with_exposure = np.sum(has_expo)
    _num_bins = shape[0] * shape[1]

    pseudocount = np.sqrt(_num_bins_with_exposure / _num_bins)
    assert pseudocount <= 1.0

    if pseudocount == 0:
        # this can not be saved
        return rel_unc, abs_unc

    rel_unc[no_expo] = 1.0 / np.sqrt(pseudocount)
    abs_unc[no_expo] = pseudocount

    return rel_unc, abs_unc


def apply_confusion_matrix(x, confusion_matrix, x_unc=None):
    """
    Parameters
    ----------
    x : 1D-array
            E.g. Effective acceptance vs. true energy.
    confusion_matrix : 2D-array
            Confusion between e.g. true and reco. energy.
            The rows are expected to be notmalized:
            CM[i, :] == 1.0
    """
    cm = confusion_matrix
    n = cm.shape[0]
    assert cm.shape[1] == n
    assert x.shape[0] == n

    # assert confusion matrix is normalized
    for i in range(n):
        s = np.sum(cm[i, :])
        assert np.abs(s - 1) < 1e-3 or s < 1e-3

    y = np.zeros(shape=(n))
    for r in range(n):
        for t in range(n):
            y[r] += cm[t, r] * x[t]

    return y