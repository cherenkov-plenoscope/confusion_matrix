from .version import __version__
import numpy as np


def init(
    ax0_key,
    ax0_values,
    ax0_bin_edges,
    ax1_key,
    ax1_values,
    ax1_bin_edges,
    weights=None,
    min_exposure_ax0=100,
    default_low_exposure=np.nan,
):
    """
    Populates a confusion-matrix based on K confusion-pairs.

    Confusion-pair:
        - ax0_value
        - ax1_value
        - weight (optional)

    Parameters
    ----------
    ax0_key : str
        Name of axis-0.
    ax0_values : array of K floats.
        Axis-0 value of the confusion-pairs.
    ax0_bin_edges : array of (M+1) floats
        Edges of the bins used to histogram ax0.
    ax1_key : str
        Name of axis-1.
    ax1_values : array of K floats.
        Axis-0 value of the confusion-pairs.
    ax1_bin_edges : array of (N+1) floats
        Edges of the bins used to histogram ax1.
    weights : array of K floats.
        Weights of the confusion-pairs.
    min_exposure_ax0 : float
        Minimal exposure counts on axis-0 in order to normalize on axis-0.
    default_low_exposure : float
        Default value for counts_normalized when min_exposure_ax0 is
        not satisfied.

    Returns
    -------
    confusion_matrix : dict
        - counts : array(M x N)
            Weighted confusion-matrix.
        - counts_au : array(M x N)
            Absolute uncertainty of counts.
        - counts_ru : array(M x N)
            Relative uncertainty of counts.
        - counts_normalized_on_ax0 : array(M x N)
            Counts but normalized on ax0.
            I.e. sum(counts_normalized_on_ax0, axis=0) = 1.
        - counts_normalized_on_ax0_au : array(M x N)
            Absolute uncertainty of counts_normalized_on_ax0.
        - counts_ax0 : array(M)
            Projection of counts on ax0.
        - exposure_ax0 : array(M)
            Projection of exposure on ax0. Like counts_ax0 but without
            the confusion-pairs weigths.
    """
    assert len(ax0_values) == len(ax1_values)
    if weights is not None:
        assert len(ax0_values) == len(weights)
        assert np.all(weights) >= 0.0

    num_bins_ax0 = len(ax0_bin_edges) - 1
    assert num_bins_ax0 >= 1

    num_bins_ax1 = len(ax1_bin_edges) - 1
    assert num_bins_ax1 >= 1

    # histogram
    # ---------
    counts = np.histogram2d(
        ax0_values,
        ax1_values,
        weights=weights,
        bins=[ax0_bin_edges, ax1_bin_edges],
    )[0]

    exposure = np.histogram2d(
        ax0_values,
        ax1_values,
        bins=[ax0_bin_edges, ax1_bin_edges],
    )[0]

    # uncertainty
    # -----------
    counts_ru, counts_au = estimate_relative_and_absolute_uncertainties(
        counts=counts,
        exposure=exposure,
    )

    # normalize
    # ---------
    counts_normalized_on_ax0, counts_normalized_on_ax0_au = normalize_on_ax0(
        counts=counts,
        counts_au=counts_au,
        exposure=exposure,
        min_exposure_ax0=min_exposure_ax0,
        default_low_exposure=default_low_exposure,
    )

    return {
        "ax0_key": ax0_key,
        "ax1_key": ax1_key,
        "ax0_bin_edges": ax0_bin_edges,
        "ax1_bin_edges": ax1_bin_edges,
        "counts": counts,
        "counts_ru": counts_ru,
        "counts_au": counts_au,
        "counts_normalized_on_ax0": counts_normalized_on_ax0,
        "counts_normalized_on_ax0_au": counts_normalized_on_ax0_au,
        "exposure_ax0": np.sum(exposure, axis=1),
        "counts_ax0": np.sum(counts, axis=1),
        "min_exposure_ax0": min_exposure_ax0,
    }


def estimate_relative_and_absolute_uncertainties(counts, exposure):
    """
    A simple estimator for the uncertainty of a confusion-matrix made from
    confusion-pairs.
    It uses 1/sqrt in the frequency-regime and falls back to a pseudocount
    estimatior for bins without any exposure.

    Parameters
    ----------
    counts : array(M x N) of floats
        Histogram of weighted confusion-pairs.
    exposure : array(M x N) of floats
        Histogram of confusion-pairs without any weights.

    Returns
    -------
    (ru, au) : (array(M x N) of floats, array(M x N) of floats)
        Two matrices with first the relative, and second the
        absolute uncertainty of the counts.
    """
    assert np.all(exposure >= 0)
    assert counts.shape == exposure.shape
    shape = counts.shape

    rel_unc = np.nan * np.ones(shape=shape)
    abs_unc = np.nan * np.ones(shape=shape)

    has_expo = exposure > 0
    no_expo = exposure == 0

    # frequency regime
    # ----------------
    rel_unc[has_expo] = 1.0 / np.sqrt(exposure[has_expo])
    abs_unc[has_expo] = counts[has_expo] * rel_unc[has_expo]

    # no frequency regime, have to approximate
    # ----------------------------------------
    _num_bins_with_exposure = np.sum(has_expo)
    _num_bins = shape[0] * shape[1]

    pseudocount = np.sqrt(_num_bins_with_exposure / _num_bins)
    assert pseudocount <= 1.0

    if pseudocount == 0:
        # this can not be recovered
        return rel_unc, abs_unc

    rel_unc[no_expo] = 1.0 / np.sqrt(pseudocount)
    abs_unc[no_expo] = pseudocount

    return rel_unc, abs_unc


def normalize_on_ax0(
    counts, counts_au, exposure, min_exposure_ax0, default_low_exposure=np.nan
):
    assert counts.shape == counts_au.shape
    assert counts.shape == exposure.shape
    assert np.all(exposure >= 0)
    assert min_exposure_ax0 > 0
    num_bins_ax0 = exposure.shape[0]
    num_bins_ax1 = exposure.shape[1]

    counts_normalized_on_ax0 = counts.copy()
    counts_normalized_on_ax0_au = counts_au.copy()

    for i0 in range(num_bins_ax0):
        if np.sum(exposure[i0, :]) >= min_exposure_ax0:
            axsum = np.sum(counts[i0, :])
            counts_normalized_on_ax0[i0, :] /= axsum
            counts_normalized_on_ax0_au[i0, :] /= axsum
        else:
            counts_normalized_on_ax0[i0, :] = (
                np.ones(num_bins_ax1) * default_low_exposure
            )

    return counts_normalized_on_ax0, counts_normalized_on_ax0_au


def apply(x, confusion_matrix, x_unc=None):
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
