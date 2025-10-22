# src/placecells/place_setup.py
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# ---------------------------
# 1) Track / Environment
# ---------------------------

def make_track(L=1.0, dx=0.001):
    """
    Build a 1D line [0, L] with spacing dx.
    Returns: x (np.ndarray of shape [n_positions])
    """
    x = np.arange(0.0, L + dx/2, dx)
    return x


# ---------------------------
# 2) Experience distribution P(x)
# ---------------------------

def make_experience_distribution(x, kind="uniform", **kwargs):
    """
    Create a probability distribution P(x) over positions.
    kind:
      - "uniform": flat over [x.min(), x.max()]
      - "gaussian": normal bump centered at 'center' with std 'sigma'
      - "mixture": sum of K Gaussians (centers, sigmas, weights)
    Returns: P (np.ndarray) normalized to sum(P)=1
    """
    x = np.asarray(x)

    if kind == "uniform":
        P = np.ones_like(x, dtype=float)

    elif kind == "gaussian":
        center = kwargs.get("center", (x[0] + x[-1]) / 2.0)
        sigma = kwargs.get("sigma", 0.1 * (x[-1] - x[0]))
        P = np.exp(-0.5 * ((x - center) / sigma) ** 2)

    elif kind == "mixture":
        centers = np.asarray(kwargs.get("centers", [(x[0] + x[-1]) / 3.0,
                                                    2*(x[0] + x[-1]) / 3.0]))
        sigmas = np.asarray(kwargs.get("sigmas", [0.08 * (x[-1]-x[0])] * len(centers)))
        weights = np.asarray(kwargs.get("weights", [1.0] * len(centers)), dtype=float)
        weights = weights / weights.sum()

        P = np.zeros_like(x, dtype=float)
        for c, s, w in zip(centers, sigmas, weights):
            P += w * np.exp(-0.5 * ((x - c) / s) ** 2)

    else:
        raise ValueError(f"Unknown kind='{kind}'")

    # avoid degenerate distributions
    P = np.clip(P, 1e-12, None)
    P = P / P.sum()
    return P


def sample_positions(x, P, n_samples=10_000, rng=rng):
    """
    Draw positions according to P(x). Returns array of sampled positions.
    """
    idx = rng.choice(len(x), size=n_samples, p=P, replace=True)
    return x[idx]


# ---------------------------
# 3) Place fields (static, environment stage)
# ---------------------------

def make_place_fields(x, n_cells=20, width=0.05, centers=None, jitter=0.0):
    """
    Create Gaussian place fields along the line.
    - x: 1D track object
    - n_cells: number of neurons
    - width: sigma of each field (in same units as x)
    - centers: optional array of field centers (len=n_cells), else spaced evenly
    - jitter: random jitter added to centers (std, same units as x)
    Returns: tuning matrix F (n_cells, len(x)) and centers (n_cells,)
    """
    L = x[-1] - x[0]
    if centers is None:  # generate evenly spaced centers if none present
        centers = np.linspace(x[0] + 0.05 * L, x[-1] - 0.05 * L, n_cells)
        if jitter > 0:
            centers = centers + rng.normal(0, jitter, size=n_cells)

    centers = np.asarray(centers)
    # Gaussian tuning (no wrap); peak ~1.0
    F = np.exp(-0.5 * ((x[None, :] - centers[:, None]) / width) ** 2)
    return F, centers


# ---------------------------
# 4) Visualization helpers
# ---------------------------

def plot_experience(x, P, samples=None, bins=80, title_suffix=""):
    """
    Visualize P(x) and (optionally) histogram of sampled positions.
    """
    fig, ax = plt.subplots(2 if samples is not None else 1, 1, figsize=(8, 5), sharex=True)

    if samples is None:
        ax = [ax]  # normalize to list

    # PDF line
    ax[0].plot(x, P, lw=2)
    ax[0].set_title(f"Experience distribution P(x){title_suffix}")
    ax[0].set_ylabel("Probability density (norm.)")

    # Samples histogram
    if samples is not None:
        ax[1].hist(samples, bins=bins, density=True)
        ax[1].set_title("Sampled positions (per 'day')")
        ax[1].set_xlabel("Position along track (m)")
        ax[1].set_ylabel("Density")

    plt.tight_layout()
    plt.show()


def plot_tuning_curves(x, F, centers, max_curves=20):
    """Line plot of a subset of tuning curves + centers."""
    k = min(max_curves, F.shape[0])
    plt.figure(figsize=(8, 4))
    for i in range(k):
        plt.plot(x, F[i], alpha=0.9)
    plt.scatter(centers[:k], np.ones(k)*1.02, marker='|', s=120)  # marks centers
    plt.ylim(0, 1.1)
    plt.xlabel("Position on track (m)")
    plt.ylabel("Tuning (a.u.)")
    plt.title(f"{k} place-field tuning curves")
    plt.tight_layout()
    plt.show()


def plot_population_heatmap(x, F):
    """Heatmap: neurons x position."""
    plt.figure(figsize=(8, 4))
    plt.imshow(F, aspect='auto', extent=[x[0], x[-1], 0, F.shape[0]], origin='lower')
    plt.colorbar(label="Tuning (a.u.)")
    plt.xlabel("Position on track (m)")
    plt.ylabel("Neuron index")
    plt.title("Population tuning heatmap")
    plt.tight_layout()
    plt.show()


# ---------------------------
# 5) Demo
# ---------------------------

if __name__ == "__main__":
    # A) Track
    x = make_track(L=1.0, dx=0.001)

    # B) Experience distributions
    P_uniform = make_experience_distribution(x, kind="uniform")
    P_bias = make_experience_distribution(x, kind="gaussian", center=0.3, sigma=0.08)

    # C) Sample a "day" of exploration
    samples_uniform = sample_positions(x, P_uniform, n_samples=15_000)
    samples_bias = sample_positions(x, P_bias, n_samples=15_000)

    # D) Place cells
    F, centers = make_place_fields(
        x,
        n_cells=30,
        width=0.05,         # try 0.03–0.10 to see coverage change
        centers=None,       # or pass custom center array
        jitter=0.01         # small irregularity looks more realistic
    )

    # E) Visuals
    plot_experience(x, P_uniform, samples_uniform, title_suffix=" (uniform)")
    plot_experience(x, P_bias, samples_bias, title_suffix=" (biased → 0.3 m)")
    plot_tuning_curves(x, F, centers, max_curves=20)
    plot_population_heatmap(x, F)
