# src/placecells/simulate_env.py
from place_setup import (
    make_track, make_experience_distribution, sample_positions,
    make_place_fields, plot_experience, plot_tuning_curves, plot_population_heatmap
)

def main():
    x = make_track(1.0, 0.001)

    P = make_experience_distribution(x, kind="gaussian", center=0.3, sigma=0.08)
    samples = sample_positions(x, P, n_samples=20_000)

    F, centers = make_place_fields(x, n_cells=40, width=0.05, jitter=0.01)

    plot_experience(x, P, samples, title_suffix=" (biased)")
    plot_tuning_curves(x, F, centers, max_curves=25)
    plot_population_heatmap(x, F)

if __name__ == "__main__":
    main()
