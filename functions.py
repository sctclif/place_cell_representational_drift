from modules import *

def get_path(folder_name, file_name): #just makes directories if they don't exist
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    grandparent_dir = os.path.dirname(parent_dir)
    folder_path = os.path.join(grandparent_dir, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, file_name)
    return file_path

def position_gaussian(N, x, amp=2.0, sigma=0.1, baseline=0.0, L=1.0):
    """
    Compute a bank of N Gaussian place fields over a 1-D line [0, L] instead of the previous circular space.
    """
    centers = np.linspace(0, L, N)  # receptive field centers
    d = np.abs(x - centers)
    y = amp * np.exp(-(d**2) / (2 * sigma**2)) + baseline
    return y

def place_matrix_different_vars(N, vars, L=1.0, diagonal=True):
    """
    Build an (N x N) basis matrix of Gaussian place fields with per-neuron sigma=vars[i].
    Column i is the tuning of input neuron i sampled at N evenly spaced positions
    """
    x = np.linspace(0, L, N)
    matrix = np.zeros((N, N))
    for i in range(N):
        mean = x[i]
        matrix[:, i] = stats.norm.pdf(x, mean, vars[i])
        if diagonal is False:
            matrix[i, i] = 0.0
    return matrix

def linear_distance(x, y):
    """
    Absolute distance between arrays x and y on a line (no circularity).
    """
    return np.abs(x - y)

def signed_linear_distance(x, y):
    """
    Signed distance on a line (x - y). This is mostly here to create parity to the circular version.
    """
    return x - y

def convert_to_array(*args):
    """
    Convert any inputs to numpy arrays and return them.
    (Fixes original bug where re-binding didn't affect returned tuple.)
    """
    return tuple(np.array(a) for a in args)

def propensity(w, a):
    return np.tanh(a*w)

def get_preferred_positions(N, W, n_positions, L=1.0, amp=1.0, sigma=0.05, baseline=0.0):
    """
    Estimate each postsynaptic neuron's preferred position (PP) by brute-force sweep.
    - Build inputs u(x) (Gaussian bank) at n_positions along [0, L]
    - Compute v(x) = W^T u(x); pick argmax_x per postsyn neuron.
    Returns: array of length N with preferred positions in [0, L].
    """
    xs = np.linspace(0.0, L, n_positions)
    posts = np.zeros((N, n_positions))
    for i, x in enumerate(xs):
        u = position_gaussian(N, x, amp=amp, sigma=sigma, baseline=baseline, L=L)
        posts[:, i] = W.T.dot(u)
    idx = np.argmax(posts, axis=1)
    return xs[idx]

def initialise_W(N, vars, L=1.0):
    """
    Initialize W with a tiling of place fields (heterogeneous widths).
    Then normalize each column so inputs to each postsynaptic unit sum to 1.
    """
    W = place_matrix_different_vars(N, vars, L=L, diagonal=True) / N
    W /= (np.sum(W, axis=0, keepdims=True) + 1e-10)
    return W

def single_hebbian_component(N, W_old, x_stim, stim_type, L=1.0):
    """
    Produce the Hebbian outer-product u v^T for a single sample.
    - stim_type in {'baseline','test','deprivation'} (was 'stripe_rearing')
    - Baseline/Test: sample x ~ Uniform(0, L)
    - Deprivation: clamp x = x_stim (the experienced position)
    """
    if stim_type in ('baseline', 'test'):
        x = np.random.uniform(0.0, L)
    elif stim_type in ('deprivation',):
        x = x_stim
    else:
        raise ValueError(f"Unknown stim_type: {stim_type}")

    u = position_gaussian(N, x, amp=0.62, sigma=0.20 * L, baseline=0.0, L=L)  # sigma scaled to track length
    v = W_old.T.dot(u)
    return np.outer(u, v)

def normalisation(W):
    """
    Column-wise divisive normalization (homeostasis).
    In-place; keeps columns summing to ~1 and prevents runaway growth.
    """
    W /= (np.sum(W, axis=0, keepdims=True) + 1e-10)

def prerun(W_init, x_stim, a, hebb_scaling, rand_scaling, learning_rate, n_steps_per_norm, n_trials, L=1.0):
    """
    Short stabilization run under baseline sampling before the main schedule.
    """
    N = W_init.shape[0]
    W = W_init
    for t in range(n_trials):
        H = single_hebbian_component(N, W, x_stim, stim_type='baseline', L=L)
        eta = np.random.randn(N, N)
        rho = propensity(W, a)
        W = W + (hebb_scaling * H * rho + rand_scaling * eta * rho) * learning_rate
        if t % n_steps_per_norm == 0:
            normalisation(W)
    return W


def get_PPs_over_trials(W_init, n_steps, a, hebb_scaling, rand_scaling, learning_rate,
                        x_stim, n_steps_per_norm, n_norm_per_day, n_test_positions,
                        stim_type, L=1.0):
    """
    Evolve weights and periodically (per 'day') test preferred positions (PPs).
    Returns: list of PP arrays (one per test day).
    """
    N = W_init.shape[0]
    PPs = []
    W = np.zeros((N, N, n_steps + 1))
    W[:, :, 0] = W_init

    for t in tqdm(range(n_steps)):
        W_old = W[:, :, t]
        H = single_hebbian_component(N, W_old, x_stim, stim_type=stim_type, L=L)
        eta = np.random.randn(N, N)
        rho = propensity(W_old, a)
        W_new = W_old + (hebb_scaling * H * rho + rand_scaling * eta * rho) * learning_rate

        if t % n_steps_per_norm == 0:
            normalisation(W_new)
            if t % (n_steps_per_norm * n_norm_per_day) == 0:
                PP = get_preferred_positions(N, W_new, n_positions=n_test_positions, L=L)
                PPs.append(PP)  # save per 'day'

        W[:, :, t + 1] = W_new

    return PPs


def get_metrics(N, n_days, x_stim, PPs, L=1.0):
    """
    Compute:
    - drift_magnitude(day): |PP_day - PP_initial|
    - drift_rate(day->day+1): |PP_{d+1} - PP_d|
    - convergence(day): decrease in |PP_day - x_stim| relative to initial distance
    """
    preferences = np.array(PPs).T  # shape (N, n_days)
    initial_preferences = np.linspace(0.0, L, N)

    drift_magnitude = np.array([
        linear_distance(preferences[:, day], initial_preferences)
        for day in range(n_days)
    ])

    drift_rate = np.array([
        linear_distance(preferences[:, day + 1], preferences[:, day])
        for day in range(n_days - 1)
    ])

    initial_distances = np.abs(initial_preferences - x_stim)
    distances = np.abs(preferences - x_stim)
    convergence = np.array([
        initial_distances - distances[:, day]
        for day in range(n_days - 1)
    ])

    return drift_magnitude, drift_rate, convergence

def evolve_weights(N, W_old, t, stim_type, x_stim, a, learning_rate,
                   hebb_scaling, rand_scaling, n_steps_per_norm, L=1.0):
    """
    One learning step with Hebbian + volatility noise and periodic normalization.
    """
    H = single_hebbian_component(N, W_old, x_stim, stim_type=stim_type, L=L)
    eta = np.random.randn(N, N)
    rho = propensity(W_old, a)
    W_new = W_old + (hebb_scaling * H * rho + rand_scaling * eta * rho) * learning_rate
    if t % n_steps_per_norm == 0:
        normalisation(W_new)
    return W_new

def get_r_values(PPs, x_stim, n_days, N, L=1.0):
    """
    Spearman correlation between initial |PP - x_stim| and total drift magnitude,
    per day. Mirrors the orientation analysis but on a line.
    """
    preferences = np.array(PPs).T  # (N, n_days)
    initial_preferences = np.linspace(0.0, L, N)
    initial_distances = np.abs(initial_preferences - x_stim)

    total_drift = np.array([
        linear_distance(preferences[:, day], initial_preferences)
        for day in range(n_days)
    ]).T  # (N, n_days)

    r_values = np.array([
        stats.spearmanr(initial_distances, total_drift[:, day])[0]
        for day in range(n_days)
    ])
    return r_values

