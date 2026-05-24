import numpy as np
import matplotlib.pyplot as plt
import arviz as az


def plot_team_effects(trace, teams, model_type="basic"):
    """
    Attack vs defence scatter plot with quadrant labels.

    Parameters
    ----------
    trace : arviz.InferenceData
    teams : list[str]
    model_type : str  — "basic" uses 'att'/'def'; anything else uses
                        'att_centered'/'def_centered' (mixture model).

    Returns
    -------
    matplotlib.figure.Figure
    """
    if model_type == "basic":
        att_means = trace.posterior["att"].mean(dim=["chain", "draw"]).values
        def_means = trace.posterior["def"].mean(dim=["chain", "draw"]).values
    else:
        att_means = trace.posterior["att_centered"].mean(dim=["chain", "draw"]).values
        def_means = trace.posterior["def_centered"].mean(dim=["chain", "draw"]).values

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(att_means, def_means, s=100, alpha=0.7)

    for i, team in enumerate(teams):
        ax.annotate(
            team, (att_means[i], def_means[i]),
            xytext=(5, 5), textcoords="offset points", fontsize=8, alpha=0.8,
        )

    ax.axhline(0, color="k", linestyle="--", alpha=0.5)
    ax.axvline(0, color="k", linestyle="--", alpha=0.5)
    ax.set_xlabel("Attack Effect")
    ax.set_ylabel("Defence Effect")
    ax.set_title(f"Team Attack vs Defence  ({model_type.title()} Model)")
    ax.grid(True, alpha=0.3)

    quadrants = [
        (0.02, 0.98, "top",    "left",  "Poor Attack\nPoor Defence",  "lightcoral"),
        (0.98, 0.98, "top",    "right", "Good Attack\nPoor Defence",  "lightgray"),
        (0.02, 0.02, "bottom", "left",  "Poor Attack\nGood Defence",  "lightyellow"),
        (0.98, 0.02, "bottom", "right", "Good Attack\nGood Defence",  "lightgreen"),
    ]
    for x, y, va, ha, label, color in quadrants:
        ax.text(x, y, label, va=va, ha=ha, transform=ax.transAxes,
                bbox=dict(boxstyle="round", facecolor=color, alpha=0.5))

    plt.tight_layout()
    return fig


def plot_traceplots(trace, model_type="basic", var_names=None, figsize=(15, 12)):
    """
    ArviZ trace plots for key global parameters.

    Parameters
    ----------
    trace : arviz.InferenceData
    model_type : str  — used to pick sensible defaults when var_names is None.
    var_names : list[str] | None
    figsize : tuple

    Returns
    -------
    matplotlib axes array (from az.plot_trace)
    """
    if var_names is None:
        if model_type == "basic":
            var_names = ["home_advantage", "mu_att", "mu_def", "tau_att", "tau_def"]
        elif model_type == "mixture":
            var_names = [
                "home_advantage",
                "mu_att_1", "mu_att_2", "mu_att_3",
                "mu_def_1", "mu_def_2", "mu_def_3",
            ]
        elif model_type == "team_home":
            var_names = ["mu_home", "tau_home"]
        else:  # covariate
            var_names = [
                "home_base", "beta_stadium", "beta_distance",
                "beta_friday", "beta_saturday", "beta_sunday",
                "beta_season", "beta_travel_fatigue",
            ]

    available = list(trace.posterior.data_vars)
    var_names = [v for v in var_names if v in available]

    axes = az.plot_trace(trace, var_names=var_names, figsize=figsize,
                         combined=False, compact=False)
    plt.suptitle(f"Trace Plots — {model_type.title()} Model", fontsize=14, y=1.01)
    plt.tight_layout()
    return axes


def plot_team_effect_traceplots(trace, teams, model_type="basic", n_teams=5):
    """
    Per-team attack / defence trace and posterior distribution panels.

    Parameters
    ----------
    trace : arviz.InferenceData
    teams : list[str]
    model_type : str
    n_teams : int  — number of teams to plot (first n_teams in alphabetical order)

    Returns
    -------
    matplotlib.figure.Figure
    """
    att_param = "att"       if model_type == "basic" else "att_centered"
    def_param = "def"       if model_type == "basic" else "def_centered"

    if att_param not in trace.posterior.data_vars:
        raise ValueError(f"'{att_param}' not found in trace posterior.")

    selected = teams[:n_teams]
    n = len(selected)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for row, team in enumerate(selected):
        ti = teams.index(team)
        att_s = trace.posterior[att_param].values[:, :, ti]   # (chains, draws)
        def_s = trace.posterior[def_param].values[:, :, ti]

        for chain in range(att_s.shape[0]):
            axes[row, 0].plot(att_s[chain], alpha=0.7, label=f"Chain {chain}")
            axes[row, 2].plot(def_s[chain], alpha=0.7, label=f"Chain {chain}")

        axes[row, 0].set_title(f"{team} — Attack (trace)")
        axes[row, 0].legend(fontsize=7)
        axes[row, 2].set_title(f"{team} — Defence (trace)")
        axes[row, 2].legend(fontsize=7)

        axes[row, 1].hist(att_s.flatten(), bins=50, alpha=0.7, density=True)
        axes[row, 1].set_title(f"{team} — Attack (posterior)")
        axes[row, 3].hist(def_s.flatten(), bins=50, alpha=0.7, density=True)
        axes[row, 3].set_title(f"{team} — Defence (posterior)")

        for ax in axes[row]:
            ax.grid(True, alpha=0.3)

    plt.suptitle(f"Team Effect Trace Plots — {model_type.title()} Model", fontsize=14)
    plt.tight_layout()
    return fig


def plot_covariate_effects(trace, model_type="full"):
    """
    Posterior histograms for all beta coefficients in the covariate model.

    Parameters
    ----------
    trace : arviz.InferenceData
    model_type : str  — label used in the figure title only.

    Returns
    -------
    matplotlib.figure.Figure
    """
    beta_params = [
        ("beta_stadium",        "Stadium quality"),
        ("beta_distance",       "Distance (home adv)"),
        ("beta_friday",         "Friday games"),
        ("beta_saturday",       "Saturday games"),
        ("beta_sunday",         "Sunday games"),
        ("beta_season",         "Season phase"),
        ("beta_travel_fatigue", "Away travel fatigue"),
    ]

    present = [(p, l) for p, l in beta_params if p in trace.posterior.data_vars]
    n = len(present)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for i, (param, label) in enumerate(present):
        samples = trace.posterior[param].values.flatten()
        axes[i].hist(samples, bins=50, alpha=0.7, density=True, color="steelblue")
        axes[i].axvline(0, color="red", linestyle="--", alpha=0.8)
        axes[i].axvline(float(np.mean(samples)), color="blue", linestyle="-", alpha=0.8)
        sig = np.percentile(samples, 2.5) > 0 or np.percentile(samples, 97.5) < 0
        axes[i].set_title(f"{label}\n{'* significant' if sig else 'not significant'}")
        axes[i].set_xlabel("Effect size (per 1 SD)")
        axes[i].grid(True, alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f"Standardised Covariate Effects — {model_type.title()} Model", fontsize=14)
    plt.tight_layout()
    return fig


def plot_team_home_comparison(basic_trace, team_home_trace, teams):
    """
    Compare fixed home advantage (basic model) with team-specific estimates.

    Left panel: fixed scalar posterior distribution.
    Right panel: per-team posterior means with 95 % credible intervals,
    sorted descending.

    Parameters
    ----------
    basic_trace : arviz.InferenceData
    team_home_trace : arviz.InferenceData
    teams : list[str]

    Returns
    -------
    matplotlib.figure.Figure
    """
    fixed_samples = basic_trace.posterior["home_advantage"].values.flatten()
    team_samples  = team_home_trace.posterior["home_advantage"]  # (chains, draws, teams)

    team_means = team_samples.mean(dim=["chain", "draw"]).values
    team_ci_lo = np.quantile(team_samples.values.reshape(-1, len(teams)), 0.025, axis=0)
    team_ci_hi = np.quantile(team_samples.values.reshape(-1, len(teams)), 0.975, axis=0)

    order = np.argsort(team_means)[::-1]
    sorted_teams  = [teams[i] for i in order]
    sorted_means  = team_means[order]
    sorted_ci_lo  = team_ci_lo[order]
    sorted_ci_hi  = team_ci_hi[order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(6, len(teams) * 0.4 + 2)))

    # Left: fixed home advantage posterior
    ax1.hist(fixed_samples, bins=60, alpha=0.7, density=True, color="steelblue")
    ax1.axvline(float(np.mean(fixed_samples)), color="red", linestyle="--", alpha=0.8,
                label=f"Mean = {np.mean(fixed_samples):.3f}")
    ax1.set_title("Fixed Home Advantage\n(Basic Model)")
    ax1.set_xlabel("Home advantage (log scale)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: team-specific home advantages
    y_pos = np.arange(len(teams))
    ax2.barh(y_pos, sorted_means, xerr=[sorted_means - sorted_ci_lo,
                                         sorted_ci_hi - sorted_means],
             alpha=0.7, color="steelblue", error_kw={"capsize": 3})
    ax2.axvline(float(np.mean(fixed_samples)), color="red", linestyle="--",
                alpha=0.8, label="Fixed estimate")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(sorted_teams, fontsize=8)
    ax2.set_title("Team-Specific Home Advantages\n(TeamHome Model)")
    ax2.set_xlabel("Home advantage (log scale)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Home Advantage: Fixed vs Team-Specific", fontsize=14)
    plt.tight_layout()
    return fig
