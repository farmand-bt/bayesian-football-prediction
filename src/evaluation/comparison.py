import numpy as np
import pandas as pd
import arviz as az


def get_realistic_model_predictions(trace, data, teams, model_type, n_simulations=1500):
    """
    Simulate full seasons from posterior theta_g1 / theta_g2 samples.

    For each of n_simulations posterior draws, every match is replayed by
    sampling Poisson goals from the draw's scoring intensities.  Season totals
    (points, goals scored/conceded, W/D/L) are recorded and the median across
    simulations is returned — matching the approach in the original paper.

    Parameters
    ----------
    trace : arviz.InferenceData
        Fitted posterior from any model that stores theta_g1 / theta_g2.
    data : pd.DataFrame
        Data as returned by FootballDataLoader (normalised column names).
    teams : list[str]
        Sorted list of team name strings.
    model_type : str
        Label used as a column prefix in the returned list of dicts.
    n_simulations : int
        Number of posterior draws to use (resampled with replacement if fewer
        are available).

    Returns
    -------
    list[dict]  — one entry per team, with keys:
        team, {model_type}_points, _scored, _conceded, _wins, _draws, _losses
    """
    np.random.seed(42)

    if "theta_g1" not in trace.posterior.data_vars:
        raise ValueError(f"theta_g1 not in posterior for '{model_type}' model.")

    theta1 = trace.posterior["theta_g1"].values   # (chains, draws, games)
    theta2 = trace.posterior["theta_g2"].values

    n_chains, n_draws, n_games = theta1.shape
    theta1_flat = theta1.reshape(-1, n_games)
    theta2_flat = theta2.reshape(-1, n_games)
    n_available = len(theta1_flat)

    if n_available < n_simulations:
        idx = np.random.choice(n_available, size=n_simulations, replace=True)
    else:
        idx = np.arange(n_simulations)

    theta1_sim = theta1_flat[idx]
    theta2_sim = theta2_flat[idx]

    pred_stats = []

    for team in teams:
        team_games = data[(data["home_team"] == team) | (data["away_team"] == team)].copy()

        pts, scored, conceded, wins, draws, losses = [], [], [], [], [], []

        for sim_idx in range(n_simulations):
            sp = ss = sc = sw = sd = sl = 0

            for _, match in team_games.iterrows():
                gi = match.name   # original row index == game index in theta arrays
                hg = np.random.poisson(theta1_sim[sim_idx, gi])
                ag = np.random.poisson(theta2_sim[sim_idx, gi])

                tg, og = (hg, ag) if match["home_team"] == team else (ag, hg)

                ss += tg
                sc += og
                if tg > og:
                    sp += 3; sw += 1
                elif tg == og:
                    sp += 1; sd += 1
                else:
                    sl += 1

            pts.append(sp); scored.append(ss); conceded.append(sc)
            wins.append(sw); draws.append(sd); losses.append(sl)

        pred_stats.append({
            "team": team,
            f"{model_type}_points":   int(np.median(pts)),
            f"{model_type}_scored":   int(np.median(scored)),
            f"{model_type}_conceded": int(np.median(conceded)),
            f"{model_type}_wins":     int(np.median(wins)),
            f"{model_type}_draws":    int(np.median(draws)),
            f"{model_type}_losses":   int(np.median(losses)),
        })

    return pred_stats


def compute_observed_stats(data, teams):
    """
    Calculate actual season results from raw match data.

    Returns
    -------
    list[dict]  — one entry per team, with keys:
        team, obs_points, obs_scored, obs_conceded, obs_wins, obs_draws, obs_losses
    """
    observed = []

    for team in teams:
        team_games = data[(data["home_team"] == team) | (data["away_team"] == team)].copy()
        pts = scored = conceded = wins = draws = losses = 0

        for _, match in team_games.iterrows():
            if match["home_team"] == team:
                gf, ga = int(match["y1"]), int(match["y2"])
            else:
                gf, ga = int(match["y2"]), int(match["y1"])

            scored += gf
            conceded += ga

            if gf > ga:
                pts += 3; wins += 1
            elif gf == ga:
                pts += 1; draws += 1
            else:
                losses += 1

        observed.append({
            "team": team,
            "obs_points":   pts,
            "obs_scored":   scored,
            "obs_conceded": conceded,
            "obs_wins":     wins,
            "obs_draws":    draws,
            "obs_losses":   losses,
        })

    return observed


def create_comparison_table(observed, *model_predictions):
    """
    Merge observed stats with one or more model prediction lists.

    Parameters
    ----------
    observed : list[dict]
        Output of compute_observed_stats().
    *model_predictions : list[dict]
        One or more outputs of get_realistic_model_predictions(); pass None to
        skip a model.

    Returns
    -------
    pd.DataFrame sorted by observed points (descending).
    """
    result = pd.DataFrame(observed).set_index("team")

    for preds in model_predictions:
        if preds is None:
            continue
        pred_df = pd.DataFrame(preds).set_index("team")
        result = result.join(pred_df, how="left")

    return result.reset_index().sort_values("obs_points", ascending=False).reset_index(drop=True)


def print_mae_comparison(comparison_df, model_names):
    """
    Print a mean absolute error table for each model against observed results.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Output of create_comparison_table().
    model_names : list[str]
        Labels matching the column prefixes in comparison_df.
    """
    metrics = ["points", "scored", "conceded", "wins", "draws", "losses"]
    header = f"{'Model':20s}  " + "  ".join(f"{m:>8}" for m in metrics) + f"  {'Total':>8}"
    print("\n" + header)
    print("-" * len(header))

    for name in model_names:
        row_mae = []
        for m in metrics:
            obs_col  = f"obs_{m}"
            pred_col = f"{name}_{m}"
            if obs_col in comparison_df.columns and pred_col in comparison_df.columns:
                mae = float(np.mean(np.abs(comparison_df[obs_col] - comparison_df[pred_col])))
            else:
                mae = float("nan")
            row_mae.append(mae)

        total = float(np.nansum(row_mae))
        vals = "  ".join(f"{v:8.2f}" for v in row_mae)
        print(f"{name:20s}  {vals}  {total:8.2f}")


def compare_information_criteria(traces, model_names):
    """
    Compute WAIC and LOO for each fitted trace via ArviZ.

    Parameters
    ----------
    traces : list[arviz.InferenceData]
    model_names : list[str]

    Returns
    -------
    pd.DataFrame indexed by model name with columns waic, waic_se, loo, loo_se.
    """
    rows = []
    for name, trace in zip(model_names, traces):
        try:
            waic = az.waic(trace)
            loo  = az.loo(trace)
            rows.append({
                "model":   name,
                "waic":    float(waic.waic),
                "waic_se": float(waic.se),
                "loo":     float(loo.loo),
                "loo_se":  float(loo.se),
            })
            print(f"{name:30s}  WAIC={waic.waic:8.1f} (±{waic.se:.1f})  "
                  f"LOO={loo.loo:8.1f} (±{loo.se:.1f})")
        except Exception as exc:
            print(f"{name}: could not compute IC — {exc}")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("model")
    print(f"\nBest by WAIC : {df['waic'].idxmin()}")
    print(f"Best by LOO  : {df['loo'].idxmin()}")
    return df
