import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from src.data_loader import FootballDataLoader


class TeamHomeModel:
    """
    Contribution 1: team-specific home advantage.

    Replaces the single home_advantage scalar of the basic model with a vector
    home_advantage[j] ~ Normal(mu_home, tau_home) for j = 1..n_teams.
    Partial pooling lets the data estimate how much teams genuinely differ in
    home effect while shrinking noisy estimates toward the population mean.
    """

    def __init__(self, loader: FootballDataLoader):
        self.data = loader.data
        self.teams = loader.teams
        self.n_teams = loader.n_teams
        self.n_games = loader.n_games
        self.team_to_idx = loader.team_to_idx
        self.idx_to_team = loader.idx_to_team
        self.model = None
        self.trace = None

    def build_team_home_model(self):
        home_team_idx = self.data["home_team_idx"].values
        away_team_idx = self.data["away_team_idx"].values
        y1_data = self.data["y1"].values
        y2_data = self.data["y2"].values

        with pm.Model() as model:
            # Hierarchical prior: team-specific home advantages
            mu_home = pm.Normal("mu_home", mu=0, tau=0.0001)
            tau_home = pm.Gamma("tau_home", alpha=0.01, beta=0.01)
            home_advantage = pm.Normal(
                "home_advantage", mu=mu_home, tau=tau_home, shape=self.n_teams
            )

            mu_att = pm.Normal("mu_att", mu=0, tau=0.0001)
            mu_def = pm.Normal("mu_def", mu=0, tau=0.0001)
            tau_att = pm.Gamma("tau_att", alpha=0.01, beta=0.01)
            tau_def = pm.Gamma("tau_def", alpha=0.01, beta=0.01)

            att_star = pm.Normal("att_star", mu=mu_att, tau=tau_att, shape=self.n_teams)
            def_star = pm.Normal("def_star", mu=mu_def, tau=tau_def, shape=self.n_teams)

            att = pm.Deterministic("att", att_star - pt.mean(att_star))
            def_ = pm.Deterministic("def", def_star - pt.mean(def_star))

            # home_advantage is now indexed by home team, not a scalar
            log_theta_g1 = home_advantage[home_team_idx] + att[home_team_idx] + def_[away_team_idx]
            log_theta_g2 = att[away_team_idx] + def_[home_team_idx]

            theta_g1 = pm.Deterministic("theta_g1", pt.exp(log_theta_g1))
            theta_g2 = pm.Deterministic("theta_g2", pt.exp(log_theta_g2))

            pm.Poisson("y1", mu=theta_g1, observed=y1_data)
            pm.Poisson("y2", mu=theta_g2, observed=y2_data)

        self.model = model
        return model

    def fit_team_home_model(self, draws=2000, tune=2000, chains=4, cores=1, random_seed=42):
        if self.model is None:
            self.build_team_home_model()

        with self.model:
            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                random_seed=random_seed,
                return_inferencedata=True,
                target_accept=0.95,
            )
            self.trace.extend(pm.sample_posterior_predictive(self.trace))

        return self.trace

    def analyze_team_home_results(self) -> pd.DataFrame:
        """
        Return a DataFrame ranking teams by posterior mean home advantage,
        with 95 % credible intervals and the implied goal-rate multiplier.
        """
        if self.trace is None:
            raise RuntimeError("Call fit_team_home_model() before analyze_team_home_results().")

        mu_home_mean = float(self.trace.posterior["mu_home"].mean())
        mu_home_ci = (
            float(self.trace.posterior["mu_home"].quantile(0.025)),
            float(self.trace.posterior["mu_home"].quantile(0.975)),
        )
        sigma_home = float(1.0 / np.sqrt(float(self.trace.posterior["tau_home"].mean())))

        print("Population-level home advantage:")
        print(f"  mu_home    : {mu_home_mean:.4f}  95% CI [{mu_home_ci[0]:.4f}, {mu_home_ci[1]:.4f}]")
        print(f"  sigma_home : {sigma_home:.4f}")

        home_samples = self.trace.posterior["home_advantage"]
        home_means = home_samples.mean(dim=["chain", "draw"]).values

        rows = []
        for i, team in enumerate(self.teams):
            flat = home_samples.values[:, :, i].flatten()
            rows.append(
                {
                    "team": team,
                    "home_advantage": home_means[i],
                    "home_multiplier": float(np.exp(home_means[i])),
                    "ci_low": float(np.quantile(flat, 0.025)),
                    "ci_high": float(np.quantile(flat, 0.975)),
                }
            )

        return pd.DataFrame(rows).sort_values("home_advantage", ascending=False).reset_index(drop=True)
