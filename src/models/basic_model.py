import numpy as np
import pymc as pm
import pytensor.tensor as pt

from src.data_loader import FootballDataLoader


class BasicModel:
    """
    Section 2 of Baio & Blangiardo (2010): single home_advantage scalar with
    hierarchical attack/defence effects centred via sum-to-zero constraint.

    Note on convergence
    -------------------
    R-hat ~2.1 has been observed for mu_att and mu_def in some runs.  If this
    occurs, try: increasing tune (>= 2000), using more chains, or switching to
    a non-centred parameterisation for att_star / def_star.
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

    def build_basic_model(self):
        home_team_idx = self.data["home_team_idx"].values
        away_team_idx = self.data["away_team_idx"].values
        y1_data = self.data["y1"].values
        y2_data = self.data["y2"].values

        with pm.Model() as model:
            home_advantage = pm.Normal("home_advantage", mu=0, tau=0.0001)

            mu_att = pm.Normal("mu_att", mu=0, tau=0.0001)
            mu_def = pm.Normal("mu_def", mu=0, tau=0.0001)
            tau_att = pm.Gamma("tau_att", alpha=0.01, beta=0.01)
            tau_def = pm.Gamma("tau_def", alpha=0.01, beta=0.01)

            att_star = pm.Normal("att_star", mu=mu_att, tau=tau_att, shape=self.n_teams)
            def_star = pm.Normal("def_star", mu=mu_def, tau=tau_def, shape=self.n_teams)

            # Sum-to-zero centring
            att = pm.Deterministic("att", att_star - pt.mean(att_star))
            def_ = pm.Deterministic("def", def_star - pt.mean(def_star))

            # log(theta_g1) = home + att[h(g)] + def[a(g)]
            # log(theta_g2) =        att[a(g)] + def[h(g)]
            log_theta_g1 = home_advantage + att[home_team_idx] + def_[away_team_idx]
            log_theta_g2 = att[away_team_idx] + def_[home_team_idx]

            theta_g1 = pm.Deterministic("theta_g1", pt.exp(log_theta_g1))
            theta_g2 = pm.Deterministic("theta_g2", pt.exp(log_theta_g2))

            pm.Poisson("y1", mu=theta_g1, observed=y1_data)
            pm.Poisson("y2", mu=theta_g2, observed=y2_data)

        self.model = model
        return model

    def fit_basic_model(self, draws=2000, tune=2000, chains=4, cores=1, random_seed=42):
        if self.model is None:
            self.build_basic_model()

        with self.model:
            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                random_seed=random_seed,
                return_inferencedata=True,
                target_accept=0.97,
            )
            self.trace.extend(pm.sample_posterior_predictive(self.trace))

        return self.trace
