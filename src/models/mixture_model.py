import numpy as np
import pymc as pm
import pytensor.tensor as pt

from src.data_loader import FootballDataLoader


class MixtureModel:
    """
    Section 4 of Baio & Blangiardo (2010): three-group Dirichlet / Categorical
    mixture with StudentT(nu=4) team effects.

    Groups: bottom (poor attack, poor defence), mid-table, top (good attack,
    good defence).  Each team is assigned to a group via a latent Categorical
    variable, and team effects are drawn from the group-specific StudentT.

    Note on convergence
    -------------------
    Discrete latent variables (grp_att, grp_def) can slow MCMC mixing.
    Increasing tune (>= 2000) and running more chains is recommended.  R-hat
    values slightly above 1.1 for group-membership parameters are common and
    do not necessarily indicate a problem with the continuous parameters.
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

    def build_mixture_model(self):
        home_team_idx = self.data["home_team_idx"].values
        away_team_idx = self.data["away_team_idx"].values
        y1_data = self.data["y1"].values
        y2_data = self.data["y2"].values

        with pm.Model() as model:
            home_advantage = pm.Normal("home_advantage", mu=0, tau=0.0001)

            # Group membership probabilities for each team (3 groups)
            p_att = pm.Dirichlet("p_att", a=np.ones(3), shape=(self.n_teams, 3))
            p_def = pm.Dirichlet("p_def", a=np.ones(3), shape=(self.n_teams, 3))
            grp_att = pm.Categorical("grp_att", p=p_att, shape=self.n_teams)
            grp_def = pm.Categorical("grp_def", p=p_def, shape=self.n_teams)

            # Group 1 — bottom teams: poor attack, poor defence
            mu_att_1 = pm.TruncatedNormal("mu_att_1", mu=0, tau=0.001, lower=-3, upper=0)
            mu_def_1 = pm.TruncatedNormal("mu_def_1", mu=0, tau=0.001, lower=0, upper=3)
            tau_att_1 = pm.Gamma("tau_att_1", alpha=0.01, beta=0.01)
            tau_def_1 = pm.Gamma("tau_def_1", alpha=0.01, beta=0.01)

            # Group 2 — mid-table teams
            tau_att_2 = pm.Gamma("tau_att_2", alpha=0.01, beta=0.01)
            tau_def_2 = pm.Gamma("tau_def_2", alpha=0.01, beta=0.01)
            mu_att_2 = pm.Normal("mu_att_2", mu=0, tau=tau_att_2)
            mu_def_2 = pm.Normal("mu_def_2", mu=0, tau=tau_def_2)

            # Group 3 — top teams: good attack, good defence
            mu_att_3 = pm.TruncatedNormal("mu_att_3", mu=0, tau=0.001, lower=0, upper=3)
            mu_def_3 = pm.TruncatedNormal("mu_def_3", mu=0, tau=0.001, lower=-3, upper=0)
            tau_att_3 = pm.Gamma("tau_att_3", alpha=0.01, beta=0.01)
            tau_def_3 = pm.Gamma("tau_def_3", alpha=0.01, beta=0.01)

            mu_att_groups = pt.stack([mu_att_1, mu_att_2, mu_att_3])
            mu_def_groups = pt.stack([mu_def_1, mu_def_2, mu_def_3])
            tau_att_groups = pt.stack([tau_att_1, tau_att_2, tau_att_3])
            tau_def_groups = pt.stack([tau_def_1, tau_def_2, tau_def_3])

            # Team effects: StudentT(nu=4) drawn from the assigned group
            att_effects, def_effects = [], []
            for t in range(self.n_teams):
                att_mu_t = pt.switch(
                    pt.eq(grp_att[t], 0), mu_att_groups[0],
                    pt.switch(pt.eq(grp_att[t], 1), mu_att_groups[1], mu_att_groups[2]),
                )
                att_tau_t = pt.switch(
                    pt.eq(grp_att[t], 0), tau_att_groups[0],
                    pt.switch(pt.eq(grp_att[t], 1), tau_att_groups[1], tau_att_groups[2]),
                )
                def_mu_t = pt.switch(
                    pt.eq(grp_def[t], 0), mu_def_groups[0],
                    pt.switch(pt.eq(grp_def[t], 1), mu_def_groups[1], mu_def_groups[2]),
                )
                def_tau_t = pt.switch(
                    pt.eq(grp_def[t], 0), tau_def_groups[0],
                    pt.switch(pt.eq(grp_def[t], 1), tau_def_groups[1], tau_def_groups[2]),
                )
                att_effects.append(pm.StudentT(f"att_raw_{t}", nu=4, mu=att_mu_t, lam=att_tau_t))
                def_effects.append(pm.StudentT(f"def_raw_{t}", nu=4, mu=def_mu_t, lam=def_tau_t))

            att = pt.stack(att_effects)
            def_ = pt.stack(def_effects)

            att_centered = pm.Deterministic("att_centered", att - pt.mean(att))
            def_centered = pm.Deterministic("def_centered", def_ - pt.mean(def_))

            log_theta_g1 = home_advantage + att_centered[home_team_idx] + def_centered[away_team_idx]
            log_theta_g2 = att_centered[away_team_idx] + def_centered[home_team_idx]

            theta_g1 = pm.Deterministic("theta_g1", pt.exp(log_theta_g1))
            theta_g2 = pm.Deterministic("theta_g2", pt.exp(log_theta_g2))

            pm.Poisson("y1", mu=theta_g1, observed=y1_data)
            pm.Poisson("y2", mu=theta_g2, observed=y2_data)

        self.model = model
        return model

    def fit_mixture_model(self, draws=2000, tune=2000, chains=4, cores=1, random_seed=42):
        if self.model is None:
            self.build_mixture_model()

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
