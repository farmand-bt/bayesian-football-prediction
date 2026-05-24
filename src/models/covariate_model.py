import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from src.data_loader import FootballDataLoader


class CovariateModel:
    """
    Contribution 2: stadium / distance / temporal covariate model.

    Merges the best of TrulyFixedEnhancedModelAddon and FullyFixedEnhancedModel
    (the latter is the primary source).  Key design decisions:

    * Single composite stadium_quality index — eliminates multicollinearity
      between capacity, utilisation, and absolute attendance.
      Formula: 0.6 * utilisation^1.5 + 0.25 * log(capacity/25000)
                                      + 0.15 * log(attendance/15000)

    * ALL covariates z-score standardised before entering the model (including
      binary weekday indicators), so all beta coefficients are directly
      comparable as "effect per 1 SD change".

    * Team-level:  home_advantage[j] = home_base + beta_stadium * sq_std[j]
    * Game-level:  beta_distance, beta_friday, beta_saturday, beta_sunday,
                   beta_season adjustments to the dynamic home advantage.
    * Away team:   beta_travel_fatigue * distance_std reduces away scoring.
    """

    _W_UTIL = 0.60
    _W_CAP  = 0.25
    _W_ATT  = 0.15

    def __init__(self, loader: FootballDataLoader):
        self.data = loader.data
        self.teams = loader.teams
        self.n_teams = loader.n_teams
        self.n_games = loader.n_games
        self.team_to_idx = loader.team_to_idx
        self.idx_to_team = loader.idx_to_team

        self._raw: dict = {}
        self._std: dict = {}

        self._prepare_single_stadium_metric()
        self._prepare_distance_covariates()
        self._prepare_temporal_covariates()
        self._standardize_all_covariates()

        self.model = None
        self.trace = None

    # ------------------------------------------------------------------
    # Private covariate preparation
    # ------------------------------------------------------------------

    def _prepare_single_stadium_metric(self) -> None:
        df = self.data
        team_data = {}

        for idx, team in enumerate(self.teams):
            home_games = df[df["home_team"] == team]

            if len(home_games) > 0:
                capacity = self._first_valid(home_games, "stadium_capacity", 40_000.0)
                attendance = self._mean_valid(
                    home_games, ["average_attendance", "attendance"], capacity * 0.70
                )
                utilization = self._mean_valid(
                    home_games, ["capacity_utilization"],
                    min(attendance / capacity, 1.0),
                )
            else:
                capacity, attendance, utilization = 40_000.0, 28_000.0, 0.70

            utilization = min(float(utilization), 1.0)
            quality = (
                self._W_UTIL * utilization ** 1.5
                + self._W_CAP  * np.log(max(float(capacity), 1) / 25_000)
                + self._W_ATT  * np.log(max(float(attendance), 1) / 15_000)
            )
            team_data[idx] = {
                "team": team,
                "capacity": float(capacity),
                "attendance": float(attendance),
                "utilization": float(utilization),
                "stadium_quality": quality,
            }

        self._raw["stadium"] = team_data

    def _prepare_distance_covariates(self) -> None:
        df = self.data
        dist_cols = [c for c in df.columns if "distance" in c.lower()]

        if dist_cols:
            distances = df[dist_cols[0]].values.astype(float)
            median = float(np.nanmedian(distances))
            distances = np.where(np.isnan(distances), median, distances)
        else:
            distances = np.full(self.n_games, 300.0)

        self._raw["distance"] = distances

    def _prepare_temporal_covariates(self) -> None:
        df = self.data
        date_cols    = [c for c in df.columns if "date"    in c.lower()]
        weekday_cols = [c for c in df.columns if "weekday" in c.lower()]
        temporal: dict = {}
        parsed = False

        if date_cols:
            try:
                dates = pd.to_datetime(df[date_cols[0]], dayfirst=True)
                temporal["is_friday"]    = (dates.dt.dayofweek == 4).astype(int).values
                temporal["is_saturday"]  = (dates.dt.dayofweek == 5).astype(int).values
                temporal["is_sunday"]    = (dates.dt.dayofweek == 6).astype(int).values
                months = dates.dt.month.values
                temporal["season_phase"] = np.where(
                    (months >= 8) | (months <= 10), 0,
                    np.where(months <= 2, 1, 2),
                )
                parsed = True
            except Exception:
                pass

        if not parsed and weekday_cols:
            try:
                _wmap = {
                    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
                    "friday": 4, "saturday": 5, "sunday": 6,
                    "mon": 0, "tue": 1, "wed": 2, "thu": 3,
                    "fri": 4, "sat": 5, "sun": 6,
                }
                raw = df[weekday_cols[0]].values
                nums = (
                    np.array([_wmap.get(str(d).lower(), 0) for d in raw])
                    if isinstance(raw[0], str)
                    else raw.astype(int)
                )
                temporal["is_friday"]   = (nums == 4).astype(int)
                temporal["is_saturday"] = (nums == 5).astype(int)
                temporal["is_sunday"]   = (nums == 6).astype(int)
                np.random.seed(42)
                temporal["season_phase"] = np.random.choice([0, 1, 2], self.n_games)
                parsed = True
            except Exception:
                pass

        if not parsed:
            np.random.seed(42)
            days = np.random.choice(
                range(7), self.n_games, p=[0.05, 0.05, 0.05, 0.10, 0.15, 0.30, 0.30]
            )
            temporal["is_friday"]    = (days == 4).astype(int)
            temporal["is_saturday"]  = (days == 5).astype(int)
            temporal["is_sunday"]    = (days == 6).astype(int)
            temporal["season_phase"] = np.random.choice([0, 1, 2], self.n_games)

        self._raw["temporal"] = temporal

    def _standardize_all_covariates(self) -> None:
        """Z-score every covariate — including binary weekday indicators."""

        def _z(values):
            arr = np.asarray(values, dtype=float)
            mu, sigma = arr.mean(), arr.std()
            return ((arr - mu) / sigma if sigma > 0 else np.zeros_like(arr)), mu, sigma

        qualities = [self._raw["stadium"][i]["stadium_quality"] for i in range(self.n_teams)]
        vals, mu, sigma = _z(qualities)
        self._std["stadium_quality"] = {"values": vals, "mean": mu, "std": sigma}

        vals, mu, sigma = _z(self._raw["distance"])
        self._std["distance"] = {"values": vals, "mean": mu, "std": sigma}

        for key in ("is_friday", "is_saturday", "is_sunday", "season_phase"):
            vals, mu, sigma = _z(self._raw["temporal"][key])
            self._std[key] = {"values": vals, "mean": mu, "std": sigma}

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _first_valid(df: pd.DataFrame, col: str, fallback: float) -> float:
        if col in df.columns:
            v = df[col].iloc[0]
            return float(v) if pd.notna(v) else fallback
        return fallback

    @staticmethod
    def _mean_valid(df: pd.DataFrame, cols: list, fallback: float) -> float:
        for col in cols:
            if col in df.columns:
                v = df[col].mean()
                return float(v) if pd.notna(v) else fallback
        return fallback

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def build_covariate_model(self):
        home_team_idx = self.data["home_team_idx"].values
        away_team_idx = self.data["away_team_idx"].values
        y1_data = self.data["y1"].values
        y2_data = self.data["y2"].values

        sq_std  = np.array(self._std["stadium_quality"]["values"])
        d_std   = self._std["distance"]["values"]
        fri_std = self._std["is_friday"]["values"]
        sat_std = self._std["is_saturday"]["values"]
        sun_std = self._std["is_sunday"]["values"]
        sea_std = self._std["season_phase"]["values"]

        with pm.Model() as model:
            # ---- team-level home advantage (stadium baseline) ----
            home_base    = pm.Normal("home_base",    mu=0.3, sigma=0.1)
            beta_stadium = pm.Normal("beta_stadium", mu=0,   sigma=0.1)
            team_home_baseline = pm.Deterministic(
                "team_home_baseline",
                home_base + beta_stadium * sq_std,
            )

            # ---- game-level contextual adjustments (all standardised) ----
            beta_distance = pm.Normal("beta_distance", mu=0, sigma=0.05)
            beta_friday   = pm.Normal("beta_friday",   mu=0, sigma=0.05)
            beta_saturday = pm.Normal("beta_saturday", mu=0, sigma=0.05)
            beta_sunday   = pm.Normal("beta_sunday",   mu=0, sigma=0.05)
            beta_season   = pm.Normal("beta_season",   mu=0, sigma=0.05)

            game_context = pm.Deterministic(
                "game_context",
                beta_distance * d_std
                + beta_friday   * fri_std
                + beta_saturday * sat_std
                + beta_sunday   * sun_std
                + beta_season   * sea_std,
            )

            dynamic_home_advantage = pm.Deterministic(
                "dynamic_home_advantage",
                team_home_baseline[home_team_idx] + game_context,
            )

            # ---- away team travel fatigue ----
            beta_travel_fatigue = pm.Normal("beta_travel_fatigue", mu=0, sigma=0.03)

            # ---- standard hierarchical team effects ----
            mu_att = pm.Normal("mu_att", mu=0, tau=0.0001)
            mu_def = pm.Normal("mu_def", mu=0, tau=0.0001)
            tau_att = pm.Gamma("tau_att", alpha=0.01, beta=0.01)
            tau_def = pm.Gamma("tau_def", alpha=0.01, beta=0.01)

            att_star = pm.Normal("att_star", mu=mu_att, tau=tau_att, shape=self.n_teams)
            def_star = pm.Normal("def_star", mu=mu_def, tau=tau_def, shape=self.n_teams)

            att  = pm.Deterministic("att",  att_star - pt.mean(att_star))
            def_ = pm.Deterministic("def",  def_star - pt.mean(def_star))

            log_theta_g1 = (
                dynamic_home_advantage
                + att[home_team_idx]
                + def_[away_team_idx]
            )
            log_theta_g2 = (
                att[away_team_idx]
                + def_[home_team_idx]
                + beta_travel_fatigue * d_std  # longer trip → lower away scoring
            )

            theta_g1 = pm.Deterministic("theta_g1", pt.exp(log_theta_g1))
            theta_g2 = pm.Deterministic("theta_g2", pt.exp(log_theta_g2))

            pm.Poisson("y1", mu=theta_g1, observed=y1_data)
            pm.Poisson("y2", mu=theta_g2, observed=y2_data)

        self.model = model
        return model

    def fit_covariate_model(self, draws=2000, tune=2000, chains=4, cores=1, random_seed=42):
        if self.model is None:
            self.build_covariate_model()

        with self.model:
            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                random_seed=random_seed,
                return_inferencedata=True,
                target_accept=0.90,
            )
            self.trace.extend(pm.sample_posterior_predictive(self.trace))

        return self.trace

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def analyze_standardized_effects(self) -> pd.DataFrame:
        """Print and return posterior summaries for all beta coefficients."""
        if self.trace is None:
            raise RuntimeError("Call fit_covariate_model() first.")

        params = [
            ("beta_stadium",        "Stadium quality (per 1 SD)"),
            ("beta_distance",       "Distance home-advantage (per 1 SD)"),
            ("beta_friday",         "Friday games (per 1 SD)"),
            ("beta_saturday",       "Saturday games (per 1 SD)"),
            ("beta_sunday",         "Sunday games (per 1 SD)"),
            ("beta_season",         "Season phase (per 1 SD)"),
            ("beta_travel_fatigue", "Away travel fatigue (per 1 SD)"),
        ]

        rows = []
        for param, desc in params:
            if param not in self.trace.posterior.data_vars:
                continue
            flat = self.trace.posterior[param].values.flatten()
            mean  = float(np.mean(flat))
            ci_lo = float(np.percentile(flat, 2.5))
            ci_hi = float(np.percentile(flat, 97.5))
            sig   = ci_lo > 0 or ci_hi < 0
            rows.append({"parameter": param, "description": desc,
                          "mean": mean, "ci_low": ci_lo, "ci_high": ci_hi,
                          "significant": sig})

        df = pd.DataFrame(rows)
        print("\nStandardised Beta Coefficients  (effect per 1 SD change):")
        for _, r in df.iterrows():
            tag = "*" if r["significant"] else " "
            print(f"  {tag} {r['parameter']:25s}  {r['mean']:+.4f}  "
                  f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]  {r['description']}")
        return df

    def get_team_home_advantages(self) -> pd.DataFrame:
        """Return teams ranked by posterior mean home advantage baseline."""
        if self.trace is None:
            raise RuntimeError("Call fit_covariate_model() first.")

        means = self.trace.posterior["team_home_baseline"].mean(dim=["chain", "draw"]).values
        rows = []
        for i, team in enumerate(self.teams):
            rows.append({
                "team": team,
                "home_advantage": float(means[i]),
                "home_multiplier": float(np.exp(means[i])),
                "stadium_quality": self._raw["stadium"][i]["stadium_quality"],
                "capacity":        self._raw["stadium"][i]["capacity"],
                "utilization":     self._raw["stadium"][i]["utilization"],
            })
        return (pd.DataFrame(rows)
                .sort_values("home_advantage", ascending=False)
                .reset_index(drop=True))
