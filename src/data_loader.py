import os
import pandas as pd
import numpy as np


class FootballDataLoader:
    """
    Unified data loader for all four Italian Serie A datasets.

    Normalises column name differences across seasons so the rest of the
    codebase always sees: home_team, away_team, home_team_idx, away_team_idx,
    y1, y2 (plus any extra covariate columns that are present).

    Parameters
    ----------
    file_name : str
        Filename (e.g. "italy_serie-a_1991-1992.xlsx") or an absolute path.
        When relative, it is resolved against data_dir.
    season : str
        One of "1991-92", "2007-08", "2022-23".  Used for context / validation.
    data_dir : str
        Base directory that contains the .xlsx files.  Ignored when file_name
        is an absolute path.  Default: "data/".
    """

    def __init__(self, file_name: str, season: str, data_dir: str = "data/"):
        self.season = season
        self.data_dir = data_dir

        if os.path.isabs(file_name):
            self.file_path = file_name
        else:
            self.file_path = os.path.join(data_dir, file_name)

        self.data = self._load_and_normalise()
        self._build_team_mappings()

    # ------------------------------------------------------------------
    # Public attributes set by _build_team_mappings:
    #   self.teams        – sorted list of team name strings
    #   self.n_teams      – int
    #   self.n_games      – int
    #   self.team_to_idx  – dict[str, int]
    #   self.idx_to_team  – dict[int, str]
    # ------------------------------------------------------------------

    def _load_and_normalise(self) -> pd.DataFrame:
        df = pd.read_excel(self.file_path)
        df.columns = df.columns.str.strip()
        df = self._normalise_columns(df)
        return df

    def _normalise_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Map every known column name variant to a canonical name.

        Known quirks handled here:
        * 1991-92:  'home team' / 'away team'  (spaces, no underscore)
        * 2022-23:  'hometeam_name+'            (trailing '+' typo)
        * 2022-23:  'attendance'                (renamed to 'average_attendance')
        """
        rename_map = {}

        for col in df.columns:
            # Strip trailing '+' then whitespace to get the base name
            base = col.rstrip("+").strip()

            if base in ("home team", "hometeam_name"):
                rename_map[col] = "home_team"
            elif base in ("away team", "awayteam_name"):
                rename_map[col] = "away_team"

        # Unify attendance column name (2022-23 uses 'attendance', others use
        # 'average_attendance'; normalise so covariate models see one name)
        if "attendance" in df.columns and "average_attendance" not in df.columns:
            rename_map["attendance"] = "average_attendance"

        return df.rename(columns=rename_map)

    def _build_team_mappings(self) -> None:
        all_teams = pd.concat(
            [self.data["home_team"], self.data["away_team"]]
        ).unique()

        self.teams = sorted(all_teams)
        self.n_teams = len(self.teams)
        self.n_games = len(self.data)
        self.team_to_idx = {team: idx for idx, team in enumerate(self.teams)}
        self.idx_to_team = {idx: team for idx, team in enumerate(self.teams)}

        self.data["home_team_idx"] = self.data["home_team"].map(self.team_to_idx)
        self.data["away_team_idx"] = self.data["away_team"].map(self.team_to_idx)

        missing_home = self.data["home_team_idx"].isna().sum()
        missing_away = self.data["away_team_idx"].isna().sum()
        if missing_home or missing_away:
            raise ValueError(
                f"Team mapping failed: {missing_home} home / {missing_away} away "
                "entries could not be mapped to an index."
            )
