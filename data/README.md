# Data

All four datasets cover Italian Serie A seasons and are open-source public
football statistics.  They are committed directly to the repository.

---

## Files

| File | Season | Shape | Description |
|------|--------|-------|-------------|
| `italy_serie-a_1991-1992.xlsx` | 1991/92 | 306 × 7 | Basic match results used to verify the model against Table 2 of Baio & Blangiardo (2010) |
| `final dataset 2007-08.xlsx` | 2007/08 | 380 × 9 | Main replication dataset |
| `final_dataset_2007-08_stadium&distance&date.xlsx` | 2007/08 | 380 × 17 | Enhanced dataset for contribution models — adds stadium, distance, and date covariates |
| `final_dataset_2022-23_stadium&distance&date.xlsx` | 2022/23 | 380 × 17 | Generalisation test dataset, same structure as the 2007/08 enhanced file |

---

## Column reference

### Common columns (all files)

| Canonical name | Raw name(s) | Meaning |
|----------------|-------------|---------|
| `home_team` | `home team` (1991/92), `hometeam_name` (others), `hometeam_name+` (2022/23 typo) | Home team name |
| `away_team` | `away team` (1991/92), `awayteam_name` (others) | Away team name |
| `y1` | `y1` | Goals scored by home team |
| `y2` | `y2` | Goals scored by away team |

The `FootballDataLoader` normalises all raw variants to the canonical names
shown above before any modelling code sees the data.

### Additional columns (enhanced datasets only)

| Column | Meaning |
|--------|---------|
| `stadium_capacity` | Maximum stadium capacity (seats) |
| `average_attendance` | Mean attendance per home match (`attendance` in 2022/23, renamed by loader) |
| `capacity_utilization` | `average_attendance / stadium_capacity` |
| `distance` | Road distance in km between the two clubs' home cities |
| `date` | Match date (day-first format) |
| `weekday` | Day-of-week string (e.g. "Sunday") |
| `matchday` | Matchday number within the season |

---

## Known quirks

1. **`hometeam_name+` typo** (2022/23 file): the home team column has a
   trailing `+` character.  The loader strips it silently.

2. **`attendance` vs `average_attendance`** (2022/23 file): the 2022/23
   enhanced file uses `attendance` where all other files use
   `average_attendance`.  The loader renames it on load.

3. **Column spacing** (1991/92 file): column names use spaces (`home team`,
   `away team`) rather than underscores.  The loader normalises these.

---

## Stadium quality index

The covariate model collapses `stadium_capacity`, `average_attendance`, and
`capacity_utilization` into a single composite index to avoid multicollinearity:

```
stadium_quality = 0.60 * utilisation^1.5
                + 0.25 * log(capacity / 25 000)
                + 0.15 * log(attendance / 15 000)
```

Higher values indicate better home-advantage potential.  The index is
z-score standardised before entering the model so the `beta_stadium`
coefficient is interpretable as "effect per 1 SD of stadium quality".
