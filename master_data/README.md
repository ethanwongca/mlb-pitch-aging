# pitching_master.csv — Data Dictionary

Master dataset for the MLB Pitch Aging study. Each row represents one pitcher's
aggregated Statcast metrics for a single pitch type in a single season.

**Unit of observation:** pitcher × season × pitch type

## Identifiers

| Column | Type | Description |
|--------|------|-------------|
| `pitcher` | int | MLB Advanced Media (MLBAM) player ID — primary key for joining to other sources |
| `player_name` | str | Player's full name as reported by Statcast |
| `pitch_type` | str | Pitch type code (FF, SL, SI, CH, CU, FC) |
| `pitch_name` | str | Full pitch type name (e.g. 4-Seam Fastball, Slider) |
| `p_throws` | str | Pitcher handedness — R (right) or L (left) |
| `year` | int | MLB season year |

## Stuff Metrics (Season Averages)

All metrics are aggregated from pitch-level Statcast data. Minimum 150 pitches
per pitcher-season-pitchtype required for inclusion.

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `mean_velo` | float | mph | Mean release velocity |
| `std_velo` | float | mph | Standard deviation of release velocity — measures consistency |
| `mean_spin_rate` | float | rpm | Mean spin rate at release |
| `std_spin_rate` | float | rpm | Standard deviation of spin rate |
| `mean_pfx_x` | float | inches | Mean horizontal movement from catcher's perspective. Negative = arm-side, positive = glove-side |
| `std_pfx_x` | float | inches | Standard deviation of horizontal movement |
| `mean_pfx_z` | float | inches | Mean induced vertical movement (gravity removed). Positive = rise, negative = drop |
| `std_pfx_z` | float | inches | Standard deviation of vertical movement |
| `mean_ext` | float | feet | Mean release extension — how far toward home plate at release point |
| `std_ext` | float | feet | Standard deviation of release extension |
| `mean_eff_speed` | float | mph | Mean effective speed — perceived velocity accounting for extension |
| `std_eff_speed` | float | mph | Standard deviation of effective speed |
| `n_pitches` | int | count | Total pitches thrown of this type in this season — used as reliability weight |

## Spin Axis (2020+ only)

Spin axis data is unavailable prior to 2020. All values are NaN for 2015–2019.

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `mean_spin_axis` | float | degrees | Mean spin axis in the 2D X-Z plane (0–360). 180° = pure backspin (fastball), 0° = pure topspin (12-6 curveball) |
| `std_spin_axis` | float | degrees | Standard deviation of spin axis |

## Age Variables

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `age` | int | years | Pitcher age at season midpoint — computed as `year - birthYear` from Lahman People.csv |
| `age_sq` | int | years² | Age squared — `age²` |
| `age_c` | float | years | Age centered at sample mean (~28) — `age - mean(age)`. Used as primary predictor in mixed effects models to improve numerical stability and interpretability |
| `age_c_sq` | float | years² | Centered age squared — `age_c²`. Enables quadratic aging curve estimation |

## Notes
- **2020 season:** Shortened to 60 games due to COVID-19. Fewer pitcher-seasons and higher variance expected.
- **2021 spin rate:** MLB introduced sticky substance enforcement mid-2021. Population-wide spin rate drop visible in this season. Year is included as a fixed effect in all models to absorb this.
- **pfx_x handedness:** Not normalized for handedness. RHP arm-side run is negative, LHP arm-side run is positive. Account for `p_throws` when analyzing horizontal movement across handedness.
- **Source:** Statcast via pybaseball, birth years from Lahman Database (People.csv) bridged via Chadwick Bureau register.