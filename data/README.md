# Data

Everything under `data/raw/` and `data/processed/` is created by the scripts and is not committed.

## 1. Scripted downloads

```bash
python scripts/download_data.py
```

| Folder in `data/raw/` | Source | Licence |
| --- | --- | --- |
| `eloundou/` | Eloundou et al. (2024), github.com/openai/GPTs-are-GPTs | MIT |
| `microsoft/` | Tomlinson et al. (2025), github.com/microsoft/working-with-ai | CC BY 4.0 |
| `aioe/` | Felten, Raj & Seamans (2021), github.com/AIOE-Data/AIOE | see repo |
| `indeed/` | Indeed Hiring Lab job postings tracker and AI tracker | CC BY 4.0 |
| `crosswalk/` | BLS SOC 2010 → 2018 crosswalk (needed for AIOE) | public domain |
| `oews/<year>/` | BLS OEWS national, May 2022–2025 | public domain |
| `onet/` | O*NET database (text version) | CC BY 4.0 |
| `anthropic/` | Anthropic Economic Index, huggingface.co/datasets/Anthropic/EconomicIndex | CC BY |

Every file is logged in `data/raw/MANIFEST.csv` with its URL and download date.

**BLS blocks scripts without a contact.** Set `USER_AGENT` in `scripts/download_data.py`
to a string with your e-mail. If a BLS or O*NET download still fails, download the
file by hand and put it in the folder shown above.

## 2. Manual downloads (free accounts)

### Pew Research Center, American Trends Panel Wave 157
1. Create an account at pewresearch.org and open
   https://www.pewresearch.org/dataset/american-trends-panel-wave-157/
2. Download the SPSS file and put the `.sav` in `data/raw/pew/`.
3. Open the codebook and fill in:
   - `PEW_COLUMNS` in `labor_sim/survey/personas.py` (age, gender, education, income, occupation, weight)
   - the exact wording, options and `pew_var` of each item in `labor_sim/survey/questions.yaml`,
     then set `verified: true`.

### IPUMS CPS
1. Register at https://cps.ipums.org/cps/
2. Select the **basic monthly** samples January 2021 – December 2025.
3. Variables: `YEAR, MONTH, CPSIDP, AGE, EMPSTAT, OCC, WTFINL`.
4. Download as CSV and save as `data/raw/ipums/cps.csv` (or `cps.csv.gz`).
5. CPS uses Census occupation codes. Save the Census → SOC list as
   `data/raw/crosswalk/census_occ_to_soc.csv` with columns `census_occ,soc`
   (from the Census Bureau's occupation code list for the 2018 codes).

## 3. Build the processed tables

```bash
python scripts/build_occupations.py   # -> data/processed/occupations.csv (+ coverage report)
python scripts/build_targets.py       # -> indeed_targets.csv, ipums_targets.csv
```

`build_occupations.py` works with whatever is present. Until OEWS May 2022–2025 are
downloaded it uses the May 2021 file from the Eloundou repo, which is fine for
development but **not** for results.

`data/processed/occupations.csv` is committed on purpose: it was built from OEWS May 2022-2025 on a laptop
(BLS blocks cloud machines such as Colab), and the Colab setup reuses it instead of rebuilding.
Re-commit it (`git add -f data/processed/occupations.csv`) whenever `build_occupations.py` is re-run with new data.
