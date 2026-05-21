# PXR pEC50 Prediction with TabPFN

A compact TabPFN-based workflow for the OpenADMET PXR pEC50 prediction task.

This project takes molecular SMILES, converts them into simple molecular features, trains a TabPFN regression model, evaluates it on a local validation split, then retrains on the full training set to create a submission file.

---

## 1. What this script does

The pipeline is intentionally direct:

```text
train/test CSV
    ↓
SMILES cleanup
    ↓
Morgan fingerprint + RDKit 2D descriptors
    ↓
feature cleaning / imputation / scaling
    ↓
TabPFNRegressor
    ↓
validation report + final submission
```

The goal is not to build a complicated ensemble. The goal is to test whether a pure TabPFN table-model route can give a strong and reproducible baseline from molecular features.

---

## 2. Project layout

Expected folder structure:

```text
project_root/
  main_tabpfn.py
  README.md
  data/
    openadmet_train_clean.csv
    openadmet_test_clean.csv
  outputs/
  submissions/
```

The `outputs/` and `submissions/` folders are created automatically if they do not already exist.

---

## 3. Input files

### Training CSV

Default path:

```text
data/openadmet_train_clean.csv
```

Required columns:

| Column | Meaning |
|---|---|
| `Molecule Name` | molecule identifier |
| `SMILES` | original SMILES string |
| `canonical_smiles` | canonicalized SMILES string, if already available |
| `pEC50` | regression target |

### Test CSV

Default path:

```text
data/openadmet_test_clean.csv
```

Required columns:

| Column | Meaning |
|---|---|
| `Molecule Name` | molecule identifier |
| `SMILES` | original SMILES string |
| `canonical_smiles` | canonicalized SMILES string, if already available |

When `canonical_smiles` is missing, the script tries to create it from `SMILES` with RDKit.

---

## 4. Installation

Use Python 3.10 or 3.11 if possible.

```bash
pip install -U tabpfn-client rdkit pandas numpy scipy scikit-learn
```

If RDKit installation fails through pip, install RDKit using the package method that works best for your environment, then rerun the command without `rdkit`.

---

## 5. API token setup

The script should read the TabPFN API token from an environment variable.

### PowerShell

```powershell
$env:TABPFN_CLIENT_TOKEN="your_token_here"
```

### CMD

```cmd
set TABPFN_CLIENT_TOKEN=your_token_here
```

### Linux / macOS shell

```bash
export TABPFN_CLIENT_TOKEN="your_token_here"
```

Do not hard-code the token into `main_tabpfn.py`. Do not commit the token to GitHub.

---

## 6. Basic run

From the project root:

```bash
python main_tabpfn.py
```

This uses the default files:

```text
data/openadmet_train_clean.csv
data/openadmet_test_clean.csv
```

---

## 7. Custom run examples

Use a different train/test file:

```bash
python main_tabpfn.py --train_csv data/my_train.csv --test_csv data/my_test.csv
```

Try fewer Morgan bits:

```bash
python main_tabpfn.py --n_bits 512
```

Try the default Morgan size used in the script:

```bash
python main_tabpfn.py --n_bits 1024
```

Try a larger fingerprint:

```bash
python main_tabpfn.py --n_bits 2048
```

Change the validation split:

```bash
python main_tabpfn.py --val_size 0.15
```

Change the random seed:

```bash
python main_tabpfn.py --seed 123
```

Enable TabPFN thinking mode, if supported by your installed client and account:

```bash
python main_tabpfn.py --thinking --thinking_timeout_s 600
```

---

## 8. Main parameters

| Parameter | Default | Meaning |
|---|---:|---|
| `--train_csv` | `data/openadmet_train_clean.csv` | training file |
| `--test_csv` | `data/openadmet_test_clean.csv` | test file |
| `--seed` | `42` | random seed |
| `--val_size` | `0.2` | local validation ratio |
| `--n_bits` | `1024` | Morgan fingerprint length |
| `--radius` | `2` | Morgan fingerprint radius |
| `--thinking` | off | optional TabPFN thinking mode |
| `--thinking_timeout_s` | `600` | timeout for thinking mode |

---

## 9. Output files

After running, the script writes:

```text
outputs/validation_metrics_tabpfn.csv
outputs/validation_predictions_tabpfn.csv
outputs/test_predictions_tabpfn.csv
submissions/submission_tabpfn.csv
```

### `validation_metrics_tabpfn.csv`

Local validation metrics.

Expected columns include:

| Metric | Meaning |
|---|---|
| `MAE` | mean absolute error |
| `RMSE` | root mean squared error |
| `R2` | coefficient of determination |
| `RAE` | relative absolute error |
| `Spearman` | rank correlation |
| `Kendall` | rank correlation |

### `validation_predictions_tabpfn.csv`

Per-molecule validation predictions.

Useful for checking:

- which molecules are badly predicted
- whether errors cluster by scaffold
- whether the model is too conservative
- whether high or low pEC50 values are compressed

### `test_predictions_tabpfn.csv`

Detailed prediction file for the test set.

This is mainly for debugging and analysis.

### `submission_tabpfn.csv`

Final submission file.

Expected format:

```text
SMILES,Molecule Name,pEC50
```

This is the file to submit.

---

## 10. How to interpret the validation result

A good local validation score does not always guarantee a good leaderboard score. This script uses a simple random validation split by default, so it mainly answers:

> Can TabPFN learn a reasonable mapping from the current molecular features to pEC50?

It does not fully answer:

> Will the model generalize to new analog series or hidden test scaffolds?

For more trustworthy testing, run the script with several seeds and compare the stability of the metrics.

Example:

```bash
python main_tabpfn.py --seed 1
python main_tabpfn.py --seed 2
python main_tabpfn.py --seed 3
python main_tabpfn.py --seed 4
python main_tabpfn.py --seed 5
```

If validation metrics swing heavily between seeds, the split is not stable enough to judge small improvements.

---

## 11. Notes on features

The feature matrix is built from two parts:

### Morgan fingerprint

A binary molecular fingerprint similar to ECFP4 when `radius=2`.

It captures local substructure patterns.

### RDKit 2D descriptors

A set of computed molecular descriptors from RDKit.

They provide continuous chemistry-related properties such as molecular weight, polarity, ring counts, and other descriptor values.

Before training, the script removes constant descriptor columns and handles invalid numeric values.

---

## 12. Practical tuning suggestions

Recommended first tests:

```bash
python main_tabpfn.py --n_bits 512 --seed 42
python main_tabpfn.py --n_bits 1024 --seed 42
python main_tabpfn.py --n_bits 2048 --seed 42
```

Then test seed stability on the best-looking setting:

```bash
python main_tabpfn.py --n_bits 1024 --seed 1
python main_tabpfn.py --n_bits 1024 --seed 2
python main_tabpfn.py --n_bits 1024 --seed 3
```

For this task, a slightly smaller feature dimension may work better than a very large one because TabPFN is operating as a tabular learner, not as a molecular graph model.

---

## 13. Troubleshooting

### `TABPFN_CLIENT_TOKEN is missing`

Set the token before running:

```powershell
$env:TABPFN_CLIENT_TOKEN="your_token_here"
```

Then run the script in the same terminal window.

### `Cannot find SMILES column`

Check that the CSV contains at least one of:

```text
SMILES
canonical_smiles
```

### RDKit cannot parse some molecules

The script skips invalid molecules during feature generation. If too many are skipped, inspect the SMILES column for formatting issues.

### API or network error

The TabPFN client needs network access. Check:

- token is valid
- internet connection is available
- package version is current
- request did not time out

### Validation works but submission has the wrong number of rows

Check whether invalid test molecules were removed during feature generation. The test file should contain valid SMILES for all molecules expected by the competition.

---

## 14. Reproducibility checklist

Before submitting, record:

```text
script name: main_tabpfn.py
train file: data/openadmet_train_clean.csv
test file: data/openadmet_test_clean.csv
seed: 42
n_bits: 1024
radius: 2
val_size: 0.2
output: submissions/submission_tabpfn.csv
```

Also save:

```text
outputs/validation_metrics_tabpfn.csv
outputs/validation_predictions_tabpfn.csv
outputs/test_predictions_tabpfn.csv
```

These files make it easier to compare later experiments.

---

## 15. One-line summary

This is a minimal TabPFN regression pipeline for PXR pEC50 prediction: molecular features in, local validation metrics out, final submission generated from the full training set.
