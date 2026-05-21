# OpenADMET PXR TabPFN

This project uses:

- Morgan fingerprints
- RDKit 2D descriptors
- TabPFN regressor
- a simple train/validation split
- full-train prediction for the final test set

## Files

Expected project layout:

```text
project/
  main.py
  data/
    openadmet_train_clean.csv
    openadmet_test_clean.csv
  outputs/
  submissions/
```

## Input data

The train CSV should contain:

```text
Molecule Name
SMILES
pEC50
```

The test CSV should contain:

```text
Molecule Name
SMILES
```

If `canonical_smiles` already exists, the script can use it.  
Otherwise, it will generate canonical SMILES from `SMILES`.

## Install

```bash
pip install -U tabpfn-client rdkit pandas numpy scipy scikit-learn
```

## Set TabPFN API token

PowerShell:

```powershell
$env:TABPFN_CLIENT_TOKEN="your_token_here"
```

CMD:

```cmd
set TABPFN_CLIENT_TOKEN=your_token_here
```

Do not put the token directly into the Python file.

## Run

```bash
python main.py
```

Optional examples:

```bash
python main.py --n_bits 512
python main.py --n_bits 1024
python main.py --val_size 0.2
python main.py --seed 42
```

If the script supports thinking mode:

```bash
python main.py --thinking --thinking_timeout_s 600
```

## Output files

After running, the script saves:

```text
outputs/validation_metrics_tabpfn.csv
outputs/validation_predictions_tabpfn.csv
outputs/test_predictions_tabpfn.csv
submissions/submission_tabpfn.csv
```

The main submission file is:

```text
submissions/submission_tabpfn.csv
```

It contains:

```text
SMILES
Molecule Name
pEC50
```

## Method

The script first reads the train and test CSV files.

Then it builds molecular features:

```text
SMILES
  -> Morgan fingerprint
  -> RDKit 2D descriptors
  -> final feature table
```

It trains TabPFN on the training split and checks validation metrics.

Finally, it trains again on all training data and predicts the test set.

## Metrics

The validation output includes common regression metrics:

```text
MAE
RMSE
R2
RAE
Spearman
Kendall
```

## Notes

This is a simple baseline.

It does not use external training data, GNN models, mol2vec, or complicated ensembling.

The goal is to keep the pipeline short and easy to modify.
