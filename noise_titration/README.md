# Noise Titration Minimal Repro

This directory is a clean copy for reproducing the saved noise titration result and the plot in `result/Detection/noise_titration_scores.json`.

It now also includes the bdjscc reproduction assets under `ckpt/dpjscc/` so the experiment can be run entirely from this standalone repo.

## What is included

- Result JSON: `result/Detection/noise_titration_scores.json`
- Plot script: `result/Detection/plot.py`
- Reproduction script: `result/Detection/Noise_Titration.py`
- Minimal model code: `models/models_mae.py`, `models/models_all.py`, `models/model_BDJSCC.py`, `util/pos_embed.py`
- Required weights:
  - `save/resnet18_cifar10.pth`
  - `ckpt/dpjscc/bdjscc_R1_3_cleanTrue_datasetCIFAR_ratio0.01_snr20_ChannelAWGN/checkpoint-99.pth`
  - `ckpt/dpjscc/bdjscc_R1_3_cleanFalse_datasetCIFAR_ratio0.01_snr20_ChannelAWGN/checkpoint-99.pth`

## Dependencies

Create a local environment with `uv` and install the packages listed in `requirements.txt`:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r noise_titration/requirements.txt
```

The project only needs the following runtime packages for this result:

- PyTorch
- torchvision
- timm 0.3.2
- numpy
- matplotlib

## Parameters used in the saved result

The saved JSON was generated with these settings:

- `start_power = 0.1`
- `end_power = 1.0`
- `step = 0.1`
- `threshold = 0.40`
- `min_consistency = 3`
- `N_r_k = 128`
- `r` scan: `0.1 ... 1.0` with step `0.1`
- `chi` scan: `0.0 ... 1.0` with step `0.2`

## Models used

- Benign model: `bdjscc_R1_3`
  - checkpoint: `ckpt/dpjscc/bdjscc_R1_3_cleanTrue_datasetCIFAR_ratio0.01_snr20_ChannelAWGN/checkpoint-99.pth`
- Backdoor model: `bdjscc_R1_3`
  - checkpoint: `ckpt/dpjscc/bdjscc_R1_3_cleanFalse_datasetCIFAR_ratio0.01_snr20_ChannelAWGN/checkpoint-99.pth`
- bdjscc latent layout:
  - encoder output is inferred directly from the checkpointed model at runtime
  - the titration scan feeds random noise with the same latent shape into `decoder`

The oracle classifier is a ResNet-18 trained on CIFAR-10:

- `save/resnet18_cifar10.pth`

## Run

Generate the default JSON files again:

```bash
python noise_titration/Noise_Titration.py
```

Run only the BDJSCC reproduction scan with noise power from `0` to `2` in steps of `0.1`:

```bash
python noise_titration/Noise_Titration.py \
  --model-kind bdjscc \
  --r-start 0 \
  --r-end 2 \
  --r-step 0.1
```

Run only ViT-JSCC:

```bash
python noise_titration/Noise_Titration.py --model-kind vit
```

Use an explicit noise grid when needed:

```bash
python noise_titration/Noise_Titration.py \
  --model-kind bdjscc \
  --r-values 0,0.05,0.1,0.2,0.5,1,2
```

Render the plot from the saved JSON:

```bash
python result/Detection/plot.py result/Detection/bdjscc_noise_titration_scores.json
```
