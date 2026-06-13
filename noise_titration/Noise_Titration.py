import json
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR if (SCRIPT_DIR / "models").is_dir() else SCRIPT_DIR.parent
ROOT = PROJECT_ROOT
MODELS_DIR = ROOT / "models"
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

import models_all


def torch_load_weights(path: Path, device: torch.device):
    try:
        return torch.load(str(path), map_location=device, weights_only=True)
    except Exception:
        return torch.load(str(path), map_location=device, weights_only=False)


class OracleScorer:
    def __init__(self, checkpoint_path: str, device: torch.device, input_size: int = 224):
        self.device = device
        self.input_size = input_size
        self.resize = transforms.Resize((input_size, input_size))
        model = resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 10)
        state = torch_load_weights(Path(checkpoint_path), device)
        model.load_state_dict(state)
        self.model = model.to(device).eval()

    def calculate_pi_r_chi(self, img_batch, chi=0.7):
        confidences = self.calculate_confidences(img_batch)
        return (confidences > chi).float().mean().item()

    def calculate_confidences(self, img_batch):
        with torch.no_grad():
            if img_batch.shape[-2:] != (self.input_size, self.input_size):
                img_batch = self.resize(img_batch)
            logits = self.model(img_batch)
            probs = torch.softmax(logits, dim=1)
            y_bar_r, _ = torch.max(probs, dim=1)
            return y_bar_r


def _scaled_real_noise(shape, device, noise_power):
    return torch.randn(shape, device=device) * torch.sqrt(noise_power / 2)


def _scaled_complex_noise(shape, device, noise_power):
    real = _scaled_real_noise(shape, device, noise_power)
    imag = _scaled_real_noise(shape, device, noise_power)
    return torch.complex(real, imag)


def _decode_vit_noise(jscc_model, n_samples, noise_power):
    device = next(jscc_model.parameters()).device

    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224, device=device)
        latent, recover_size = jscc_model.forward_encoder(dummy)

    z_noise = _scaled_complex_noise((n_samples, *latent.shape[1:]), device, noise_power)
    recover_size = list(recover_size)
    recover_size[0] = n_samples

    with torch.no_grad():
        pred = jscc_model.forward_decoder(z_noise, recover_size)
        return jscc_model.unpatchify(pred)


def _decode_bdjscc_noise(jscc_model, n_samples, noise_power, input_size):
    device = next(jscc_model.parameters()).device

    with torch.no_grad():
        dummy = torch.zeros(1, 3, input_size, input_size, device=device)
        latent = jscc_model.encoder(dummy)
        if latent.dim() == 3:
            latent = latent.unsqueeze(0)
        latent_shape = latent.shape[1:]

    z_noise = _scaled_real_noise((n_samples, *latent_shape), device, noise_power)

    with torch.no_grad():
        return jscc_model.decoder(z_noise)


def run_paper_titration(jscc_model, oracle, N_r_k=128, r=0.1, chi=0.2, bdjscc_input_size=32):
    device = next(jscc_model.parameters()).device
    noise_p = torch.tensor(r, device=device, dtype=torch.float32)

    if hasattr(jscc_model, "forward_decoder"):
        x_out = _decode_vit_noise(jscc_model, N_r_k, noise_p)
    else:
        x_out = _decode_bdjscc_noise(jscc_model, N_r_k, noise_p, bdjscc_input_size)

    return oracle.calculate_pi_r_chi(x_out, chi=chi)


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    for base in (ROOT, SCRIPT_DIR):
        candidate = base / path
        if candidate.exists():
            return candidate
    return ROOT / path


def load_model(ctor_name: str, ckpt_path: Path, device: torch.device):
    model = models_all.__dict__[ctor_name]().to(device)
    state = torch_load_weights(ckpt_path, device)
    if isinstance(state, dict):
        state = state.get("model", state.get("state_dict", state))
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def run_experiment(
    ctor_name: str,
    ckpt_path: str,
    oracle: OracleScorer,
    device: torch.device,
    r_values,
    chi_values,
    n_samples: int,
    bdjscc_input_size: int,
):
    model = load_model(ctor_name, resolve_path(ckpt_path), device)
    history = []
    for r in r_values:
        noise_p = torch.tensor(float(r), device=device, dtype=torch.float32)
        if hasattr(model, "forward_decoder"):
            x_out = _decode_vit_noise(model, n_samples, noise_p)
        else:
            x_out = _decode_bdjscc_noise(model, n_samples, noise_p, bdjscc_input_size)
        confidences = oracle.calculate_confidences(x_out)

        for chi in chi_values:
            pi_val = (confidences > float(chi)).float().mean().item()
            history.append({"r": float(r), "chi": float(chi), "pi_val": float(pi_val)})

    return history


def model_kind_from_ctor(ctor_name: str) -> str:
    if ctor_name.startswith("bdjscc"):
        return "bdjscc"
    return "vit"


def parse_float_values(value: str):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def make_scan_values(values, start, end, step):
    if values:
        return parse_float_values(values)
    return np.arange(start, end + step / 2, step).round(10).tolist()


def build_experiments():
    return [
        {
            "class_key": "cleanTrue",
            "exp_name": "dpjscc/bdjscc_R1_3_cleanTrue_datasetCIFAR_ratio0.01_snr20_ChannelAWGN",
            "ctor": "bdjscc_R1_3",
            "checkpoint": "ckpt/dpjscc/bdjscc_R1_3_cleanTrue_datasetCIFAR_ratio0.01_snr20_ChannelAWGN/checkpoint-99.pth",
        },
        {
            "class_key": "cleanFalse",
            "exp_name": "dpjscc/bdjscc_R1_3_cleanFalse_datasetCIFAR_ratio0.01_snr20_ChannelAWGN",
            "ctor": "bdjscc_R1_3",
            "checkpoint": "ckpt/dpjscc/bdjscc_R1_3_cleanFalse_datasetCIFAR_ratio0.01_snr20_ChannelAWGN/checkpoint-99.pth",
        },
        {
            "class_key": "cleanTrue",
            "exp_name": "experiment1/vit_jscc_lager_1_4_patch16_cleanTrue_datasetImageNet_ratio0.1_snr15_ChannelAWGN",
            "ctor": "vit_jscc_lager_1_4_patch16",
            "checkpoint": "ckpt/experiment1/ckpt/vit_jscc_lager_1_4_patch16_cleanTrue_datasetImageNet_ratio0.1_snr15_ChannelAWGN/checkpoint-119.pth",
        },
        {
            "class_key": "cleanFalse",
            "exp_name": "experiment1/vit_jscc_lager_1_6_patch16_cleanFalse_datasetCIFAR_ratio0.1_snr15_ChannelAWGN",
            "ctor": "vit_jscc_lager_1_6_patch16",
            "checkpoint": "ckpt/experiment1/ckpt/vit_jscc_lager_1_6_patch16_cleanFalse_datasetCIFAR_ratio0.1_snr15_ChannelAWGN/checkpoint-99.pth",
        },
    ]


def parse_args():
    parser = argparse.ArgumentParser(description="Run noise titration for BDJSCC and ViT-JSCC checkpoints.")
    parser.add_argument("--model-kind", choices=["all", "bdjscc", "vit"], default="all")
    parser.add_argument("--n-samples", type=int, default=128, help="Number of random latent samples per r/chi point.")
    parser.add_argument("--r-values", default=None, help="Comma separated noise powers, e.g. 0.05,0.1,0.2.")
    parser.add_argument("--r-start", type=float, default=0.1)
    parser.add_argument("--r-end", type=float, default=1.0)
    parser.add_argument("--r-step", type=float, default=0.1)
    parser.add_argument("--chi-values", default=None, help="Comma separated oracle thresholds, e.g. 0.2,0.4,0.6.")
    parser.add_argument("--chi-start", type=float, default=0.0)
    parser.add_argument("--chi-end", type=float, default=1.0)
    parser.add_argument("--chi-step", type=float, default=0.2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--oracle-checkpoint", default="save/resnet18_cifar10.pth")
    parser.add_argument("--oracle-input-size", type=int, default=224)
    parser.add_argument(
        "--bdjscc-input-size",
        type=int,
        default=32,
        help="Image size used to infer BDJSCC latent shape. CIFAR BDJSCC uses 32.",
    )
    parser.add_argument("--output-dir", default=str(SCRIPT_DIR / "result/Detection"))
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    grouped_results = {}
    oracle = OracleScorer(str(resolve_path(args.oracle_checkpoint)), device, input_size=args.oracle_input_size)
    r_values = make_scan_values(args.r_values, args.r_start, args.r_end, args.r_step)
    chi_values = make_scan_values(args.chi_values, args.chi_start, args.chi_end, args.chi_step)

    experiments = [
        exp for exp in build_experiments()
        if args.model_kind == "all" or model_kind_from_ctor(exp["ctor"]) == args.model_kind
    ]

    for exp in experiments:
        ctor_name = exp["ctor"]
        history = run_experiment(
            ctor_name,
            exp["checkpoint"],
            oracle,
            device,
            r_values,
            chi_values,
            args.n_samples,
            args.bdjscc_input_size,
        )
        model_kind = model_kind_from_ctor(ctor_name)
        grouped_results.setdefault(model_kind, {"cleanTrue": {}, "cleanFalse": {}})
        grouped_results[model_kind][exp["class_key"]][exp["exp_name"]] = {
            "ctor": ctor_name,
            "checkpoint": exp["checkpoint"],
            "scan": history,
        }

    output_dir = Path(args.output_dir)
    for model_kind, results in grouped_results.items():
        output_path = output_dir / f"{model_kind}_noise_titration_scores.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
