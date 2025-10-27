from pathlib import Path
from typing import List, Set

import torch
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS

from BNN_KWS_GSC import BNN_KWS, waveform_to_logmel

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_ROOT = SCRIPT_DIR / "speech_commands"
DATASET_ROOT.mkdir(parents=True, exist_ok=True)

ckpt_path = SCRIPT_DIR / "bnn_KWS.pt"
ckpt = torch.load(str(ckpt_path), map_location="cpu")
args = ckpt["args"]
model = BNN_KWS(
    num_classes=ckpt["num_classes"],
    bin_first=args["bin_first"],
    bin_last=args["bin_last"],
    use_scale=not args["no_scale"],
)
model.load_state_dict(ckpt["model"])
model.eval()

# 确保 Speech Commands 数据集已下载到脚本所在目录
SPEECHCOMMANDS(root=str(DATASET_ROOT), download=True)


def load_labels_from_lists(dataset_root: Path) -> List[str]:
    """Attempt to collect ordered labels by reading the split list files."""
    version_dir = dataset_root / "speech_commands_v0.02"
    if not version_dir.exists():
        return []

    label_set: Set[str] = set()
    for list_name in ("training_list.txt", "validation_list.txt", "testing_list.txt"):
        list_path = version_dir / list_name
        if not list_path.exists():
            continue
        with list_path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                label = line.split("/", 1)[0]
                if label:
                    label_set.add(label)

    return sorted(label_set)


def resolve_labels(dataset_root: Path) -> List[str]:
    labels = load_labels_from_lists(dataset_root)
    if labels:
        return labels

    # Fallback: instantiate the dataset once to infer labels.
    dataset = SPEECHCOMMANDS(str(dataset_root), subset="training", download=False)
    return sorted({sample[2] for sample in dataset})

waveform_path = SCRIPT_DIR / "yes.wav"
waveform, sr = torchaudio.load(str(waveform_path))
feat = waveform_to_logmel(waveform, sr).unsqueeze(0)  # [1, 1, 40, 98]
with torch.no_grad():
    logits = model(feat)
    pred_idx = logits.argmax(dim=1).item()

print(pred_idx)

labels = resolve_labels(DATASET_ROOT)
if not labels:
    print("未能解析标签列表。")
else:
    if 0 <= pred_idx < len(labels):
        pred_label = labels[pred_idx]
        print(pred_label)
    else:
        print(f"预测索引 {pred_idx} 超出标签范围 (0-{len(labels) - 1}).")

    print("\n标签顺序：")
    for idx, label in enumerate(labels):
        print(f"{idx:2d}: {label}")
