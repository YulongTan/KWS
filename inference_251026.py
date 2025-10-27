from pathlib import Path

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

waveform_path = SCRIPT_DIR / "no.wav"
waveform, sr = torchaudio.load(str(waveform_path))
feat = waveform_to_logmel(waveform, sr).unsqueeze(0)  # [1, 1, 40, 98]
with torch.no_grad():
    logits = model(feat)
    pred_idx = logits.argmax(dim=1).item()

print(pred_idx)

# 可选：恢复标签字符串
labels = sorted(
    {
        sample[2]
        for sample in SPEECHCOMMANDS(
            str(DATASET_ROOT), subset="training", download=False
        )
    }
)
pred_label = labels[pred_idx]
print(pred_label)
