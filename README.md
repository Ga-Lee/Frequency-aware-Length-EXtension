# ♾️ Train Short, Inference Long: Training-free Horizon Extension for Autoregressive Video Generation

<p align="center">
  <a href="https://ga-lee.github.io/FLEX_demo/"><strong>🌐 Project Page</strong></a> •
  <a href="https://arxiv.org/abs/2602.14027"><strong>📄 arXiv</strong></a> 
</p>

---

## 🎬 Long-Horizon Extension

<p align="center">
  <img src="asset/stitched_052_labeled_small_v2.gif" width="1000">
</p>

FLEX enables stable long video generation beyond the predefined generation horizon — without retraining or fine-tuning.

*Note:* The GIF above is compressed for GitHub display and may suffer from reduced visual quality.  
For higher-resolution videos and more demos, please visit **our Project Page**: https://ga-lee.github.io/FLEX_demo/

---

## ✨ Overview

Autoregressive video diffusion models suffer from severe extrapolation failure, where rapid error accumulation leads to significant temporal degradation and motion collapse when extending beyond training horizons. We propose **FLEX** (Frequency-aware Length EXtension), a *training-free, inference-time* framework that bridges the gap between *short-term training* and *long-term inference*.

---

## 📊 Highlights

- 🔹 **30s video generation (6× extrapolation for 5s models)** outperforming SOTA open-source autoregressive methods
- 🔹 **60s video generation (12× extrapolation for 5s models)** competitive with long-video fine-tuned baselines
- 🔹 Training-free, plug-and-play integration
- 🔹 Support for minute-level video generation

---

## 💻 Code

This release provides FLEX inference with the official **Self Forcing** checkpoint.

### Installation

Use Linux, Python 3.10, and an NVIDIA GPU supporting BF16. Run from the repository root:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m pip install packaging ninja
python -m pip install flash-attn==2.7.3 --no-build-isolation
python -m pip install -e .
python -m pip check
```

Choose `cu121` or `cu118` instead of `cu124` if needed for your CUDA environment. Building FlashAttention requires a compatible CUDA toolkit and C++ compiler.

### Download weights

```bash
hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir wan_models/Wan2.1-T2V-1.3B
hf download gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt --local-dir .
```

### Generate videos

```bash
# Approximately 30 seconds
bash scripts/infer_self_forcing.sh

# Approximately 60 seconds
NUM_OUTPUT_FRAMES=240 bash scripts/infer_self_forcing.sh

# Custom prompts (one per line) and output directory
PROMPTS=prompts/my_prompts.txt OUTPUT_FOLDER=outputs/demo bash scripts/infer_self_forcing.sh
```

Videos are generated at **832×480, 16 fps**. The default 126 latent frames produce 501 RGB frames (~31.3s); 240 produce 957 (~59.8s). Results are saved under `outputs/` by default.

FLEX settings are in [`configs/self_forcing_flex.yaml`](configs/self_forcing_flex.yaml). Use `python inference.py --help` for direct CLI options, including component overrides and `--low_memory` for text-encoder offloading.

Built on [Self Forcing](https://github.com/guandeh17/Self-Forcing), [Wan2.1](https://github.com/Wan-Video/Wan2.1), and [FramePack](https://github.com/lllyasviel/FramePack). Code is licensed under [Apache-2.0](LICENSE).

---

## 📄 Citation

If you find this work useful, please cite:

```bibtex
@misc{li2026trainshortinferencelong,
      title={Train Short, Inference Long: Training-free Horizon Extension for Autoregressive Video Generation}, 
      author={Jia Li and Xiaomeng Fu and Xurui Peng and Weifeng Chen and Youwei Zheng and Tianyu Zhao and Jiexi Wang and Fangmin Chen and Xing Wang and Hayden Kwok-Hay So},
      year={2026},
      eprint={2602.14027},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2602.14027}, 
}
```
