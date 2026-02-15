# ♾️ Train Short, Inference Long: Training-free Horizon Extension for Autoregressive Video Generation

<p align="center">
  <a href="https://ga-lee.github.io/FLEX_demo/"><strong>🌐 Project Page</strong></a> •
  <a href="#"><strong>📄 arXiv</strong></a> 
</p>

---

## 🎬 Long-Horizon Extension

<p align="center">
  <img src="asset/stitched_052_labeled_small_v2.gif" width="1000">
</p>

FLEX enables stable long video generation beyond the predefined generation horizon — without retraining or finetuning.

*Note:* The GIF above is compressed for GitHub display and may suffer from reduced visual quality.  
For higher-resolution videos and more demo, please visit **Project Page**: https://ga-lee.github.io/FLEX_demo/

---

## ✨ Overview

Autoregressive video diffusion models suffer from severe extrapolation failure, where rapid error accumulation leads to significant temporal degradation and motion collapse when extending beyond training horizons. we propose **FLEX** (Frequency-aware Length EXtension) , a *training-free inference-time* framework that bridges the gap between *short-term training* and *long-term inference*.

---

## 📊 Highlights

- 🔹 **30s video generation (6× extrapolation for 5s models)** outperforming SOTA opensource autoregressive methods  
- 🔹 **60s video generation (12× extrapolation for 5s models)** competitive with long-video fine-tuned baselines  
- 🔹 Training free, plug-and-play integration  
- 🔹 Further support minute-level video generation  

---

## 💻 Code

Code release is coming soon.  
We are preparing a clean and well-documented implementation.

---

## 📄 Citation

If you find this work useful, please cite:

```bibtex
@article{li2025flex,
  title={Train Short, Inference Long: Training-free Horizon Extension for Autoregressive Video Generation},
  author={Li, Jia and Fu, Xiaomeng and Peng, Xurui and Chen, Weifeng and Zheng, Youwei and Zhao, Tianyu and Wang, Jiexi and Chen, Fangmin and Wang, Xing and So, Hayden Kwok-Hay},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2025}
}
