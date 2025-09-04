# End-to-End ASR with SqueezeFormer + CTC + KenLM

An automatic speech recognition (ASR) system built on a **SqueezeFormer** acoustic model, trained with **CTC loss**, decoded via **greedy/beam search**, and optionally **rescored with an n-gram language model (KenLM)**.

- **Framework:** TensorFlow / Keras  
- **Techniques:** SqueezeFormer (Transformer-based), CTC loss, WER metric, n-gram LM (KenLM)  
- **Results (example runs):**  
  - **WER < 10%** with KenLM  
  - **WER < 30%** without LM  

---

## Features
- Temporal U-Net style **downsampling–upsampling** with skip connections (SqueezeFormer).
- **Depthwise-separable convolutional subsampling** and **post-LayerNorm** blocks.
- **CTC training**, **beam search decoding**, optional **KenLM** for LM fusion.
- Reproducible **training/validation/test loops** with WER tracking & plots.


