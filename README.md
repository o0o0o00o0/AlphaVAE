# AlphaVAE: Unified End-to-End RGBA Image Reconstruction and Generation with Alpha-Aware Representation Learning
[![arXiv](https://img.shields.io/badge/arXiv-2503.10522-brightgreen.svg?style=flat-square)](https://arxiv.org/abs/2507.09308)
[![🤗 Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/o0o0o00o0/AlphaVAE/tree/main)
[![🤗 Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Data-blue)](https://huggingface.co/datasets/o0o0o00o0/AlphaTest)

---

**This is the official repository for "[AlphaVAE: Unified End-to-End RGBA Image Reconstruction and Generation with Alpha-Aware Representation Learning](https://arxiv.org/abs/2507.09308)".**

![grid](qualitative_t2i.png)

---

## 📦 Installation

```bash
conda create -n AlphaVAE python=3.10
conda activate AlphaVAE
pip install torch torchvision torchaudio
pip install -r requirements.txt
pip install taming-transformers-rom1504 # A portable, easy-to-install packaging of taming-transformers (CompVis)
#eval
pip install pyiqa
pip install tokenizers==0.21.1 transformers==4.51.1
```

## 🔽 Model Download

Pretrained model checkpoints are available at:

🔗 **[FLUX](https://huggingface.co/black-forest-labs/FLUX.1-dev)**
🔗 **[AlphaVAE](https://huggingface.co/o0o0o00o0/AlphaVAE/tree/main)**

```bash
models/
├── FLUX.1-dev/
├── finetune_VAE/
│   ├── config.json
│   ├── diffusion_pytorch_model.safetensors
│   └── finetune_diffusion/
│       └── pytorch_lora_weights.safetensors
└── convert.py
```

## 🚀 Inference

To run inference using pretrained models:

```bash
# VAE
bash inference/infer_vae.sh
# T2I
bash inference/infer_t2i.sh
```

## 🏋️ Training

Run the following command to start training:

```bash
# convert the original VAE to support 4-channel RGBA input
BNB_CUDA_VERSION=122 python models/convert.py --src /data/raw/zhanjiabo/models/Z-Image/vae --dst /data/raw/zhanjiabo/models/Z-Image/rgba_vae
# VAE
bash train/train_vae.sh
# Diffusion
bash train/train_diffusion_lora.sh
```

## 🧪 Evaluation

> 📂 Before running evaluation, please download the dataset from **[Huggingface](https://huggingface.co/datasets/o0o0o00o0/AlphaTest)**.
>
> ```bash
> tar -xzvf data.tar.gz
> ```

To evaluate model performance:

```bash
# VAE
bash validation_pipeline/vae_eval_pipeline.sh
bash validation_pipeline/vae_generation_and_eval_pipeline.sh

# Diffusion
bash validation_pipeline/t2i_eval_pipeline.sh
bash validation_pipeline/diffusion_generation_and_eval_pipeline.sh
```

## Citation
```bibtex
@misc{wang2025alphavaeunifiedendtoendrgba,
      title={AlphaVAE: Unified End-to-End RGBA Image Reconstruction and Generation with Alpha-Aware Representation Learning}, 
      author={Zile Wang and Hao Yu and Jiabo Zhan and Chun Yuan},
      year={2025},
      eprint={2507.09308},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.09308}, 
}
@article{yu2025omnialpha0,
  title   = {OmniAlpha: A Sequence-to-Sequence Framework for Unified Multi-Task RGBA Generation},
  author  = {Hao Yu and Jiabo Zhan and Zile Wang and Jinglin Wang and Huaisong Zhang and Hongyu Li and Xinrui Chen and Yongxian Wei and Chun Yuan},
  year    = {2025},
  journal = {arXiv preprint arXiv: 2511.20211}
}
```
