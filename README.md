# AlphaVAE

## 📦 Installation

```bash
conda create -n AlphaVAE python=3.10
conda activate AlphaVAE
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install git+https://github.com/CompVis/taming-transformers.git
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
python models/convert.py --src models/FLUX.1-dev/vae --dst models/FLUX.1-dev/rgba_vae
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

