qllm - Quantization with LLM

## Training Pipeline

Run the steps in order. All scripts read a YAML config via `--config-path` (default
`configs.yaml`); edit that file to select the model, categories, and hyperparameters.

### Step 0 — (Amazon 2014 only) Download the dataset

The Amazon **2023** dataset is pulled automatically from the HuggingFace Hub, so you can
skip this step for it. The Amazon **2014** dataset must be downloaded first, since it is
distributed as local per-category `meta_<Category>.json.gz` files.

```bash
# Download metadata for the categories you want (see --list for valid names)
python download_amazon_2014.py --out ./amazon2014_meta Beauty Digital_Music
# ...or grab every known category:
python download_amazon_2014.py --out ./amazon2014_meta --all
```

Then point `configs.yaml` at the downloaded folder and switch the loader to 2014:

```yaml
dataset_version: 2014
amazon_2014_dir: "./amazon2014_meta"
# amazon_2014_filename_template: "meta_{category}.json.gz"   # optional override
categories:
  - "Beauty"            # 2014 category names (same ones passed to the download script)
  - "Digital_Music"
```

For the 2023 dataset, leave `dataset_version` unset (or `2023`) and use the 2023 category
names (e.g. `All_Beauty`).

### Step 1 — Train the quantizer (semantic-ID) model

Learns to map product text to semantic IDs.

```bash
python train_quant.py --config-path configs.yaml
```

### Step 2 — Train the recommender

Uses the semantic-ID representation from Step 1 for the recommendation task.

```bash
python train_rec.py --config-path configs.yaml
```

### Notes

- Both training scripts share the same data loader (`data_module.build_amazon_datasets`),
  so the dataset config from Step 0 applies to both.
- For multi-GPU / large models (e.g. Qwen3-8B) use DeepSpeed ZeRO via the launcher:
  `NUM_GPUS=4 CONFIG_PATH=configs_qwen8b.yaml bash train_distributed.sh`.
- Evaluation utilities live in `quant_eval.py` / `evaluation.py`.

## RunPod Deployment

### Prerequisites
- RunPod account with GPU access
- Docker installed locally
- Project files pushed to git repository

### Building and Pushing to RunPod

1. **Build the Docker image locally:**
```bash
docker build -t qllm-train:latest .
```

2. **Tag for RunPod registry (optional):**
```bash
docker tag qllm-train:latest docker.io/your-username/qllm-train:latest
docker push docker.io/your-username/qllm-train:latest
```

3. **Deploy on RunPod:**
   - Go to RunPod dashboard
   - Create new pod with GPU (recommend RTX 4000 Ada or higher)
   - Select "Custom Docker Image"
   - Use your built image or public registry image
   - Set environment variables:
     - `WANDB_API_KEY`: Your Weights & Biases API key
     - `CONFIG_PATH`: Path to config file (default: configs.yaml)
   - Set volume mapping for outputs: `/workspace/outputs` to persistent storage
   - Start the pod

### Using RunPod CLI

Alternatively, use RunPod CLI for automated deployment:
```bash
# Install RunPod CLI
pip install runpod

# Deploy training job
runpodctl create gpu \
  --name qllm-training \
  --image qllm-train:latest \
  --gpu-type RTX_4000_ADA \
  --volume-size 50 \
  --env WANDB_API_KEY=your_key_here
```

### Monitoring Training

- **WandB**: View training metrics at https://wandb.ai/quocdat32461997
- **MLflow**: If configured, access at `http://localhost:5000` (port forward required)
- **Logs**: View pod logs in RunPod dashboard or via SSH

### Retrieving Results

After training completes:
1. Download outputs from persistent volume
2. Or use SCP to copy files:
```bash
scp -r user@pod-ip:/workspace/outputs ./outputs
```

### Configuration

Modify `configs.yaml` before building to adjust:
- Model selection
- Training hyperparameters
- LoRA settings
- Data categories
