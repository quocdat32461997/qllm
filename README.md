qllm - Quantization with LLM

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
