# MAAS Integration in MASLab

## Overview

MAAS (Multi-agent Architecture Search via Agentic Supernet) has been integrated into MASLab. This integration allows you to use MAAS's advanced multi-agent architecture search capabilities within the MASLab framework.

## Key Parameters

### Round and Sample Parameters
- **round**: Specifies which training round's model to use (default: 1)
- **sample**: Specifies which sample size model to use (default: 4)

Pre-trained models are stored in:
```
methods/maas/maas/ext/maas/scripts/optimized/{DATASET}/train/round_{round}/{DATASET}_controller_sample{sample}.pth
```

For example, with round=1 and sample=4 for MATH dataset:
```
methods/maas/maas/ext/maas/scripts/optimized/MATH/train/round_1/MATH_controller_sample4.pth
```

## Usage

### 1. Inference Mode (Default)

To run inference with a pre-trained MAAS model:

```bash
# Using default round=1, sample=4
python inference.py --method_name maas --test_dataset_name MATH --debug

# Or specify custom round and sample in config
python inference.py --method_name maas --test_dataset_name MATH --method_config_name config_custom --debug
```

**Note**: Ensure the pre-trained model file exists for your specified round and sample parameters.

### 2. Training Mode

To train a MAAS model:

```bash
# First, build your dataset
python datasets/build_test_dataset.py --dataset_name MATH

# Then run training
python inference.py --method_name maas --test_dataset_name MATH --method_config_name config_train --require_val
```

### 3. Batch Inference on Dataset

For full dataset evaluation:

```bash
# Sequential processing
python inference.py --method_name maas --test_dataset_name MATH --sequential

# Parallel processing
python inference.py --method_name maas --test_dataset_name MATH
```

## Configuration Options

Create custom configuration files in `methods/maas/configs/`:

```yaml
# Example: config_custom.yaml
is_test: true      # true for inference, false for training
sample: 4          # Must match your pre-trained model
round: 1           # Must match your pre-trained model
batch_size: 4      # Batch size for training
lr: 0.01           # Learning rate
is_textgrad: false # Whether to use textgrad optimization
```

### Pre-configured Files:
- `config_main.yaml`: Default inference configuration (round=1, sample=4)
- `config_train.yaml`: Training configuration

## Supported Datasets

MAAS currently supports:
- **MATH**: Mathematical problem solving
- **GSM8K**: Grade school math problems
- **HumanEval**: Code generation tasks

## Data Preparation

The integration automatically handles data conversion:
1. MASLab JSON format → MAAS JSONL format
2. Automatic copying to MAAS data directory
3. Both train and test splits handled

Data flow:
```
datasets/data/{dataset}.json → methods/maas/maas/ext/maas/data/{dataset}_{train|test}.jsonl
```

## Pre-trained Models

### Available Models
Currently available pre-trained model:
- MATH dataset: round_1, sample_4

### Using Pre-trained Models
1. Ensure your config matches the available model (round and sample)
2. The model file must exist in the expected path
3. If model not found, you'll see an error message with the expected path

## Example Workflows

### Quick Test with Pre-trained Model
```bash
# Test on MATH dataset with existing model (round=1, sample=4)
python inference.py --method_name maas --test_dataset_name MATH --model_name gpt-4o-mini --debug
```

### Train New Model
```bash
# 1. Prepare dataset
python datasets/build_test_dataset.py --dataset_name GSM8K

# 2. Train with custom parameters
# Create config_gsm8k_train.yaml with desired parameters
python inference.py --method_name maas --test_dataset_name GSM8K --method_config_name config_gsm8k_train --require_val
```

## Troubleshooting

### "No pre-trained model found"
- Check if the model file exists in the expected path
- Verify round and sample parameters in your config
- Train a model first if needed

### API Configuration Issues
- Ensure your model API configuration is properly set
- Check API keys and endpoints
- Use a valid model configuration file

## Limitations

1. **Single Sample Inference**: MAAS is designed for batch processing. Single sample inference returns a simplified response.
2. **Model Dependency**: Inference requires pre-trained models with matching round and sample parameters.
3. **Training Time**: MAAS training can be computationally intensive due to architecture search.

## Future Improvements

1. Implement proper single-sample inference support
2. Add automatic model downloading/sharing
3. Support for more datasets
4. Integrate MAAS's visualization capabilities
5. Support for distributed training