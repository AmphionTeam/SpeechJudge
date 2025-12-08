# SpeechJudge: Towards Human-Level Judgment for Speech Naturalness

[![arXiv](https://img.shields.io/badge/arXiv-2511.07931-b31b1b.svg)](https://arxiv.org/abs/2511.07931)
[![Demo Page](https://img.shields.io/badge/Project-Demo_Page-blue)](https://speechjudge.github.io/)
[![GitHub](https://img.shields.io/badge/GitHub-SpeechJudge-black?logo=github)](https://github.com/AmphionTeam/SpeechJudge)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/RMSnow/SpeechJudge-GRM)
[![Data](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Data-yellow)](https://huggingface.co/datasets/RMSnow/SpeechJudge-Data)

Aligning large generative models with human feedback is a critical challenge. In speech synthesis, this is particularly pronounced due to the lack of a large-scale human preference dataset, which hinders the development of models that truly align with human perception. To address this, we introduce **SpeechJudge**, a comprehensive suite comprising a dataset, a benchmark, and a reward model centered on ***naturalness***—one of the most fundamental subjective metrics for speech synthesis:

- **SpeechJudge-Data**: a large-scale human feedback corpus of 99K speech pairs. The dataset is constructed using a diverse set of advanced zero-shot text-to-speech (TTS) models across diverse speech styles and multiple languages, with human annotations for both intelligibility and naturalness preference. 
- **SpeechJudge-Eval**: a challenging benchmark for speech naturalness judgment.
- **SpeechJudge-GRM**: a generative reward model (GRM) based on Qwen2.5-Omni-7B. It is trained on SpeechJudge-Data via a two-stage post-training process: Supervised Fine-Tuning (SFT) with Chain-of-Thought rationales followed by Reinforcement Learning (RL) with GRPO on challenging cases.

## TODO

We plan to release the following components in the future:

- [x] **SpeechJudge-Data** and **SpeechJudge-Eval**: Release the 99K speech pairs dataset with human annotations.
- [x] **SpeechJudge-GRM**: 
    - [x] Inference pipeline for pairwise speech comparison.
    - [x] Add inference-time scaling support via vLLM.
    - [ ] The two-stage "SFT+RL" training pipeline.

Stay tuned for updates!


## SpeechJudge-Data and SpeechJudge-Eval

The SpeechJudge-Data and SpeechJudge-Eval datasets are released at HuggingFace (see the [dataset page](https://huggingface.co/datasets/RMSnow/SpeechJudge-Data) for detailed documentation). 

You can load the dataset directly using the Hugging Face `datasets` library:

```python
from datasets import load_dataset

# Load the entire dataset (all splits)
ds = load_dataset("RMSnow/SpeechJudge-Data")

# Load a specific split, e.g., the SpeechJudge-Eval benchmark (test split)
test_ds = load_dataset("RMSnow/SpeechJudge-Data", split="test")
```

## SpeechJudge-GRM

### Features

- **Automated Naturalness Evaluation**: Compare two TTS audio outputs and receive quantitative scores.
- **Multi-Criteria Speech Assessment**: Evaluates based on:
  - Prosody and Intonation
  - Pacing and Rhythm
  - Articulation and Clarity
  - Overall Naturalness
- **Chain-of-Thought Reasoning**: Provides explainable analysis with detailed reasoning process
- **Inference-time Scaling**: Optional inference-time scaling for enhanced judgment accuracy。

### Installation

1. Clone this repository:
```bash
git clone https://github.com/AmphionTeam/SpeechJudge.git
cd SpeechJudge
```

2. Install the required dependencies:
```bash
pip install transformers==4.52.3
pip install accelerate
pip install qwen-omni-utils
```

### Usage

#### Basic Usage

The main entry point is `infer/main_grm.py`. Here's a basic example:

```python
from infer.main_grm import load_model, compare_wavs

# Load the SpeechJudge-GRM model (The checkpoint will be downloaded from https://huggingface.co/RMSnow/SpeechJudge-GRM.)
model_path = "pretrained/SpeechJudge-GRM"  # The local dir to save the model
model, processor = load_model(model_path)

# The compared two speeches (and the corresponding text)
target_text = "Your target text here"
wav_path_a = "path/to/audio_a.wav"
wav_path_b = "path/to/audio_b.wav"

# Compare the two audio outputs
rating, result = compare_wavs(processor, model, target_text, wav_path_a, wav_path_b)

print(f"Output A score: {rating['output_a']}")
print(f"Output B score: {rating['output_b']}")
print(f"\nDetailed Analysis:\n{result}")
```

#### Running the Example

The repository includes example audio files in `infer/examples/`. To run the provided example:

```bash
cd infer
python main_grm.py
```

#### Inference with vLLM

For enhanced performance and efficiency, SpeechJudge-GRM also supports inference via vLLM, which enables inference-time scaling for improved judgment accuracy. The implementation follows [vLLM's official documentation for Qwen2.5-Omni](https://docs.vllm.ai/en/v0.9.2/examples/offline_inference/qwen2_5_omni.html). To run the example with vLLM:

```bash
cd infer
python main_grm_vllm.py
```

## Citation

If you use SpeechJudge in your research, please cite our paper:

```bibtex
@article{zhang2025speechjudge,
  title={SpeechJudge: Towards Human-Level Judgment for Speech Naturalness},
  author={Zhang, Xueyao and Wang, Chaoren and Liao, Huan and Li, Ziniu and Wang, Yuancheng and Wang, Li and Jia, Dongya and Chen, Yuanzhe and Li, Xiulin and Chen, Zhuo and Wu, Zhizheng},
  journal={arXiv preprint arXiv:2511.07931},
  year={2025}
}
```
