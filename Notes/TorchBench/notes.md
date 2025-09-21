TorchBench: Benchmarking PyTorch with High API Surface Coverage

This paper proposes a test bench for the PyTorch stack (Model (nn.Conv2d), Framework (torch.nn.functional.conv2d), and Acceleration (CUDA APIs) layers)

The reason this is necessary is existing benchmark suites aim to compare a wider variety of models unlike existing benches like MLPerf

TorchBench aims to give a comprehensive and deep analysis of PyTorch software stack, while MLPerf aims to compare models running atop different frameworks.


Torch bench differs:
- TorchBench benchmarks ONLY optimized for PyTorch API's while MLPerf optimized ENTIRE MODEL w/ end-to-end execution (includes preporcessing, etc.)
- TorchBench benchmarks PyTorch only, while MLPerf benchmarks different deep learning frameworks
- TorchBench includes 84 DL models in six domains, while MLPerf has only five models in five domains with PyTorch. TorchBench covers 2.3× more PyTorch APIs than MLPerf.

TorchBench aims to:
- run benchmarks with different configurations
- collect various performance statistics
- be ready for any continuous integration systems

What it DOES:
- Covers 2.3x more PyTorch API surface
- Integrates built in tools to config to enviornment for stats
- CI/CD pipeline for checking performance on official PyTorch repo (reverted changes when detect bad perf)

Models included (84 models, 6 domains {CV, NLP, etc.}):
- Classic Models: ResNet, VGG, Mobile Net (used as foundation of SOTA)
- Popular models: Yolo, pig2, T5, docTR
- Important Industry Models: Detectron2, BERT
- Diverse Models: Variety of domains


How they identify performance:
- Uses execution time as metric bc common and straightforward
- Ratios time of GPU Idleness, Data movement, and GPU active

Some models perform better on training v inference:
- Different input sizes
- Training can require higher precision, than inferene
- Torch may invoke different GPU kernels for training and inference even when they have the same input