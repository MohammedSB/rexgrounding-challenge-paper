# ReXGrounding Challenge paper: team method description (Appendix A)

## Data
We used the challenge training set (2,992 CT scans), comprising volumes, free-text findings, and ground-truth masks; the 200-scan validation set was used only for model selection. Original annotations were retained without modification. No additional external or in-house CT datasets, pseudo-labels, or derived anatomy masks were used in the final system. The only externally pretrained component was Qwen3-Embedding-4B (Apache 2.0), used frozen; the image network was trained from scratch. We did not use challenge test-set scans or finding texts for training or model selection. Whether any such material was included in Qwen's upstream pretraining data is unverified.

## Pre-processing
Following nnU-Net preprocessing, CTs were oriented to LPS and resampled to approximately 0.7 × 0.7 × 1.0 mm³. Intensities were clipped to [−1000, 197] HU and normalized with mean −549 HU and standard deviation 313 HU. Training used random foreground-centered crops (foreground oversampling probability 1.0) with zero padding. Patch sizes ranged from 224 × 224 × 128 to 448 × 448 × 256 voxels.

## Architecture and text encoder
Grounder uses a 3D nnU-Net ResEncL encoder and five-stage decoder. Frozen Qwen3-Embedding-4B (approximately 4B parameters) converts each finding into a 2,560-dimensional embedding. Two-layer MLPs generate feature-wise scale and shift parameters to condition decoder features through FiLM: F′ = γF + β.

## Training
We optimized Dice plus cross-entropy losses using AdamW, a fixed learning rate of 10⁻⁴, and batch size one. Models generally converged within 20–30 epochs. Image augmentation included rotation, scaling, and intensity transformations, but not flipping. In-distribution text augmentation substituted real findings from other training cases matched by anatomical location and pathology/finding type, retaining the original CT and mask. Each model trained on one NVIDIA H200 GPU for approximately 4–7 days depending on patch size; experiments ran in parallel across four GPUs.

## Ensembling
The final ensemble comprised 13 independently trained models differing in patch size, selected using validation performance. Their voxel-wise predicted probabilities were averaged equally.

## Post-processing
The averaged probabilities were thresholded at 0.5. No connected-component filtering, anatomical restrictions, or inference-time text rewriting/LLM processing was applied.

## Inference time and hardware
Inference required approximately 5 seconds per CT scan per finding on an NVIDIA H200 GPU.

## Final submission and lessons
The final leaderboard submission was **Grounder_1.4n** (Dice 0.379; hit rate 0.781; instance F1 0.366). Larger anatomical context, in-distribution text augmentation, and diverse patch sizes improved grounding. Adding TotalSegmentator lobe maps or category embeddings did not meaningfully help; negative-text augmentation was unreliable with partial annotations.

## Code and weights availability
The public repository is https://github.com/khuhm/Grounder (MIT license). Code and trained weights have not yet been uploaded and are planned for future release.

## Figure (optional)
none

## References
- Isensee et al., *nnU-Net: A Self-configuring Method for Deep Learning-Based Biomedical Image Segmentation*, Nature Methods, 2021. https://doi.org/10.1038/s41592-020-01008-z
- Isensee et al., *nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image Segmentation*, MICCAI, 2024. https://arxiv.org/abs/2404.09556
- Zhang et al., *Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models*, arXiv:2506.05176, 2025. https://arxiv.org/abs/2506.05176
- Perez et al., *FiLM: Visual Reasoning with a General Conditioning Layer*, AAAI, 2018. https://doi.org/10.1609/aaai.v32i1.11671
- Rokuss et al., *VoxTell: Free-Text Promptable Universal 3D Medical Image Segmentation*, arXiv:2511.11450, 2025. https://arxiv.org/abs/2511.11450
