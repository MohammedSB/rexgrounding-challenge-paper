# ReXGrounding Challenge paper: team method description (Appendix A)


## Data

For training, we used the challenge training data, 200 validation scans, COVID1920 (180), OmniAbnormCT (52), sparse ILD (91), and CT-RATE (18,652, test set scans excluded), with PatchChest annotations attached to 768 CT-RATE scans.

We reviewed and edited 146 findings across 117 scans. Common errors were:

- Artifacts, partially due to the annotation program
- Annotation–prompt location mismatches
- Incomplete annotations, such as missed nodules
- Annotating artery as atelectasis / fibrosis
- Drawing the entire lung as emphysema 

## Pre-processing

CTs were resampled to 0.75×0.75×1.25 mm, anatomically cropped using totalsegmentator lung masks with 15 mm margins. CT intensities were clipped to [−1024,1000] HU and normalized as (HU+232)/281. 

## Architecture and text encoder

We used VoxTell 1.1v, using frozen Qwen3-Embedding-4B embeddings. Input crop sizes were 288×288×192.

## Training

We trained models using an effective batch of 8 across two RTX PRO 6000 GPUs. We used AdamW with an initial learning rate of 1e-5 and cosine decay unless stated otherwise.

**1) Supervised training.** We initialized two models from VoxTell v1.1 and trained them using the challenge training set and complementary halves of the validation set, retaining the other half for checkpoint selection. The loss combined 0.3 BCE, 0.7 Dice and 0.1 location KL with deep supervision. Augmentation included intensity changes, noise, sharpening and prompt paraphrases.

**2) Semi-supervised training with external data.** We continued training with extended supervision set including COVID1920, OmniAbnormCT and sparse ILD data. In addition, we performed semi-supervised training using CT-RATE scans. For this purpose, we extracted positive findings from radiology reports using MedGemma and applied two losses: 

1. positive-presence loss (weight 0.1): log-sum-exp pooling of voxel logits followed by binary cross-entropy against finding presence (whether it is a positive finding).

2. EMA-teacher pseudo-mask loss (weight 0.2; EMA decay 0.995). 

**3) Fine-tuning on the full validation set.** We fine-tuned each selected model using only the 200 validation scans and supervised losses for 500 steps at a constant learning rate of 3e-7.
## Ensembling

We ensembeled the two models from each fold using probability averaging.

## Post-processing

For prompts explicitly stating left / right, we removed predictions that were confidently assigned to the wrong lobe based on TotalSegmentator mask.

## Inference time and hardware

Across 300 scans/582 findings, the prediction time for the model averaged 11.45 s/scan (5.90 s/finding) using RTX 3090, excluding I/O and post-processing.

## Final submission and lessons

We submitted the ensemble model described above.

What helped

- Increasing crop size
- Fine-tuning with curated dataset

What did not

- Re-annotating non-focal categories, such as bronchiectasis
- Using Tyversky loss to address "partial annotation" problem

## Code and weights availability

Edited data, code, and weights will not be publicly released, but available on requests.

## Figure

The semi-supervised training (stage 2) is illustrated in the figure.

## References

1. **VoxTell.** Rokuss, M., Langenberg, M., Kirchhoff, Y., Isensee, F., Hamm, B., Ulrich, C., Regnery, S., Bauer, L., Katsigiannopulos, E., Norajitra, T., and Maier-Hein, K. “VoxTell: Free-Text Promptable Universal 3D Medical Image Segmentation.” In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 37538–37557, 2026. [Publisher record](https://openaccess.thecvf.com/content/CVPR2026/html/Rokuss_VoxTell_Free-Text_Promptable_Universal_3D_Medical_Image_Segmentation_CVPR_2026_paper.html).

2. **ReXGroundingCT.** Baharoon, M., Luo, L., Moritz, M., Kumar, A., Kim, S. E., Zhang, X., Zhu, M., Alabbad, M. H., Alhazmi, M. S., Mistry, N. P., et al. “ReXGroundingCT: A 3D Chest CT Dataset for Segmentation of Findings from Free-Text Reports.” NEJM AI, 2026, AIdbp2501220. doi:10.1056/AIdbp2501220.

3. **CT-RATE.** Hamamci, I. E., Er, S., Wang, C., Almas, F., Simsek, A. G., Esirgun, S. N., Dogan, I., Durugol, O. F., Hou, B., Shit, S., et al. “Generalist foundation models from a multimodal dataset for 3D computed tomography.” *Nature Biomedical Engineering*, 10(8):1610–1628, 2026. [doi:10.1038/s41551-025-01599-y](https://doi.org/10.1038/s41551-025-01599-y).

4. **PatchChestCT.** Li, Y., Zhang, H., Xu, M., Zhang, L., Lei, N., Zhang, Y., Zhu, X., Lu, Y., Wei, W., and Zhou, S. K. “PatchChestCT: A patch-level spatial annotation dataset for nine abnormalities in chest CT.” *Scientific Data*, 13:1180, 2026. [doi:10.1038/s41597-026-07793-0](https://doi.org/10.1038/s41597-026-07793-0).

5. **COVID-19-20 (COVID1920).** Roth, H. R., Xu, Z., Tor-Díez, C., Sanchez Jacob, R., Zember, J., Molto, J., Li, W., Xu, S., Turkbey, B., Turkbey, E., et al. “Rapid artificial intelligence solutions in a pandemic—The COVID-19-20 Lung CT Lesion Segmentation Challenge.” *Medical Image Analysis*, 82:102605, 2022. [doi:10.1016/j.media.2022.102605](https://doi.org/10.1016/j.media.2022.102605).

6. **OmniAbnorm-CTRATE (OmniAbnormCT).** Zhao, Z., Dai, L., Zhang, Y., Xie, W., and Wang, Y. “Rethinking Whole-Body CT Image Interpretation: An Abnormality-Centric Approach.” In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 5179–5189, 2026.

7. **HUG/MedGIFT ILD.** Depeursinge, A., Vargas, A., Platon, A., Geissbuhler, A., Poletti, P.-A., and Müller, H. “Building a reference multimedia database for interstitial lung diseases.” *Computerized Medical Imaging and Graphics*, 36(3):227–238, 2012. [doi:10.1016/j.compmedimag.2011.07.003](https://doi.org/10.1016/j.compmedimag.2011.07.003).
