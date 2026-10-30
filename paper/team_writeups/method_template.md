# ReXGrounding Challenge paper: team method description (Appendix A)

Please keep this to about half a page of text (300–450 words) plus one optional figure. Keep the headings in this order and write "none" where something does not apply. We will typeset it and may lightly edit for length and style. If you prefer LaTeX, use method_template.tex instead.

## Data
Everything the model saw in training, with sizes and licences: challenge training set (2,992 scans) and validation set (200 scans); external public datasets; in-house data; pretrained weights; derived labels (pseudo-labels, LLM-extracted text, anatomy masks). State explicitly whether any test-set scans, in image or text form, were used at any stage, including in the pretraining of components you did not train yourselves.

## Pre-processing
Resampling, orientation, intensity windowing or normalisation, cropping, patch size.

## Architecture and text encoder
Image backbone, text encoder, how the text conditions the image model, number of parameters if known. Name the base framework (for example VoxTell, nnU-Net).

## Training
Loss, optimiser, learning-rate schedule, batch size, epochs or steps, data augmentation (image and text), hardware and training time, whether the validation set was used for training and how.

## Ensembling
Number of models in the final submission, how they differ, how predictions were combined.

## Post-processing
Thresholding, component selection, anatomical restriction (lobe or side), any use of an LLM at inference.

## Inference time and hardware
Average time per scan per finding at inference, and the hardware used.

## Final submission and lessons
Which submission is the final one (model name as shown on the leaderboard), and one or two sentences on what helped most and what did not help.

## Code and weights availability
Links to code and weights with licence, or a statement that they are not released.

## Figure (optional)
One architecture or pipeline figure, vector PDF preferred.

## References
Plain text; we will format them.
