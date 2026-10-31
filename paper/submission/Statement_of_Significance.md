# Statement of Significance

Manuscript: Grounding Radiology Reports in 3D Chest CT: Results of the ReXGrounding Challenge
Corresponding author: Mohammed Baharoon (MohammedSalimAB@outlook.com)

Items in [square brackets] need the authors' input before upload.

## Question 1

Peer-Review: Has this paper been previously submitted to this journal or to another journal? If yes, please provide the journal name and a detailed explanation of what has been done to improve the prior submission to justifying re-submission to MedIA. (300 words maximum). Please note that re-submission to MedIA of papers previously rejected by MedIA is not permitted.

Answer: No. This manuscript has not been submitted to Medical Image Analysis or to any other journal.

## Question 2

Has a version of the paper previously been published as a peer-reviewed paper (e.g. conference)? If yes, please upload the article as an attachment on the previous page and briefly outline here the improvements over that article. Please also confirm that the improvements are explicitly mentioned in the introduction. (300 words maximum)

Answer: No. No version of this paper has been published as a peer-reviewed paper. The challenge design was registered with MICCAI before the challenge and deposited on Zenodo (Baharoon et al., 2026, https://doi.org/10.5281/zenodo.19737020); that document is not peer-reviewed and contains no results. The dataset on which the challenge builds was described in a separate peer-reviewed article (Baharoon et al., 2026, NEJM AI), which the introduction cites and distinguishes from the present work.

## Question 3

Please provide 3 to 5 Associate Editors of this journal whom you consider best suited to judge the contributions of your manuscript, together with your rationale for each selection (https://www.sciencedirect.com/journal/medical-image-analysis/about/editorial-board). Please note that each nominee should be based in countries different from the authors. Please DO NOT select any names with whom any of the listed authors have a conflict of interest, i.e., prior professional collaborations, such as co-authoring publications. If you select an Associate Editor who has previously collaborated with a listed author, please explain why no conflict of interest exists. Please DO NOT propose Executive Committee or Senior Editor names.

Answer: [To be completed from the editorial board page. Constraints: nominees must be based outside the authors' countries (USA, Saudi Arabia, China, South Korea, the Netherlands, Germany and Belgium) and must not have co-authored with any of the 56 listed authors. Suitable profiles: Associate Editors who have organized or analysed MICCAI challenges, or who work on vision-language models or text-guided segmentation for CT. Suggested format for each: Name, institution, country. Rationale: one sentence. No prior collaboration with any listed author.]

## Question 4

Novelty: To help the editors understand your intended contribution to the literature, please state clearly (in 1-2 sentences) the methodological innovation that is contained within your paper. (300 words maximum)

Answer: This paper reports the first international challenge on grounding free-text radiology report findings as 3D segmentations in chest CT, establishing a benchmark of 300 exhaustively annotated test scans with hidden reference masks and a public leaderboard, and analysing the methods of 48 teams. Beyond the ranking, it introduces a controlled prompt-editing experiment in which the best methods are rerun on systematically edited sentences, which quantifies for the first time how text-conditioned 3D segmentation models use the side, lobe and size information in a sentence.

## Question 5

How is this work different from prior publications of any of the listed authors? Please list at least one published article (full reference) by any listed authors and briefly note the difference between your effort and that paper. If there is no such publication, please state so. (300 words maximum)

Answer:

Baharoon, M., Luo, L., Moritz, M., et al., Rajpurkar, P., 2026. ReXGroundingCT: A 3D chest CT dataset for segmentation of findings from free-text reports. NEJM AI. [volume, article number and DOI to add]. That article describes the construction of the ReXGroundingCT dataset (3,142 scans, 100 test scans) and shows that existing text-promptable segmentation models perform poorly on it. The present paper reports the challenge built on that dataset: 350 additional validation and test scans were annotated exhaustively, 48 teams developed methods for the task, and the paper analyses their results, the stability of the ranking, performance by finding type and the methods' use of language. None of this appears in the dataset paper.

Baharoon, M., Luo, L., Moritz, M., et al., Rajpurkar, P., 2025. State-of-the-art text-prompted medical segmentation models struggle to ground chest CT findings. Proceedings of the Machine Learning for Healthcare Conference (MLHC). That paper benchmarks existing models zero-shot on the original test set. The present paper reports methods developed specifically for the task and their evaluation on a three times larger hidden test set.

Rokuss, M., Langenberg, M., Kirchhoff, Y., et al., Maier-Hein, K., 2026. VoxTell: Free-text promptable universal 3D medical image segmentation. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. That paper proposes a single method, which several teams fine-tuned in the challenge. The present paper evaluates it alongside 47 other entries and is a challenge report, not a method paper.

## Question 6

Please list 3-5 published papers that you believe are the efforts closest to your work, briefly noting the difference between your effort and each paper if possible. List the papers including title; DOI; first author name and last author name; journal name; and year of publication. (300 words maximum)

Answer:

1. VoxTell: Free-text promptable universal 3D medical image segmentation. Rokuss, M. (first), Maier-Hein, K. (last). Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026. [DOI to confirm]. Proposes one text-promptable 3D segmentation model and was the best model before the challenge; our paper is a challenge report that compares 48 methods on a hidden test set.

2. BiomedParse: a biomedical foundation model for image parsing of everything everywhere all at once. Zhao, T. (first), Poon, H. (last). Nature Methods, 2025. https://doi.org/10.1038/s41592-024-02499-w [to confirm]. Text-guided segmentation of biomedical images, mainly 2D, prompted with object names; our task grounds full report sentences in 3D volumes.

3. Generalist foundation models from a multimodal dataset for 3D computed tomography (CT-RATE and CT-CLIP). Hamamci, I.E. (first), Menze, B. (last). Nature Biomedical Engineering, 2026. [DOI to confirm]. Provides the CT volumes and reports on which the challenge builds and trains vision-language models for classification and retrieval; it does not localize findings.

4. Unleashing the strengths of unlabelled data in deep learning-assisted pan-cancer abdominal organ quantification: the FLARE22 challenge. Ma, J. (first), Wang, B. (last). The Lancet Digital Health, 2024. [DOI to confirm]. A MICCAI challenge report on CT segmentation of fixed anatomical classes; our challenge targets free-text findings and adds a language analysis.

5. BIAS: Transparent reporting of biomedical image analysis challenges. Maier-Hein, L. (first), Kopp-Schneider, A. (last). Medical Image Analysis, 2020. https://doi.org/10.1016/j.media.2020.101796. The reporting guideline our paper follows; it does not report a challenge.

## Question 7

Is the proposed method significantly better than the state of the art? If so, briefly describe how the gains were quantified and verified (metrics, datasets, baselines, and statistical tests).

Answer: This is a challenge report rather than a single proposed method. The challenge substantially advanced the state of the art. On the 100 test scans of the pre-challenge benchmark, the winning method reached a Dice of 0.390 against 0.285 for the best model before the challenge (VoxTell), and 23 of the 48 teams exceeded 0.285. General text-promptable segmentation models (SAT, SegVol, BiomedParse-V) scored close to zero on this task, and SAT fine-tuned on the training set reached 0.205. Metrics were Dice per finding averaged per case (primary), hit rate (Dice of at least 0.1) and instance F1, all on the full 300-scan hidden test set. Precision was quantified with 95% percentile bootstrap confidence intervals over 1,000 resamples of the test scans, ranking stability with bootstrap rank distributions and Kendall's tau, and pairwise differences among the ten best teams with one-sided paired Wilcoxon signed-rank tests on per-case Dice with Holm correction. The best methods match the agreement between two independent groups of radiologists on Dice and hit rate.

## Question 8

Please state whether your method operates on full-resolution, full-field-of-view images or not. For example, histology methods may use entire whole-slide images or tiled patches and MRI analyses may be volumetric (3D) or slice-based (2D). If input is not based on full-resolution, full-field-of-view images, please provide scientific justification.

Answer: Participants received the full chest CT volumes at native resolution, orientation and intensity, and predictions were evaluated on the native voxel grid of each scan. All winning methods are volumetric (3D); their internal resampling, cropping and patching choices are described in Appendix B.

## Question 9

Please confirm that the manuscript includes figures showing findings across the full spatial extent of the input, e.g., findings derived from 3D MRIs need to be shown from at least two orthogonal planes (e.g., axial and coronal).

Answer: Confirmed. Reference and predicted masks are shown as 3D surface renderings of the whole lung volume (Figures 5, 7 and D.1), which display the full three-dimensional extent of each finding, and as axial slices (Figure 1). [If the editors require two orthogonal 2D planes, a coronal view can be added to Figure 5.]

## Question 10

Please confirm that any information about novelty (i.e., Questions 4 to 7) is explicitly described in the manuscript.

Answer: Confirmed. The contribution is stated in Section 1 (Introduction), the comparison with the state of the art in Sections 3.3 and 4.1, the statistical methods in Section 2.4, and the controlled prompt experiment in Sections 2.4 and 3.7.

## Question 11

Reproducibility: How did you ensure reproducibility of your findings? If this work relies on machine learning, Did you test your method on multiple data sets of which at least one is publicly available? If not, please justify. Did you apply stratified cross-validation or test on external data sets? If not, please justify. How did you ensure that there was no data leakage between training and testing data? For example, with respect to cross-validation, was the data of a subject confined to one-fold?

Answer: All methods were evaluated by the organizers on a single hidden test set of 300 scans with the public evaluation code, so every reported score is reproducible from a team's submitted masks. The test set is from the same source as the training data because ReXGroundingCT is the only dataset with free-text finding segmentations in 3D CT; this is stated as a limitation, and an inter-rater study on 50 scans is used to put the scores in context. Cross-validation does not apply to a challenge with a fixed held-out test set; instead, the paper compares the ranking on the 150 publicly scored scans with the ranking on all 300 scans and reports bootstrap confidence intervals and rank distributions. Training, validation and test sets are disjoint sets of scans, no scan appears in more than one split, and the test masks were never released. Some teams pretrained on the public CT-RATE collection, which contains the test images without any challenge masks or sentences; this is documented in the method descriptions (Appendix B), and the one team that identified test studies in its pretraining pool retrained with those studies removed.

## Question 12

For each data set, specify if they are publicly available, the sample size, and if all samples of the data set were used. If they were not all used, please describe selection criteria. Please number of samples used for training and testing (unless cross-validated is performed).

Answer: ReXGroundingCT (public, https://huggingface.co/datasets/rajpurkarlab/ReXGroundingCT): training set 2,992 scans with 7,687 findings, validation set 200 scans with 381 findings, test set 300 scans with 582 findings. The test images and finding sentences are public; the test masks are withheld. All samples were used. The inter-rater analysis uses a published subset of 50 validation and test scans (96 findings) annotated twice. Additional public datasets used by individual teams for pretraining or auxiliary supervision (CT-RATE, COVID-19-20, OmniAbnorm-CT, the HUG ILD database, PatchChestCT) are listed with sample counts in Appendix B.

## Question 13

What are potential biases in the data and how are those biases accounted for in the analysis?

Answer: The data come from a single centre in Turkey, six scanner models and non-contrast chest CT only, so results may not transfer to other institutions, protocols or contrast-enhanced imaging. The report sentences were machine-translated from Turkish and rewritten and split by large language models, with radiologist review on samples. Focal findings make up 87% of the test set and several categories contain fewer than ten findings. Training annotations cover at most three instances per finding whereas the test annotations are exhaustive, which can teach models to stop after a few instances. The reference standard comes from single annotators with modest inter-rater agreement (Dice 0.32). These biases are addressed by reporting results per finding category with bootstrap confidence intervals, by analysing performance against the number of instances, by comparing all scores with the inter-rater agreement, by describing the sources of error in Section 2.2 and the limitations in Section 4.2, and by keeping the leaderboard open so that future methods are compared on the same data. Patient age and sex are reported for all splits; performance was not stratified by sex or age because of the sample size, and this is noted as a limitation. [Add this sentence to Section 4.2 if not already present.]

## Question 14

Is the data and code publicly available? If not, please justify.

Answer: The training and validation data, including findings and reference masks, the test images and finding sentences, and the evaluation code are public at https://huggingface.co/datasets/rajpurkarlab/ReXGroundingCT under a CC BY-NC-SA 4.0 licence. The test reference masks are withheld so that the public leaderboard at https://rexrank.ai/ReXGroundingCT can continue to evaluate new methods on an uncontaminated test set; any method can be scored on it by submission. The analysis code and derived results underlying the figures and tables of this paper are available at [https://github.com/MohammedSB/rexgrounding-challenge-paper, to be made public on acceptance]. Participating teams were encouraged but not required to release their code; the links they provided are given in Appendix B.

## Question 15

Please confirm that any information about reproducibility (i.e., Questions 9 to 12) is explicitly described in the manuscript.

Answer: Confirmed. Data, splits, annotation and sources of error are described in Section 2.2, metrics and statistical analysis in Section 2.4, the team methods and the data they used in Appendix B, and data and code availability in the Declarations.

## Question 16

Other: Has the manuscript been written with the help of a Large Language Model (such as ChatGPT)? If yes, please specify which ones was used, how it was used, and how you ensured the scientific integrity of the writing (including that the citations provided by the model are correct). Please confirm that this information is included in the manuscript.

Answer: Yes. The authors used Claude (Opus 5.5 and Fable 5.1, Anthropic) to help draft and edit the text, to write analysis and figure code, and to cross-check the reported numbers against the analysis outputs. Every number in the manuscript was traced to its source file, and the authors reviewed and edited every sentence. All references were selected and entered by the authors from the original publications; none was generated by the model. Separately, Claude Opus 5.5 was used as a research tool to rewrite test sentences in the controlled prompt experiment, which is described in Section 2.4. The use of the model in preparing the manuscript is declared in the section "Declaration of generative AI and AI-assisted technologies in the manuscript preparation process" before the reference list.

## Question 17

Please confirm that the submission does not include any patient identifying information.

Answer: Confirmed. All scans and reports come from CT-RATE, a public de-identified dataset; the manuscript contains no patient identifying information.

## Question 18

Please enter any additional comments you would like to send to the publication office. These comments will not appear directly in your submission.

Answer: This manuscript is a Challenge Report. The required Bias Report is included as the BIAS checklist in Appendix E, and the paper follows the BIAS guideline throughout. The author list has 56 authors, as is usual for challenge reports (organizers, the four winning teams, the annotators and the authors of the baseline model); we would be grateful if the vitae requirement could be waived or limited to the corresponding and senior authors. Competing interests of individual team members are declared in the manuscript.
