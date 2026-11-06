# Data-leakage check: which of our datasets MedSAM was trained on (P1-13)

Source: Ma et al., *Segment anything in medical images*, Nature Communications 15, 654 (2024),
<https://doi.org/10.1038/s41467-024-44824-z>.
- **Supplementary Information** (PDF, 27 pages):
  <https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-024-44824-z/MediaObjects/41467_2024_44824_MOESM1_ESM.pdf>
- **Main text** (arXiv v3): <https://arxiv.org/abs/2304.12306v3>

Both were accessed on 2026-11-06.

How to read the supplementary tables. Supplementary Tables 1–4 list every dataset MedSAM used. Their captions say: "Datasets marked with ∗ denote external validation sets and the remaining datasets are used for internal validation." The Methods section ("Training protocol and experimental setting") explains what that means:
- **Internal datasets:** pooled and "randomly split … into 80%, 10%, and 10% as training, tuning, and validation". So every internal dataset was at least partly trained on.
- **External datasets:** "all datasets were hold-out and did not appear during model training".

We cannot tell which images of an internal dataset fell into the 80% training part. So a whole internal dataset counts as **seen**.

| Dataset | In MedSAM training data? | Evidence |
|---|---|---|
| Kvasir-SEG | no | Supp. Table 4 (Endoscopy): "Kvasir∗ [91] Endoscopy Polyp 1000". The ∗ marks an external validation set. Ref. [91] is the Kvasir-SEG paper (Jha et al., MMM 2020), and the 1000 images match our 1000 pairs. Methods: external datasets "did not appear during model training". Caveat: other polyp datasets were trained on (BKAI-IGH NeoPolyp, PolypGen; Supp. Table 4); image overlap between them and Kvasir-SEG was not checked. |
| ISIC 2018 | yes | Supp. Table 4 (Dermoscopy): "ISIC [84] [85] [86] Dermoscopy Skin cancer 3694". There is no ∗, so it is an internal (training) dataset. Ref. [86] is the ISIC 2018 challenge paper (Codella et al., arXiv:1902.03368); refs. [84]–[85] are HAM10000 and ISIC 2017. Supp. Table 17 links <https://challenge.isic-archive.com/data/>. 3694 = 2594 + 100 + 1000, the ISIC 2018 Task 1 train + val + test sizes. The train and test counts are from the ISIC data page; we counted the validation images in the official zip. Matching the total is our inference, so the official test split may be seen too. |
| BUSI | yes | Supp. Table 3 (Ultrasound): "Breast Cancer [77] Ultrasound Benign & malignant breast lesion 647". There is no ∗, so it is an internal (training) dataset. Ref. [77] is Al-Dhabyani et al., "Dataset of breast ultrasound images", Data in Brief 28, 104863 (2020), i.e. BUSI. Supp. Table 17 links <https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset>. 647 matches the benign + malignant images (BUSI's "normal" images have no lesion), which is our inference. |
| COCO val2017 | no | Not listed in Supp. Tables 1–4, which list only medical datasets. MedSAM is fine-tuned from SAM; SAM's own pre-training data (SA-1B) is not checked here. |

**Consequence for the analysis:** Kvasir-SEG and COCO are **unseen** by MedSAM. ISIC 2018 and BUSI are **seen**, so their MedSAM results are reported separately (CLAUDE.md 6.9). `scripts/make_report_tables.py` reads these labels from this file.
