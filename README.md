# RSNA Knee Abnormality Detection

My work on the [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) Kaggle competition (2026): predict 12 knee MRI findings (ACL, MCL, menisci, osteoarthritis, effusion, synovitis, Baker's cyst, contusion, fracture) from multi-series DICOM studies.

- **Task:** multi-label classification, scored by macro-averaged ROC AUC over the 12 labels
- **Format:** code competition. Inference runs in a Kaggle notebook with no internet and a 9-hour limit.

## What the data looks like

Findings from `src/eda.py`:

- **4,407 training studies, but only 58 have labels.** Those 58 are labeled for all 12 findings, and the rest are unlabeled.
- **Every training study has a radiology report, but test studies don't.** The reports can supervise training, but inference has to work from images alone.
- **The reports are multilingual** (Dutch, Spanish, English, German, French, Bulgarian, Turkish, …).
- Each study has 3–14 series (median 5), tagged by plane (sagittal / coronal / axial) and fluid-sensitive / fat-suppressed.
- Findings are correlated, e.g. medial OA with medial meniscus tears.

## Planned approach

1. **Report → labels.** A multilingual open-weights LLM reads each report and outputs 12 soft labels, validated against the 58 human-labeled studies.
2. **DICOM preprocessing.** Order slices by `ImagePositionPatient`, normalize intensity by percentiles, crop and resize, and pick the series per plane from the metadata.
3. **Image model.** A pretrained 2.5D CNN per plane, pooled across slices and planes, with 12 sigmoid outputs trained with BCE on the soft labels.
4. **Validation.** GroupKFold on the pseudo-labels, plus a final check against the 58 human labels.

## Results so far

| Submission | Public LB |
|---|---|
| Constant 0.5 baseline (`src/baseline_submission.py`) | 0.500 |
| Fork of a public notebook, *not my work* (see below) | _pending_ |

The public notebook I ran is [RSNA_Versia_7 by Evgenii Dvorkin](https://www.kaggle.com/code/evgendvorkin/rsna-versia-7), an ensemble of public CoAtNet / ConvNeXt / Raptor models. I ran it unmodified. Credit for that score belongs to its authors.

## Data

Competition data is **not included**. The competition rules forbid redistributing it. To run the code, attach the competition dataset to a Kaggle notebook.
