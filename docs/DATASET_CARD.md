# CBC-12,618 dataset card (as used in this study)

## Structure
| Attribute | Value |
|:--|:--|
| File | `data/hematology_cbc_dataset.csv` (SHA-256 in `data/README.md`) |
| Records | 12,618 |
| Columns | 23 numerical CBC measurements + 1 label (`Disorder_Class`) |
| Missing numerical entries | 18,247 of 290,214 (6.29%) |
| Exact duplicate rows | 0 |
| Identifier-like columns | none found (audit §2.1) |
| Columns > 90% missing | none (all 23 features retained) |

## Class distribution (Table 1)
| Category | Records | Share |
|:--|--:|--:|
| Microcytic Hypochromic Anaemia | 2,994 | 23.73% |
| Normal / Non-specific | 2,541 | 20.14% |
| Mild-Moderate Leukocytosis | 2,501 | 19.82% |
| Thrombocytopaenia | 2,468 | 19.56% |
| Severe Leukocytosis Flag | 2,114 | 16.75% |

## Variables
White-cell parameters (WBC, five absolute differential counts, five differential percentages), red-cell parameters (RBC, haemoglobin, HCT, MCV, MCH, MCHC, RDW), platelet parameters (platelet count, MPV, PCT, PDW) and ESR. See [`data/data_dictionary.csv`](../data/data_dictionary.csv). Missingness ranges from 0% (HCT, MCV, MCH, MCHC, RDW, MPV, PCT, PDW) to 14.56% (Basophil %); it is handled by median imputation with missing-indicator flags fitted inside each training fold.

## Interpretation and quality notes (from the audit, `results/audit/`)
* **Arithmetic consistency (Table S4):** HCT ≈ RBC × MCV / 10 and MCH ≈ Hb / RBC × 10 hold with Pearson r = 1.000 (median relative error ≤ 0.0002); MCHC ≈ Hb / HCT × 100 gives r = 0.939; differential percentages sum to 100 on complete rows.
* **WBC recoverability (Table S5):** each absolute differential count divided by its percentage reproduces WBC (r ≥ 0.997), so WBC magnitude is encoded redundantly across features.
* **Label definition (Table S6):** *Severe Leukocytosis Flag* is separable by WBC alone (AUC = 1.000); *Microcytic Hypochromic Anaemia* by MCV (0.976) and *Thrombocytopaenia* by platelet count (0.974).
* The labels are CBC-pattern categories; high scores on a threshold-defined class describe the class definition rather than learned pathophysiology.

## Provenance and stewardship
See [DATA_GOVERNANCE.md](DATA_GOVERNANCE.md); provenance, ethics and licence statements are to be completed by the authors.
