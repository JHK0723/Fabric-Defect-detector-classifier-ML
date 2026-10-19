# Fabric Defect Detector + Classifier

Classify fabric defects into **hole, stain, broken_thread, irregular_yarn** (plus **non_defective**) with YOLO11.

This repo contains a trained **patch classifier** (`models/fabric_classifier_best.pt`), the code that builds its training
data from two public datasets, my corrected labels, and the scripts to retrain it on Google Colab.
An oriented-box **detector** (YOLO11-OBB) is also set up, but **it has not been trained yet** (see "Not done yet").

## Result (classifier)

`yolo11s-cls`, 256x256 grayscale patches, 45 epochs on a Colab T4. Evaluated on **558 held-out test patches** that were
never used for training or model selection. The numbers were reproduced on a second machine (CPU) from the saved weights.

| Class | Precision | Recall | F1 | Test patches |
|---|---|---|---|---|
| broken_thread | 0.966 | 1.000 | 0.983 | 28 |
| hole | 0.953 | 0.946 | 0.949 | 129 |
| irregular_yarn | 0.941 | 0.931 | 0.936 | 102 |
| non_defective | 0.910 | 0.942 | 0.926 | 139 |
| stain | 0.942 | 0.919 | 0.930 | 160 |
| **overall accuracy** | | | **0.937** | 558 |

By source: Bangladesh set 0.935 (n=199), TILDA 0.939 (n=359). Validation accuracy was 0.938, so there is no sign of overfitting.

Confusion matrix (rows = true, columns = predicted):

| | broken_thread | hole | irregular_yarn | non_defective | stain |
|---|---|---|---|---|---|
| broken_thread | 28 | 0 | 0 | 0 | 0 |
| hole | 0 | 122 | 6 | 0 | 1 |
| irregular_yarn | 0 | 4 | 95 | 1 | 2 |
| non_defective | 1 | 1 | 0 | 131 | 6 |
| stain | 0 | 1 | 0 | 12 | 147 |

The most common mistake is a faint stain called clean (12 of 160).

### Please read before relying on these numbers

* **This is in-distribution accuracy.** The test patches come from the same two datasets as the training patches. It does not
  tell you how the model performs on a different camera, lighting or fabric. Measure that on your own images before trusting it.
* **It classifies 256x256 patches, not whole frames.** Scanning a whole image with a sliding window works (`infer_classifier.py`)
  but gives **false alarms near cloth edges and seams**: a vertical cloth edge looks like a broken thread to the model.
  Mask the cloth edges (region of interest) and require a defect to persist over several frames.
* **`broken_thread` is the easy subset.** Only 228 images could be labelled (see below) and only 28 test patches exist, so the
  recall of 1.00 is a weak claim for faint broken threads or checked/plaid fabrics.
* **Grayscale.** The model was trained on grayscale images. Convert input to gray first (`infer_classifier.py` and
  `vision/detector.py` do this). Colour was removed on purpose: otherwise the model could recognise the *source dataset*
  (e.g. TILDA is grayscale, the Bangladesh set is colour) instead of the defect.
* The `non_defective` patches are clean regions cut from the **same images** that contain defects, for the same reason.

## Use the model

```bash
pip install -r requirements.txt
python infer_classifier.py path/to/image.jpg --save annotated.jpg      # scan a whole image
python infer_classifier.py patch.jpg --patch                          # one 256x256 patch -> class probabilities
```

Classes: `broken_thread, hole, irregular_yarn, non_defective, stain`.

`vision/detector.py` is a `FabricDetector` class (works with normal and oriented-box YOLO models, converts frames to gray)
for a live camera pipeline.

## Data

The datasets are **not** stored here. Both are CC BY 4.0 on Mendeley Data:

| Source | Used for | Reference |
|---|---|---|
| Fabric Anomaly Recognition Dataset (Hamza, Tabassum, Hossain, Sara, Hossain; NITER) | hole, stain (oil), broken_thread | Mendeley Data, v1, 2025, doi:10.17632/h2ytgy2pmb.1 |
| TILDA-OBB (Md Riadul Islam; oriented boxes on the TILDA texture database) | hole (+cut), stain, irregular_yarn (thread defect) | Mendeley Data, v1, 2026, doi:10.17632/8y46pm2m6y.1 |

Changes made: images converted to grayscale; patches cropped; classes remapped (TILDA: cut and hole -> `hole`,
stain -> `stain`, thread defect -> `irregular_yarn`, foreign object images dropped); the Bangladesh labels replaced as
described below.

### Why the original Bangladesh labels were replaced

* **Hole:** the shipped boxes mostly did not sit on the hole (median IoU 0.15 against the re-drawn boxes; 220 of 649 had
  almost no overlap with them). All 649 images were re-boxed by eye in two passes (`data/label_pack.json.gz`).
* **Oil (-> stain):** the dataset has no oil labels. 564 images with a clearly visible stain were boxed by eye; 278 images
  with no stain I could box with confidence were left out.
* **Broken end (-> broken_thread):** the shipped boxes do not sit on the thread (only 58 of 664 within 15 px of the line a detector
  finds). `training/broken_end_autolabel.py` finds the real vertical line per image with a simple high-pass column profile
  stacked over the image height. 228 images pass strict checks (verified visually on 48 random ones: 48/48 correct).
  Checked fabrics and cloth-edge frames are skipped.
* All of these were boxed by a single annotator and have **not been independently audited**; expect errors of a few percent of
  the image size.

`data/label_pack.json.gz` holds these labels (keys are the original image names, values are YOLO `class xc yc w h`).

## Retrain on Google Colab

1. Create the Drive folder `MyDrive/fabric_models` and put in it: `data/label_pack.json.gz`, `training/build_dataset.py`
   and `colab/run_classifier.py` (and `colab/run_detector.py` for the detector).
2. Open `colab/fabric_classifier.ipynb` in Colab, select a **T4 GPU**, run all cells (about 15 minutes).
   The notebook downloads the two datasets from Mendeley, builds the data, trains, prints the test report and saves
   the weights to `fabric_models/cls/`.

`training/make_label_pack.py` rebuilds the label pack from the corrected label folders and the auto-label JSON.

## Not done yet

* **The OBB detector (`colab/run_detector.py`) has not been run.** Its dataset step is the same code that built the classifier
  data (tested), but training and evaluation of the detector are untested. The data it would use: 2,601 labelled images
  (1,237 hole, 1,202 stain, 228 broken_thread, 523 irregular_yarn boxes).
* **No data from the intended camera rig.** Collecting some, with clean (defect-free) frames, is the most valuable next step.
* More `broken_thread` and `irregular_yarn` examples on varied fabrics.

## Licence note

Model weights are fine-tuned from Ultralytics YOLO11 (AGPL-3.0). Datasets keep their own CC BY 4.0 licences and must be
credited as above. No licence has been chosen for the code in this repo yet.
