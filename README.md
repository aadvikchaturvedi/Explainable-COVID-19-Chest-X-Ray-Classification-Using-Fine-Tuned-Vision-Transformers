# Explainable COVID-19 Chest X-Ray Classification with a Fine-Tuned DeiT

A Vision Transformer (DeiT-base) fine-tuned to classify chest X-rays as **COVID-19**, **Normal** or
**Viral Pneumonia**, plus a comparison of two saliency methods with quantitative faithfulness
metrics (entropy, deletion/insertion AUC, AOPC).

**Result:** 96.78% accuracy on a held-out test set of 435 images (145 per class).

## Pipeline

```
PNG chest X-ray
  -> resize 224x224, ImageNet normalization (+ flip / rotate ±10° / brightness jitter in training)
  -> DeiT-base/16 (timm deit_base_patch16_224, ImageNet-pretrained)
       blocks 0-7 frozen, blocks 8-11 + norm layers + 3-class head trained
  -> softmax over {COVID, Normal, Viral Pneumonia}
  -> saliency: input-gradient map and Grad-CAM on the last block
  -> faithfulness: entropy, deletion / insertion curves, AOPC
```

| Script | What it does | Outputs |
|---|---|---|
| `train.py` | fine-tunes DeiT-base, keeps the checkpoint with the best val accuracy | `deit_best.pth`, `training_curves.png` |
| `evaluation_scores.py` | scores `deit_best.pth` on the test split | `evaluation_metrics.{txt,json,csv}`, `confusion_matrix_eval.png` |
| `explainable_model.py` | saliency maps for selected test images and their faithfulness metrics | `explainability_comparison.png`, `explainability_curves.png`, `explainability_results.{txt,json,csv}` |
| `xai_evaluation.py` | metric functions: `calculate_entropy`, `deletion_insertion_aopc` | |

## Data

`COVID_19_dataset/` in `ImageFolder` layout, balanced across classes:

| Split | COVID | Normal | Viral Pneumonia | Total |
|---|---:|---:|---:|---:|
| train | 1000 | 1000 | 1000 | 3000 |
| val | 200 | 200 | 200 | 600 |
| test | 145 | 145 | 145 | 435 |

## Model and training

- **Backbone**: `deit_base_patch16_224` from timm, ImageNet-pretrained, 16x16 patches
  (196 patch tokens + CLS), new 3-class linear head.
- **Partial fine-tuning**: all parameters frozen, then transformer blocks 8-11, all LayerNorms
  and the head unfrozen. The early blocks keep their generic ImageNet features.
- **Loss**: cross-entropy with inverse-frequency class weights (the training split is balanced, so
  the weights come out equal; the code handles imbalanced splits).
- **Optimizer**: AdamW, lr 3e-4, weight decay 1e-4, cosine annealing to 1e-6 over 25 epochs,
  batch size 64, gradient clipping at 1.0.
- **Model selection**: the checkpoint with the best validation accuracy is saved. Validation
  accuracy peaked at about 0.978 at epoch 6 and stayed between roughly 0.925 and 0.965 afterwards.
- Runs on Apple MPS, CUDA or CPU (picked automatically).

![Training Curves](training_curves.png)

## Results (test set, 435 images)

From `evaluation_metrics.txt`:

| Class | Precision | Recall | F1 | Support |
|---|:---:|:---:|:---:|:---:|
| COVID | 0.9474 | 0.9931 | 0.9697 | 145 |
| Normal | 0.9781 | 0.9241 | 0.9504 | 145 |
| Viral Pneumonia | 0.9795 | 0.9862 | 0.9828 | 145 |
| **Macro avg** | **0.9683** | **0.9678** | **0.9676** | **435** |

Overall accuracy **0.9678**. COVID recall is the highest (0.9931); most errors are Normal images
predicted as another class (Normal recall 0.9241).

![Confusion Matrix](confusion_matrix.png)

## Explainability

### Methods

1. **Grad-CAM** (`pytorch-grad-cam`) on `blocks[-1].norm1`. A `reshape_transform` drops the CLS
   token and reshapes the 196 patch tokens into a 14x14 grid, so the CNN-style method works on
   the transformer's token sequence.
2. **Input-gradient saliency**, labelled "Attention Rollout" in the code and figures. The function
   `get_attention_rollout` does not multiply attention matrices across layers; it backpropagates
   the predicted class logit to the input and takes the channel-mean absolute gradient. It is a
   vanilla-gradient saliency map, which explains the high-frequency, patch-grid texture in the
   middle column below.

![Explainability Comparison](explainability_comparison.png)

*Columns: original | input-gradient map ("Attention Rollout") | Grad-CAM overlay. Rows are
outlined green for correct and red for wrong predictions.*

### Faithfulness metrics (`xai_evaluation.py`)

| Metric | How it is computed | Better |
|---|---|---|
| Entropy | Shannon entropy of the heatmap normalized to a distribution | lower = more focused |
| Deletion AUC | sort pixels by saliency, zero the top 5% at a time over 20 steps, average the target-class confidence | lower |
| Insertion AUC | start from a blank image, add pixels in saliency order, average the confidence | higher |
| AOPC | mean drop from the original confidence over the deletion steps | higher |

Results from `explainability_results.json` (6 test images: 3 correct and 3 wrong predictions,
drawn from a random 100-image scan):

| Method | Avg mean activation | Entropy | Deletion AUC | Insertion AUC | AOPC |
|---|---:|---:|---:|---:|---:|
| Input gradient ("Attention Rollout") | 0.0327 | 10.2065 | 0.4606 | 0.3533 | 0.0000 |
| Grad-CAM | 0.1650 | 10.2044 | 0.4606 | 0.4218 | 0.0000 |

Grad-CAM has the higher insertion AUC (0.4218 vs 0.3533): adding pixels in Grad-CAM order recovers
the predicted class faster. Entropy is almost identical for the two methods.

![Insertion and deletion curves](explainability_curves.png)

**Read these numbers with care.** They come from one small run and have problems that should be
fixed before drawing conclusions:

- In the saved results, every image's deletion AUC equals its original confidence and AOPC is
  exactly 0 for both methods, which means the recorded deletion curves were flat. The committed
  `explainability_curves.png` shows deletion curves that do drop (from about 0.46 to 0.31), so the
  plot and the result files do not agree; they were likely produced by different runs.
- Prediction confidences in this run are 40-53%, close to chance for 3 classes. That does not fit
  a model with 96.78% test accuracy. `explainable_model.py` falls back to ImageNet weights with an
  untrained head when `deit_best.pth` is missing (it prints a warning and continues), so this run
  may not have used the fine-tuned checkpoint.
- Six images is too few for a method comparison.

## Setup and usage

```bash
git clone https://github.com/aadvikchaturvedi/Explainable-COVID-19-Chest-X-Ray-Classification-Using-Fine-Tuned-Vision-Transformers.git
cd Explainable-COVID-19-Chest-X-Ray-Classification-Using-Fine-Tuned-Vision-Transformers
pip install -r requirements.txt

python train.py                 # writes deit_best.pth and training_curves.png
python evaluation_scores.py     # test-set metrics and confusion matrix
python explainable_model.py     # saliency figure, curves and XAI metrics
```

`deit_best.pth` is not committed, so run `train.py` first. The scripts expect to be run from the
repository root with `COVID_19_dataset/{train,val,test}/<class>/` present. `train.py` calls the
macOS `caffeinate` command to keep the laptop awake; on other systems it prints a warning and
continues.

## Limitations and next steps

- Rename the input-gradient method or implement real attention rollout (fuse the per-layer
  attention matrices with the residual identity, e.g. via forward hooks on `attn_drop`).
- Re-run `explainable_model.py` with the fine-tuned checkpoint loaded (fail instead of falling
  back), on a larger stratified sample, and check why the deletion scores in the saved results are flat.
- Deletion fills removed pixels with 0 in normalized space (the dataset mean colour, not black);
  a blurred baseline is a common alternative.
- No patient-level split information is available, so leakage between splits cannot be ruled out.
  The model is a course/research exercise and not validated for clinical use.

---
Created for the DTU Explainable Vision task. Author: Aadvik Chaturvedi.
