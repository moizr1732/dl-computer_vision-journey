\# Deepfake Face-Swap Detector



A deep learning model that detects \*\*face-swap style deepfakes\*\* in images, built with transfer learning on EfficientNet-B0 and trained on the FaceForensics++ dataset. Includes a live Streamlit demo and an honest evaluation of how well the model generalizes to manipulation types it wasn't trained on.



\## What this is (and isn't)



This is a \*\*binary classifier\*\*: given a face image, it predicts whether the face has been altered using a \*\*face-swap\*\* technique (identity replacement), or is authentic/unaltered. It does \*\*not\*\* measure how "human-looking" or realistic an image is, and it is not a general-purpose AI-generated-image detector — it is specifically trained on and tuned for face-swap manipulation detection.



\## Key finding: cross-manipulation generalization



The headline result of this project isn't the accuracy number — it's what happens when the model faces a manipulation method it's never seen:



| Test set | Manipulation type | Accuracy | Recall (catching actual fakes) |

|---|---|---|---|

| Same-method test set | Deepfakes (trained on this) | \*\*73.7%\*\* | 74.0% |

| Unseen manipulation test | Face2Face (never seen in training) | \*\*45.2%\*\* | 37.9% |



A model trained only on Deepfakes-style face-swaps loses roughly \*\*28 points of accuracy\*\* and misses \*\*6 out of 10 fakes\*\* when tested on Face2Face, a manipulation method that alters expressions rather than swapping identity. Qualitative review of the missed cases shows sharp, well-lit, artifact-free faces — consistent with the idea that the model learned to detect face-swap \*blending seams\* specifically, a signature Face2Face manipulations don't produce (since it's the same identity, just reenacted).



This is a known, documented limitation in deepfake detection research (not a bug), and it's exactly the kind of result most beginner projects never test for.



\## Methodology



1\. \*\*Dataset\*\*: \[FaceForensics++](https://github.com/ondyari/FaceForensics) — real YouTube videos + Deepfakes/Face2Face manipulated videos, c40 compression.

2\. \*\*Face extraction\*\*: MTCNN detects and crops faces from sampled video frames (`src/extract\_faces.py`).

3\. \*\*Splitting\*\*: Train/dev/test splits are done \*\*by video ID\*\*, not by image, to prevent the same identity leaking across splits (`src/make\_splits.py`).

4\. \*\*Model\*\*: EfficientNet-B0 (ImageNet-pretrained), fine-tuned via transfer learning (`src/train.py`). Three configurations were compared:

&#x20;  - Frozen backbone (classification head only): 70.3% dev accuracy

&#x20;  - \*\*1-block fine-tuned (chosen production model): 74.7% dev accuracy\*\* — best balance of accuracy and stability

&#x20;  - 2-block fine-tuned: 75.7% dev accuracy but severely overfit (train accuracy hit 95%, dev loss diverged) — rejected in favor of the more stable 1-block model

5\. \*\*Threshold tuning\*\*: default 0.5 cutoff was compared against alternatives; 0.45 gives the best precision/recall balance (F1 = 0.763).

6\. \*\*Generalization test\*\*: the trained model was evaluated on Face2Face, an unseen manipulation method, to measure real-world robustness (see table above).



\## Results summary



\- \*\*Precision (fake, same-method)\*\*: \~79%

\- \*\*Recall (fake, same-method)\*\*: \~74–82% depending on threshold

\- \*\*Precision (fake, cross-method)\*\*: 80.6%

\- \*\*Recall (fake, cross-method)\*\*: 37.9% ← the key limitation



\## Demo



A Streamlit app (`demo/app.py`) provides live inference:

\- Upload any photo — MTCNN automatically detects and crops the face

\- Color-coded REAL/FAKE verdict with a plain-language confidence explanation

\- Labeled Real↔Fake confidence bar

\- Built-in explanation of what a deepfake is and the model's known limitations



\### Running the demo



```bash

python -m venv venv

venv\\Scripts\\activate       # Windows

pip install -r requirements.txt

python -m streamlit run demo/app.py

```



\## Project structure



```

deepfake-detection/

├── src/

│   ├── extract\_faces.py    # MTCNN face extraction from videos

│   ├── make\_splits.py      # Train/dev/test split by video ID (no leakage)

│   ├── dataset.py          # PyTorch Dataset class

│   └── train.py            # Transfer learning training script

├── demo/

│   ├── app.py               # Streamlit inference app

│   └── best\_model\_v2\_moredata.pt   # Trained model weights

├── download.py              # FaceForensics++ dataset download script

└── requirements.txt

```



\## Limitations



\- Trained on a single manipulation method (Deepfakes); generalization to other methods (Face2Face, FaceSwap, NeuralTextures) is measurably weaker, as documented above.

\- Trained on c40 (compressed) video quality only.

\- Dataset size (2,000 real + 2,000 fake images) is modest; more data and/or multi-manipulation training would likely improve both accuracy and generalization.

\- A natural extension: training on multiple manipulation methods simultaneously, which is known in the literature to improve cross-method generalization.



\## Built as part of



Andrew Ng's Deep Learning Specialization (Course 3: Structuring Machine Learning Projects) — this project applies train/dev/test methodology, bias-variance analysis, and error analysis principles from that course to a real, adversarial computer vision problem.

