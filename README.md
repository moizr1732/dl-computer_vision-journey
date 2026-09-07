# Deep Learning & Computer Vision Journey

This repository documents my hands-on journey into deep learning and computer vision — built from first principles alongside Andrew Ng's [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning), and applied to real, non-trivial problems rather than toy datasets.

## About me

I'm Moiz Ur Rehman, currently:
- Working through the **Agentic & Robotic AI Engineer** program at PIAIC, Air University, Islamabad
- Completing Andrew Ng's Deep Learning Specialization (currently on Course 3: Structuring ML Projects)
- Machine learning intern at FlyRank AI
- Freelancing in web development, automation, and applied AI/ML for B2B clients

This repo is where I turn what I'm learning into real, working projects — including the messy parts (debugging, iteration, and honest reporting of what didn't work), not just polished final numbers.

## Philosophy behind this repo

Most beginner ML portfolios show a single accuracy number and stop there. I'm trying to do something different here: every project in this repo aims to include —
- **Real methodology**, not just running a tutorial (proper train/dev/test splitting, avoiding data leakage, bias-variance analysis)
- **Honest evaluation**, including where and why a model fails, not just where it succeeds
- **A working demo**, so the project can actually be tried, not just read about
- **Applied Course 3 thinking** — structuring ML projects the way Andrew Ng's course teaches, applied to real, sometimes adversarial problems

## Projects

### 1. [Deepfake Face-Swap Detector](./deepfake-detection)

A deep learning model that detects face-swap deepfakes, built on FaceForensics++ and EfficientNet-B0 transfer learning. Includes a live Streamlit demo.

**The key finding**: the model hits 73.7% accuracy on the manipulation method it was trained on (Deepfakes), but drops to 45.2% accuracy on a manipulation method it's never seen (Face2Face) — a real, measured demonstration of a well-known generalization problem in deepfake detection, rather than a cherry-picked success number.

[→ Full project details, methodology, and results](./deepfake-detection/README.md)

---

*More projects will be added here as I progress through the specialization and take on new applied CV/DL challenges.*

## Tech stack across this repo

- **PyTorch** / **TensorFlow** — model development
- **Transfer learning** — EfficientNet, and other pretrained backbones as projects require
- **Computer vision tooling** — OpenCV, MTCNN/facenet-pytorch for face detection
- **Streamlit** — interactive demos for finished models
- **Google Colab** — GPU-accelerated training

## Connect

If you're reviewing this repo as part of a hiring process, freelance inquiry, or just want to discuss any of the projects — feel free to reach out via [LinkedIn] or [GitHub](https://github.com/moizr1732).
