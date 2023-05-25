## YOLO-NAS

YOLO-NAS is a new State of the Art, foundation model for object detection inspired by YOLOv6 and YOLOv8.

Here's why you've got to give it a try:

🧱 New Quantization-Friendly Block: Improving on previous models, YOLO-NAS features a novel basic block that's tailor-made for quantization.

🚀 Advanced Training Scheme: YOLO-NAS undergoes pre-training on the Object365 dataset, leverages pseudo-labeled data, and benefits from knowledge distillation using a pre-trained teacher model.

🎯 Post-Training Quantization (PTQ): The network is converted to INT8 after training, making it even more efficient.

🧬 AutoNac Optimized: Three final networks emerge from applying AutoNac on the architecture space, all while using the equivalent GPU time of training just 5 networks.

💾 Pre-Trained on Top Datasets: YOLO-NAS comes pre-trained on COCO, Objects365, and Roboflow 100, setting you up for success in downstream object detection tasks.

Deci's open-source, PyTorch-based computer vision library, SuperGradients, makes YOLO-NAS easy to train, and has advanced techniques like Distributed Data Parallel, Exponential Moving Average, Automatic Mixed Precision, and Quantization Aware Training.

🌟 Head over to GitHub, star SuperGradients, and explore the starter notebook to see YOLO-NAS in action!

## Installation
```ruby
# Create environmet 
conda create -n yolonas python=3.8
conda activate yolonas

# Install packages and dependencies
chmod +x install.sh
bash ./install.sh

# Run inference.py
python3 inference.py
```

<img src="https://github.com/TunaUlusoy/YOLO-NAS/blob/master/readme.png" alt= “http://url/to/img.png” width="500" height="500">

## Links

**Technical blog**: https://deci.ai/blog/YOLO-NAS-object-detection-foundation-model <br />
**GitHub repo**: https://github.com/Deci-AI/super-gradients/blob/master/YOLONAS.md <br />
**Starter Notebook**: https://colab.research.google.com/drive/1q0RmeVRzLwRXW-h9dPFSOchwJkThUy6d
