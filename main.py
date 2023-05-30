import os

from inference import Inference
from train import Train
from evaluation import Evaluation

# Select init model
model = "yolo_nas_l"

# Train, Evaluation, Inference
operation = "Inference" 

# Youtube video id
video_id="DSxNv_nKW4Y" 

# Root checkpoint directory
ckpt_root_dir = "checkpoints/"

# Experiment_name
ckpt_path = "checkpoints/yolonas_train_01/ckpt_best.pth"

# Create checkpoints folder
if not os.path.exists(os.path.join(os.getcwd(), ckpt_root_dir)):
        os.mkdir(ckpt_root_dir)

# Create dataset folder
if not os.path.exists(os.path.join(os.getcwd(), "dataset/")):
        os.mkdir("dataset/")

# Operations
if operation == "Train":
        train = Train(model=model, ckpt_root_dir=ckpt_root_dir)
        train.main()

elif operation == "Evaluation":
        evaluation = Evaluation(model=model, experiment_name="yolonas_train_1", ckpt_root_dir=ckpt_root_dir, checkpoint_path=ckpt_path)
        evaluation.main()

elif operation == "Inference":
        inference = Inference(model=model, video_id=video_id, checkpoint_path=ckpt_path) 
        inference.main()