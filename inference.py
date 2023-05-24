import os
import torch

from super_gradients.training import models
from torchinfo import summary

device = 'cuda' if torch.cuda.is_available() else "cpu"

image_flag = False
video_flag = True

yolo_nas_l = models.get("yolo_nas_l", pretrained_weights="coco")

summary(model=yolo_nas_l, 
        input_size=(16, 3, 640, 640),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"],
        verbose=False
)

if image_flag:
        url = "https://previews.123rf.com/images/freeograph/freeograph2011/freeograph201100150/158301822-group-of-friends-gathering-around-table-at-home.jpg"
        yolo_nas_l.predict(url, conf=0.25).show()


if video_flag:
        video_id_list = ["HgFGjJqmU0U", "KCsPNKjHFuY", "zKDfhGpRy3w"]

        for video_id in video_id_list:
                video_url = f'https://www.youtube.com/watch?v={video_id}'  
                command = f"python3 -m youtube_dl -f 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4' {video_url} -o 'videos/%(title)s.%(ext)s'"
                os.system(command)
        
        videos = [file for file in os.listdir() if file.endswith("mp4")]
        
        for video in videos:
                input_video_path = video
                output_video_path = f"detections/{video}.mp4"

                yolo_nas_l.to(device).predict(input_video_path).save(output_video_path)
