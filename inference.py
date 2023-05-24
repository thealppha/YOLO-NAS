import os
import torch

from super_gradients.training import models
from torchinfo import summary

class Inference():
        
        def __init__(self, model, video_id):
                self.model = model             
                self.video_id = video_id

        def get_device(self):
                device = 'cuda' if torch.cuda.is_available() else "cpu"

                return device

        def get_model(self):
                model = models.get(self.model, pretrained_weights="coco")

                summary(model=model, 
                        input_size=(16, 3, 640, 640),
                        col_names=["input_size", "output_size", "num_params", "trainable"],
                        col_width=20,
                        row_settings=["var_names"],
                        verbose=False
                )
                
                return model
        
        def get_video(self):
                video_url = f'https://www.youtube.com/watch?v={self.video_id}'  

                command = f"python3 -m youtube_dl -f 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4' {video_url}"
                os.system(command)

                video_name = [file for file in os.listdir() if file.endswith("mp4")][0]

                return video_name

        def main(self):
                device = self.get_device()
                video = self.get_video()
                model = self.get_model()

                input_video_path = video
                output_video_path = f"detection/{video}"

                model.to(device).predict(input_video_path).save(output_video_path)

if __name__ == "__main__":
        inference = Inference("yolo_nas_l", "HgFGjJqmU0U")
        inference.main()

        os.remove([file for file in os.listdir() if file.endswith("mp4")][0])