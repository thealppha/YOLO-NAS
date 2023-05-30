import os
import torch

from super_gradients.training import models
from params import dataset_params

class Inference():
        def __init__(self, model, video_id, checkpoint_path):
                self.model = model             
                self.video_id = video_id
                self.checkpoint_path = checkpoint_path

        def get_device(self):
                device = 'cuda' if torch.cuda.is_available() else "cpu"

                return device

        def get_model(self):
                model = models.get(self.model,
                                num_classes=len(dataset_params['classes']),
                                checkpoint_path=self.checkpoint_path)
                              
                return model
        
        def get_video(self):
                video_url = f'https://www.youtube.com/watch?v={self.video_id}'  

                command = f"python -m youtube_dl --format mp4 -o 'prediction/input/{self.video_id}.mp4' {video_url}"
                os.system(command)

                video_name = f'{self.video_id}.mp4'

                return video_name

        def main(self):
                if not os.path.exists(os.path.join(os.getcwd(), "prediction/input")):
                        os.makedirs("prediction/input")

                if not os.path.exists(os.path.join(os.getcwd(), "prediction/output")):
                        os.makedirs("prediction/output")

                device = self.get_device()
                video = self.get_video()
                model = self.get_model()

                input_video_path = f"prediction/input/{video}"
                output_video_path = f"prediction/output/{video}"

                model.to(device).predict(input_video_path).save(output_video_path)