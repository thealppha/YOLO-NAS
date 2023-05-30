import os

from super_gradients.training import Trainer
from super_gradients.training.dataloaders.dataloaders import coco_detection_yolo_format_train, coco_detection_yolo_format_val
from super_gradients.training import models

from params import dataset_params, train_params

class Train():
    def __init__(self, model, ckpt_root_dir):
        self.model = model
        self.ckpt_root_dir = ckpt_root_dir

    def get_train_data(self):
        train_data = coco_detection_yolo_format_train(
                        dataset_params={
                            'data_dir': dataset_params['data_dir'],
                            'images_dir': dataset_params['train_images_dir'],
                            'labels_dir': dataset_params['train_labels_dir'],
                            'classes': dataset_params['classes']
                        },
                        dataloader_params={
                            'batch_size':train_params["batch_size"],
                            'num_workers':train_params["num_workers"]
                        }
                    )
        return train_data
    
    def get_val_data(self):
        val_data = coco_detection_yolo_format_train(
                        dataset_params={
                            'data_dir': dataset_params['data_dir'],
                            'images_dir': dataset_params['val_images_dir'],
                            'labels_dir': dataset_params['val_labels_dir'],
                            'classes': dataset_params['classes']
                        },
                        dataloader_params={
                            'batch_size':train_params["batch_size"],
                            'num_workers':train_params["num_workers"]
                        }
                    )
        return val_data

    def get_model(self):
        model = models.get(self.model, 
                        num_classes=len(dataset_params['classes']), 
                        pretrained_weights="coco")

        return model

    def main(self): 
        checkpoint_list = os.listdir(os.path.join(os.getcwd(), "checkpoints/"))

        if len(checkpoint_list) == 0:
            experiment_name = "yolonas_train_01"
        else:
            last_experiment = max(checkpoint_list)
            train_number = int(last_experiment.split("_")[-1]) + 1
            experiment_name = f"yolonas_train_{train_number:02}"

        train_data = self.get_train_data()
        val_data = self.get_val_data()

        model = self.get_model()

        trainer = Trainer(experiment_name=experiment_name, ckpt_root_dir=self.ckpt_root_dir)

        trainer.train(model=model, 
                      training_params=train_params, 
                      train_loader=train_data, 
                      valid_loader=val_data
                      )