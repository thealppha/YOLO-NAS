import os

from super_gradients.training import Trainer
from super_gradients.training import models
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
from super_gradients.training.dataloaders.dataloaders import coco_detection_yolo_format_val

from params import dataset_params, train_params

class Evaluation():
    def __init__(self, model, experiment_name, ckpt_root_dir, checkpoint_path):
        self.model = model
        self.experiment_name = experiment_name
        self.ckpt_root_dir = ckpt_root_dir
        self.checkpoint_path = checkpoint_path

    def get_best_model(self):
        best_model = models.get(self.model,
                                num_classes=len(dataset_params['classes']),
                                checkpoint_path=self.checkpoint_path)
                
        return best_model

    def get_test_data(self):
        test_data = coco_detection_yolo_format_val(
                        dataset_params={
                            'data_dir': dataset_params['data_dir'],
                            'images_dir': dataset_params['test_images_dir'],
                            'labels_dir': dataset_params['test_labels_dir'],
                            'classes': dataset_params['classes']
                        },
                        dataloader_params={
                            'batch_size':train_params["batch_size"],
                            'num_workers':train_params["num_workers"]
                        }
                    )
        return test_data
    
    def evaluate_with_test_data(self, best_model, trainer, test_data):
        evaluation_result = trainer.test(model=best_model,
                            test_loader=test_data,
                            test_metrics_list=DetectionMetrics_050(score_thres=0.1, 
                                                            top_k_predictions=300, 
                                                            num_cls=len(dataset_params['classes']), 
                                                            normalize_targets=True, 
                                                            post_prediction_callback=PPYoloEPostPredictionCallback(score_threshold=0.01, 
                                                                                                                    nms_top_k=1000, 
                                                                                                                    max_predictions=300,                                                                              
                                                                                                                    nms_threshold=0.7)
                                                            ))
        
        text = "Evaluation Result with Test Data"
        padding = "#" * 42
        result = f"{padding} {text} {padding}"
        print(f"\n{result}\n\n", evaluation_result)

    def main(self):
        trainer = Trainer(experiment_name=self.experiment_name, ckpt_root_dir=self.ckpt_root_dir)

        best_model = self.get_best_model()
        test_data = self.get_test_data()

        self.evaluate_with_test_data(best_model=best_model, trainer=trainer, test_data=test_data)