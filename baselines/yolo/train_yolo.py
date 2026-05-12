try:
    import comet_ml
except ModuleNotFoundError:
    comet_ml = None
import yaml
import torch
from ultralytics import YOLO
import os


def run():
    YAML_FILE = './baselines/yolo/custom_joined.yaml'  # Set to './baselines/yolo/custom.yaml' to train with all the classes (instead of joint ones)
    run_name = 'biodcase_baseline' # Change to the name of your run

    # Check if CUDA is available
    print('CUDA device count:')
    print(torch.cuda.device_count())

    # Read the config file
    with open(YAML_FILE, 'r') as file:
        config = yaml.safe_load(file)

    if "COMET_API_KEY" in os.environ and comet_ml is not None:
        experiment = comet_ml.Experiment(
            project_name="biodcase",
        )

    # Load a model
    model = YOLO('yolo11s.pt')

    # train the model
    best_params = {
        'iou': 0.3,
        'imgsz': 640,
        'hsv_s': 0,
        'hsv_v':  0,
        'degrees': 0,
        'translate': 0,
        'scale': 0,
        'shear': 0,
        'perspective': 0,
        'flipud': 0,
        'fliplr': 0,
        'bgr': 0,
        'mosaic': 0,
        'mixup': 0,
        'copy_paste': 0,
        'erasing': 0,
        'crop_fraction': 0,
    }
    model.train(epochs=20, batch=32, data=YAML_FILE, seed=24,
                project=config['path'] + '/runs/' + run_name, resume=False, **best_params, device=0)

    if "COMET_API_KEY" in os.environ and comet_ml is not None:
        experiment.end()


if __name__ == '__main__':
    run()
