# Training a YOLOv11 object detector

YOLOv11 is an object detector model which can be re-trained on custom data. 
To do so, the challenge data will have to be converted to the YOLO format. 

> [!IMPORTANT]  
> All the scripts from this baseline need to be run from the baselines/yolo folder (otherwise the configuration files 
> specified in relative paths will not be found)

## Create the environment using poetry
You will need to install poetry first. 
```bash
poetry install
```

## Preprocess dataset
To convert the dataset to the YOLO format, it needs to be pre-processed.
The following steps will guide you through the creation of a YOLO-format dataset.

First you will need to adjust the dataset_config.json parameters if desired. 
The options are to modify the spectrogram parameters: 

```json 
{
"duration" : 30,
"overlap" : 0.5,
"desired_fs" : 250,
"channel" : 0,
"log": false,
"color": false,

"nfft" : 512,
"win_len" : 256,
"hop_len" : 20,

"class_encoding": {"bma": 0,"bmb": 1, "bmz": 2, "bmd": 3, "bpd": 4, "bp20": 5, "bp20plus": 6}
}
```

If you want to already have the classes joined during training, specify: 

```json 
{
"class_encoding": {"bma": 0,"bmb": 0, "bmz": 0, "bmd": 1, "bpd": 1, "bp20": 2, "bp20plus": 2}
}
```

Then you will have to run
1. Run the preprocess_data.py script and pass the folder to the training folder. 
When asked if it is for training, answer yes (y). This will only select and process a proportion of background images. 
2. Run the preprocess_data.py script and pass the folder to the validation folder 
When asked if it is for training, answer yes (y). This will only select and process a proportion of background images. 
3. Run the preprocess_data.py script and pass the folder to the test folder
When asked if it is for training, answer no (n). This will process all the background images. 

This will create a dataset for the validation which does not include all the backgrounds. Therefore, this dataset should NOT be used to evaluate the performance on the dataset. 
In case you want to evaluate the performance on the validation dataset for real, (or you want to train with the full validation set), you should answer (n) when creating the validation set.


## Train the YOLO model
First, adjust the paths on the custom.yaml file to point to the train and valid folders.
You will need to specify (see custom.yaml and custom_joined.yaml files): 
```yaml
path:  './biodcase'
train: './biodcase/biodcase_development_set/train'
test: './biodcase/biodcase_evaluation_set'
val: './biodcase_development_set/validation'

#classes names
nc: 3

#replace all class names list with your classes names
names:
  0: bmabz
  1: d
  2: bp
```

> [!IMPORTANT]  
> Make sure the names match the ones in your configuration file. If you are using the joined strategy in the json 
> configuration file, you should also have them in the yaml file. If you're using the 7 classes, make sure you list them
> accordingly and you change the nc to 7 in the yaml file

Two yaml config files examples are provided, the custom.yaml and the custom_joined.yaml to use with or without the 
joining strategey.

Then, run the train_yolo.py script 
```bash
python train_yolo.py 
```
This will generate the automatic YOLO output in the folder specified in the "path" argument of the yaml config file. 


## Predict and convert the YOLO output to the required format
To be able to compare the YOLO results with the ground truth annotations we will need to re-process the obtained results. 
To do that, run the convert_yolo_output.py script 

```bash 
python predict_and_convert_yolo_output.py 
```

When asked, point to the evaluation folder path and the model which should be used for evaluation. 
