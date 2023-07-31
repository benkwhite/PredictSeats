# Airlines Seats Prediction based on Route and Quarter
## Introduction
This project is to predict the number of seats that will be sold for a given route and quarter. The model used the RNNs neural network to train the data. The data is from Diio Mi.

## How to train the model
1. Download the data from Diio Mi and put it in the data folder, including airline performance data in previous quarters and airline schedule data in the future quarters. This inlcudes:
   - Schedule_Monthly_Summary_Report_Conti.csv
   - Airline_Performance_Report_Conti.csv
   - Schedule_Monthly_Summary_2023Q1234.csv

2. Put the code files in a new folder. It could be a local folder, a folder in Colab, or a folder in Azure Databricks. The nessary files are:
   - RNN_model.py
   - RNN_apply_ind.py
   - parameters.json

3. Adjust the parameters in `parameters.json` as needed. 

## Running platform
### On local machine (not recommended, very slow, but can be used to test the code)

1. Run the code the directly or create a new Jupyter Notebook and run the code in the notebook.

```python
    # Set basic parameters
    folder_path = r'C:\Users\qilei.zhang\OneDrive - Frontier Airlines\Documents\Data\USconti'
    seats_file_name = r'\Schedule_Monthly_Summary_Report_Conti.csv'
    perf_file_name = r'\Airline_Performance_Report_Conti.csv'

    # Check if parameters.json file exists, if not create one with default values.
    if not os.path.exists('parameters.json'):
        parameters = {
            "resume_training": False,
            "MSE_or_GaussianNLLLoss": "MSE",
            "pred_num_quarters": 4,
            "seq_num": 10,
            "if_add_time_info": False,
            "learning_rate": 1e-04,
            "momentum": 0.95,
            "batch_size": 32,
            "epochs": 20,
            "num_workers": 4,
            "shuffle": True,
            "fixed_seed": True,
            "rnn_type": "LSTM",
            "n_layers": 4,
            "drop_prob": 0.35,
            "num_heads": 6,
            "start_year": 2004,
            "checkpoint_file_name": "checkpoint.pth",
            "bidirectional": False, 
            "if_skip": False, 
            "if_feed_drop": True, 
            "if_feed_norm": True,
        }
        with open('parameters.json', 'w') as f:
            json.dump(parameters, f)
    
    # Load parameters from the JSON file.
    with open('parameters.json', 'r') as f:
        args = argparse.Namespace(**json.load(f))
    
    main_program(args, folder_path, seats_file_name, perf_file_name)
```

2. This is test part.
```python
import RNN_apply_ind, os, json, argparse

folder_path = r'C:\Users\qilei.zhang\OneDrive - Frontier Airlines\Documents\Data\USconti'
seats_file_name = r'\Schedule_Monthly_Summary_Report_Conti.csv'
perf_file_name = r'\Airline_Performance_Report_Conti.csv'
apply_file_name = '\Schedule_Monthly_Summary_2023Q1234.csv'
# Load parameters from the JSON file.
if not os.path.exists('parameters.json'):
    print("parameters.json does not exist, Find the file and put it in the same folder as this file")
with open('parameters.json', 'r') as f:
    args = argparse.Namespace(**json.load(f))

RNN_apply_ind.main_apply(args, folder_path, seats_file_name, perf_file_name, apply_file_name)
``` 

### On Colab
1. Upload the code files to the Colab folder. Data files are already in the Colab folder.
2. Change the runtime type to GPU.
3. Load google drive `from google.colab import drive
drive.mount('/content/drive')`
4. Uploaed the parameters.json file to the Colab folder.
```python
gpu_info = !nvidia-smi
gpu_info = '\n'.join(gpu_info)
if gpu_info.find('failed') >= 0:
  print('Not connected to a GPU')
else:
  print(gpu_info)

from psutil import virtual_memory
ram_gb = virtual_memory().total / 1e9
print('Your runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))

if ram_gb < 20:
  print('Not using a high-RAM runtime')
else:
  print('You are using a high-RAM runtime!')

"""
Use it only on Google Colab.
"""

from google.colab import files

uploaded = files.upload()

for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))

import argparse
import json
```
5. Run the training code
```python

import RNN_model

folder_path = r'/content/drive/MyDrive/Data/'
seats_file_name = r'Schedule_Monthly_Summary_Report_Conti.csv'
perf_file_name = r'Airline_Performance_Report_Conti.csv'

# Load parameters from the JSON file.
with open('parameters.json', 'r') as f:
    args = argparse.Namespace(**json.load(f))

RNN_model.main_program(args, folder_path, seats_file_name, perf_file_name)
```

6. Run the validation code
```python
import RNN_apply_ind, os
# Load parameters from the JSON file.s
if not os.path.exists('parameters.json'):
    print("parameters.json does not exist, Find the file and put it in the same folder as this file")
with open('parameters.json', 'r') as f:
    args = argparse.Namespace(**json.load(f))

RNN_apply_ind.main_apply(args, folder_path, seats_file_name, perf_file_name, apply_file_name)
```

7. Remember to download the model and checkpoint file to local machine. The model file is in the folder `model` and the checkpoint file is in the folder `checkpoint`. The model file is named `model.pth` and the checkpoint file is named `checkpoint.pth`. The model file is used for prediction and the checkpoint file is used for future training purpose.


### On Azure Databricks

1. cd to the folder where the file is located `%cd /dbfs/FileStore/SeatPre/RunModelvX`
2. Install the required packages 
   ```python
   !pip install airportsdata
   !pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # if the GPU cluster is used
   !pip install dask
   import argparse
   import json
   ```
3. Make sure GPU is used
   ```python
   import torch
   torch.cuda.is_available()
   ```
4. Run the training code
   ```python
    import RNN_model

    folder_path = r'/dbfs/FileStore/SeatPre/'
    seats_file_name = r'Schedule_Monthly_Summary_Report_Conti.csv'
    perf_file_name = r'Airline_Performance_Report_Conti.csv'
    apply_file_name = r'Schedule_Monthly_Summary_2023Q1234.csv'

    # Load parameters from the JSON file.
    with open('parameters.json', 'r') as f:
        args = argparse.Namespace(**json.load(f))

    RNN_model.main_program(args, folder_path, seats_file_name, perf_file_name)
   ```
5. RUn the validation code
   ```python
    import RNN_apply_ind, os
    # Load parameters from the JSON file.s
    if not os.path.exists('parameters.json'):
        print("parameters.json does not exist, Find the file and put it in the same folder as this file")
    with open('parameters.json', 'r') as f:
        args = argparse.Namespace(**json.load(f))

    RNN_apply_ind.main_apply(args, folder_path, seats_file_name, perf_file_name, apply_file_name)
   ```


#### Ways to download files from Azure Databricks to local machine

1. Use the Databricks CLI to download the file to local machine. Install `pip install databricks-cli` first. Set up access token `databricks configure --token`. Check if success `databricks fs ls dbfs:/`. Finally, the command is `databricks fs cp dbfs:/FileStore/test.txt ./test.txt` 

2. Use web browser to download the file to local machine. The url is `https://community.cloud.databricks.com/?o=<unique ID>`. For example, `https://adb-7094xxxxx.11.azuredatabricks.net/files/SeatPre/RunModelv5/model/model.pth?o=7094xxxxxx`


## Things to know
1. Don't turn learning rate too high, making it no great than 0.001 is a good choice.

## Expectation?