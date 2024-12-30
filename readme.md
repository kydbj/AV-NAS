## Catalogue
- [Getting Started](#getting-started)
- [Data Processing](#data-processing)
- [Project Structure](#Project-Structure)
- [Search](#Search)
- [Re-train](#Re-train)
## Getting Started
1. Clone this repository:

2. Create a conda environment and install the dependencies:

```
conda env create -f environment.yml
```

## Data Processing
1. Download Datasets

   Datasets of FCVID and ActivityNet are kindly uploaded by the authors. You can download them from the following links.
   
	| *Dataset*   | *Link*                                                   |
	| ----------- | ------------------------------------------------------- |
	| FCVID       | [Link](https://pan.baidu.com/s/1sFY6bE1M7xiomOBEWI26og?pwd=mvej) |
	| ActivityNet | [Link](https://pan.baidu.com/s/1sFY6bE1M7xiomOBEWI26og?pwd=mvej) |

2. Feauture Extraction

    2.1 **Video Processing**:
    - For each video, uniformly extract 25 frames.
    - Uniformly divide the audio into 25 segments.

    2.2 **Feature Extraction**:

    - For each frame, use the [CLIP](https://github.com/openai/CLIP) model to generate a 768-dimensional vector.
    - For each audio segment, use the [AST](https://github.com/YuanGongND/ast) model to generate a 768-dimensional vector.

    2.3 **Data Storage**:
    - Each video has `(25, 768)` image features and `(25, 768)` audio features.
    
    - Store the features of each video in one HDF5 file.
    
    - Store all audio features in another HDF5 file.
    
      The structure of the HDF5 file storing video features is as follows:
      ```
      ActivityNet_image.h5
      ├── v_---9CpRcKoU
      │ └── vectors
      │ └── Type: float32
      ├── v_--0edUL8zmA
      │ └── vectors
      │ ├── Shape: (25, 768)
      │ └── Type: float32
      ├── v_--1DO2V4K74
      │ └── vectors
      │ ├── Shape: (25, 768)
      │ └── .....
      └── .....
      ```
      
      The structure of the HDF5 file storing audio features is as follows:
       ```
      ActivityNet_audio.h5
      ├── v_---9CpRcKoU
      │ └── vectors
      │ └── Type: float32
      ├── v_--0edUL8zmA
      │ └── vectors
      │ ├── Shape: (25, 768)
      │ └── Type: float32
      ├── v_--1DO2V4K74
      │ └── vectors
      │ ├── Shape: (25, 768)
      │ └── .....
      └── .....
       ```

	  In above structure:
      - Each video's ID serves as the top-level group name.
      - Each group contains a dataset named vectors.
      - The vectors dataset contains the image or audio features of the video, with a shape of (25, 768) and a type of float32.
    
3. Dataset Splitting
   Split the dataset into training, validation, and test sets evenly based on categories. The file IDs for different data splits are stored in `train.txt`, `test.txt`, and `val.txt` files respectively.
   
   **ActivityNet** contents are as follows, with each line including: videoname, video frame count, category.
   ```
   v_JDg--pjY5gg,3153,10
   v_DFAodsf1dWk,6943,10,10,10
   v_J__1J4MmH4w,3370,10,10,10,10
   ...
   ```

	 **FCVID** contents are as follows, with each line including: videoname, category.
   ```
   --0K_j-zexM,76
   --1DKnUmLNQ,163
   --45hTBwKRI,117
   ...
   ```
   
4. Configure the **Anet.json** and **fcvid.json** file in ./Json/
   ```
   {
   "dataset":  dataset ("actnet" or "fcvid")
   "data_path": path to the image frames floder,
   "num_class":  dataset classes, 200(actnet) or 239(fcivd)
   "train_list": "path to the train set file",
   "val_list": "path to the validation set file",
   "test_list": "path to the test set file",
   "retrieval_list": "path to the databese set (train set) file"
   }
   ```
   

  ## Project Structure

  This is the overall structure of the project:

```
AV-NAS/
│
├── json/               		# Dataset configurations
│   ├── Anet.json				# ActivityNet dataset configuration
│   └── FCVID.json      		# FCVID dataset configuration
│
├── loader/            			# Data loading
│   ├── dataset_per_label.py	# Load data		
│   └── path.py      			# Load data and log paths
│
├── utils/                      # Utilities
│   ├── calc_probs.py           # Calculate operator probabilities
│   ├── eval.py                 # Metric calculation
│   ├── log.py                  # Log files
│   └── ops_adapter_new.py      # Alternative operators
│
├── model/
├── model/                      # Model files
│   ├── cls.pt                  # CLS token
│   ├── config.json             # Configuration file
│   ├── modules_new1.py         # Basic operations
│   ├── pscan.py                # Mamba model
│   ├── full_model_AttDetail.py # Specific model used for retraining
│   ├── hygr_model_AttDetail.py # Search space used for searching
│   └── mixed_AttDetail.py      # Searchable cells
|
├── loss/						# Loss function
│   └── loss.py
│
├── search_p_AttDetail.py       # Search function
├── train_p_AttDetail.py		# Retrain function
├── environment.yml             # Conda environment configuration
└── README.md          			# Project documentation (this file)
```

  ## Search

  To search on FCVID:

  ```bash
  self.DATASET_CONFIG = './json/FCVID.json'  #  modify the DATASET_CONFIG attribute in the Path class located in ./loader/path.py.
  python search_p_AttDetail.py
  ```

  To search on ActivityNet:

  ```bash
  self.DATASET_CONFIG = './json/Anet.json'  #  modify the DATASET_CONFIG attribute in the Path class located in ./loader/path.py.
  python search_p_AttDetail.py
  ```
**Note**: During the search phase, a different learning rate is typically used compared to retraining. Ensure that the learning rate for the search phase is appropriately configured in your model or configuration file. In our paper, the learning rate for search was set to 0.001.


  ## Retrain

  To retrain on FCVID:
  ```bash
  self.DATASET_CONFIG = './json/FCVID.json'  #  modify the DATASET_CONFIG attribute in the CfgSearch class located in train_p_AttDetail.py.
  python train_p_AttDetail.py
  ```
  To retrain on ActivityNet: 
  ```bash
  self.DATASET_CONFIG = './json/Anet.json'  #  modify the DATASET_CONFIG attribute in the CfgSearch class located in train_p_AttDetail.py.
  python train_p_AttDetail.py
  ```

**Note**: During the retrain phase, a different learning rate is typically used compared to search.. In our paper, the learning rate for retrain was set to 0.0001.
