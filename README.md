# MCV-C1 Project: Content Based Image Retrieval | Team3
## Project Description
This project contains all the code necessary to build an end-to-end image retrieval system. The system allows users to query images and retrieve the most visually similar results from a database using various feature extraction and distance functions.
The figure below illustrates the main idea:
![Image Retrieval Pipeline](assets/main)

## Project Organization
The project organitzation is the following:

```
Team3/
├── README.md                         <- Main project description and usage instructions
├── requirements.txt                  <- Python dependencies required to run the project
├── .gitignore                        <- Files and folders ignored by Git
│
├── config/                           <- Configuration files for the entire pipeline
│   ├── color_descriptors_config.py   <- Defines descriptor search space, grid configs, and mixed setups
│   ├── general_config.py             <- Global parameters to control the entire pipeline
│   └── io_config.py                  <- Input/output paths and directory structure used across modules
│
├── data/                             <- Datasets used for development, testing, and retrieval evaluation
│   ├── BBDD/                         <- Image database used as retrieval reference
│   ├── qsd1_w1/                      <- Query set for development phase
│   └── qst1_w1/                      <- Query set for testing phase
│
├── descriptors/                      <- Feature extraction modules
│   └── color_descriptors/            <- Implementation of color descriptors
│       ├── color_descriptors_func.py <- Functions to compute color descriptors and store histograms
│       └── stored_color_descriptors/ <- Precomputed descriptors saved as .txt files
│
├── exploratory_analysis/             <- Exploratory notebooks and visual analyses
│   └── dataset_analysis.ipynb        <- Sample visualizations
│
├── pipeline/                         <- Core pipeline scripts for descriptor computation and evaluation
│   ├── descriptor_creator.py         <- Generates and stores descriptors for all datasets
│   ├── dev_pipeline.py               <- Runs evaluation over development queries
│   └── test_pipeline.py              <- Runs final predictions and stores ranked retrieval results
│
├── results/                          <- Generated outputs, scores, and evaluation metrics
│   ├── histograms/                   <- Saved histogram visualizations
│   ├── method1/                      <- Results for the first retrieval configuration (in the test set)
│   │   └── result.pkl
│   ├── method2/                      <- Results for the second retrieval configuration (in the test set)
│   │   └── result.pkl
│   ├── dev_scores.csv                <- Summary table with metrics for all the descriptors and distances selected
│   ├── obtained_scores_k1.png        <- Plot showing mAP@1 performance for each descriptor-distance
│   └── obtained_scores_k5.png        <- Plot showing mAP@5 performance for each descriptor-distance
│
├── utils/                            <- Utility functions shared across modules
│   ├── common.py                     <- Helpers for file I/O, descriptor loading, and formatting
│   ├── logging_formater.py           <- Custom log formatting for debugging and reproducibility
│   ├── logging.ini                   <- Logging configuration file
│   └── metrics.py                    <- Implementation of distance functions
│
├── main.py                           <- Entry point to run descriptor generation, development, or test phases
```

## Instructions 
To run the code follow these instructions: 

**Clone the repository**

Clone the github repository by running the following command:

```bash
git clone https://github.com/MCV-2025-C1-Project/Team3.git
cd Team3
```

**Create virtual environemnt**

The next thing you will need is python3.12 or higher to create a virtual environment. There are several ways to do this but in our team we did the following steps. First you will need to create a virtual environment with the following command and run it.

```bash
python3 -m venv venv
source ./venv/bin/activate
```

**Install dependencies**
Once the virtual environment is active, install the required Python libraries listed in the requirements.txt file:
```bash
pip install -r requirements.txt
```
This command will install all dependencies necessary to run the entire pipeline.



## Contact
- Gerard Asbert Marcos: gerardasbert@gmail.com
- Xavier Pacheco Bach: xavipba@gmail.com
- Mohammed Oussama Ammouri: am.oussama10@gmail.com
- Aleix Armero Rofes: aleix.armero73@gmail.com
