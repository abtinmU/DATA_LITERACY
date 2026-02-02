# ML4102 - Data Literacy Project

## Project Overview
This repository contains the code and analysis pipeline for our project, **"Everyday Context and Cognition: Links Between Household Resources and Neural Control,"** conducted for the Data Literacy course at the University of Tübingen (Winter 2025/26). 

The study examines the relationship between recent household food insecurity and neural indices of target-focused attention, specifically P3b amplitude and latency, using EEG data collected during a visual oddball task. Our analysis revealed that while P3b amplitude demonstrated variable relationships across socioeconomic strata, higher food insecurity was consistently linked to slower target evaluation, with an increase in P3b latency per category increase.

## Contributors

- **Abtin Mogharabin** <a href="https://github.com/abtinmU" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub Badge" /></a> <a href="https://www.linkedin.com/in/abtinmu/" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/LinkedIn-%230077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn Badge" /></a>

- **Mina Mikhael** <a href="https://github.com/Mina-88" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub Badge" /></a> <a href="https://www.linkedin.com/in/minawmikhael/?originalSubdomain=egk" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/LinkedIn-%230077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn Badge" /></a>

- **Seyedmehdi Hosseini** <a href="https://github.com/mhdihso" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub Badge" /></a> <a href="https://www.linkedin.com/in/mehdi-hoseyni/" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/LinkedIn-%230077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn Badge" /></a>

- **Kourosh Sharifi** <a href="https://github.com/KouroshKSH" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub Badge" /></a> <a href="https://www.linkedin.com/in/kouroshsharifi/" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/LinkedIn-%230077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn Badge" /></a>

---

## Repository Contents
### Core Features:
- **Streamlit Dashboard for EEG Analysis**: 
  This interactive dashboard performs EEG quality control and visualization tasks, including:
  - **Raw Data Inspection**: Visualizing EEG signal segments to detect noise and artifacts.  
  - **Preprocessed Data Examination**: Plotting spectra, inspecting channel loss, and reviewing ICA classifications.  
  - **ERP Analysis**: Generating averaged event-related potentials (e.g., P3b) from epoched data.  
  - **Pipeline Control**: Filtering subjects by demographics, task, or other attributes using configurable sidebars.

- **Main Analysis Pipeline**:  
  Includes preprocessing scripts, data linking, and regression models to analyze relationships between socioeconomic indicators and EEG markers.

## Dataset
The analysis uses publicly available EEG data from a study from [Isbell et al.](https://openneuro.org/datasets/ds005863/versions/1.0.0) performing a visual oddball task. Key dependent variables include **P3b amplitude** and **latency**, as neural indicators of attention.

## Summary of Findings
- Higher levels of household food insecurity were associated with slower cognitive processing (evidenced by increased P3b latency).  
- Variable relationships between P3b amplitude and socioeconomic indicators were observed.  
- The findings suggest socioeconomic strain affects cognitive bandwidth and stress response.

---

## Getting Started
1. Clone this repository:
    ```bash
    git clone https://github.com/mhdihso/DATA_LITERACY.git
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Launch the Streamlit dashboard:
    ```bash
    cd website
    streamlit run project_steamlite.py
    ```

## Acknowledgments
This project was developed as a part of the **Data Literacy course** at the University of Tübingen. We thank the course instructors and the creators of the public EEG dataset used in this project.  

---  
