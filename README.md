# ML4102 - Data Literacy Project
## Everyday Context and Cognition: Links Between Household Resources and Neural Control

This repository is for the group project of the course ML4102 - _Data Literacy_, in the winter semester of 2025-26.

---

## Project Overview
This repository contains the code and analysis pipeline for our project, **"Everyday Context and Cognition: Links Between Household Resources and Neural Control,"** conducted for the Data Literacy course at the University of Tübingen (Winter 2025/26). 

The study examines the relationship between recent household food insecurity and neural indices of target-focused attention—specifically P3b amplitude and latency—using EEG data collected during a visual oddball task. Our analysis revealed that while P3b amplitude demonstrated variable relationships across socioeconomic strata, higher food insecurity was consistently linked to slower target evaluation, with a 9.3 ms increase in P3b latency per category increase.

## Contributors
- **Abtin Mogharabin**  
- **Mina Mikhael**  
- **Seyedmehdi Hosseini**  
- **Kourosh Sharifi**

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

### Key Files:
- `website/project_steamlite.py`: The Streamlit-based dashboard for visualizing EEG metrics and quality control.  
- Additional Python analysis scripts (if provided), used for preprocessing and statistical modeling.  

## Dataset
The analysis uses publicly available EEG data from a study of 127 adults performing a visual oddball task. Key dependent variables include **P3b amplitude** and **latency**, as neural indicators of attention.

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
