import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mne
import os
import glob
import re
import altair as alt

# --- Configuration & Pathing ---
PROJECT = os.path.join(os.getcwd(), "Preprocessed_VisualOddball")

# --- FILTER CONFIG (for sidebar) ---
FILTER_CONFIG = {
    'participant_id': {'type': 'multiselect', 'options': ['sub-001', 'sub-002', 'sub-003', 'sub-004', 'sub-005', 'sub-006', 'sub-007', 'sub-008', 'sub-009', 'sub-010', 'sub-011', 'sub-012', 'sub-013', 'sub-014', 'sub-015', 'sub-016', 'sub-017', 'sub-018', 'sub-019', 'sub-020', 'sub-021', 'sub-022', 'sub-023', 'sub-024', 'sub-025', 'sub-026', 'sub-027', 'sub-028', 'sub-029', 'sub-030', 'sub-031', 'sub-032', 'sub-033', 'sub-034', 'sub-035', 'sub-036', 'sub-037', 'sub-038', 'sub-039', 'sub-040', 'sub-041', 'sub-042', 'sub-043', 'sub-044', 'sub-045', 'sub-046', 'sub-047', 'sub-048', 'sub-049', 'sub-050', 'sub-051', 'sub-052', 'sub-053', 'sub-054', 'sub-055', 'sub-056', 'sub-057', 'sub-058', 'sub-059', 'sub-060', 'sub-061', 'sub-062', 'sub-063', 'sub-064', 'sub-065', 'sub-066', 'sub-067', 'sub-068', 'sub-069', 'sub-070', 'sub-071', 'sub-072', 'sub-073', 'sub-074', 'sub-075', 'sub-076', 'sub-077', 'sub-078', 'sub-079', 'sub-080', 'sub-081', 'sub-082', 'sub-083', 'sub-084', 'sub-085', 'sub-086', 'sub-087', 'sub-088', 'sub-089', 'sub-090', 'sub-091', 'sub-092', 'sub-093', 'sub-094', 'sub-095', 'sub-096', 'sub-097', 'sub-098', 'sub-099', 'sub-100', 'sub-101', 'sub-102', 'sub-103', 'sub-104', 'sub-105', 'sub-106', 'sub-107', 'sub-108', 'sub-109', 'sub-110', 'sub-111', 'sub-112', 'sub-113', 'sub-114', 'sub-115', 'sub-116', 'sub-117', 'sub-118', 'sub-119', 'sub-120', 'sub-121', 'sub-122', 'sub-123', 'sub-124', 'sub-125', 'sub-126', 'sub-127']},
    # Numerical Ranges (for st.slider)
    'Age': {'type': 'range', 'min': 18, 'max': 80, 'step': 1},
    'Household_Members': {'type': 'range', 'min': 1, 'max': 10, 'step': 1},
    
    # Categorical Options (for st.multiselect) - Real values extracted from TSV
    'Gender': {'type': 'multiselect', 'options': ['female', 'male', 'other']},
    'Handedness': {'type': 'multiselect', 'options': ['right', 'left', 'ambidextrous']},
    'Highest_Edu': {'type': 'multiselect', 'options': ["bachelor's degree (for example: ba, bs)", "high school graduate...", "1 or more years of college...", "professional or graduate degree"]},
    'Occupation': {'type': 'multiselect', 'options': ['student', 'software engineer', 'janitor', 'nanny', 'unemployed', 'peer advisor (oia)']},
    'Employed': {'type': 'multiselect', 'options': ['yes', 'no']},
    'Employed_Yes': {'type': 'multiselect', 'options': ['full-time', 'part-time', 'self-employed']},
    'Income': {'type': 'multiselect', 'options': ['less than $5,000', '10,000 - 12,499', '75,000 - 99,999', '100,000 or more']},
    'Project': {'type': 'multiselect', 'options': ['COCOA', 'SASA', 'PILOT']},
    'EEG_Tasks': {'type': 'multiselect', 'options': ['Flanker (FL), Visual Search (VS), Visual Oddball (VO)', 'Passive Auditory Oddball (TONE)', 'MIST']},
    'fs1': {'type': 'multiselect', 'options': ['never true', 'sometimes true', 'often true', 'very often true']},
    
    # Placeholder for other text/object columns using the generic scale
    'default_multiselect_options': ['Value A', 'Value B', 'Value C', 'N/A']
}

# List of all active filter columns (derived from the new FILTER_CONFIG keys)
ALL_FILTER_COLUMNS = [key for key in FILTER_CONFIG if key != 'default_multiselect_options']

# --- Helper Functions ---
def find_any_set(folder, pattern="*.set", participant_id=None):
    if participant_id:
        search_path = os.path.join(PROJECT, folder, f"{participant_id}*{pattern.replace('*', '')}")
    else:
        search_path = os.path.join(PROJECT, folder, pattern)
    files = sorted(glob.glob(search_path))
    return files[0] if files else None

def plot_segment_st(raw_obj, title, ch_name="Pz", duration=10.0):
    if ch_name in raw_obj.ch_names:
        picks = [raw_obj.ch_names.index(ch_name)]
    else:
        picks = [0]
        ch_name = raw_obj.ch_names[0]
    sfreq = float(raw_obj.info["sfreq"])
    data, times = raw_obj[picks, : int(sfreq * duration)]
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(times, data[0] * 1e6, lw=0.7)
    ax.set_title(f"{title} ({ch_name})")
    ax.set_ylabel("µV")
    st.pyplot(fig)
    plt.close(fig)

# --- QC Functions ---
def qc_1_raw(p_id):
    st.header("[1] Raw Continuous Dataset")
    path = find_any_set("01_raw", "*raw.set", p_id)
    if not path: return st.warning("Raw file not found.")
    
    raw = mne.io.read_raw_eeglab(path, preload=True, verbose="ERROR")
    st.write(f"**File:** {os.path.basename(path)}")
    st.info("Expected: Visible drifts, line noise, and large blinks.")
    plot_segment_st(raw, "Raw EEG Segment", "Pz")

def qc_2_preprocessed(p_id):
    st.header("[2] Preprocessed Dataset")
    path = find_any_set("02_preprocessed", "*preprocessed.set", p_id)
    if not path: return st.warning("Preprocessed file not found.")
    
    preproc = mne.io.read_raw_eeglab(path, preload=True, verbose="ERROR")
    st.write(f"**File:** {os.path.basename(path)}")
    plot_segment_st(preproc, "Preprocessed Segment", "Pz")
    
    fig = preproc.compute_psd(fmin=0.1, fmax=45.0).plot(show=False)
    st.pyplot(fig)
    st.write("Expected: 1/f shape and alpha peak (8-12 Hz).")

def qc_3_preICA_loss(p_id):
    st.header("[3] Pre ICA Bad Channel Detection")
    loss_xlsx = os.path.join(PROJECT, "03_preICA", "COCOA_preICAextremeloss.xlsx")
    if not os.path.exists(loss_xlsx): return st.warning("Loss table not found.")
    
    df = pd.read_excel(loss_xlsx)
    subj_data = df[df['ID'].astype(str).str.contains(p_id)]
    if subj_data.empty: return st.write("No loss data for this subject.")
    
    ch_cols = [c for c in df.columns if c != "ID"]
    values = subj_data[ch_cols].values.flatten().astype(float)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(ch_cols)), values)
    ax.set_ylabel("% Extreme Artifact")
    ax.set_title(f"Artifact Loss per Channel: {p_id}")
    st.pyplot(fig)

def qc_4_ICLabel(p_id):
    st.header("[4] ICLabel Classification")
    ic_dir = os.path.join(PROJECT, "05_ICLabel")
    class_files = glob.glob(os.path.join(ic_dir, f"{p_id}*_ICclassifications.xlsx"))
    
    if class_files:
        df = pd.read_excel(class_files[0])
        cols = ["Brain", "Muscle", "Eye", "Heart", "Line_Noise", "Channel_Noise", "Other"]
        mean_probs = df[cols].mean()
        fig, ax = plt.subplots()
        mean_probs.plot(kind='bar', ax=ax)
        st.pyplot(fig)
    else:
        st.warning("ICLabel file not found for this participant.")

def qc_5_postICA(p_id):
    st.header("[5] Post ICA & Corrected EOG")
    path = find_any_set("06_postICA", "*postICA.set", p_id)
    if not path: return st.warning("Post ICA file not found.")
    
    try:
        raw = mne.io.read_raw_eeglab(path, preload=True, verbose="ERROR")
        for ch in ["CVEOGR", "CHEOG"]:
            if ch in raw.ch_names:
                st.write(f"**Channel:** {ch}")
                plot_segment_st(raw, f"Corrected {ch}", ch)
    except Exception as e:
        st.error(f"MNE could not open file: {e}")

def qc_6_epoched(p_id):
    st.header("[6] Epoched Data & ERP")
    path = find_any_set("08_AR", "*autoAR.set", p_id) or find_any_set("07_epoched", "*epoched.set", p_id)
    if not path: return st.warning("Epoched file not found.")
    
    epochs = mne.io.read_epochs_eeglab(path, verbose="ERROR")
    target_code = next((k for k in epochs.event_id.keys() if '11' in str(k)), list(epochs.event_id.keys())[0])
    evoked = epochs[target_code].average()
    fig = evoked.plot(picks="Pz", show=False) if "Pz" in evoked.ch_names else evoked.plot(show=False)
    st.pyplot(fig)
    st.write(f"ERP for condition: {target_code}. Expected: P3b deflection 300-500ms.")

def qc_7_final_features():
    st.header("[7] Final P3b Feature Distribution")
    feat_csv = os.path.join(PROJECT, "COCOA_VO_P3b_trial_features_with_meta.csv")
    if os.path.exists(feat_csv):
        df = pd.read_csv(feat_csv)
        fig, ax = plt.subplots()
        df["roi_mean_300_600"].hist(bins=30, ax=ax)
        ax.set_title("P3b Mean Amplitude Distribution (All Trials)")
        st.pyplot(fig)
    else:
        st.warning("Feature CSV not found.")

# --- Run QC Analysis ---
def run_qc_analysis(filter_selections, task_option):
    participant_id = filter_selections.get("participant_id", [])
    if not participant_id:
        st.warning("Select at least one participant to run the analysis.")
        return
    
    # Loop through selected participants
    for p_id in participant_id:
        st.markdown(f"## Participant: {p_id}")
        qc_1_raw(p_id)
        st.divider()
        qc_2_preprocessed(p_id)
        st.divider()
        qc_3_preICA_loss(p_id)
        st.divider()
        qc_4_ICLabel(p_id)
        st.divider()
        qc_5_postICA(p_id)
        st.divider()
        qc_6_epoched(p_id)
        st.divider()
        # qc_7_final_features()
        st.markdown("---")

# --- Streamlit Layout ---
st.set_page_config(layout="wide")
st.title("Visual Oddball EEG Analysis Dashboard (Redesigned)")

# Sidebar Filters
# filter_selections = {}
# with st.sidebar:
#     st.header("Controls and Inputs")
#     for col in ALL_FILTER_COLUMNS:
#         config = FILTER_CONFIG[col]
#         filter_selections[col] = st.multiselect(
#             f"Select {col}:", 
#             options=config['options'], 
#             default=[], 
#             placeholder="Search and select..."
#         )
    
#     st.markdown("---")
#     st.subheader("Task Selection")
#     task_option = st.selectbox("Select Task:", ["Visual Oddball"], index=0)
    
#     generate_button = st.button("Generate Analysis", type="primary")

filter_selections = {}

with st.sidebar:
    st.header("Controls and Inputs")
    st.markdown("---")

    # --- 1. Filter Section (Organized into Expanders) ---
    st.subheader("Filter Task Bar (Searchable)")
    
    # Helper function to generate filters based on config
    def generate_filters_for_group(cols):
        for col in cols:
            config = FILTER_CONFIG.get(col)
            
            if config and config.get('type') == 'range':
                # Slider/Range filter
                filter_selections[col] = st.slider(
                    f'Filter by {col} Range:',
                    min_value=config['min'], 
                    max_value=config['max'], 
                    value=(config['min'], config['max']), 
                    step=config.get('step', 1),
                    key=f'slider_{col}'
                )
            elif col in ALL_FILTER_COLUMNS:
                # Multiselect filter (Searchable)
                options = config['options'] if config and 'options' in config else FILTER_CONFIG['default_multiselect_options']
                
                filter_selections[col] = st.multiselect(
                    f'Select {col}:',
                    options=sorted(options),
                    default=[],
                    key=f'multi_{col}',
                    placeholder="Search and select options..."
                )

    # Group 1: Identifiers and Core Demographics
    with st.expander("ID & Core Demographics", expanded=True):
        id_cols = ['participant_id', 'Age', 'Gender', 'Handedness', 'Highest_Edu']
        generate_filters_for_group(id_cols)

    # Group 2: Socio-Economic Status and Household
    with st.expander("SES and Household Info"):
        ses_cols = ['Occupation', 'Employed', 'Employed_Yes', 'Income', 'Household_Members']
        generate_filters_for_group(ses_cols)

    # Group 3: Questionnaire and Session Details
    with st.expander("Questionnaire and Session Details"):
        qs_cols = ['fs1', 'Project', 'EEG_Tasks']
        generate_filters_for_group(qs_cols)
                        
    # Removed Adult/Relative Information and Questionnaires (fs, hnc, ASRS) & Language 
    # to keep only active filters.


    st.markdown("---")
    
    # 2. ML Model Selection (From Diagram)
    st.subheader("ML Model")
    model_option = st.selectbox(
        'Select ML Model:',
        ['Bayesian', 'Regression'],
        key='model_select'
    )
    st.markdown("---")

    # 3. Task Selection (From Diagram)
    st.subheader("Task")
    task_option = st.selectbox(
        'Select Task:',
        ['Flanker', 'Visual Oddball'],
        key='task_select'
    )
    st.markdown("---")

    # 4. Generate Button
    generate_button = st.button("Generate Analysis", type="primary")



# Main Execution
col_plots, col_text = st.columns([3, 1])

if generate_button:
    # --- Run Analysis and Display Outputs ---
    with col_plots:
        plot_container = st.container(border=True)
        with plot_container:
            # Pass user selections to the analysis function
            run_qc_analysis(filter_selections, task_option)

    with col_text:
        text_container = st.container(border=True)
        with text_container:
            st.markdown("---")
            st.markdown(f"**ML Model Selected:** {model_option}")
            st.markdown(f"**Task Selected:** {task_option}")
            st.markdown("---")
            st.code("Detailed metrics are in the plot area.", language="markdown")
else:
    # Initial state
    with col_plots:
        st.subheader("Output Plot(s)")
        st.info("Select options from the sidebar and press 'Generate Analysis' to run the analysis.")
    
    with col_text:
        st.subheader("Text outputs (if any)")
        st.code("Awaiting analysis results...", language="markdown")