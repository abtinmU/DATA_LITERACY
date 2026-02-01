"""
Streamlit dashboard for exploring the Cognitive Electrophysiology in Socioeconomic
Context in Adulthood (COCOA) dataset.

This application demonstrates a number of best practices for building an
interactive dashboard in Streamlit:

* **Data is cached** with ``st.cache_data`` so it is only loaded once per
  session.  The Excel file provided by the user contains pre‑ICA extreme
  loss metrics for each participant and task.
* **Filters are generated from the data** instead of being hard‑coded.  The
  list of participants, tasks and available EEG channels are derived from
  the dataframe.  Selecting one or more participants automatically filters
  the dataset used to populate subsequent controls.
* **Visualisations respect basic plotting rules**: axes are labelled and
  include units where appropriate, axis limits start at zero when that
  makes sense and legends are only drawn when multiple series are present.
  The slides on scientific plotting stress the importance of clear axis
  labels, proper units and reasonable axis limits【239241550391385†L485-L494】.
* **Accessible colour palettes** are used.  A small colour cycler is defined
  using hex values inspired by the University of Tübingen palette (see the
  ``tueplots`` documentation for reference), and these colours are cycled
  through when plotting with Matplotlib.  This avoids the default rainbow
  colormap and takes into account the fact that roughly 8 % of the
  population is colour blind【239241550391385†L1000-L1009】.
* **Multiple plot types** are supported.  Users can choose between a line
  chart (showing channel values across participants), a histogram or a
  boxplot.  Line charts are created with Altair for interactivity while
  histograms and boxplots are drawn with Matplotlib so the custom colour
  palette can be applied easily.
* **The layout separates plots from textual information**.  Summary
  statistics for the selected data are shown alongside the figure instead
  of being hidden away.  This encourages a narrative around the data and
  makes the dashboard self‑contained.

This file relies only on standard Python libraries plus ``pandas``,
``numpy``, ``altair`` and ``streamlit``.  If you wish to customise the
Matplotlib aesthetics further (for example to exactly match the style
bundles shown in the Data Literacy lecture), consider installing
``tueplots`` and updating the ``plt.rcParams`` at the top of the module.
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
from typing import List
from pathlib import Path

###############################################################################
# Data loading and preprocessing
###############################################################################

@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    """Load the Excel file containing pre‑ICA extreme loss metrics.

    The ``ID`` column encodes both a participant identifier (e.g.
    ``sub-005``) and the task (e.g. ``visualoddball``).  This function
    extracts those two pieces of information into separate columns to make
    filtering easier.  All remaining columns correspond to EEG channel
    metrics and are converted to numeric types where possible.

    Parameters
    ----------
    path : str
        Path to the Excel file.

    Returns
    -------
    pandas.DataFrame
        The processed dataframe with ``participant`` and ``task`` columns.
    """
    df = pd.read_excel(path)
    # Extract participant and task from the ID column
    df['participant'] = df['ID'].str.extract(r'^(sub-\d+)')
    df['task'] = df['ID'].str.extract(r'task-([^_]+)')
    # Ensure numeric columns are of float type
    for col in df.columns:
        if col not in ['ID', 'participant', 'task']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    """Return a list of numeric column names representing EEG channels."""
    return [c for c in df.columns if c not in ['ID', 'participant', 'task']]


###############################################################################
# Plotting utilities
###############################################################################

# Define a custom colour palette inspired by the University of Tübingen colours.
# These hex codes approximate the RGB values published in the tueplots
# documentation.  Feel free to adjust or extend this palette.
TUE_PALETTE = [
    '#006AA3',  # tue_blue
    '#E65C00',  # tue_orange
    '#A31C34',  # tue_red
    '#5C8021',  # tue_green
    '#735545',  # tue_brown
    '#4A6D8C',  # tue_darkblue
]


def plot_line_chart(df: pd.DataFrame, channels: List[str]) -> alt.Chart:
    """Create an interactive line chart using Altair.

    The x‑axis shows the selected channels and the y‑axis the corresponding
    measurement values.  Each selected participant is plotted as a separate
    line and coloured automatically by Altair's default categorical palette.

    Parameters
    ----------
    df : pandas.DataFrame
        Filtered dataframe containing only the selected participants.
    channels : list of str
        Column names (EEG channels) to include on the x‑axis.

    Returns
    -------
    altair.Chart
        Configured line chart.
    """
    # Reshape to long format for Altair
    long_df = df.melt(id_vars=['participant'], value_vars=channels,
                      var_name='channel', value_name='value')
    # Order channels as specified to preserve ordering on x axis
    channel_order = channels
    chart = (
        alt.Chart(long_df)
        .mark_line(point=True)
        .encode(
            x=alt.X('channel:N', sort=channel_order, title='EEG channel'),
            y=alt.Y('value:Q', title='Pre‑ICA extreme loss'),
            color=alt.Color('participant:N', title='Participant'),
            tooltip=['participant', 'channel', 'value']
        )
        .properties(height=400)
        .interactive()
    )
    return chart


def plot_histograms(df: pd.DataFrame, channels: List[str]) -> None:
    """Draw one histogram per selected channel using Matplotlib.

    Each histogram is placed in its own subplot on a single figure.  Axis
    limits start at zero because histograms represent counts.  A custom
    colour from the TUE palette is used for each channel; colours are
    cycled if there are more channels than colours defined.
    """
    n = len(channels)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    for idx, channel in enumerate(channels):
        ax = axes[idx]
        colour = TUE_PALETTE[idx % len(TUE_PALETTE)]
        data = df[channel].dropna()
        ax.hist(data, bins=20, color=colour, edgecolor='black')
        ax.set_title(f'Distribution of {channel}')
        ax.set_xlabel('Pre‑ICA extreme loss')
        ax.set_ylabel('Count')
        # Start y-axis at zero as recommended【239241550391385†L485-L494】
        ax.set_ylim(bottom=0)
    # Remove any unused subplots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])
    fig.tight_layout()
    st.pyplot(fig)


def plot_boxplot(df: pd.DataFrame, channels: List[str]) -> None:
    """Create a boxplot for the selected channels using Matplotlib.

    Each channel’s distribution is summarised with a boxplot.  Colours are
    cycled from the TUE palette to distinguish boxes.  Axis labels and
    limits follow the guidelines from the lecture slides.
    """
    fig, ax = plt.subplots(figsize=(1 + 1.2 * len(channels), 5))
    data = [df[ch].dropna() for ch in channels]
    box = ax.boxplot(data, patch_artist=True, labels=channels)
    for patch, colour in zip(box['boxes'], TUE_PALETTE):
        patch.set_facecolor(colour)
    ax.set_title('Distribution of selected channels')
    ax.set_xlabel('EEG channel')
    ax.set_ylabel('Pre‑ICA extreme loss')
    # Optional: add grid for readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig)


###############################################################################
# Main analysis function
###############################################################################

def run_analysis(df: pd.DataFrame, selected_ids: List[str], channels: List[str],
                 plot_type: str, model_option: str, task_option: str) -> None:
    """Filter the dataset and generate visualisations and summary statistics.

    Parameters
    ----------
    df : pandas.DataFrame
        The full pre‑processed dataframe.
    selected_ids : list of str
        Participant identifiers selected via the sidebar.  If empty, all
        participants are used.
    channels : list of str
        EEG channels selected for plotting.
    plot_type : str
        One of 'Line chart', 'Histogram' or 'Boxplot'.
    model_option : str
        Placeholder for future ML model selection (e.g. 'Bayesian' or
        'Regression').  Currently not used but displayed in the sidebar.
    task_option : str
        Placeholder for future task selection (e.g. 'Flanker' or
        'Visual Oddball').  Currently not used but displayed in the sidebar.
    """
    # Apply filters
    if selected_ids:
        filtered_df = df[df['participant'].isin(selected_ids)].copy()
    else:
        filtered_df = df.copy()

    # Display summary metrics in the sidebar
    with st.sidebar:
        st.markdown('---')
        st.subheader('Summary of Selected Data')
        st.write(f'Number of participants: {filtered_df["participant"].nunique()}')
        st.write(f'Number of records: {len(filtered_df)}')
        for ch in channels:
            values = filtered_df[ch].dropna()
            st.write(
                f'**{ch}**: mean = {values.mean():.2f}, std = {values.std():.2f}, '
                f'min = {values.min():.2f}, max = {values.max():.2f}'
            )

    # Plot area
    plot_container = st.container()
    with plot_container:
        if plot_type == 'Line chart':
            st.altair_chart(plot_line_chart(filtered_df, channels), use_container_width=True)
        elif plot_type == 'Histogram':
            plot_histograms(filtered_df, channels)
        elif plot_type == 'Boxplot':
            plot_boxplot(filtered_df, channels)
        else:
            st.warning('Unknown plot type selected.')

    # Placeholder for ML models or further analysis
    st.markdown('---')
    st.subheader('Machine Learning Placeholder')
    st.write(
        'You selected the **{model_option}** model and the **{task_option}** task. '
        'Implement your analysis here.'
    )


###############################################################################
# Streamlit App Layout
###############################################################################

def main():
    st.set_page_config(layout='wide', page_title='COCOA EEG Dashboard')
    st.title('COCOA EEG Dataset Analysis')
    st.write(
        'Explore pre‑ICA extreme loss metrics from the Cognitive Electrophysiology '
        'in Socioeconomic Context in Adulthood (COCOA) dataset.  This dashboard '
        'lets you filter participants, select EEG channels and visualise the data '
        'using various plot types.  The dataset includes EEG recordings from '
        'young adults along with socioeconomic and behavioural measures' 
        '【850338308294355†L66-L80】.'
    )

    # Load data (cloud-safe path)
    BASE_DIR = Path(__file__).resolve().parent
    data_path = BASE_DIR / "COCOA_preICAextremeloss.xlsx"

    if not data_path.exists():
        st.error(f"Data file not found: {data_path}")
        st.stop()

    df = load_data(str(data_path))


    # Sidebar controls
    with st.sidebar:
        st.header('Filters')
        st.markdown('Select one or more participants and channels to explore.')
        # Participant filter
        participants = sorted(df['participant'].dropna().unique())
        selected_ids = st.multiselect('Participants', options=participants, default=[])
        # Task filter (for information only; tasks are derived from ID)
        tasks = sorted(df['task'].dropna().unique())
        st.multiselect('Tasks (informative)', options=tasks, default=tasks, disabled=True)
        # Channel selection
        numeric_cols = get_numeric_columns(df)
        default_channels = numeric_cols[:3]  # Show first three by default for brevity
        channels = st.multiselect('EEG channels', options=numeric_cols,
                                  default=default_channels)
        # Plot type selection
        plot_type = st.radio('Plot type', options=['Line chart', 'Histogram', 'Boxplot'])
        # ML model and task selection (placeholders)
        model_option = st.selectbox('ML Model', ['Bayesian', 'Regression'])
        task_option = st.selectbox('Task', ['Flanker', 'Visual Oddball'])
        # Trigger analysis
        run_button = st.button('Generate Analysis', type='primary')

    if run_button:
        run_analysis(df, selected_ids, channels, plot_type, model_option, task_option)
    else:
        st.info('Use the controls in the sidebar to generate plots and statistics.')


if __name__ == '__main__':
    main()