"""
Unified Streamlit dashboard for the COCOA project.

Features:
- Pre-ICA extreme-loss exploration (Excel)
- Visual Oddball QC workflow (EEGLAB .set files)
- Demographic filters (Age, Gender, Income, etc.) applied FIRST to determine the eligible participants.
- Participant selection is optional: if empty, use all eligible participants.

Plot styling:
- Uses tueplots if available (recommended). Falls back gracefully if not installed.
"""

import os
import glob
import re
from typing import List, Optional, Dict, Any

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt

try:
    import mne  # noqa: F401
except Exception:
    mne = None

# Try tueplots (optional). If not available, we fallback to a clean rcParams style.
try:
    from tueplots import bundles  # type: ignore
    TUEPLOTS_AVAILABLE = True
except Exception:
    bundles = None
    TUEPLOTS_AVAILABLE = False


###############################################################################
# Configuration
###############################################################################

PREICA_XLSX = "COCOA_preICAextremeloss.xlsx"
VO_PROJECT = os.path.join(os.getcwd(), "Preprocessed_VisualOddball")

# Tübingen-inspired palette (fallback / also used for consistency)
TUE_PALETTE = [
    "#006AA3",  # blue
    "#E65C00",  # orange
    "#A31C34",  # red
    "#5C8021",  # green
    "#735545",  # brown
    "#4A6D8C",  # dark blue
]

FILTER_CONFIG = {
    "participant_id": {"type": "multiselect", "options": []},  # dynamic
    "Age": {"type": "range", "min": 18, "max": 80, "step": 1},
    "Household_Members": {"type": "range", "min": 1, "max": 10, "step": 1},
    "Gender": {"type": "multiselect", "options": ["female", "male", "other"]},
    "Handedness": {"type": "multiselect", "options": ["right", "left", "ambidextrous"]},
    "Highest_Edu": {
        "type": "multiselect",
        "options": [
            "bachelor's degree (for example: ba, bs)",
            "high school graduate...",
            "1 or more years of college...",
            "professional or graduate degree",
        ],
    },
    "Occupation": {
        "type": "multiselect",
        "options": ["student", "software engineer", "janitor", "nanny", "unemployed", "peer advisor (oia)"],
    },
    "Employed": {"type": "multiselect", "options": ["yes", "no"]},
    "Employed_Yes": {"type": "multiselect", "options": ["full-time", "part-time", "self-employed"]},
    "Income": {
        "type": "multiselect",
        "options": ["less than $5,000", "10,000 - 12,499", "75,000 - 99,999", "100,000 or more"],
    },
    "Project": {"type": "multiselect", "options": ["COCOA", "SASA", "PILOT"]},
    "EEG_Tasks": {
        "type": "multiselect",
        "options": [
            "Flanker (FL), Visual Search (VS), Visual Oddball (VO)",
            "Passive Auditory Oddball (TONE)",
            "MIST",
        ],
    },
    "fs1": {"type": "multiselect", "options": ["never true", "sometimes true", "often true", "very often true"]},
    "default_multiselect_options": ["Value A", "Value B", "Value C", "N/A"],
}

ALL_FILTER_COLUMNS = [k for k in FILTER_CONFIG if k != "default_multiselect_options"]


###############################################################################
# Styling (tueplots if available)
###############################################################################

def apply_plot_style() -> None:
    """
    Apply a global matplotlib style.
    - If tueplots is available: use an ICML-like bundle
    - Else: a clean fallback rcParams
    """
    if "plot_style_applied" in st.session_state:
        return

    if TUEPLOTS_AVAILABLE:
        # Pick a bundle that works well for paper-like figures.
        # If you prefer half-width, change column="half".
        plt.rcParams.update(bundles.icml2024(column="full", nrows=1, ncols=1))
    else:
        # Fallback: clean defaults emphasizing readability
        plt.rcParams.update(
            {
                "figure.dpi": 120,
                "savefig.dpi": 300,
                "axes.grid": True,
                "grid.alpha": 0.25,
                "grid.linestyle": "--",
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.labelsize": 11,
                "axes.titlesize": 12,
                "legend.fontsize": 10,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "lines.linewidth": 1.4,
            }
        )

    st.session_state["plot_style_applied"] = True


###############################################################################
# Metadata loading + filtering (dynamic participant list)
###############################################################################

META_COL_ALIASES: Dict[str, List[str]] = {
    "participant_id": ["participant_id", "subject_id", "sub_id", "id"],
    "Age": ["age", "Age"],
    "Household_Members": ["household_members", "householdmembers", "household_members_count"],
    "Gender": ["gender", "sex"],
    "Handedness": ["handedness"],
    "Highest_Edu": ["highest_edu", "highest_education", "education", "edu"],
    "Occupation": ["occupation"],
    "Employed": ["employed"],
    "Employed_Yes": ["employed_yes", "employment_type"],
    "Income": ["income", "household_income", "income_range"],
    "Project": ["project"],
    "EEG_Tasks": ["eeg_tasks", "eeg_task", "tasks"],
    "fs1": ["fs1"],
}


def _find_first_col(df: pd.DataFrame, aliases: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for a in aliases:
        if a.lower() in cols_lower:
            return cols_lower[a.lower()]
    return None


@st.cache_data(show_spinner=False)
def load_participants_metadata() -> Optional[pd.DataFrame]:
    candidates = [
        os.path.join(os.getcwd(), "participants.tsv"),
        os.path.join(VO_PROJECT, "participants.tsv"),
        os.path.join(VO_PROJECT, "..", "participants.tsv"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            try:
                return pd.read_csv(path, sep="\t")
            except Exception:
                return None
    return None


def apply_demographic_filters_to_meta(df_meta: pd.DataFrame, selections: dict) -> pd.DataFrame:
    out = df_meta.copy()

    # Age (range)
    age_sel = selections.get("Age")
    if age_sel:
        col = _find_first_col(out, META_COL_ALIASES["Age"])
        if col:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            a_min, a_max = age_sel
            out = out[(out[col] >= a_min) & (out[col] <= a_max)]

    # Household members (range)
    hh_sel = selections.get("Household_Members")
    if hh_sel:
        col = _find_first_col(out, META_COL_ALIASES["Household_Members"])
        if col:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            h_min, h_max = hh_sel
            out = out[(out[col] >= h_min) & (out[col] <= h_max)]

    # Categorical fields
    categorical_fields = [
        "Gender", "Handedness", "Highest_Edu", "Occupation",
        "Employed", "Employed_Yes", "Income", "Project", "EEG_Tasks", "fs1",
    ]
    for field in categorical_fields:
        chosen = selections.get(field) or []
        if not chosen:
            continue
        col = _find_first_col(out, META_COL_ALIASES.get(field, [field]))
        if not col:
            continue
        series = out[col].astype(str).str.strip().str.lower()
        allowed = {str(x).strip().lower() for x in chosen}
        out = out[series.isin(allowed)]

    return out


def meta_participant_ids(df_meta: pd.DataFrame) -> List[str]:
    col = _find_first_col(df_meta, META_COL_ALIASES["participant_id"])
    if not col:
        return []
    return sorted(df_meta[col].astype(str).str.strip().unique().tolist())


def generate_filter_widgets_for_group(cols: List[str], filter_selections: dict, *, prefix: str) -> None:
    """Create widgets for non-participant filters (Age, Gender, etc.)."""
    for col in cols:
        cfg = FILTER_CONFIG.get(col)
        if not cfg:
            continue

        if cfg.get("type") == "range":
            filter_selections[col] = st.slider(
                f"{col} range",
                min_value=cfg["min"],
                max_value=cfg["max"],
                value=(cfg["min"], cfg["max"]),
                step=cfg.get("step", 1),
                key=f"{prefix}_slider_{col}",
            )
        else:
            options = cfg.get("options") or FILTER_CONFIG["default_multiselect_options"]
            filter_selections[col] = st.multiselect(
                f"{col}",
                options=sorted(options),
                default=[],
                key=f"{prefix}_multi_{col}",
                placeholder="Search and select...",
            )


def user_changed_any_demographic_filter(selections: dict) -> bool:
    """Detect if user deviated from defaults (so we only warn when it matters)."""
    # Range defaults = full range
    age = selections.get("Age")
    if age and age != (FILTER_CONFIG["Age"]["min"], FILTER_CONFIG["Age"]["max"]):
        return True
    hh = selections.get("Household_Members")
    if hh and hh != (FILTER_CONFIG["Household_Members"]["min"], FILTER_CONFIG["Household_Members"]["max"]):
        return True

    # Any categorical selection non-empty
    for k, v in selections.items():
        if k in ("Age", "Household_Members"):
            continue
        if isinstance(v, list) and len(v) > 0:
            return True
    return False


###############################################################################
# Pre-ICA data helpers
###############################################################################

@st.cache_data(show_spinner=False)
def load_preica_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df["participant"] = df["ID"].astype(str).str.extract(r"^(sub-\d+)")
    df["task"] = df["ID"].astype(str).str.extract(r"task-([^_]+)")
    for c in df.columns:
        if c not in ["ID", "participant", "task"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def get_preica_channels(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in ["ID", "participant", "task"]]


def plot_preica_line(df: pd.DataFrame, channels: List[str]) -> alt.Chart:
    long_df = df.melt(id_vars=["participant"], value_vars=channels, var_name="channel", value_name="value")
    return (
        alt.Chart(long_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("channel:N", sort=channels, title="EEG channel"),
            y=alt.Y("value:Q", title="Pre-ICA extreme loss"),
            color=alt.Color("participant:N", title="Participant"),
            tooltip=["participant", "channel", "value"],
        )
        .properties(height=420)
        .interactive()
    )


def plot_preica_histograms(df: pd.DataFrame, channels: List[str]) -> None:
    apply_plot_style()
    n = len(channels)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.ravel(axes) if isinstance(axes, np.ndarray) else [axes]
    for i, ch in enumerate(channels):
        ax = axes[i]
        color = TUE_PALETTE[i % len(TUE_PALETTE)]
        data = df[ch].dropna()
        ax.hist(data, bins=20, color=color, edgecolor="black")
        ax.set_title(f"{ch}")
        ax.set_xlabel("Pre-ICA extreme loss")
        ax.set_ylabel("Count")
        ax.set_ylim(bottom=0)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_preica_boxplot(df: pd.DataFrame, channels: List[str]) -> None:
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(1 + 1.2 * len(channels), 5))
    data = [df[ch].dropna() for ch in channels]
    box = ax.boxplot(data, patch_artist=True, labels=channels)
    for patch, color in zip(box["boxes"], TUE_PALETTE):
        patch.set_facecolor(color)
    ax.set_title("Distribution of selected channels")
    ax.set_xlabel("EEG channel")
    ax.set_ylabel("Pre-ICA extreme loss")
    st.pyplot(fig)
    plt.close(fig)


###############################################################################
# Visual Oddball QC helpers
###############################################################################

def list_vo_participants(project_dir: str) -> List[str]:
    if not os.path.isdir(project_dir):
        return []
    files = glob.glob(os.path.join(project_dir, "**", "*.set"), recursive=True)
    ids = set()
    for f in files:
        m = re.search(r"(sub-\d+)", os.path.basename(f))
        if m:
            ids.add(m.group(1))
    return sorted(ids)


def find_vo_file(folder: str, pattern: str, participant_id: str) -> Optional[str]:
    if not os.path.isdir(VO_PROJECT):
        return None
    d = os.path.join(VO_PROJECT, folder)
    if not os.path.isdir(d):
        return None
    glob_pattern = f"{participant_id}*{pattern.replace('*','')}"
    hits = sorted(glob.glob(os.path.join(d, glob_pattern)))
    return hits[0] if hits else None


@st.cache_data(show_spinner=False)
def load_raw(path: str):
    if mne is None or not path or not os.path.exists(path):
        return None
    try:
        return mne.io.read_raw_eeglab(path, preload=True, verbose="ERROR")
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_epochs(path: str):
    if mne is None or not path or not os.path.exists(path):
        return None
    try:
        return mne.io.read_epochs_eeglab(path, verbose="ERROR")
    except Exception:
        return None


def plot_segment(raw_obj, title: str, channel: str = "Pz", duration: float = 5.0) -> None:
    apply_plot_style()
    picks = [raw_obj.ch_names.index(channel)] if channel in raw_obj.ch_names else [0]
    ch_name = channel if channel in raw_obj.ch_names else raw_obj.ch_names[0]
    sfreq = float(raw_obj.info["sfreq"])
    data, times = raw_obj[picks, : int(sfreq * duration)]
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(times, data[0] * 1e6, lw=0.9, color=TUE_PALETTE[0])
    ax.set_title(f"{title} ({ch_name})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (µV)")
    ax.axhline(0, color="gray", lw=0.5)
    st.pyplot(fig)
    plt.close(fig)


def plot_psd_overlay(raw_obj, preproc_obj, channel: str = "Pz") -> None:
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(8, 4))

    # --- New MNE: compute_psd returns a Spectrum object ---
    try:
        spec_raw = raw_obj.compute_psd(fmin=0.5, fmax=45.0, picks=channel, verbose="ERROR")
        spec_pre = preproc_obj.compute_psd(fmin=0.5, fmax=45.0, picks=channel, verbose="ERROR")

        freqs = spec_raw.freqs
        psd_raw_db = 10 * np.log10(spec_raw.get_data().squeeze())
        psd_pre_db = 10 * np.log10(spec_pre.get_data().squeeze())

    except Exception:
        # --- Old MNE fallback: psd_welch returns (psds, freqs) ---
        from mne.time_frequency import psd_welch

        psd_raw, freqs = psd_welch(raw_obj, fmin=0.5, fmax=45.0,
                                   picks=[channel] if channel in raw_obj.ch_names else None,
                                   verbose="ERROR")
        psd_pre, _ = psd_welch(preproc_obj, fmin=0.5, fmax=45.0,
                               picks=[channel] if channel in preproc_obj.ch_names else None,
                               verbose="ERROR")

        psd_raw_db = 10 * np.log10(psd_raw.squeeze())
        psd_pre_db = 10 * np.log10(psd_pre.squeeze())

    ax.plot(freqs, psd_raw_db, label="Raw", color=TUE_PALETTE[1])
    ax.plot(freqs, psd_pre_db, label="Preprocessed", color=TUE_PALETTE[2])
    ax.set_title(f"PSD overlay @ {channel}")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (dB, V²/Hz)")
    ax.legend()

    st.pyplot(fig)
    plt.close(fig)



def plot_preica_extremeloss(subj_df: pd.DataFrame) -> None:
    apply_plot_style()
    ch_cols = [c for c in subj_df.columns if c not in ["ID", "participant", "task"]]
    vals = subj_df[ch_cols].values.flatten().astype(float)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(ch_cols)), vals, color=TUE_PALETTE[3])
    ax.axhline(10, color="gray", linestyle="--", label="10% threshold")
    ax.set_xticks(range(len(ch_cols)))
    ax.set_xticklabels(ch_cols, rotation=90)
    ax.set_ylabel("% extreme artifact")
    ax.set_title("Pre-ICA extreme-loss per channel")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)


def plot_iclabel_summary(ic_df: pd.DataFrame) -> None:
    apply_plot_style()
    cols = ["Brain", "Muscle", "Eye", "Heart", "Line_Noise", "Channel_Noise", "Other"]
    if not set(cols).issubset(ic_df.columns):
        st.warning("ICLabel summary missing required columns.")
        return
    mean_probs = ic_df[cols].mean()
    fig, ax = plt.subplots()
    mean_probs.plot(
        kind="bar",
        ax=ax,
        color=[TUE_PALETTE[i % len(TUE_PALETTE)] for i in range(len(cols))],
    )
    ax.set_ylabel("Mean probability (%)")
    ax.set_title("Mean ICLabel class probabilities")
    st.pyplot(fig)
    plt.close(fig)


def qc_workflow(participant_id: str) -> None:
    st.markdown(f"### Participant: {participant_id}")

    st.subheader("1. Raw continuous data")
    raw_path = find_vo_file("01_raw", "*raw.set", participant_id)
    if raw_path:
        raw = load_raw(raw_path)
        if raw is not None:
            plot_segment(raw, "Raw EEG segment", "Pz", 5.0)
        else:
            st.error("Failed to load raw file.")
    else:
        st.warning("Raw file not found.")

    st.subheader("2. Preprocessed data and PSD overlay")
    pre_path = find_vo_file("02_preprocessed", "*preprocessed.set", participant_id)
    if pre_path:
        pre = load_raw(pre_path)
        if pre is not None:
            plot_segment(pre, "Preprocessed segment", "Pz", 5.0)
            if raw_path and raw is not None:
                plot_psd_overlay(raw, pre, "Pz")
        else:
            st.error("Failed to load preprocessed file.")
    else:
        st.warning("Preprocessed file not found.")

    st.subheader("3. Pre-ICA extreme-loss")
    loss_xlsx = os.path.join(VO_PROJECT, "03_preICA", "COCOA_preICAextremeloss.xlsx")
    if os.path.exists(loss_xlsx):
        df_loss = pd.read_excel(loss_xlsx)
        df_loss["participant"] = df_loss["ID"].astype(str).str.extract(r"^(sub-\d+)")
        df_loss["task"] = df_loss["ID"].astype(str).str.extract(r"task-([^_]+)")
        subj = df_loss[df_loss["ID"].astype(str).str.contains(participant_id)]
        if not subj.empty:
            plot_preica_extremeloss(subj)
        else:
            st.info("No loss data for this participant.")
    else:
        st.warning("Loss table not found.")

    st.subheader("4. ICLabel classification")
    ic_dir = os.path.join(VO_PROJECT, "05_ICLabel")
    ic_files = glob.glob(os.path.join(ic_dir, f"{participant_id}*_ICclassifications.xlsx"))
    if ic_files:
        ic_df = pd.read_excel(ic_files[0])
        plot_iclabel_summary(ic_df)
    else:
        st.warning("ICLabel file not found for this participant.")

    st.subheader("5. Post-ICA & corrected EOG")
    post_path = find_vo_file("06_postICA", "*postICA.set", participant_id)
    if post_path:
        post = load_raw(post_path)
        if post is not None:
            for ch in ["CVEOGR", "CHEOG"]:
                if ch in post.ch_names:
                    plot_segment(post, f"Corrected {ch}", ch, 5.0)
        else:
            st.error("Failed to load post-ICA file.")
    else:
        st.warning("Post-ICA file not found.")

    st.subheader("6. Epoched data & ERP")
    ep_path = find_vo_file("08_AR", "*autoAR.set", participant_id) or find_vo_file("07_epoched", "*epoched.set", participant_id)
    if ep_path:
        epochs = load_epochs(ep_path)
        if epochs is not None:
            keys = list(epochs.event_id.keys())
            target = next((k for k in keys if "11" in str(k)), keys[0])
            try:
                evk = epochs[target].average()
                fig = evk.plot(picks="Pz" if "Pz" in evk.ch_names else None, show=False)
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.error(f"Could not plot ERP: {e}")
        else:
            st.error("Failed to load epochs file.")
    else:
        st.warning("Epoched file not found.")

    st.markdown("---")


###############################################################################
# Views
###############################################################################

def preica_view() -> None:
    st.header("Pre-ICA extreme-loss exploration")

    if not os.path.exists(PREICA_XLSX):
        st.error(f"Missing {PREICA_XLSX} in current directory.")
        return

    df = load_preica_data(PREICA_XLSX)
    all_participants_from_xlsx = sorted(df["participant"].dropna().unique().tolist())
    channels = get_preica_channels(df)

    meta = load_participants_metadata()

    st.sidebar.header("Pre-ICA Filters")

    filter_selections: dict = {}

    with st.sidebar.expander("ID & core demographics", expanded=True):
        generate_filter_widgets_for_group(["Age", "Gender", "Handedness", "Highest_Edu"], filter_selections, prefix="preica_core")

    with st.sidebar.expander("SES & household info"):
        generate_filter_widgets_for_group(
            ["Occupation", "Employed", "Employed_Yes", "Income", "Household_Members"],
            filter_selections,
            prefix="preica_ses",
        )

    with st.sidebar.expander("Questionnaire & session details"):
        generate_filter_widgets_for_group(["fs1", "Project", "EEG_Tasks"], filter_selections, prefix="preica_qs")

    # Compute eligible participant list
    if meta is not None:
        filtered_meta = apply_demographic_filters_to_meta(meta, filter_selections)
        eligible = meta_participant_ids(filtered_meta)
        eligible = [p for p in eligible if p in all_participants_from_xlsx]
    else:
        eligible = all_participants_from_xlsx
        # Only warn if user actually changed any demographic filter away from defaults
        if user_changed_any_demographic_filter(filter_selections):
            st.sidebar.warning("participants.tsv not found → demographic filters cannot be applied (showing all participants).")

    if not eligible:
        st.warning("No participants match current demographic filters.")
        return

    # Participant selection is OPTIONAL
    prev_selected = st.session_state.get("preica_selected_participants", [])
    pruned_default = [p for p in prev_selected if p in eligible]

    selected_participants = st.sidebar.multiselect(
        "Participants (optional)",
        options=eligible,
        default=pruned_default,
        key="preica_selected_participants",
        placeholder="Leave empty = use all filtered participants",
    )

    st.sidebar.markdown("---")

    default_channels = channels[:3]
    selected_channels = st.sidebar.multiselect("EEG channels", channels, default=default_channels, key="preica_channels")
    plot_type = st.sidebar.radio("Plot type", ["Line chart", "Histogram", "Boxplot"], key="preica_plot_type")
    generate = st.sidebar.button("Generate analysis", type="primary", key="preica_generate")

    if not generate:
        st.info("Set demographic filters first (they filter the participant list). Click **Generate analysis**.")
        return

    # IMPORTANT: If user did NOT select participants, use ALL eligible
    active_participants = selected_participants if selected_participants else eligible
    filtered_df = df[df["participant"].isin(active_participants)].copy()

    st.subheader("Summary statistics")
    st.write(f"Eligible participants (after demographic filters): {len(eligible)}")
    st.write(f"Used participants: {len(active_participants)}")
    st.write(f"Records: {len(filtered_df)}")

    stats_rows = []
    for ch in selected_channels:
        vals = filtered_df[ch].dropna()
        stats_rows.append(
            {
                "channel": ch,
                "mean": float(vals.mean()) if len(vals) else np.nan,
                "std": float(vals.std()) if len(vals) else np.nan,
                "min": float(vals.min()) if len(vals) else np.nan,
                "max": float(vals.max()) if len(vals) else np.nan,
            }
        )
    st.dataframe(pd.DataFrame(stats_rows), use_container_width=True)

    st.subheader("Visualisation")
    if plot_type == "Line chart":
        # Altair isn't tueplots, but it is interactive. If you want “all tueplots” strictly,
        # switch this to Matplotlib too. Tell me and I’ll do it.
        st.altair_chart(plot_preica_line(filtered_df, selected_channels), use_container_width=True)
    elif plot_type == "Histogram":
        plot_preica_histograms(filtered_df, selected_channels)
    else:
        plot_preica_boxplot(filtered_df, selected_channels)

    if not TUEPLOTS_AVAILABLE:
        st.caption("Note: `tueplots` is not installed → using fallback Matplotlib styling. Install tueplots to match paper-ready style.")


def vo_qc_view() -> None:
    st.header("Visual Oddball QC workflow")

    if mne is None:
        st.error("MNE is not available. Install `mne` to enable QC plots.")
        return

    participants = list_vo_participants(VO_PROJECT)
    if not participants:
        st.warning("No Visual Oddball data found. Put EEGLAB .set files under Preprocessed_VisualOddball/")
        return

    st.sidebar.header("Visual Oddball Filters")
    selected_ids = st.sidebar.multiselect("Participants", participants, default=participants[:1], key="vo_participants")
    run_button = st.sidebar.button("Run QC workflow", type="primary", key="vo_run")

    if not run_button:
        st.info("Select participants and click **Run QC workflow**.")
        return

    for p in selected_ids:
        qc_workflow(p)

    if not TUEPLOTS_AVAILABLE:
        st.caption("Note: `tueplots` is not installed → using fallback Matplotlib styling. Install tueplots to match paper-ready style.")


###############################################################################
# Main
###############################################################################

def main() -> None:
    st.set_page_config(layout="wide", page_title="COCOA Dashboard")
    st.title("COCOA EEG Analysis Dashboard")

    view = st.sidebar.selectbox("Select view", ["Pre-ICA Metrics", "Visual Oddball QC"], key="main_view")

    if view == "Pre-ICA Metrics":
        preica_view()
    else:
        vo_qc_view()


if __name__ == "__main__":
    main()
