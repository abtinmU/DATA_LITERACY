"""
Unified Streamlit dashboard for the COCOA project.

Features:
- Pre-ICA extreme-loss exploration (Excel)
- Visual Oddball QC workflow (EEGLAB .set files) if present
- Demographic filters (Age, Gender, Income, etc.) applied FIRST to determine eligible participants
- Participant selection is optional: if empty, use all eligible participants
- Uses tueplots if available (optional). Falls back gracefully if not installed.

Cloud-safe:
- Uses paths relative to this file (Path(__file__).parent) instead of os.getcwd()
"""

from __future__ import annotations

import re
import glob
from pathlib import Path
from typing import List, Optional, Dict

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt

# Optional MNE for EEGLAB support
try:
    import mne  # noqa: F401
except Exception:
    mne = None

# Optional tueplots for style bundles
try:
    from tueplots import bundles  # type: ignore
    TUEPLOTS_AVAILABLE = True
except Exception:
    bundles = None
    TUEPLOTS_AVAILABLE = False


###############################################################################
# Paths (cloud-safe)
###############################################################################

BASE_DIR = Path(__file__).resolve().parent

PREICA_XLSX = BASE_DIR / "COCOA_preICAextremeloss.xlsx"
PARTICIPANTS_TSV = BASE_DIR / "participants.tsv"
VO_PROJECT = BASE_DIR / "Preprocessed_VisualOddball"  # from your screenshot


###############################################################################
# Palette
###############################################################################

TUE_PALETTE = [
    "#006AA3",  # blue
    "#E65C00",  # orange
    "#A31C34",  # red
    "#5C8021",  # green
    "#735545",  # brown
    "#4A6D8C",  # dark blue
]


###############################################################################
# Filters config (same idea as your unified file)
###############################################################################

FILTER_CONFIG = {
    "Age": {"type": "range", "min": 18, "max": 80, "step": 1},
    "Household_Members": {"type": "range", "min": 1, "max": 10, "step": 1},
    "Gender": {"type": "multiselect", "options": ["female", "male", "other"]},
    "Handedness": {"type": "multiselect", "options": ["right", "left", "ambidextrous"]},
    "Highest_Edu": {"type": "multiselect", "options": []},   # will be derived if present
    "Occupation": {"type": "multiselect", "options": []},    # will be derived if present
    "Employed": {"type": "multiselect", "options": ["yes", "no"]},
    "Employed_Yes": {"type": "multiselect", "options": []},  # derived if present
    "Income": {"type": "multiselect", "options": []},        # derived if present
    "Project": {"type": "multiselect", "options": []},       # derived if present
    "EEG_Tasks": {"type": "multiselect", "options": []},     # derived if present
    "fs1": {"type": "multiselect", "options": []},           # derived if present
}

META_COL_ALIASES: Dict[str, List[str]] = {
    "participant_id": ["participant_id", "subject_id", "sub_id", "id", "participant", "subject"],
    "Age": ["age", "Age"],
    "Household_Members": ["household_members", "householdmembers", "household_members_count", "household_members"],
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


###############################################################################
# Plot style
###############################################################################

def apply_plot_style() -> None:
    if st.session_state.get("_plot_style_applied"):
        return

    if TUEPLOTS_AVAILABLE:
        plt.rcParams.update(bundles.icml2024(column="full", nrows=1, ncols=1))
    else:
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

    st.session_state["_plot_style_applied"] = True


###############################################################################
# Loaders
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


@st.cache_data(show_spinner=False)
def load_participants_metadata(tsv_path: str) -> Optional[pd.DataFrame]:
    if not Path(tsv_path).is_file():
        return None
    try:
        return pd.read_csv(tsv_path, sep="\t")
    except Exception:
        return None


###############################################################################
# Demographic filtering
###############################################################################

def _derive_options_if_possible(meta: pd.DataFrame, field: str) -> List[str]:
    col = _find_first_col(meta, META_COL_ALIASES.get(field, [field]))
    if not col:
        return []
    vals = (
        meta[col]
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", np.nan)
        .dropna()
        .unique()
        .tolist()
    )
    # Keep it tidy
    vals_sorted = sorted(vals, key=lambda x: x.lower())
    # Avoid insane lists
    return vals_sorted[:200]


def apply_demographic_filters_to_meta(df_meta: pd.DataFrame, selections: dict) -> pd.DataFrame:
    out = df_meta.copy()

    # Age range
    age_sel = selections.get("Age")
    if age_sel:
        col = _find_first_col(out, META_COL_ALIASES["Age"])
        if col:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            a_min, a_max = age_sel
            out = out[(out[col] >= a_min) & (out[col] <= a_max)]

    # Household members range
    hh_sel = selections.get("Household_Members")
    if hh_sel:
        col = _find_first_col(out, META_COL_ALIASES["Household_Members"])
        if col:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            h_min, h_max = hh_sel
            out = out[(out[col] >= h_min) & (out[col] <= h_max)]

    # Categorical
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


def generate_filter_widgets(meta: Optional[pd.DataFrame], selections: dict, *, prefix: str) -> None:
    # Range sliders
    selections["Age"] = st.slider(
        "Age range",
        min_value=FILTER_CONFIG["Age"]["min"],
        max_value=FILTER_CONFIG["Age"]["max"],
        value=(FILTER_CONFIG["Age"]["min"], FILTER_CONFIG["Age"]["max"]),
        step=FILTER_CONFIG["Age"]["step"],
        key=f"{prefix}_age",
    )
    selections["Household_Members"] = st.slider(
        "Household members range",
        min_value=FILTER_CONFIG["Household_Members"]["min"],
        max_value=FILTER_CONFIG["Household_Members"]["max"],
        value=(FILTER_CONFIG["Household_Members"]["min"], FILTER_CONFIG["Household_Members"]["max"]),
        step=FILTER_CONFIG["Household_Members"]["step"],
        key=f"{prefix}_hh",
    )

    # Categorical filters
    cat_fields = ["Gender", "Handedness", "Highest_Edu", "Occupation", "Employed", "Employed_Yes",
                  "Income", "Project", "EEG_Tasks", "fs1"]

    for field in cat_fields:
        options = FILTER_CONFIG[field].get("options") or []
        if meta is not None and not options:
            # derive from metadata if possible
            options = _derive_options_if_possible(meta, field)

        # keep Gender/Handedness defaults if metadata missing
        if not options and FILTER_CONFIG[field].get("options"):
            options = FILTER_CONFIG[field]["options"]

        selections[field] = st.multiselect(
            field,
            options=options,
            default=[],
            key=f"{prefix}_{field}",
            placeholder="Search and select...",
        )


###############################################################################
# Pre-ICA plotting
###############################################################################

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
# Visual Oddball QC helpers (optional)
###############################################################################

def list_vo_participants(project_dir: Path) -> List[str]:
    if not project_dir.is_dir():
        return []
    files = project_dir.rglob("*.set")
    ids = set()
    for f in files:
        m = re.search(r"(sub-\d+)", f.name)
        if m:
            ids.add(m.group(1))
    return sorted(ids)


def find_vo_file(folder: str, pattern: str, participant_id: str) -> Optional[str]:
    d = VO_PROJECT / folder
    if not d.is_dir():
        return None
    # pattern example: "*raw.set" => we just use it as suffix-ish
    hits = sorted(glob.glob(str(d / f"{participant_id}*{pattern.replace('*','')}")))
    return hits[0] if hits else None


@st.cache_data(show_spinner=False)
def load_raw(path: str):
    if mne is None or not path or not Path(path).exists():
        return None
    try:
        return mne.io.read_raw_eeglab(path, preload=True, verbose="ERROR")
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_epochs(path: str):
    if mne is None or not path or not Path(path).exists():
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

    try:
        spec_raw = raw_obj.compute_psd(fmin=0.5, fmax=45.0, picks=channel, verbose="ERROR")
        spec_pre = preproc_obj.compute_psd(fmin=0.5, fmax=45.0, picks=channel, verbose="ERROR")
        freqs = spec_raw.freqs
        psd_raw_db = 10 * np.log10(spec_raw.get_data().squeeze())
        psd_pre_db = 10 * np.log10(spec_pre.get_data().squeeze())
    except Exception:
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


def qc_workflow(participant_id: str) -> None:
    st.markdown(f"### Participant: {participant_id}")

    st.subheader("1) Raw continuous data")
    raw_path = find_vo_file("01_raw", "*raw.set", participant_id)
    if raw_path:
        raw = load_raw(raw_path)
        if raw is not None:
            plot_segment(raw, "Raw EEG segment", "Pz", 5.0)
        else:
            st.error("Failed to load raw file.")
    else:
        st.warning("Raw file not found.")

    st.subheader("2) Preprocessed data + PSD overlay")
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

    st.subheader("3) Epoched data & ERP (if available)")
    ep_path = find_vo_file("08_AR", "*autoAR.set", participant_id) or find_vo_file("07_epoched", "*epoched.set", participant_id)
    if ep_path:
        epochs = load_epochs(ep_path)
        if epochs is not None:
            keys = list(epochs.event_id.keys())
            target = keys[0]
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

    if not PREICA_XLSX.exists():
        st.error(f"Missing file: {PREICA_XLSX.name} (expected next to this script).")
        st.stop()

    df_preica = load_preica_data(str(PREICA_XLSX))
    channels = get_preica_channels(df_preica)

    meta = load_participants_metadata(str(PARTICIPANTS_TSV))

    st.sidebar.header("Demographic filters (apply first)")
    filter_selections: dict = {}
    generate_filter_widgets(meta, filter_selections, prefix="demo")

    # Eligible list comes from metadata, then intersect with Excel participants
    all_from_xlsx = sorted(df_preica["participant"].dropna().unique().tolist())

    if meta is not None:
        filtered_meta = apply_demographic_filters_to_meta(meta, filter_selections)
        eligible = meta_participant_ids(filtered_meta)
        eligible = [p for p in eligible if p in all_from_xlsx]
    else:
        eligible = all_from_xlsx
        st.sidebar.warning("participants.tsv not found → demographic filters cannot be applied (showing all participants).")

    if not eligible:
        st.warning("No participants match the current demographic filters.")
        return

    st.sidebar.markdown("---")
    prev_selected = st.session_state.get("preica_selected_participants", [])
    pruned_default = [p for p in prev_selected if p in eligible]

    selected_participants = st.sidebar.multiselect(
        "Participants (optional)",
        options=eligible,
        default=pruned_default,
        key="preica_selected_participants",
        placeholder="Leave empty = use all eligible participants",
    )

    default_channels = channels[:3] if channels else []
    selected_channels = st.sidebar.multiselect(
        "EEG channels",
        options=channels,
        default=default_channels,
        key="preica_channels",
    )

    plot_type = st.sidebar.radio(
        "Plot type",
        options=["Line chart", "Histogram", "Boxplot"],
        key="preica_plot_type",
    )

    generate = st.sidebar.button("Generate analysis", type="primary", key="preica_generate")

    if not generate:
        st.info("Set demographic filters first. Then click **Generate analysis**.")
        return

    if not selected_channels:
        st.warning("Select at least one EEG channel.")
        return

    # If user did NOT select participants, use ALL eligible
    active_participants = selected_participants if selected_participants else eligible
    filtered_df = df_preica[df_preica["participant"].isin(active_participants)].copy()

    st.subheader("Summary")
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
        st.altair_chart(plot_preica_line(filtered_df, selected_channels), use_container_width=True)
    elif plot_type == "Histogram":
        plot_preica_histograms(filtered_df, selected_channels)
    else:
        plot_preica_boxplot(filtered_df, selected_channels)

    if not TUEPLOTS_AVAILABLE:
        st.caption("Note: `tueplots` not installed → using fallback Matplotlib styling.")


def vo_qc_view() -> None:
    st.header("Visual Oddball QC workflow")

    if mne is None:
        st.error("MNE is not available. Add `mne` to requirements.txt to enable QC plots.")
        return

    if not VO_PROJECT.is_dir():
        st.warning(f"No folder found: {VO_PROJECT.name} (expected next to this script).")
        return

    participants = list_vo_participants(VO_PROJECT)
    if not participants:
        st.warning("No .set files found under Preprocessed_VisualOddball/.")
        return

    st.sidebar.header("Visual Oddball QC")
    selected_ids = st.sidebar.multiselect(
        "Participants",
        options=participants,
        default=participants[:1],
        key="vo_participants",
    )
    run_button = st.sidebar.button("Run QC workflow", type="primary", key="vo_run")

    if not run_button:
        st.info("Select participants and click **Run QC workflow**.")
        return

    for p in selected_ids:
        qc_workflow(p)

    if not TUEPLOTS_AVAILABLE:
        st.caption("Note: `tueplots` not installed → using fallback Matplotlib styling.")


###############################################################################
# Main
###############################################################################

def main() -> None:
    st.set_page_config(layout="wide", page_title="COCOA Dashboard")
    st.title("COCOA EEG Analysis Dashboard")

    view = st.sidebar.selectbox(
        "Select view",
        options=["Pre-ICA Metrics", "Visual Oddball QC"],
        key="main_view",
    )

    if view == "Pre-ICA Metrics":
        preica_view()
    else:
        vo_qc_view()


if __name__ == "__main__":
    main()
