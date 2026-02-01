"""
COCOA Unified Streamlit Dashboard (Streamlit Cloud-ready)

Views
- Pre-ICA metrics (Excel): local if present, else fetched from Drive by filename
- Visual Oddball QC (Drive-backed): downloads EEGLAB .set + .fdt on-demand to /tmp cache

Secrets required (Streamlit Cloud -> App -> Settings -> Secrets)
- GDRIVE_VO_FOLDER_ID = "<folder id>"
- [gcp_service_account] ... (service account fields including private_key)

Notes
- If you set GDRIVE_VO_FOLDER_ID to the parent folder (DataLiteracyProject),
  the app will automatically locate the "Preprocessed_VisualOddball" child folder.
"""

from __future__ import annotations

import io
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt

# Optional MNE (needed for .set files)
try:
    import mne  # noqa: F401
except Exception:
    mne = None

# Google Drive API (required for Drive-backed views)
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload


# -----------------------------
# Local paths (repo)
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
LOCAL_PREICA_XLSX = BASE_DIR / "COCOA_preICAextremeloss.xlsx"
LOCAL_PARTICIPANTS_TSV = BASE_DIR / "participants.tsv"  # optional; not required by current UI

# Cache folder on Streamlit Cloud
CACHE_ROOT = Path("/tmp/cocoa_cache")
CACHE_ROOT.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Palette / style
# -----------------------------
TUE_PALETTE = ["#006AA3", "#E65C00", "#A31C34", "#5C8021", "#735545", "#4A6D8C"]


import os
from pathlib import Path

# expects:files_index
# - files_index: List[Dict]  (Drive recursive index)
# - CACHE_ROOT = Path("/tmp/cocoa_cache")
# - _download(file_id: str, dest: Path) -> Path
# - _find_by_name(files_index, name) -> Optional[Dict]

from pathlib import Path

# expects:
# - CACHE_ROOT = Path("/tmp/cocoa_cache")
# - _download(file_id: str, dest: Path) -> Path
# - _find_by_name(files_index, name) -> Optional[Dict]

def find_any_set(files_index, folder, pattern="*.set", participant_id=None):
    """
    Google Drive-backed version of your original find_any_set.

    Returns:
        local_path (str) to downloaded .set file in /tmp cache, or None
    """
    suffix = pattern.replace("*", "")
    if suffix == "":
        suffix = ".set"

    if participant_id:
        candidates = [
            f for f in files_index
            if f.get("name", "").startswith(str(participant_id)) and f["name"].endswith(suffix)
        ]
    else:
        candidates = [f for f in files_index if f.get("name", "").endswith(suffix)]

    candidates.sort(key=lambda x: x["name"])
    if not candidates:
        return None

    hit = candidates[0]
    set_name = hit["name"]

    local_set = CACHE_ROOT / folder / set_name
    _download(hit["id"], local_set)

    # Download matching .fdt if present
    if set_name.lower().endswith(".set"):
        fdt_name = set_name[:-4] + ".fdt"
        fdt_meta = _find_by_name(files_index, fdt_name)
        if fdt_meta:
            local_fdt = CACHE_ROOT / folder / fdt_name
            _download(fdt_meta["id"], local_fdt)

    return str(local_set)


def apply_plot_style() -> None:
    if st.session_state.get("_plot_style_applied"):
        return
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


# ============================================================
# 0) Drive auth + services
# ============================================================
def _drive_service():
    creds_info = dict(st.secrets["gcp_service_account"])
    creds = service_account.Credentials.from_service_account_info(
        creds_info,
        scopes=["https://www.googleapis.com/auth/drive.readonly"],
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def drive_smoke_test(folder_id: str) -> None:
    """Fail fast if secrets / sharing are wrong."""
    try:
        sa_email = st.secrets["gcp_service_account"]["client_email"]
        svc = _drive_service()

        # Check folder exists & is visible
        meta = svc.files().get(
            fileId=folder_id,
            fields="id,name,mimeType",
            supportsAllDrives=True,
        ).execute()

        if meta.get("mimeType") != "application/vnd.google-apps.folder":
            st.error("GDRIVE_VO_FOLDER_ID is not a folder.")
            st.stop()

        st.caption(f"Drive connected as: {sa_email}")
        st.caption(f"Root folder: {meta.get('name')} ({meta.get('id')})")

    except Exception as e:
        st.error("❌ Drive access failed. Check folder sharing + secrets.")
        st.exception(e)
        st.stop()


# ============================================================
# 1) Drive indexing + downloading
# ============================================================
@st.cache_data(show_spinner=False)
def drive_list_children(folder_id: str) -> List[Dict]:
    """List direct children of folder_id."""
    svc = _drive_service()
    items: List[Dict] = []
    page_token = None
    while True:
        resp = svc.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields="nextPageToken, files(id,name,mimeType)",
            pageToken=page_token,
            pageSize=1000,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
        items.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return items


@st.cache_data(show_spinner=False)
def drive_index_recursive(folder_id: str) -> List[Dict]:
    """Recursively index all files under folder_id. Returns [{id,name,mimeType}, ...]."""
    svc = _drive_service()
    out: List[Dict] = []
    queue = [folder_id]

    while queue:
        fid = queue.pop()
        page_token = None
        while True:
            resp = svc.files().list(
                q=f"'{fid}' in parents and trashed=false",
                fields="nextPageToken, files(id,name,mimeType)",
                pageToken=page_token,
                pageSize=1000,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            ).execute()

            for f in resp.get("files", []):
                if f["mimeType"] == "application/vnd.google-apps.folder":
                    queue.append(f["id"])
                else:
                    out.append(f)

            page_token = resp.get("nextPageToken")
            if not page_token:
                break

    return out


def _download(file_id: str, dest: Path) -> Path:
    """Download a Drive file to dest (cached)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    svc = _drive_service()
    request = svc.files().get_media(fileId=file_id, supportsAllDrives=True)

    with io.FileIO(dest, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()

    return dest


def _find_by_name(files: List[Dict], name: str) -> Optional[Dict]:
    return next((f for f in files if f["name"] == name), None)


def find_first_match(files: List[Dict], *, contains: str, endswith: str) -> Optional[Dict]:
    hits = [f for f in files if contains in f["name"] and f["name"].endswith(endswith)]
    hits.sort(key=lambda x: x["name"])
    return hits[0] if hits else None


def participants_from_index(files: List[Dict]) -> List[str]:
    ids = set()
    for f in files:
        m = re.search(r"(sub-\d+)", f["name"])
        if m:
            ids.add(m.group(1))
    return sorted(ids)


def download_set_and_pair(files: List[Dict], set_file: Dict, stage: str) -> Path:
    """
    Download .set and (if present) matching .fdt into the same stage folder.
    Returns local path to the .set file.
    """
    set_name = set_file["name"]
    local_set = CACHE_ROOT / stage / set_name
    _download(set_file["id"], local_set)

    if set_name.lower().endswith(".set"):
        fdt_name = set_name[:-4] + ".fdt"
        fdt_file = _find_by_name(files, fdt_name)
        if fdt_file:
            local_fdt = CACHE_ROOT / stage / fdt_name
            _download(fdt_file["id"], local_fdt)

    return local_set


def resolve_vo_root_folder_id(root_id: str) -> str:
    """
    If root_id is the parent (DataLiteracyProject), find child folder named
    'Preprocessed_VisualOddball'. If root_id already points to that folder, return as-is.
    """
    svc = _drive_service()
    meta = svc.files().get(fileId=root_id, fields="id,name,mimeType", supportsAllDrives=True).execute()
    name = meta.get("name", "")
    if name == "Preprocessed_VisualOddball":
        return root_id

    # Look for child folder
    children = drive_list_children(root_id)
    for c in children:
        if c.get("mimeType") == "application/vnd.google-apps.folder" and c.get("name") == "Preprocessed_VisualOddball":
            return c["id"]

    # If not found, just return root_id (so error messages are meaningful later)
    return root_id


# ============================================================
# 2) Pre-ICA helpers
# ============================================================
@st.cache_data(show_spinner=False)
def load_preica_data_from_excel(path: str) -> pd.DataFrame:
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
        data = df[ch].dropna()
        ax.hist(data, bins=20, color=TUE_PALETTE[i % len(TUE_PALETTE)], edgecolor="black")
        ax.set_title(ch)
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


# ============================================================
# 3) VO QC helpers (MNE)
# ============================================================
@st.cache_data(show_spinner=False)
def load_raw_eeglab(path: str):
    if mne is None:
        return None
    try:
        return mne.io.read_raw_eeglab(path, preload=True, verbose="ERROR")
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_epochs_eeglab(path: str):
    if mne is None:
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

        psd_raw, freqs = psd_welch(
            raw_obj,
            fmin=0.5,
            fmax=45.0,
            picks=[channel] if channel in raw_obj.ch_names else None,
            verbose="ERROR",
        )
        psd_pre, _ = psd_welch(
            preproc_obj,
            fmin=0.5,
            fmax=45.0,
            picks=[channel] if channel in preproc_obj.ch_names else None,
            verbose="ERROR",
        )
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


# ============================================================
# 4) Views
# ============================================================
def preica_view(files_index: Optional[List[Dict]] = None) -> None:
    st.header("Pre-ICA extreme-loss exploration")

    excel_path: Optional[str] = None
    if LOCAL_PREICA_XLSX.exists():
        excel_path = str(LOCAL_PREICA_XLSX)
        st.caption("Using local COCOA_preICAextremeloss.xlsx")
    else:
        if files_index is not None:
            xlsx = _find_by_name(files_index, "COCOA_preICAextremeloss.xlsx")
            if xlsx:
                local = CACHE_ROOT / "preica" / xlsx["name"]
                _download(xlsx["id"], local)
                excel_path = str(local)
                st.caption("Using Drive COCOA_preICAextremeloss.xlsx (cached)")
    if excel_path is None:
        st.error("Pre-ICA Excel not found (local or Drive).")
        return

    df = load_preica_data_from_excel(excel_path)
    channels = get_preica_channels(df)
    participants = sorted(df["participant"].dropna().unique().tolist())

    st.sidebar.subheader("Pre-ICA controls")
    selected_participants = st.sidebar.multiselect(
        "Participants (optional)",
        options=participants,
        default=[],
        placeholder="Leave empty = all participants",
        key="preica_participants",
    )
    default_channels = channels[:3] if channels else []
    selected_channels = st.sidebar.multiselect(
        "EEG channels",
        options=channels,
        default=default_channels,
        key="preica_channels",
    )
    plot_type = st.sidebar.radio("Plot type", ["Line chart", "Histogram", "Boxplot"], key="preica_plot")
    run = st.sidebar.button("Generate analysis", type="primary", key="preica_run")

    if not run:
        st.info("Select channels and click **Generate analysis**.")
        return
    if not selected_channels:
        st.warning("Select at least one EEG channel.")
        return

    active = selected_participants if selected_participants else participants
    fdf = df[df["participant"].isin(active)].copy()

    st.subheader("Summary")
    st.write(f"Participants used: {len(active)}")
    st.write(f"Records: {len(fdf)}")

    stats = []
    for ch in selected_channels:
        vals = fdf[ch].dropna()
        stats.append(
            {
                "channel": ch,
                "mean": float(vals.mean()) if len(vals) else np.nan,
                "std": float(vals.std()) if len(vals) else np.nan,
                "min": float(vals.min()) if len(vals) else np.nan,
                "max": float(vals.max()) if len(vals) else np.nan,
            }
        )
    st.dataframe(pd.DataFrame(stats), use_container_width=True)

    st.subheader("Visualisation")
    if plot_type == "Line chart":
        st.altair_chart(plot_preica_line(fdf, selected_channels), use_container_width=True)
    elif plot_type == "Histogram":
        plot_preica_histograms(fdf, selected_channels)
    else:
        plot_preica_boxplot(fdf, selected_channels)


def vo_qc_view(files_index: List[Dict]) -> None:
    st.header("Visual Oddball QC (Google Drive)")

    if mne is None:
        st.error("MNE is not available. Add `mne` to requirements.txt.")
        return

    participants = participants_from_index(files_index)
    if not participants:
        st.warning("No participant files found in this Drive folder.")
        return

    st.sidebar.subheader("VO QC controls")
    pid = st.sidebar.selectbox("Participant", participants, index=0, key="vo_pid")
    run = st.sidebar.button("Run QC", type="primary", key="vo_run")

    if not run:
        st.info("Pick a participant and click **Run QC**.")
        return

    st.markdown(f"### Participant: `{pid}`")

    # Stage suffix patterns (your dataset naming)
    stages: List[Tuple[str, str]] = [
        ("01_raw", "_eeg_raw.set"),
        ("02_preprocessed", "_eeg_preprocessed.set"),
        ("03_preICA", "_eeg_preICA.set"),
        ("04_ICAweighted", "_eegICA_weighted.set"),
        ("06_postICA", "_eegpostICA.set"),
        ("06_postICApracticeremoved", "_eegpostICA_practiceremoved.set"),
        ("07_epoched", "_epoched.set"),
    ]

    # 1) Raw
    st.header("[1] Raw Continuous Dataset")

    path = find_any_set("01_raw", "*raw.set", pid)
    if not path:
        st.warning("Raw file not found.")
    else:
        try:
            raw_obj = mne.io.read_raw_eeglab(path, preload=True, verbose="ERROR")
            st.write(f"**File:** {os.path.basename(path)}")
            st.info("Expected: Visible drifts, line noise, and large blinks.")
            plot_segment_st(raw_obj, "Raw EEG Segment", "Pz")
        except Exception as e:
            st.error("MNE could not open raw file.")
            st.exception(e)

    # 2) Preprocessed + PSD overlay
    st.subheader("2) Preprocessed data and PSD overlay")
    pre_set = find_first_match(files_index, contains=pid, endswith=stages[1][1])
    if pre_set:
        pre_path = download_set_and_pair(files_index, pre_set, "02_preprocessed")
        pre_obj = load_raw_eeglab(str(pre_path))
        if pre_obj is not None:
            plot_segment(pre_obj, "Preprocessed segment", "Pz", 5.0)
            if raw_obj is not None:
                plot_psd_overlay(raw_obj, pre_obj, "Pz")
        else:
            st.error("Failed to load preprocessed .set (downloaded).")
    else:
        st.warning("Preprocessed .set not found for this participant.")

    # 3) Pre-ICA extreme-loss Excel (optional)
    st.subheader("3) Pre-ICA extreme-loss table (optional)")
    preica_xlsx = _find_by_name(files_index, "COCOA_preICAextremeloss.xlsx")
    if preica_xlsx:
        local_xlsx = CACHE_ROOT / "03_preICA" / preica_xlsx["name"]
        _download(preica_xlsx["id"], local_xlsx)
        df_loss = pd.read_excel(local_xlsx)
        if "ID" in df_loss.columns:
            subj = df_loss[df_loss["ID"].astype(str).str.contains(pid)]
            if not subj.empty:
                ch_cols = [c for c in subj.columns if c not in ["ID"]]
                vals = subj[ch_cols].values.flatten().astype(float)
                apply_plot_style()
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.bar(range(len(ch_cols)), vals, color=TUE_PALETTE[3])
                ax.set_xticks(range(len(ch_cols)))
                ax.set_xticklabels(ch_cols, rotation=90)
                ax.set_ylabel("% extreme artifact")
                ax.set_title("Pre-ICA extreme-loss per channel (from table)")
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("No loss data for this participant in the Excel table.")
        else:
            st.warning("Excel table found but missing 'ID' column.")
    else:
        st.info("COCOA_preICAextremeloss.xlsx not found in this Drive folder.")

    # 4) ICLabel xlsx (optional)
    st.subheader("4) ICLabel classification (optional)")
    ic_xlsx = find_first_match(files_index, contains=pid, endswith="_ICclassifications.xlsx")
    if ic_xlsx:
        local_ic = CACHE_ROOT / "05_ICLabel" / ic_xlsx["name"]
        _download(ic_xlsx["id"], local_ic)
        ic_df = pd.read_excel(local_ic)

        cols = ["Brain", "Muscle", "Eye", "Heart", "Line_Noise", "Channel_Noise", "Other"]
        if set(cols).issubset(ic_df.columns):
            apply_plot_style()
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
        else:
            st.warning("ICLabel file found but required columns are missing.")
            st.write("Columns:", list(ic_df.columns))
    else:
        st.info("ICLabel xlsx not found for this participant.")

    # 5) Epoched / AutoAR (optional)
    st.subheader("5) Epoched data & ERP (optional)")
    ar_hits = [f for f in files_index if pid in f["name"] and "autoAR" in f["name"] and f["name"].endswith(".set")]
    ar_hits.sort(key=lambda x: x["name"])
    ar_set = ar_hits[0] if ar_hits else None

    ep_set = find_first_match(files_index, contains=pid, endswith="_epoched.set")
    chosen = ar_set or ep_set
    stage = "08_AR" if ar_set else "07_epoched"

    if chosen:
        ep_path = download_set_and_pair(files_index, chosen, stage)
        epochs = load_epochs_eeglab(str(ep_path))
        if epochs is not None:
            keys = list(epochs.event_id.keys())
            target = keys[0] if keys else None
            if target:
                try:
                    evk = epochs[target].average()
                    fig = evk.plot(picks="Pz" if "Pz" in evk.ch_names else None, show=False)
                    st.pyplot(fig)
                    plt.close(fig)
                except Exception as e:
                    st.error(f"Could not plot ERP: {e}")
            else:
                st.warning("No events found in epochs.")
        else:
            st.error("Failed to load epochs .set (downloaded).")
    else:
        st.info("No epoched/autoAR .set found for this participant.")


# ============================================================
# 5) Sidebar tools
# ============================================================
def sidebar_tools() -> None:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Tools")

    if st.sidebar.button("Clear local cache (/tmp)", help="Deletes downloaded files under /tmp/cocoa_cache"):
        try:
            shutil.rmtree(CACHE_ROOT, ignore_errors=True)
            CACHE_ROOT.mkdir(parents=True, exist_ok=True)
            st.sidebar.success("Cache cleared.")
        except Exception as e:
            st.sidebar.error(f"Failed to clear cache: {e}")

    if st.sidebar.button("Reindex Drive", help="Clears cached Drive index and refetches"):
        drive_index_recursive.clear()
        drive_list_children.clear()
        st.sidebar.success("Drive index cache cleared. Reload to refetch.")


# ============================================================
# 6) Main
# ============================================================
def main() -> None:
    st.set_page_config(layout="wide", page_title="COCOA Dashboard")
    st.title("COCOA EEG Analysis Dashboard")

    root_folder_id = st.secrets["GDRIVE_VO_FOLDER_ID"]

    # Verify Drive access to the configured root
    drive_smoke_test(root_folder_id)

    # Resolve VO folder (in case user pointed at parent folder)
    vo_folder_id = resolve_vo_root_folder_id(root_folder_id)

    # If we failed to locate the VO folder, warn clearly
    if vo_folder_id == root_folder_id:
        # It might already be the correct folder OR it might be parent and missing VO folder
        # We can detect by listing children
        children = drive_list_children(root_folder_id)
        child_names = {c.get("name") for c in children}
        if "Preprocessed_VisualOddball" in child_names:
            vo_folder_id = next(c["id"] for c in children if c.get("name") == "Preprocessed_VisualOddball")

    # Index under VO folder for QC view, and (optionally) for pulling Excel
    files_index = drive_index_recursive(vo_folder_id)

    sidebar_tools()

    view = st.sidebar.selectbox("Select view", ["Pre-ICA Metrics", "Visual Oddball QC"], key="main_view")

    if view == "Pre-ICA Metrics":
        preica_view(files_index=files_index)
    else:
        vo_qc_view(files_index=files_index)


if __name__ == "__main__":
    main()
