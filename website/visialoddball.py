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

# EEG
try:
    import mne
except Exception:
    mne = None

# Drive
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload


# -----------------------------
# Local paths (repo)
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
LOCAL_PREICA_XLSX = BASE_DIR / "COCOA_preICAextremeloss.xlsx"

# Cache folder on Streamlit Cloud
CACHE_ROOT = Path("/tmp/cocoa_cache")
CACHE_ROOT.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Palette / style
# -----------------------------
TUE_PALETTE = ["#006AA3", "#E65C00", "#A31C34", "#5C8021", "#735545", "#4A6D8C"]


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
# Drive auth + services
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
        svc = _drive_service()
        meta = svc.files().get(
            fileId=folder_id,
            fields="id,name,mimeType",
            supportsAllDrives=True,
        ).execute()
        if meta.get("mimeType") != "application/vnd.google-apps.folder":
            st.error("GDRIVE_VO_FOLDER_ID must point to a folder.")
            st.stop()
        st.caption(f"✅ Drive folder reachable: **{meta.get('name')}**")
    except Exception as e:
        st.error("❌ Drive access failed. Check folder sharing + secrets.")
        st.exception(e)
        st.stop()


@st.cache_data(show_spinner=False)
def drive_list_children(folder_id: str) -> List[Dict]:
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


def resolve_preprocessed_vo_folder_id(root_id: str) -> str:
    """
    If root_id points to DataLiteracyProject, find the Preprocessed_VisualOddball child.
    If it already points to Preprocessed_VisualOddball, return as-is.
    """
    svc = _drive_service()
    meta = svc.files().get(fileId=root_id, fields="id,name,mimeType", supportsAllDrives=True).execute()
    if meta.get("name") == "Preprocessed_VisualOddball":
        return root_id

    children = drive_list_children(root_id)
    for c in children:
        if c.get("mimeType") == "application/vnd.google-apps.folder" and c.get("name") == "Preprocessed_VisualOddball":
            return c["id"]

    # If not found, return root_id (and user will see helpful warnings)
    return root_id


@st.cache_data(show_spinner=False)
def drive_index_recursive(folder_id: str) -> List[Dict]:
    """
    Recursively index ALL files under Drive folder_id.
    Returns list of dicts: {id, name, mimeType}
    """
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
    Download .set and (if present) matching .fdt into the same folder.
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


# ============================================================
# Pre-ICA helpers
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
# EEG loading + plotting (FIXED: show real errors)
# ============================================================
def require_mne() -> None:
    if mne is None:
        st.error("MNE is not installed. Add `mne` to requirements.txt.")
        st.stop()
    # Quick hint for common missing deps
    try:
        import pymatreader  # noqa
    except Exception:
        st.warning("`pymatreader` missing. Add it to requirements.txt for EEGLAB loading.")
    try:
        import h5py  # noqa
    except Exception:
        st.warning("`h5py` missing. Add it if your .set files are MATLAB v7.3 (HDF5).")


def load_raw_eeglab_strict(path: str):
    """Load Raw and raise errors (do not swallow)."""
    return mne.io.read_raw_eeglab(path, preload=True, verbose="ERROR")


def load_epochs_eeglab_strict(path: str):
    """Load Epochs and raise errors (do not swallow)."""
    return mne.io.read_epochs_eeglab(path, verbose="ERROR")


def plot_segment(raw_obj, title: str, channel: str = "Pz", duration: float = 5.0) -> None:
    apply_plot_style()
    picks = [raw_obj.ch_names.index(channel)] if channel in raw_obj.ch_names else [0]
    ch_name = channel if channel in raw_obj.ch_names else raw_obj.ch_names[0]
    sfreq = float(raw_obj.info["sfreq"])
    data, times = raw_obj[picks, : int(sfreq * duration)]
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(times, data[0] * 1e6, lw=0.9)
    ax.set_title(f"{title} ({ch_name})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (µV)")
    ax.axhline(0, color="gray", lw=0.5)
    st.pyplot(fig)
    plt.close(fig)


def compute_psd_db(raw_obj, channel: str, fmin: float = 0.5, fmax: float = 45.0) -> Tuple[np.ndarray, np.ndarray]:
    """Return (freqs, psd_db) for one channel."""
    try:
        spec = raw_obj.compute_psd(fmin=fmin, fmax=fmax, picks=channel, verbose="ERROR")
        freqs = spec.freqs
        psd_db = 10 * np.log10(spec.get_data().squeeze())
        return freqs, psd_db
    except Exception:
        from mne.time_frequency import psd_welch
        psd, freqs = psd_welch(raw_obj, fmin=fmin, fmax=fmax, picks=[channel] if channel in raw_obj.ch_names else None, verbose="ERROR")
        psd_db = 10 * np.log10(psd.squeeze())
        return freqs, psd_db


def plot_psd_overlay(raw_obj, preproc_obj, channel: str = "Pz") -> None:
    apply_plot_style()
    freqs1, psd1 = compute_psd_db(raw_obj, channel)
    freqs2, psd2 = compute_psd_db(preproc_obj, channel)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(freqs1, psd1, label="Raw")
    ax.plot(freqs2, psd2, label="Preprocessed")
    ax.set_title(f"PSD overlay @ {channel}")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (dB, V²/Hz)")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)


def bandpower_from_psd(freqs: np.ndarray, psd_db: np.ndarray, band: Tuple[float, float]) -> float:
    """Mean PSD(dB) in a band."""
    lo, hi = band
    mask = (freqs >= lo) & (freqs <= hi)
    if not np.any(mask):
        return float("nan")
    return float(np.nanmean(psd_db[mask]))


# ============================================================
# Views
# ============================================================
def preica_view(files_index: Optional[List[Dict]] = None) -> None:
    st.header("Pre-ICA extreme-loss exploration")

    excel_path: Optional[str] = None
    if LOCAL_PREICA_XLSX.exists():
        excel_path = str(LOCAL_PREICA_XLSX)
        st.caption("Using local COCOA_preICAextremeloss.xlsx")
    else:
        if files_index is not None:
            xlsx = _find_by_name(files_index, "COCOA_preICAextremeloss.xlsx") or _find_by_name(files_index, "COCOA_preICAextremeloss.xlsx")
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


def vo_qc_single(files_index: List[Dict], pid: str) -> None:
    """Single participant QC based on your folder/file naming."""
    require_mne()

    # Stage suffix patterns (based on your screenshots)
    suffix_raw = "_task-visualoddball_eeg_raw.set"
    suffix_pre = "_task-visualoddball_eeg_preprocessed.set"
    suffix_ep = "_task-visualoddball_epoched.set"

    st.subheader("1) Raw continuous data")
    raw_set = find_first_match(files_index, contains=pid, endswith=suffix_raw)
    raw_obj = None
    if raw_set:
        raw_path = download_set_and_pair(files_index, raw_set, "01_raw")
        try:
            raw_obj = load_raw_eeglab_strict(str(raw_path))
            plot_segment(raw_obj, "Raw EEG segment", "Pz", 5.0)
        except Exception as e:
            st.error("Failed to load raw .set (downloaded).")
            st.exception(e)
    else:
        st.warning("Raw .set not found.")

    st.subheader("2) Preprocessed data and PSD overlay")
    pre_set = find_first_match(files_index, contains=pid, endswith=suffix_pre)
    if pre_set:
        pre_path = download_set_and_pair(files_index, pre_set, "02_preprocessed")
        try:
            pre_obj = load_raw_eeglab_strict(str(pre_path))
            plot_segment(pre_obj, "Preprocessed segment", "Pz", 5.0)
            if raw_obj is not None:
                plot_psd_overlay(raw_obj, pre_obj, "Pz")
        except Exception as e:
            st.error("Failed to load preprocessed .set (downloaded).")
            st.exception(e)
    else:
        st.warning("Preprocessed .set not found.")

    st.subheader("3) Pre-ICA extreme-loss table (optional)")
    preica_xlsx = _find_by_name(files_index, "COCOA_preICAextremeloss.xlsx") or _find_by_name(files_index, "COCOA_preICAextremeloss.xlsx")
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
                ax.bar(range(len(ch_cols)), vals)
                ax.set_xticks(range(len(ch_cols)))
                ax.set_xticklabels(ch_cols, rotation=90)
                ax.set_ylabel("% extreme artifact")
                ax.set_title("Pre-ICA extreme-loss per channel (from table)")
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("No loss data for this participant in the Excel table.")
        else:
            st.warning("Excel table missing 'ID' column.")
    else:
        st.info("COCOA_preICAextremeloss.xlsx not found in Drive folder.")

    st.subheader("4) Epoched ERP (optional)")
    ep_set = find_first_match(files_index, contains=pid, endswith=suffix_ep)
    if ep_set:
        ep_path = download_set_and_pair(files_index, ep_set, "07_epoched")
        try:
            epochs = load_epochs_eeglab_strict(str(ep_path))
            keys = list(epochs.event_id.keys())
            target = keys[0] if keys else None
            if target:
                evk = epochs[target].average()
                fig = evk.plot(picks="Pz" if "Pz" in evk.ch_names else None, show=False)
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.warning("No events in epoched data.")
        except Exception as e:
            st.error("Failed to load epoched .set (downloaded).")
            st.exception(e)
    else:
        st.info("No epoched .set found for this participant.")


def vo_compare(files_index: List[Dict], participants: List[str]) -> None:
    """Multi-participant comparison charts."""
    require_mne()
    st.subheader("Compare participants (PSD + bandpower)")

    stage = st.selectbox("Stage", ["01_raw", "02_preprocessed"], index=1)
    channel = st.selectbox("Channel", ["Pz", "Cz", "Fz", "Oz"], index=0)
    fmin, fmax = st.slider("PSD frequency range (Hz)", 0.5, 60.0, (0.5, 45.0), 0.5)

    # bands in dB (simple)
    bands = {
        "delta (1-4)": (1.0, 4.0),
        "theta (4-8)": (4.0, 8.0),
        "alpha (8-13)": (8.0, 13.0),
        "beta (13-30)": (13.0, 30.0),
        "gamma (30-45)": (30.0, 45.0),
    }
    chosen_bands = st.multiselect("Bandpower bars", list(bands.keys()), default=["alpha (8-13)", "beta (13-30)"])

    # file suffix for the chosen stage
    suffix_map = {
        "01_raw": "_task-visualoddball_eeg_raw.set",
        "02_preprocessed": "_task-visualoddball_eeg_preprocessed.set",
    }
    suffix = suffix_map[stage]

    run = st.button("Run comparison", type="primary")

    if not run:
        st.info("Select participants + stage + channel, then click **Run comparison**.")
        return

    # Load PSDs
    psd_records = []
    overlay_data = []

    for pid in participants:
        set_file = find_first_match(files_index, contains=pid, endswith=suffix)
        if not set_file:
            psd_records.append({"participant": pid, "status": "missing .set"})
            continue

        local_set = download_set_and_pair(files_index, set_file, stage)

        try:
            raw = load_raw_eeglab_strict(str(local_set))
            if channel not in raw.ch_names:
                psd_records.append({"participant": pid, "status": f"channel {channel} missing"})
                continue

            freqs, psd_db = compute_psd_db(raw, channel, fmin=fmin, fmax=fmax)
            overlay_data.append((pid, freqs, psd_db))

            row = {"participant": pid, "status": "ok"}
            for bname in chosen_bands:
                row[bname] = bandpower_from_psd(freqs, psd_db, bands[bname])
            psd_records.append(row)

        except Exception as e:
            psd_records.append({"participant": pid, "status": f"error: {type(e).__name__}"})

    df_bands = pd.DataFrame(psd_records)

    # PSD overlay plot
    st.markdown("### PSD overlay")
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(9, 4))
    for pid, freqs, psd_db in overlay_data:
        ax.plot(freqs, psd_db, label=pid)
    ax.set_title(f"PSD overlay ({stage}) @ {channel}")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (dB, V²/Hz)")
    if len(overlay_data) <= 10:
        ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    # Bandpower bars
    st.markdown("### Bandpower comparison (mean PSD in bands, dB)")
    ok_df = df_bands[df_bands["status"] == "ok"].copy()
    if ok_df.empty:
        st.warning("No participants successfully loaded for comparison.")
        st.dataframe(df_bands, use_container_width=True)
        return

    st.dataframe(df_bands, use_container_width=True)

    if chosen_bands:
        long = ok_df.melt(id_vars=["participant"], value_vars=chosen_bands, var_name="band", value_name="mean_psd_db")
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        # simple grouped bars: pivot
        piv = long.pivot(index="participant", columns="band", values="mean_psd_db")
        piv.plot(kind="bar", ax=ax2)
        ax2.set_ylabel("Mean PSD (dB)")
        ax2.set_title(f"Bandpower ({stage}) @ {channel}")
        st.pyplot(fig2)
        plt.close(fig2)


def vo_qc_view(files_index: List[Dict]) -> None:
    st.header("Visual Oddball QC (Google Drive)")

    require_mne()

    participants_all = participants_from_index(files_index)
    if not participants_all:
        st.warning("No participant files found in the VO Drive folder.")
        return

    tabs = st.tabs(["Single participant QC", "Compare participants"])

    with tabs[0]:
        pid = st.selectbox("Participant", participants_all, index=0)
        st.markdown(f"### Participant: `{pid}`")
        vo_qc_single(files_index, pid)

    with tabs[1]:
        selected = st.multiselect(
            "Participants to compare",
            participants_all,
            default=participants_all[:3],
            help="Pick multiple participants to compare PSD and bandpower.",
        )
        if len(selected) < 2:
            st.info("Select at least 2 participants for comparison.")
        else:
            vo_compare(files_index, selected)


# ============================================================
# Sidebar tools
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
# Main
# ============================================================
def main() -> None:
    st.set_page_config(layout="wide", page_title="COCOA Dashboard")
    st.title("COCOA EEG Analysis Dashboard")

    root_folder_id = st.secrets["GDRIVE_VO_FOLDER_ID"]

    # Verify Drive access to configured folder
    drive_smoke_test(root_folder_id)

    # Resolve actual VO folder
    vo_folder_id = resolve_preprocessed_vo_folder_id(root_folder_id)

    # Helpful message if user pointed to parent folder
    if vo_folder_id != root_folder_id:
        st.caption("Detected VO folder inside parent folder: **Preprocessed_VisualOddball**")

    # Index VO folder
    files_index = drive_index_recursive(vo_folder_id)

    sidebar_tools()

    view = st.sidebar.selectbox("Select view", ["Pre-ICA Metrics", "Visual Oddball QC"], key="main_view")

    if view == "Pre-ICA Metrics":
        preica_view(files_index=files_index)
    else:
        vo_qc_view(files_index=files_index)


if __name__ == "__main__":
    main()
