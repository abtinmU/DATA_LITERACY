from __future__ import annotations

import io
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# EEG
try:
    import mne
except Exception:
    mne = None

# Drive
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload


# =========================
# Constants / Styling
# =========================

CACHE_ROOT = Path("/tmp/cocoa_cache")
CACHE_ROOT.mkdir(parents=True, exist_ok=True)

BASE_DIR = Path(__file__).resolve().parent
LOCAL_PREICA_XLSX = BASE_DIR / "COCOA_preICAextremeloss.xlsx"

TUE_PALETTE = ["#006AA3", "#E65C00", "#A31C34", "#5C8021", "#735545", "#4A6D8C"]

STAGES = {
    "01_raw": "_task-visualoddball_eeg_raw.set",
    "02_preprocessed": "_task-visualoddball_eeg_preprocessed.set",
    "03_preICA": "_task-visualoddball_eeg_preICA.set",
    "04_ICAweighted": "_task-visualoddball_eegICA_weighted.set",
    "06_postICA": "_task-visualoddball_eegpostICA.set",
    "06_postICApracticeremoved": "_task-visualoddball_eegpostICA_practiceremoved.set",
    "07_epoched": "_task-visualoddball_epoched.set",
    # 08_AR varies: contains "autoAR" and endswith .set
}

BANDS = {
    "delta (1–4 Hz)": (1.0, 4.0),
    "theta (4–8 Hz)": (4.0, 8.0),
    "alpha (8–13 Hz)": (8.0, 13.0),
    "beta (13–30 Hz)": (13.0, 30.0),
    "gamma (30–45 Hz)": (30.0, 45.0),
}


def apply_plot_style() -> None:
    if st.session_state.get("_plot_style_applied"):
        return
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 300,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "lines.linewidth": 1.4,
        }
    )
    st.session_state["_plot_style_applied"] = True


# =========================
# Drive helpers
# =========================

def _drive_service():
    creds_info = dict(st.secrets["gcp_service_account"])
    creds = service_account.Credentials.from_service_account_info(
        creds_info,
        scopes=["https://www.googleapis.com/auth/drive.readonly"],
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def drive_smoke_test(folder_id: str) -> Dict:
    """Return folder metadata if reachable; stop app otherwise."""
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
        return meta
    except Exception as e:
        st.error("❌ Drive access failed. Check: secrets + folder sharing to service account.")
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


def resolve_vo_folder_id(root_id: str) -> str:
    """
    Allow user to set either:
      - DataLiteracyProject (parent)
      - Preprocessed_VisualOddball (direct)
    We resolve to Preprocessed_VisualOddball.
    """
    svc = _drive_service()
    meta = svc.files().get(fileId=root_id, fields="id,name,mimeType", supportsAllDrives=True).execute()
    if meta.get("name") == "Preprocessed_VisualOddball":
        return root_id

    children = drive_list_children(root_id)
    for c in children:
        if c.get("mimeType") == "application/vnd.google-apps.folder" and c.get("name") == "Preprocessed_VisualOddball":
            return c["id"]

    # fallback
    return root_id


@st.cache_data(show_spinner=False)
def drive_index_recursive(folder_id: str) -> List[Dict]:
    """Recursively index all files (non-folders) under folder_id."""
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
    Download .set and matching .fdt to SAME stage folder.
    This is required for EEGLAB .set + .fdt pairs.
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


# =========================
# EEG loading + analysis
# =========================

def require_mne() -> None:
    if mne is None:
        st.error("MNE is not installed. Add `mne` to requirements.txt.")
        st.stop()
    # Friendly dependency hints
    try:
        import pymatreader  # noqa
    except Exception:
        st.warning("Missing `pymatreader` → add it to requirements.txt (needed for EEGLAB loading).")
    try:
        import h5py  # noqa
    except Exception:
        st.warning("Missing `h5py` → add it if some .set are MATLAB v7.3 (HDF5).")


def load_raw_eeglab_strict(path: str):
    # preload=False = less memory; good on Streamlit Cloud
    return mne.io.read_raw_eeglab(path, preload=False, verbose="ERROR")


def load_epochs_eeglab_strict(path: str):
    return mne.io.read_epochs_eeglab(path, verbose="ERROR")


def plot_segment(raw_obj, title: str, channel: str = "Pz", duration: float = 5.0) -> None:
    apply_plot_style()
    picks = [raw_obj.ch_names.index(channel)] if channel in raw_obj.ch_names else [0]
    ch_name = channel if channel in raw_obj.ch_names else raw_obj.ch_names[0]

    sfreq = float(raw_obj.info["sfreq"])
    n = int(sfreq * duration)

    data, times = raw_obj[picks, :n]
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(times, data[0] * 1e6)
    ax.set_title(f"{title} ({ch_name})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (µV)")
    ax.axhline(0, color="gray", lw=0.5)
    st.pyplot(fig)
    plt.close(fig)


def compute_psd_db(raw_obj, channel: str, fmin: float, fmax: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (freqs, psd_db). Works across MNE versions.
    """
    try:
        spec = raw_obj.compute_psd(fmin=fmin, fmax=fmax, picks=channel, verbose="ERROR")
        freqs = spec.freqs
        psd_db = 10 * np.log10(spec.get_data().squeeze())
        return freqs, psd_db
    except Exception:
        from mne.time_frequency import psd_welch
        psd, freqs = psd_welch(
            raw_obj,
            fmin=fmin,
            fmax=fmax,
            picks=[channel] if channel in raw_obj.ch_names else None,
            verbose="ERROR",
        )
        psd_db = 10 * np.log10(psd.squeeze())
        return freqs, psd_db


def bandpower(freqs: np.ndarray, psd_db: np.ndarray, lo: float, hi: float) -> float:
    mask = (freqs >= lo) & (freqs <= hi)
    if not np.any(mask):
        return float("nan")
    return float(np.nanmean(psd_db[mask]))


def plot_psd_overlay(psd_series: List[Tuple[str, np.ndarray, np.ndarray]], title: str) -> None:
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(9, 4))
    for pid, freqs, psd_db in psd_series:
        ax.plot(freqs, psd_db, label=pid)
    ax.set_title(title)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (dB, V²/Hz)")
    if len(psd_series) <= 10:
        ax.legend()
    st.pyplot(fig)
    plt.close(fig)


# =========================
# Pre-ICA view
# =========================

@st.cache_data(show_spinner=False)
def load_preica_data_from_excel(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df["participant"] = df["ID"].astype(str).str.extract(r"^(sub-\d+)")
    df["task"] = df["ID"].astype(str).str.extract(r"task-([^_]+)")
    for c in df.columns:
        if c not in ["ID", "participant", "task"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def preica_view(files_index: List[Dict]) -> None:
    st.header("Pre-ICA Explorer")

    # prefer local, else Drive (03_preICA contains it)
    excel_path: Optional[str] = None
    if LOCAL_PREICA_XLSX.exists():
        excel_path = str(LOCAL_PREICA_XLSX)
        st.caption("Using local COCOA_preICAextremeloss.xlsx")
    else:
        xlsx = _find_by_name(files_index, "COCOA_preICAextremeloss.xlsx")
        if xlsx:
            local = CACHE_ROOT / "03_preICA" / xlsx["name"]
            _download(xlsx["id"], local)
            excel_path = str(local)
            st.caption("Using Drive COCOA_preICAextremeloss.xlsx (cached)")

    if not excel_path:
        st.error("COCOA_preICAextremeloss.xlsx not found (local or Drive).")
        return

    df = load_preica_data_from_excel(excel_path)
    participants = sorted(df["participant"].dropna().unique().tolist())
    channels = [c for c in df.columns if c not in ["ID", "participant", "task"]]

    st.sidebar.subheader("Pre-ICA controls")
    chosen_participants = st.sidebar.multiselect("Participants", participants, default=participants[:5])
    chosen_channels = st.sidebar.multiselect("Channels", channels, default=channels[:10])
    mode = st.sidebar.radio("Plot mode", ["Single participant", "Compare participants"], index=1)

    if not chosen_channels:
        st.warning("Select at least one channel.")
        return
    if not chosen_participants:
        st.warning("Select at least one participant.")
        return

    filtered = df[df["participant"].isin(chosen_participants)].copy()

    st.markdown("### Heatmap (participant × channel)")
    hm = filtered.set_index("participant")[chosen_channels]
    hm = hm.groupby(level=0).mean()  # if multiple rows per participant
    hm_long = hm.reset_index().melt(id_vars="participant", var_name="channel", value_name="extreme_loss")

    heat = (
        alt.Chart(hm_long)
        .mark_rect()
        .encode(
            x=alt.X("channel:N", title="EEG channel"),
            y=alt.Y("participant:N", title="Participant"),
            color=alt.Color("extreme_loss:Q", title="Extreme loss"),
            tooltip=["participant", "channel", alt.Tooltip("extreme_loss:Q", format=".3f")],
        )
        .properties(height=min(500, 25 * len(hm.index)))
    )
    st.altair_chart(heat, use_container_width=True)

    st.markdown("### Distribution comparison")
    long = hm_long.dropna()
    if mode == "Compare participants":
        box = (
            alt.Chart(long)
            .mark_boxplot(extent="min-max")
            .encode(
                x=alt.X("participant:N", title="Participant"),
                y=alt.Y("extreme_loss:Q", title="Extreme loss"),
                color=alt.Color("participant:N", legend=None),
            )
            .properties(height=350)
        )
        st.altair_chart(box, use_container_width=True)
    else:
        pid = chosen_participants[0]
        one = long[long["participant"] == pid]
        hist = (
            alt.Chart(one)
            .mark_bar()
            .encode(
                x=alt.X("extreme_loss:Q", bin=alt.Bin(maxbins=30), title="Extreme loss"),
                y=alt.Y("count():Q", title="Count"),
            )
            .properties(height=300)
        )
        st.altair_chart(hist, use_container_width=True)

    st.markdown("### Table")
    st.dataframe(hm.reset_index(), use_container_width=True)


# =========================
# VO QC: Single + Compare
# =========================

def vo_single_qc(files_index: List[Dict], pid: str, channel: str) -> str:
    """
    Runs QC steps and returns a Markdown report string.
    """
    require_mne()
    report_lines = []
    report_lines.append(f"# COCOA Visual Oddball QC Report\n")
    report_lines.append(f"**Participant:** `{pid}`\n")

    # 1) raw
    raw_set = find_first_match(files_index, contains=pid, endswith=STAGES["01_raw"])
    if raw_set:
        local_raw = download_set_and_pair(files_index, raw_set, "01_raw")
        try:
            raw = load_raw_eeglab_strict(str(local_raw))
            report_lines.append(f"- ✅ Raw loaded: `{raw_set['name']}`")
            st.subheader("1) Raw continuous data")
            plot_segment(raw, "Raw EEG segment", channel, 5.0)
        except Exception as e:
            st.error("Failed to load raw .set")
            st.exception(e)
            report_lines.append(f"- ❌ Raw load failed: `{raw_set['name']}` ({type(e).__name__})")
            raw = None
    else:
        st.warning("Raw file not found.")
        report_lines.append("- ❌ Raw file not found")
        raw = None

    # 2) preprocessed + PSD overlay
    pre_set = find_first_match(files_index, contains=pid, endswith=STAGES["02_preprocessed"])
    if pre_set:
        local_pre = download_set_and_pair(files_index, pre_set, "02_preprocessed")
        try:
            pre = load_raw_eeglab_strict(str(local_pre))
            report_lines.append(f"- ✅ Preprocessed loaded: `{pre_set['name']}`")
            st.subheader("2) Preprocessed data and PSD overlay")
            plot_segment(pre, "Preprocessed segment", channel, 5.0)
            if raw is not None and channel in raw.ch_names and channel in pre.ch_names:
                freqs1, psd1 = compute_psd_db(raw, channel, 0.5, 45.0)
                freqs2, psd2 = compute_psd_db(pre, channel, 0.5, 45.0)
                plot_psd_overlay(
                    [(f"{pid}-raw", freqs1, psd1), (f"{pid}-pre", freqs2, psd2)],
                    f"PSD overlay @ {channel}",
                )
        except Exception as e:
            st.error("Failed to load preprocessed .set")
            st.exception(e)
            report_lines.append(f"- ❌ Preprocessed load failed: `{pre_set['name']}` ({type(e).__name__})")
    else:
        st.warning("Preprocessed file not found.")
        report_lines.append("- ❌ Preprocessed file not found")

    # 3) ICLabel summary (xlsx)
    st.subheader("3) ICLabel (optional)")
    ic = find_first_match(files_index, contains=pid, endswith="_ICclassifications.xlsx")
    if ic:
        local_ic = CACHE_ROOT / "05_ICLabel" / ic["name"]
        _download(ic["id"], local_ic)
        try:
            ic_df = pd.read_excel(local_ic)
            cols = ["Brain", "Muscle", "Eye", "Heart", "Line_Noise", "Channel_Noise", "Other"]
            if set(cols).issubset(ic_df.columns):
                apply_plot_style()
                means = ic_df[cols].mean()
                fig, ax = plt.subplots(figsize=(8, 3))
                means.plot(kind="bar", ax=ax)
                ax.set_ylabel("Mean probability (%)")
                ax.set_title("ICLabel mean class probability")
                st.pyplot(fig)
                plt.close(fig)
                report_lines.append("- ✅ ICLabel loaded and plotted")
            else:
                st.info("ICLabel file loaded, but expected columns missing.")
                report_lines.append("- ⚠️ ICLabel loaded, but columns missing")
        except Exception as e:
            st.exception(e)
            report_lines.append(f"- ❌ ICLabel read failed ({type(e).__name__})")
    else:
        st.info("ICLabel xlsx not found.")
        report_lines.append("- ℹ️ ICLabel not found")

    # 4) epoched ERP
    st.subheader("4) ERP (optional)")
    ep = find_first_match(files_index, contains=pid, endswith=STAGES["07_epoched"])
    if ep:
        local_ep = download_set_and_pair(files_index, ep, "07_epoched")
        try:
            epochs = load_epochs_eeglab_strict(str(local_ep))
            keys = list(epochs.event_id.keys())
            if keys:
                evk = epochs[keys[0]].average()
                fig = evk.plot(picks=channel if channel in evk.ch_names else None, show=False)
                st.pyplot(fig)
                plt.close(fig)
                report_lines.append("- ✅ Epoched loaded and ERP plotted")
            else:
                report_lines.append("- ⚠️ Epoched loaded but no events")
        except Exception as e:
            st.exception(e)
            report_lines.append(f"- ❌ Epoched read failed ({type(e).__name__})")
    else:
        st.info("Epoched file not found.")
        report_lines.append("- ℹ️ Epoched not found")

    return "\n".join(report_lines) + "\n"


def vo_compare(files_index: List[Dict], participants: List[str]) -> None:
    require_mne()
    st.header("Comparison Lab")

    stage = st.selectbox("Stage", ["01_raw", "02_preprocessed"], index=1)
    channel = st.selectbox("Channel", ["Pz", "Cz", "Fz", "Oz"], index=0)
    fmin, fmax = st.slider("PSD range (Hz)", 0.5, 60.0, (0.5, 45.0), 0.5)
    band_names = st.multiselect("Bandpower bars", list(BANDS.keys()), default=["alpha (8–13 Hz)", "beta (13–30 Hz)"])

    run = st.button("Run comparison", type="primary")
    if not run:
        st.info("Choose participants + stage + channel, then click **Run comparison**.")
        return

    suffix = STAGES[stage]
    overlay = []
    rows = []

    for pid in participants:
        f = find_first_match(files_index, contains=pid, endswith=suffix)
        if not f:
            rows.append({"participant": pid, "status": "missing"})
            continue

        local = download_set_and_pair(files_index, f, stage)
        try:
            raw = load_raw_eeglab_strict(str(local))
            if channel not in raw.ch_names:
                rows.append({"participant": pid, "status": f"missing channel {channel}"})
                continue

            freqs, psd_db = compute_psd_db(raw, channel, fmin, fmax)
            overlay.append((pid, freqs, psd_db))

            row = {"participant": pid, "status": "ok"}
            for bn in band_names:
                lo, hi = BANDS[bn]
                row[bn] = bandpower(freqs, psd_db, lo, hi)
            rows.append(row)

        except Exception as e:
            rows.append({"participant": pid, "status": f"error: {type(e).__name__}"})

    st.subheader("PSD overlay")
    plot_psd_overlay(overlay, f"PSD overlay ({stage}) @ {channel}")

    st.subheader("Bandpower table")
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    ok = df[df["status"] == "ok"].copy()
    if not ok.empty and band_names:
        st.subheader("Bandpower bar chart")
        long = ok.melt(id_vars=["participant"], value_vars=band_names, var_name="band", value_name="mean_psd_db")
        chart = (
            alt.Chart(long)
            .mark_bar()
            .encode(
                x=alt.X("participant:N", title="Participant"),
                y=alt.Y("mean_psd_db:Q", title="Mean PSD (dB)"),
                color=alt.Color("band:N", title="Band"),
                tooltip=["participant", "band", alt.Tooltip("mean_psd_db:Q", format=".2f")],
            )
            .properties(height=350)
        )
        st.altair_chart(chart, use_container_width=True)


def vo_view(files_index: List[Dict]) -> None:
    st.header("Visual Oddball QC")

    require_mne()

    all_participants = participants_from_index(files_index)
    if not all_participants:
        st.error("No participants detected in Drive index.")
        return

    tabs = st.tabs(["Single participant QC", "Comparison Lab"])

    with tabs[0]:
        pid = st.selectbox("Participant", all_participants, index=0)
        channel = st.selectbox("Channel for plots", ["Pz", "Cz", "Fz", "Oz"], index=0)
        st.markdown(f"### Participant: `{pid}`")

        report_md = vo_single_qc(files_index, pid, channel)
        st.markdown("---")
        st.subheader("Download QC report")
        st.download_button(
            "Download report (Markdown)",
            data=report_md.encode("utf-8"),
            file_name=f"{pid}_QC_report.md",
            mime="text/markdown",
        )

    with tabs[1]:
        chosen = st.multiselect(
            "Participants",
            all_participants,
            default=all_participants[:4],
            help="Select multiple participants to compare PSD and bandpower.",
        )
        if len(chosen) < 2:
            st.info("Pick at least 2 participants to compare.")
        else:
            vo_compare(files_index, chosen)


# =========================
# Overview page (wow factor)
# =========================

def cache_size_mb() -> float:
    total = 0
    for p in CACHE_ROOT.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total / (1024 * 1024)


def overview_page(root_meta: Dict, vo_folder_id: str, files_index: List[Dict]) -> None:
    st.header("Project Overview")

    cols = st.columns(4)
    cols[0].metric("Drive root", root_meta.get("name", ""))
    cols[1].metric("Indexed files", f"{len(files_index):,}")
    cols[2].metric("Participants detected", f"{len(participants_from_index(files_index)):,}")
    cols[3].metric("Local cache (MB)", f"{cache_size_mb():.1f}")

    # file type distribution
    st.subheader("Dataset composition (by file type)")
    ext_counts = {}
    for f in files_index:
        name = f["name"]
        ext = name.split(".")[-1].lower() if "." in name else "(none)"
        ext_counts[ext] = ext_counts.get(ext, 0) + 1
    df = pd.DataFrame([{"ext": k, "count": v} for k, v in sorted(ext_counts.items(), key=lambda x: -x[1])])

    chart = (
        alt.Chart(df.head(12))
        .mark_bar()
        .encode(x=alt.X("count:Q", title="Count"), y=alt.Y("ext:N", sort="-x", title="Extension"))
        .properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)

    st.subheader("Folders expected")
    st.markdown(
        """
- `01_raw` → `.set` + `.fdt`
- `02_preprocessed` → `.set` + `.fdt`
- `03_preICA` → `.xlsx` + `.set` + `.fdt`
- `05_ICLabel` → `.xlsx`
- `07_epoched` → `.set` (+ eventlist `.txt`)
- `08_AR` → `.set` + `.fdt` + `.mat`
"""
    )
    st.caption(f"Resolved VO folder id: {vo_folder_id}")


# =========================
# Sidebar tools
# =========================

def sidebar_tools() -> None:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Tools")

    if st.sidebar.button("Clear cache (/tmp)"):
        shutil.rmtree(CACHE_ROOT, ignore_errors=True)
        CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        st.sidebar.success("Cache cleared.")

    if st.sidebar.button("Reindex Drive"):
        drive_index_recursive.clear()
        drive_list_children.clear()
        st.sidebar.success("Drive index cache cleared. Reload to refetch.")


# =========================
# Main
# =========================

def main() -> None:
    st.set_page_config(layout="wide", page_title="COCOA Dashboard")
    st.title("COCOA EEG Analysis Dashboard")

    root_folder_id = st.secrets["GDRIVE_VO_FOLDER_ID"]
    root_meta = drive_smoke_test(root_folder_id)

    vo_folder_id = resolve_vo_folder_id(root_folder_id)
    if vo_folder_id != root_folder_id:
        st.caption("Auto-detected: using **Preprocessed_VisualOddball** inside the parent folder.")

    files_index = drive_index_recursive(vo_folder_id)

    sidebar_tools()
    view = st.sidebar.selectbox("Navigation", ["Overview", "Visual Oddball QC", "Pre-ICA Explorer"], index=0)

    if view == "Overview":
        overview_page(root_meta, vo_folder_id, files_index)
    elif view == "Visual Oddball QC":
        vo_view(files_index)
    else:
        preica_view(files_index)


if __name__ == "__main__":
    main()
