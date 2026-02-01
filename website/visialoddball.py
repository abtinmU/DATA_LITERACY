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

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from pymatreader import read_mat


# =========================
# Config
# =========================
CACHE_ROOT = Path("/tmp/cocoa_cache")
CACHE_ROOT.mkdir(parents=True, exist_ok=True)

BASE_DIR = Path(__file__).resolve().parent
LOCAL_PREICA_XLSX = BASE_DIR / "COCOA_preICAextremeloss.xlsx"

TUE_PALETTE = ["#006AA3", "#E65C00", "#A31C34", "#5C8021", "#735545", "#4A6D8C"]

BANDS = {
    "delta (1–4)": (1.0, 4.0),
    "theta (4–8)": (4.0, 8.0),
    "alpha (8–13)": (8.0, 13.0),
    "beta (13–30)": (13.0, 30.0),
    "gamma (30–45)": (30.0, 45.0),
}

STAGE_SUFFIX = {
    "01_raw": "_task-visualoddball_eeg_raw.set",
    "02_preprocessed": "_task-visualoddball_eeg_preprocessed.set",
    "07_epoched": "_task-visualoddball_epoched.set",
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
# Drive
# =========================
def _drive_service():
    creds_info = dict(st.secrets["gcp_service_account"])
    creds = service_account.Credentials.from_service_account_info(
        creds_info, scopes=["https://www.googleapis.com/auth/drive.readonly"]
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def drive_smoke_test(folder_id: str) -> Dict:
    try:
        svc = _drive_service()
        meta = svc.files().get(
            fileId=folder_id, fields="id,name,mimeType", supportsAllDrives=True
        ).execute()
        if meta.get("mimeType") != "application/vnd.google-apps.folder":
            st.error("GDRIVE_VO_FOLDER_ID must point to a folder.")
            st.stop()
        return meta
    except Exception as e:
        st.error("❌ Drive access failed. Check secrets + folder sharing.")
        st.exception(e)
        st.stop()


@st.cache_data(show_spinner=False)
def drive_list_children(folder_id: str) -> List[Dict]:
    svc = _drive_service()
    items: List[Dict] = []
    token = None
    while True:
        resp = svc.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields="nextPageToken, files(id,name,mimeType)",
            pageToken=token,
            pageSize=1000,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
        items.extend(resp.get("files", []))
        token = resp.get("nextPageToken")
        if not token:
            break
    return items


def resolve_vo_folder_id(root_id: str) -> str:
    """Allow root to be DataLiteracyProject; resolve to Preprocessed_VisualOddball."""
    svc = _drive_service()
    meta = svc.files().get(fileId=root_id, fields="id,name", supportsAllDrives=True).execute()
    if meta.get("name") == "Preprocessed_VisualOddball":
        return root_id

    children = drive_list_children(root_id)
    for c in children:
        if c.get("mimeType") == "application/vnd.google-apps.folder" and c.get("name") == "Preprocessed_VisualOddball":
            return c["id"]

    return root_id


@st.cache_data(show_spinner=False)
def drive_index_recursive(folder_id: str) -> List[Dict]:
    svc = _drive_service()
    out: List[Dict] = []
    queue = [folder_id]

    while queue:
        fid = queue.pop()
        token = None
        while True:
            resp = svc.files().list(
                q=f"'{fid}' in parents and trashed=false",
                fields="nextPageToken, files(id,name,mimeType)",
                pageToken=token,
                pageSize=1000,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            ).execute()

            for f in resp.get("files", []):
                if f["mimeType"] == "application/vnd.google-apps.folder":
                    queue.append(f["id"])
                else:
                    out.append(f)

            token = resp.get("nextPageToken")
            if not token:
                break

    return out


def _download(file_id: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    svc = _drive_service()
    req = svc.files().get_media(fileId=file_id, supportsAllDrives=True)

    with io.FileIO(dest, "wb") as fh:
        dl = MediaIoBaseDownload(fh, req)
        done = False
        while not done:
            _, done = dl.next_chunk()

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
    """Download .set plus matching .fdt into the same folder."""
    set_name = set_file["name"]
    local_set = CACHE_ROOT / stage / set_name
    _download(set_file["id"], local_set)

    if set_name.lower().endswith(".set"):
        fdt_name = set_name[:-4] + ".fdt"
        fdt = _find_by_name(files, fdt_name)
        if fdt:
            local_fdt = CACHE_ROOT / stage / fdt_name
            _download(fdt["id"], local_fdt)

    return local_set


# =========================
# EEGLAB Reader (no MNE, no SciPy)
# =========================
@dataclass
class EEGData:
    data: np.ndarray  # shape: (n_channels, n_times) or (n_channels, n_times, n_trials)
    sfreq: float
    ch_names: List[str]
    n_trials: int


def _try_labels(chanlocs) -> List[str]:
    # chanlocs can be list of dicts or dict-like
    names = []
    if isinstance(chanlocs, list):
        for c in chanlocs:
            if isinstance(c, dict) and "labels" in c:
                names.append(str(c["labels"]))
    return names


def load_eeglab_set(set_path: Path) -> EEGData:
    """
    Best-effort EEGLAB loader:
    - Reads .set with pymatreader
    - If EEG['data'] is numeric array -> use it
    - If EEG['data'] points to an .fdt file -> read float32 binary
    """
    mat = read_mat(str(set_path))
    if "EEG" not in mat:
        raise ValueError("This .set does not contain an 'EEG' struct.")

    eeg = mat["EEG"]

    # sampling rate
    sfreq = float(eeg.get("srate", np.nan))
    if not np.isfinite(sfreq):
        raise ValueError("Missing EEG.srate in .set")

    nbchan = int(eeg.get("nbchan", 0))
    pnts = int(eeg.get("pnts", 0))
    trials = int(eeg.get("trials", 1))

    chanlocs = eeg.get("chanlocs", None)
    ch_names = _try_labels(chanlocs)
    if not ch_names:
        ch_names = [f"Ch{i+1}" for i in range(nbchan)]

    data = eeg.get("data", None)
    if data is None:
        raise ValueError("Missing EEG.data")

    # Case 1: data already numeric array
    if isinstance(data, (list, np.ndarray)):
        arr = np.asarray(data)
        # normalize to (nbchan, pnts, trials)
        if arr.ndim == 2:
            # could be (nbchan, pnts) or (pnts, nbchan)
            if arr.shape[0] == nbchan:
                pass
            elif arr.shape[1] == nbchan:
                arr = arr.T
            else:
                # fallback
                arr = arr.reshape((nbchan, -1), order="F")
            trials = 1
        elif arr.ndim == 3:
            # ensure channel-first
            if arr.shape[0] != nbchan and arr.shape[-1] == nbchan:
                arr = np.transpose(arr, (2, 0, 1))
        return EEGData(data=arr, sfreq=sfreq, ch_names=ch_names, n_trials=trials)

    # Case 2: data is a string pointer to .fdt
    if isinstance(data, str):
        fdt_name = data.strip()
        fdt_path = (set_path.parent / fdt_name).resolve()
        if not fdt_path.exists():
            # common: fdt has same base name
            alt_fdt = set_path.with_suffix(".fdt")
            if alt_fdt.exists():
                fdt_path = alt_fdt
            else:
                raise FileNotFoundError(f"Could not locate FDT: {fdt_path}")

        raw = np.fromfile(fdt_path, dtype=np.float32)
        expected = nbchan * pnts * trials
        if raw.size != expected:
            raise ValueError(f"FDT size mismatch. expected={expected}, got={raw.size}")

        # EEGLAB commonly stores in column-major; try 'F' then fallback to 'C'
        try:
            arr = raw.reshape((nbchan, pnts, trials), order="F")
        except Exception:
            arr = raw.reshape((nbchan, pnts, trials), order="C")

        if trials == 1:
            arr = arr[:, :, 0]
        return EEGData(data=arr, sfreq=sfreq, ch_names=ch_names, n_trials=trials)

    raise ValueError(f"Unsupported EEG.data type: {type(data)}")


# =========================
# Signal processing (numpy-only)
# =========================
def welch_psd(x: np.ndarray, sfreq: float, fmin: float, fmax: float, nperseg: int = 1024, noverlap: int = 512):
    """
    Simple Welch PSD using numpy only.
    Returns freqs, psd (linear).
    """
    x = np.asarray(x, dtype=np.float64)
    x = x - np.nanmean(x)

    n = x.size
    if n < nperseg:
        nperseg = max(256, (n // 2))
        noverlap = nperseg // 2

    step = nperseg - noverlap
    if step <= 0:
        step = nperseg // 2

    window = np.hanning(nperseg)
    scale = (window**2).sum()

    psds = []
    for start in range(0, n - nperseg + 1, step):
        seg = x[start:start+nperseg] * window
        fft = np.fft.rfft(seg)
        pxx = (np.abs(fft) ** 2) / (scale * sfreq)
        psds.append(pxx)

    if not psds:
        # fallback single FFT
        seg = x[:nperseg] * window
        fft = np.fft.rfft(seg)
        pxx = (np.abs(fft) ** 2) / (scale * sfreq)
        psds = [pxx]

    psd = np.mean(psds, axis=0)
    freqs = np.fft.rfftfreq(nperseg, d=1.0/sfreq)

    mask = (freqs >= fmin) & (freqs <= fmax)
    return freqs[mask], psd[mask]


def bandpower_db(freqs: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> float:
    mask = (freqs >= lo) & (freqs <= hi)
    if not np.any(mask):
        return float("nan")
    val = float(np.mean(psd[mask]))
    return float(10 * np.log10(val + 1e-20))


# =========================
# Plots
# =========================
def plot_segment(eeg: EEGData, title: str, ch: str, seconds: float = 5.0):
    apply_plot_style()
    idx = eeg.ch_names.index(ch) if ch in eeg.ch_names else 0
    n = int(eeg.sfreq * seconds)

    if eeg.data.ndim == 3:
        x = eeg.data[idx, :n, 0]
    else:
        x = eeg.data[idx, :n]

    t = np.arange(x.size) / eeg.sfreq

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t, x, lw=0.9)
    ax.set_title(f"{title} ({eeg.ch_names[idx]})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (a.u.)")
    ax.axhline(0, lw=0.5, color="gray")
    st.pyplot(fig)
    plt.close(fig)


def plot_psd_overlay(series: List[Tuple[str, np.ndarray, np.ndarray]], title: str):
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(9, 4))
    for label, f, psd in series:
        ax.plot(f, 10*np.log10(psd + 1e-20), label=label)
    ax.set_title(title)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (dB)")
    if len(series) <= 10:
        ax.legend()
    st.pyplot(fig)
    plt.close(fig)


# =========================
# Views
# =========================
def overview(files_index: List[Dict]):
    st.header("Overview")

    st.metric("Indexed files", f"{len(files_index):,}")
    st.metric("Participants detected", f"{len(participants_from_index(files_index)):,}")

    ext_counts = {}
    for f in files_index:
        name = f["name"]
        ext = name.split(".")[-1].lower() if "." in name else "(none)"
        ext_counts[ext] = ext_counts.get(ext, 0) + 1
    df = pd.DataFrame([{"ext": k, "count": v} for k, v in sorted(ext_counts.items(), key=lambda x: -x[1])])

    chart = (
        alt.Chart(df.head(12))
        .mark_bar()
        .encode(x=alt.X("count:Q"), y=alt.Y("ext:N", sort="-x"))
        .properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)

    st.caption("This dashboard runs on Python 3.13 without SciPy/MNE (EEGLAB read is numpy-based).")


def preica_view(files_index: List[Dict]):
    st.header("Pre-ICA Explorer")

    excel_path = None
    if LOCAL_PREICA_XLSX.exists():
        excel_path = str(LOCAL_PREICA_XLSX)
    else:
        xlsx = _find_by_name(files_index, "COCOA_preICAextremeloss.xlsx")
        if xlsx:
            local = CACHE_ROOT / "03_preICA" / xlsx["name"]
            _download(xlsx["id"], local)
            excel_path = str(local)

    if not excel_path:
        st.error("COCOA_preICAextremeloss.xlsx not found (local or Drive).")
        return

    df = pd.read_excel(excel_path)
    df["participant"] = df["ID"].astype(str).str.extract(r"^(sub-\d+)")
    df["task"] = df["ID"].astype(str).str.extract(r"task-([^_]+)")

    cols = [c for c in df.columns if c not in ["ID", "participant", "task"]]
    participants = sorted(df["participant"].dropna().unique().tolist())

    sel_p = st.multiselect("Participants", participants, default=participants[:6])
    sel_c = st.multiselect("Channels", cols, default=cols[:12])

    if not sel_p or not sel_c:
        st.info("Select participants and channels.")
        return

    hm = df[df["participant"].isin(sel_p)].set_index("participant")[sel_c].groupby(level=0).mean()
    long = hm.reset_index().melt(id_vars="participant", var_name="channel", value_name="extreme_loss")

    st.subheader("Heatmap (participant × channel)")
    heat = (
        alt.Chart(long)
        .mark_rect()
        .encode(
            x="channel:N",
            y="participant:N",
            color="extreme_loss:Q",
            tooltip=["participant", "channel", alt.Tooltip("extreme_loss:Q", format=".3f")],
        )
        .properties(height=min(600, 28 * len(sel_p)))
    )
    st.altair_chart(heat, use_container_width=True)

    st.subheader("Distribution comparison")
    box = (
        alt.Chart(long.dropna())
        .mark_boxplot(extent="min-max")
        .encode(
            x="participant:N",
            y="extreme_loss:Q",
            color=alt.Color("participant:N", legend=None),
        )
        .properties(height=320)
    )
    st.altair_chart(box, use_container_width=True)

    st.subheader("Table")
    st.dataframe(hm.reset_index(), use_container_width=True)


def vo_single(files_index: List[Dict]):
    st.header("Visual Oddball QC — Single Participant")

    participants = participants_from_index(files_index)
    pid = st.selectbox("Participant", participants, index=0)

    stage_raw = "01_raw"
    stage_pre = "02_preprocessed"

    # channel selector will be filled after load; for now provide common picks
    ch_pick = st.selectbox("Channel", ["Pz", "Cz", "Fz", "Oz"], index=0)

    # Raw
    st.subheader("1) Raw continuous")
    raw_set = find_first_match(files_index, contains=pid, endswith=STAGE_SUFFIX[stage_raw])
    if not raw_set:
        st.warning("Raw .set not found.")
        return

    raw_path = download_set_and_pair(files_index, raw_set, stage_raw)
    try:
        raw = load_eeglab_set(raw_path)
        ch = ch_pick if ch_pick in raw.ch_names else raw.ch_names[0]
        plot_segment(raw, "Raw segment", ch, seconds=5.0)
    except Exception as e:
        st.error("Raw load failed.")
        st.exception(e)
        return

    # Preprocessed
    st.subheader("2) Preprocessed + PSD overlay")
    pre_set = find_first_match(files_index, contains=pid, endswith=STAGE_SUFFIX[stage_pre])
    if not pre_set:
        st.warning("Preprocessed .set not found.")
        return

    pre_path = download_set_and_pair(files_index, pre_set, stage_pre)
    try:
        pre = load_eeglab_set(pre_path)
        ch = ch_pick if ch_pick in pre.ch_names else pre.ch_names[0]
        plot_segment(pre, "Preprocessed segment", ch, seconds=5.0)

        fmin, fmax = 0.5, 45.0
        idx_r = raw.ch_names.index(ch) if ch in raw.ch_names else 0
        idx_p = pre.ch_names.index(ch) if ch in pre.ch_names else 0

        xr = raw.data[idx_r, :] if raw.data.ndim == 2 else raw.data[idx_r, :, 0]
        xp = pre.data[idx_p, :] if pre.data.ndim == 2 else pre.data[idx_p, :, 0]

        fr, psdr = welch_psd(xr, raw.sfreq, fmin, fmax)
        fp, psdp = welch_psd(xp, pre.sfreq, fmin, fmax)

        plot_psd_overlay(
            [(f"{pid}-raw", fr, psdr), (f"{pid}-pre", fp, psdp)],
            f"PSD overlay @ {ch}",
        )
    except Exception as e:
        st.error("Preprocessed load/PSD failed.")
        st.exception(e)

    # ICLabel optional
    st.subheader("3) ICLabel summary (optional)")
    ic = find_first_match(files_index, contains=pid, endswith="_ICclassifications.xlsx")
    if ic:
        local_ic = CACHE_ROOT / "05_ICLabel" / ic["name"]
        _download(ic["id"], local_ic)
        ic_df = pd.read_excel(local_ic)
        cols = ["Brain", "Muscle", "Eye", "Heart", "Line_Noise", "Channel_Noise", "Other"]
        if set(cols).issubset(ic_df.columns):
            means = ic_df[cols].mean()
            apply_plot_style()
            fig, ax = plt.subplots(figsize=(8, 3))
            means.plot(kind="bar", ax=ax)
            ax.set_ylabel("Mean probability (%)")
            ax.set_title("ICLabel mean class probability")
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("ICLabel file found but columns differ.")
            st.write(list(ic_df.columns))
    else:
        st.info("No ICLabel file found for this participant.")


def vo_compare(files_index: List[Dict]):
    st.header("Comparison Lab — Multi-participant")

    participants = participants_from_index(files_index)
    chosen = st.multiselect("Participants", participants, default=participants[:4])
    if len(chosen) < 2:
        st.info("Select at least 2 participants.")
        return

    stage = st.selectbox("Stage", ["01_raw", "02_preprocessed"], index=1)
    suffix = STAGE_SUFFIX[stage]
    channel = st.selectbox("Channel", ["Pz", "Cz", "Fz", "Oz"], index=0)
    fmin, fmax = st.slider("PSD range (Hz)", 0.5, 60.0, (0.5, 45.0), 0.5)
    band_names = st.multiselect("Bandpower bars", list(BANDS.keys()), default=["alpha (8–13)", "beta (13–30)"])

    run = st.button("Run comparison", type="primary")
    if not run:
        return

    overlay = []
    rows = []

    for pid in chosen:
        f = find_first_match(files_index, contains=pid, endswith=suffix)
        if not f:
            rows.append({"participant": pid, "status": "missing .set"})
            continue

        local = download_set_and_pair(files_index, f, stage)
        try:
            eeg = load_eeglab_set(local)
            ch = channel if channel in eeg.ch_names else eeg.ch_names[0]
            idx = eeg.ch_names.index(ch)
            x = eeg.data[idx, :] if eeg.data.ndim == 2 else eeg.data[idx, :, 0]

            freqs, psd = welch_psd(x, eeg.sfreq, fmin, fmax)
            overlay.append((pid, freqs, psd))

            row = {"participant": pid, "status": "ok", "channel_used": ch}
            for bn in band_names:
                lo, hi = BANDS[bn]
                row[bn] = bandpower_db(freqs, psd, lo, hi)
            rows.append(row)

        except Exception as e:
            rows.append({"participant": pid, "status": f"error: {type(e).__name__}"})

    st.subheader("PSD overlay")
    plot_psd_overlay([(pid, f, psd) for pid, f, psd in overlay], f"PSD overlay ({stage}) @ {channel}")

    df = pd.DataFrame(rows)
    st.subheader("Bandpower table")
    st.dataframe(df, use_container_width=True)

    ok = df[df["status"] == "ok"].copy()
    if not ok.empty and band_names:
        st.subheader("Bandpower bar chart")
        long = ok.melt(id_vars=["participant"], value_vars=band_names, var_name="band", value_name="mean_psd_db")
        chart = (
            alt.Chart(long)
            .mark_bar()
            .encode(
                x="participant:N",
                y="mean_psd_db:Q",
                color="band:N",
                tooltip=["participant", "band", alt.Tooltip("mean_psd_db:Q", format=".2f")],
            )
            .properties(height=350)
        )
        st.altair_chart(chart, use_container_width=True)


# =========================
# Tools
# =========================
def sidebar_tools():
    st.sidebar.markdown("---")
    st.sidebar.subheader("Tools")

    if st.sidebar.button("Clear cache (/tmp)"):
        shutil.rmtree(CACHE_ROOT, ignore_errors=True)
        CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        st.sidebar.success("Cache cleared.")

    if st.sidebar.button("Reindex Drive"):
        drive_index_recursive.clear()
        drive_list_children.clear()
        st.sidebar.success("Drive cache cleared. Reload to refetch.")


# =========================
# Main
# =========================
def main():
    st.set_page_config(layout="wide", page_title="COCOA Dashboard")
    st.title("COCOA EEG Analysis Dashboard")

    root_id = st.secrets["GDRIVE_VO_FOLDER_ID"]
    meta = drive_smoke_test(root_id)

    vo_id = resolve_vo_folder_id(root_id)
    if vo_id != root_id:
        st.caption("Auto-detected: using Preprocessed_VisualOddball inside DataLiteracyProject.")

    files_index = drive_index_recursive(vo_id)

    sidebar_tools()
    view = st.sidebar.selectbox("Navigation", ["Overview", "VO: Single", "VO: Compare", "Pre-ICA Explorer"], index=0)

    if view == "Overview":
        overview(files_index)
    elif view == "VO: Single":
        vo_single(files_index)
    elif view == "VO: Compare":
        vo_compare(files_index)
    else:
        preica_view(files_index)


if __name__ == "__main__":
    main()
