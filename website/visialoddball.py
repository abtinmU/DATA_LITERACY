from __future__ import annotations

import io
import os
import re
import glob
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt

import mne

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload


# =========================
# Config
# =========================

CACHE_ROOT = Path("/tmp/cocoa_cache")
CACHE_ROOT.mkdir(parents=True, exist_ok=True)

TUE_PALETTE = ["#006AA3", "#E65C00", "#A31C34", "#5C8021", "#735545", "#4A6D8C"]

# Your stage folders
STAGES = ["01_raw", "02_preprocessed", "03_preICA", "05_ICLabel", "06_postICA", "07_epoched", "08_AR"]

# Your file naming patterns (from screenshots & your functions)
PATTERNS = {
    "01_raw": "*raw.set",
    "02_preprocessed": "*preprocessed.set",
    "06_postICA": "*postICA.set",
    "07_epoched": "*epoched.set",
    "08_AR": "*autoAR.set",
}

PREICA_XLSX_NAME = "COCOA_preICAextremeloss.xlsx"
FEATURES_CSV_NAME = "COCOA_VO_P3b_trial_features_with_meta.csv"

ICLABEL_SUFFIX = "_ICclassifications.xlsx"


def apply_plot_style():
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
            "lines.linewidth": 1.3,
        }
    )
    st.session_state["_plot_style_applied"] = True


# =========================
# Drive helpers
# =========================

def _drive_service():
    creds_info = dict(st.secrets["gcp_service_account"])
    creds = service_account.Credentials.from_service_account_info(
        creds_info, scopes=["https://www.googleapis.com/auth/drive.readonly"]
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def drive_smoke(folder_id: str) -> Dict:
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
    """If root is DataLiteracyProject, resolve to Preprocessed_VisualOddball child folder."""
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
    """Index all files (non-folders) under folder_id recursively."""
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


def find_first_match(files: List[Dict], contains: str, endswith: str) -> Optional[Dict]:
    hits = [f for f in files if contains in f["name"] and f["name"].endswith(endswith)]
    hits.sort(key=lambda x: x["name"])
    return hits[0] if hits else None


def download_set_and_pair(files: List[Dict], set_file: Dict, stage: str) -> Path:
    """Download .set and matching .fdt into same stage folder."""
    set_name = set_file["name"]
    local_set = CACHE_ROOT / stage / set_name
    _download(set_file["id"], local_set)

    if set_name.lower().endswith(".set"):
        fdt_name = set_name[:-4] + ".fdt"
        fdt = _find_by_name(files, fdt_name)
        if fdt:
            _download(fdt["id"], CACHE_ROOT / stage / fdt_name)
    return local_set


# =========================
# Participant discovery
# =========================

def participants_from_index(files: List[Dict]) -> List[str]:
    ids = set()
    for f in files:
        m = re.search(r"(sub-\d+)", f["name"])
        if m:
            ids.add(m.group(1))
    return sorted(ids)


# =========================
# Plot helpers
# =========================

def plot_segment_st(raw: mne.io.BaseRaw, title: str, channel: str = "Pz", duration: float = 5.0):
    apply_plot_style()
    ch = channel if channel in raw.ch_names else raw.ch_names[0]
    sfreq = float(raw.info["sfreq"])
    n = int(sfreq * duration)
    picks = [raw.ch_names.index(ch)]
    data, times = raw[picks, :n]
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(times, data[0] * 1e6, lw=0.9)
    ax.set_title(f"{title} ({ch})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (µV)")
    ax.axhline(0, lw=0.5, color="gray")
    st.pyplot(fig)
    plt.close(fig)


def compute_psd_db(raw: mne.io.BaseRaw, channel: str, fmin: float, fmax: float) -> Tuple[np.ndarray, np.ndarray, str]:
    ch = channel if channel in raw.ch_names else raw.ch_names[0]
    spec = raw.compute_psd(fmin=fmin, fmax=fmax, picks=ch, verbose="ERROR")
    freqs = spec.freqs
    psd_db = 10 * np.log10(spec.get_data().squeeze())
    return freqs, psd_db, ch


def bandpower(freqs: np.ndarray, psd_db: np.ndarray, lo: float, hi: float) -> float:
    mask = (freqs >= lo) & (freqs <= hi)
    return float(np.nanmean(psd_db[mask])) if np.any(mask) else float("nan")


# =========================
# Your QC functions (Drive-backed)
# =========================

def find_any_set_local(files_index: List[Dict], folder: str, pattern_endswith: str, p_id: str) -> Optional[Path]:
    """
    Find a file by participant and suffix, download to cache and return local path.
    pattern_endswith should be like 'raw.set', 'preprocessed.set', 'postICA.set', 'epoched.set', 'autoAR.set'
    """
    hit = find_first_match(files_index, contains=p_id, endswith=pattern_endswith)
    if not hit:
        return None
    return download_set_and_pair(files_index, hit, folder)


def qc_1_raw(files_index: List[Dict], p_id: str):
    st.header("[1] Raw Continuous Dataset")
    path = find_any_set_local(files_index, "01_raw", "raw.set", p_id)
    if not path:
        return st.warning("Raw file not found.")
    raw = mne.io.read_raw_eeglab(str(path), preload=True, verbose="ERROR")
    st.write(f"**File:** {path.name}")
    st.info("Expected: Visible drifts, line noise, and large blinks.")
    plot_segment_st(raw, "Raw EEG Segment", "Pz")


def qc_2_preprocessed(files_index: List[Dict], p_id: str):
    st.header("[2] Preprocessed Dataset")
    path = find_any_set_local(files_index, "02_preprocessed", "preprocessed.set", p_id)
    if not path:
        return st.warning("Preprocessed file not found.")
    preproc = mne.io.read_raw_eeglab(str(path), preload=True, verbose="ERROR")
    st.write(f"**File:** {path.name}")
    plot_segment_st(preproc, "Preprocessed Segment", "Pz")

    fig = preproc.compute_psd(fmin=0.1, fmax=45.0, verbose="ERROR").plot(show=False)
    st.pyplot(fig)
    st.write("Expected: 1/f shape and alpha peak (8–12 Hz).")


def qc_3_preICA_loss(files_index: List[Dict], p_id: str):
    st.header("[3] Pre ICA Bad Channel Detection")
    xlsx = _find_by_name(files_index, PREICA_XLSX_NAME)
    if not xlsx:
        return st.warning("Loss table not found on Drive.")
    local = CACHE_ROOT / "03_preICA" / xlsx["name"]
    _download(xlsx["id"], local)

    df = pd.read_excel(local)
    subj_data = df[df["ID"].astype(str).str.contains(p_id)]
    if subj_data.empty:
        return st.write("No loss data for this subject.")

    ch_cols = [c for c in df.columns if c != "ID"]
    values = subj_data[ch_cols].values.flatten().astype(float)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(ch_cols)), values)
    ax.set_ylabel("% Extreme Artifact")
    ax.set_title(f"Artifact Loss per Channel: {p_id}")
    ax.set_xticks(range(len(ch_cols)))
    ax.set_xticklabels(ch_cols, rotation=90, fontsize=7)
    st.pyplot(fig)
    plt.close(fig)


def qc_4_ICLabel(files_index: List[Dict], p_id: str):
    st.header("[4] ICLabel Classification")
    ic = find_first_match(files_index, contains=p_id, endswith=ICLABEL_SUFFIX)
    if not ic:
        return st.warning("ICLabel file not found for this participant.")
    local = CACHE_ROOT / "05_ICLabel" / ic["name"]
    _download(ic["id"], local)

    df = pd.read_excel(local)
    cols = ["Brain", "Muscle", "Eye", "Heart", "Line_Noise", "Channel_Noise", "Other"]
    if not set(cols).issubset(df.columns):
        st.warning("ICLabel columns mismatch.")
        st.write(list(df.columns))
        return

    mean_probs = df[cols].mean()
    fig, ax = plt.subplots()
    mean_probs.plot(kind="bar", ax=ax)
    ax.set_ylabel("Mean probability (%)")
    ax.set_title(f"ICLabel mean probs: {p_id}")
    st.pyplot(fig)
    plt.close(fig)


def qc_5_postICA(files_index: List[Dict], p_id: str):
    st.header("[5] Post ICA & Corrected EOG")
    path = find_any_set_local(files_index, "06_postICA", "postICA.set", p_id)
    if not path:
        return st.warning("Post ICA file not found.")
    raw = mne.io.read_raw_eeglab(str(path), preload=True, verbose="ERROR")
    for ch in ["CVEOGR", "CHEOG"]:
        if ch in raw.ch_names:
            st.write(f"**Channel:** {ch}")
            plot_segment_st(raw, f"Corrected {ch}", ch)


def qc_6_epoched(files_index: List[Dict], p_id: str):
    st.header("[6] Epoched Data & ERP")
    path = find_any_set_local(files_index, "08_AR", "autoAR.set", p_id) or find_any_set_local(files_index, "07_epoched", "epoched.set", p_id)
    if not path:
        return st.warning("Epoched file not found.")

    epochs = mne.io.read_epochs_eeglab(str(path), verbose="ERROR")
    keys = list(epochs.event_id.keys())
    if not keys:
        return st.warning("No events found in epochs.")

    target_code = next((k for k in keys if "11" in str(k)), keys[0])
    evoked = epochs[target_code].average()

    fig = evoked.plot(picks="Pz", show=False) if "Pz" in evoked.ch_names else evoked.plot(show=False)
    st.pyplot(fig)
    st.write(f"ERP for condition: {target_code}. Expected: P3b deflection 300–500ms.")


def qc_7_final_features(files_index: List[Dict]):
    st.header("[7] Final P3b Feature Distribution")
    csv = _find_by_name(files_index, FEATURES_CSV_NAME)
    if not csv:
        return st.warning("Feature CSV not found on Drive.")
    local = CACHE_ROOT / csv["name"]
    _download(csv["id"], local)

    df = pd.read_csv(local)
    col = "roi_mean_300_600" if "roi_mean_300_600" in df.columns else None
    if not col:
        st.warning("Column roi_mean_300_600 not found.")
        st.write(list(df.columns))
        return

    fig, ax = plt.subplots()
    df[col].hist(bins=30, ax=ax)
    ax.set_title("P3b Mean Amplitude Distribution (All Trials)")
    st.pyplot(fig)
    plt.close(fig)


# =========================
# Comparison lab (ALL participants)
# =========================

def compare_psd_overlay(files_index: List[Dict], participants: List[str], stage: str, channel: str, fmin: float, fmax: float):
    pattern = PATTERNS.get(stage, "*preprocessed.set")
    curves = []

    for pid in participants:
        path = find_any_set_local(files_index, stage, pattern.replace("*", ""), pid)
        if not path:
            continue
        raw = mne.io.read_raw_eeglab(str(path), preload=False, verbose="ERROR")
        freqs, psd_db, used_ch = compute_psd_db(raw, channel, fmin, fmax)
        curves.append((pid, freqs, psd_db, used_ch))

    if not curves:
        st.warning("No PSD data found.")
        return

    apply_plot_style()
    fig, ax = plt.subplots(figsize=(10, 4))
    for pid, freqs, psd_db, _ in curves:
        ax.plot(freqs, psd_db, label=pid)
    ax.set_title(f"PSD overlay ({stage}) @ {channel}")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (dB)")
    if len(curves) <= 10:
        ax.legend()
    st.pyplot(fig)
    plt.close(fig)


def compare_bandpower(files_index: List[Dict], participants: List[str], stage: str, channel: str):
    pattern = PATTERNS.get(stage, "*preprocessed.set")
    rows = []
    for pid in participants:
        path = find_any_set_local(files_index, stage, pattern.replace("*", ""), pid)
        if not path:
            rows.append({"participant": pid, "status": "missing"})
            continue
        raw = mne.io.read_raw_eeglab(str(path), preload=False, verbose="ERROR")
        freqs, psd_db, used_ch = compute_psd_db(raw, channel, 0.5, 45.0)

        row = {"participant": pid, "status": "ok", "channel_used": used_ch}
        for name, (lo, hi) in {
            "delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)
        }.items():
            row[name] = bandpower(freqs, psd_db, lo, hi)
        rows.append(row)

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    ok = df[df["status"] == "ok"].set_index("participant")
    if ok.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 4))
    ok[["delta", "theta", "alpha", "beta", "gamma"]].plot(kind="bar", ax=ax)
    ax.set_ylabel("Mean PSD (dB)")
    ax.set_title(f"Bandpower comparison ({stage}) @ {channel}")
    st.pyplot(fig)
    plt.close(fig)


def compare_preica_heatmap(files_index: List[Dict], participants: List[str]):
    xlsx = _find_by_name(files_index, PREICA_XLSX_NAME)
    if not xlsx:
        st.warning("PreICA loss excel not found.")
        return
    local = CACHE_ROOT / "03_preICA" / xlsx["name"]
    _download(xlsx["id"], local)

    df = pd.read_excel(local)
    df["participant"] = df["ID"].astype(str).str.extract(r"^(sub-\d+)")
    df = df[df["participant"].isin(participants)]
    if df.empty:
        st.warning("No rows for selected participants.")
        return

    ch_cols = [c for c in df.columns if c not in ["ID", "participant"]]
    pivot = df.groupby("participant")[ch_cols].mean()
    long = pivot.reset_index().melt(id_vars="participant", var_name="channel", value_name="loss")

    heat = (
        alt.Chart(long)
        .mark_rect()
        .encode(
            x="channel:N",
            y="participant:N",
            color="loss:Q",
            tooltip=["participant", "channel", alt.Tooltip("loss:Q", format=".3f")],
        )
        .properties(height=min(600, 28 * len(pivot.index)))
    )
    st.altair_chart(heat, use_container_width=True)


def compare_iclabel(files_index: List[Dict], participants: List[str]):
    cols = ["Brain", "Muscle", "Eye", "Heart", "Line_Noise", "Channel_Noise", "Other"]
    rows = []
    for pid in participants:
        ic = find_first_match(files_index, contains=pid, endswith=ICLABEL_SUFFIX)
        if not ic:
            continue
        local = CACHE_ROOT / "05_ICLabel" / ic["name"]
        _download(ic["id"], local)
        df = pd.read_excel(local)
        if not set(cols).issubset(df.columns):
            continue
        means = df[cols].mean()
        rows.append({"participant": pid, **{c: float(means[c]) for c in cols}})

    if not rows:
        st.warning("No ICLabel data for selected participants.")
        return

    out = pd.DataFrame(rows).set_index("participant")
    st.dataframe(out, use_container_width=True)

    fig, ax = plt.subplots(figsize=(11, 4))
    out.plot(kind="bar", stacked=True, ax=ax)
    ax.set_ylabel("Mean probability (%)")
    ax.set_title("ICLabel composition by participant")
    st.pyplot(fig)
    plt.close(fig)


def compare_features(files_index: List[Dict], participants: List[str]):
    csv = _find_by_name(files_index, FEATURES_CSV_NAME)
    if not csv:
        st.warning("Features CSV not found.")
        return
    local = CACHE_ROOT / csv["name"]
    _download(csv["id"], local)
    df = pd.read_csv(local)

    pid_col = next((c for c in ["participant", "participant_id", "subject_id", "sub_id", "id"] if c in df.columns), None)
    if not pid_col:
        st.warning("No participant id column found in features CSV.")
        st.write(list(df.columns))
        return

    df[pid_col] = df[pid_col].astype(str).str.strip()
    df = df[df[pid_col].isin(participants)]
    if df.empty:
        st.warning("No feature rows for selected participants.")
        return

    feature_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    feature = st.selectbox("Feature", feature_cols, index=0)

    fig, ax = plt.subplots(figsize=(10, 4))
    df.boxplot(column=feature, by=pid_col, ax=ax)
    ax.set_title(f"{feature} by participant")
    ax.set_ylabel(feature)
    plt.suptitle("")
    st.pyplot(fig)
    plt.close(fig)

    summary = df.groupby(pid_col)[feature].agg(["count", "mean", "std", "min", "max"]).reset_index()
    st.dataframe(summary, use_container_width=True)


# =========================
# Overview
# =========================

def cache_size_mb() -> float:
    total = 0
    for p in CACHE_ROOT.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total / (1024 * 1024)


def overview(files_index: List[Dict], root_meta: Dict):
    st.header("Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Drive folder", root_meta.get("name", ""))
    col2.metric("Indexed files", f"{len(files_index):,}")
    col3.metric("Participants", f"{len(participants_from_index(files_index)):,}")
    col4.metric("Cache (MB)", f"{cache_size_mb():.1f}")

    # file type distribution
    ext_counts = {}
    for f in files_index:
        name = f["name"]
        ext = name.split(".")[-1].lower() if "." in name else "(none)"
        ext_counts[ext] = ext_counts.get(ext, 0) + 1

    df = pd.DataFrame([{"ext": k, "count": v} for k, v in sorted(ext_counts.items(), key=lambda x: -x[1])]).head(12)
    chart = alt.Chart(df).mark_bar().encode(x="count:Q", y=alt.Y("ext:N", sort="-x")).properties(height=320)
    st.altair_chart(chart, use_container_width=True)

    st.caption("This app runs the full MNE QC pipeline on Drive-backed EEG data (cached in /tmp).")


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
        st.sidebar.success("Drive cache cleared. Reload to reindex.")


# =========================
# Main
# =========================

def main():
    st.set_page_config(layout="wide", page_title="COCOA QC Suite")
    st.title("COCOA QC Suite")

    root_id = st.secrets["GDRIVE_VO_FOLDER_ID"]
    root_meta = drive_smoke(root_id)

    vo_id = resolve_vo_folder_id(root_id)
    if vo_id != root_id:
        st.caption("Auto-detected: using **Preprocessed_VisualOddball** inside the parent folder.")

    files_index = drive_index_recursive(vo_id)
    p_all = participants_from_index(files_index)

    sidebar_tools()

    nav = st.sidebar.radio("Navigation", ["Overview", "QC Single", "QC Full", "Comparison Lab"], index=0)

    if nav == "Overview":
        overview(files_index, root_meta)
        return

    if not p_all:
        st.error("No participants found in index.")
        return

    if nav in ["QC Single", "QC Full"]:
        pid = st.sidebar.selectbox("Participant", p_all, index=0)

        if nav == "QC Single":
            step = st.sidebar.selectbox(
                "Step",
                ["[1] Raw", "[2] Preprocessed", "[3] PreICA Loss", "[4] ICLabel", "[5] PostICA", "[6] Epoched", "[7] Features"],
                index=1,
            )
            if step == "[1] Raw":
                qc_1_raw(files_index, pid)
            elif step == "[2] Preprocessed":
                qc_2_preprocessed(files_index, pid)
            elif step == "[3] PreICA Loss":
                qc_3_preICA_loss(files_index, pid)
            elif step == "[4] ICLabel":
                qc_4_ICLabel(files_index, pid)
            elif step == "[5] PostICA":
                qc_5_postICA(files_index, pid)
            elif step == "[6] Epoched":
                qc_6_epoched(files_index, pid)
            else:
                qc_7_final_features(files_index)

        else:
            st.sidebar.info("Runs all steps in order.")
            qc_1_raw(files_index, pid)
            qc_2_preprocessed(files_index, pid)
            qc_3_preICA_loss(files_index, pid)
            qc_4_ICLabel(files_index, pid)
            qc_5_postICA(files_index, pid)
            qc_6_epoched(files_index, pid)
            qc_7_final_features(files_index)

        return

    # Comparison Lab
    st.header("Comparison Lab (Multi-participant)")
    chosen = st.sidebar.multiselect("Participants", p_all, default=p_all[:6])
    if len(chosen) < 2:
        st.info("Select at least 2 participants.")
        return

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["PSD Overlay", "Bandpower", "PreICA Heatmap", "ICLabel", "P3b Features"])

    with tab1:
        stage = st.selectbox("Stage", ["01_raw", "02_preprocessed", "06_postICA"], index=1)
        channel = st.selectbox("Channel", ["Pz", "Cz", "Fz", "Oz"], index=0)
        fmin, fmax = st.slider("PSD range (Hz)", 0.1, 60.0, (0.5, 45.0), 0.1)
        compare_psd_overlay(files_index, chosen, stage, channel, fmin, fmax)

    with tab2:
        stage = st.selectbox("Stage for bandpower", ["01_raw", "02_preprocessed"], index=1)
        channel = st.selectbox("Channel for bandpower", ["Pz", "Cz", "Fz", "Oz"], index=0)
        compare_bandpower(files_index, chosen, stage, channel)

    with tab3:
        compare_preica_heatmap(files_index, chosen)

    with tab4:
        compare_iclabel(files_index, chosen)

    with tab5:
        compare_features(files_index, chosen)


if __name__ == "__main__":
    main()
