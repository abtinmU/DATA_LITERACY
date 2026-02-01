"""
COCOA Unified Streamlit Dashboard (Streamlit Cloud-ready)

Views
- Overview (dataset health, cache, composition)
- Visual Oddball QC (Drive-backed, MNE)
- Comparison Lab (multi-participant comparisons)
- Pre-ICA Metrics (Excel explorer)

Secrets required (Streamlit Cloud -> App -> Settings -> Secrets)
- GDRIVE_VO_FOLDER_ID = "<folder id>"  (can be DataLiteracyProject or Preprocessed_VisualOddball)
- [gcp_service_account] ... (service account fields including private_key)

Notes
- If you set GDRIVE_VO_FOLDER_ID to the parent folder (DataLiteracyProject),
  the app automatically locates "Preprocessed_VisualOddball" child folder.
"""

from __future__ import annotations

import io
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt

# ---- SciPy/MNE compatibility shim for Python 3.13 ----
# New SciPy removed sph_harm; MNE still imports it.
# We recreate sph_harm using sph_harm_y with correct argument mapping.
try:
    import scipy.special as _sp

    if not hasattr(_sp, "sph_harm") and hasattr(_sp, "sph_harm_y"):
        def sph_harm(m, n, theta, phi, out=None):
            # Old sph_harm: (m, n, theta=azimuth, phi=polar)
            # New sph_harm_y: (n, m, theta=polar, phi=azimuth)
            # => swap (theta, phi) and swap (m, n)
            y = _sp.sph_harm_y(n, m, phi, theta)
            return y if out is None else np.copyto(out, y)

        _sp.sph_harm = sph_harm
except Exception:
    pass

import mne

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

BANDS = {
    "delta (1–4 Hz)": (1.0, 4.0),
    "theta (4–8 Hz)": (4.0, 8.0),
    "alpha (8–13 Hz)": (8.0, 13.0),
    "beta (13–30 Hz)": (13.0, 30.0),
    "gamma (30–45 Hz)": (30.0, 45.0),
}

FEATURES_CSV_NAME = "COCOA_VO_P3b_trial_features_with_meta.csv"
PREICA_XLSX_NAME = "COCOA_preICAextremeloss.xlsx"


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


def drive_smoke_test(folder_id: str) -> Dict:
    """Fail fast if secrets/sharing are wrong. Returns folder meta."""
    try:
        sa_email = st.secrets["gcp_service_account"]["client_email"]
        svc = _drive_service()

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
        return meta

    except Exception as e:
        st.error("❌ Drive access failed. Check folder sharing + secrets.")
        st.exception(e)
        st.stop()


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
    return next((f for f in files if isinstance(f, dict) and f.get("name") == name), None)


def find_first_match(files: List[Dict], *, contains: str, endswith: str) -> Optional[Dict]:
    valid = [f for f in files if isinstance(f, dict) and "name" in f]
    hits = [f for f in valid if contains in f["name"] and f["name"].endswith(endswith)]
    hits.sort(key=lambda x: x["name"])
    return hits[0] if hits else None


def participants_from_index(files: List[Dict]) -> List[str]:
    ids = set()
    for f in files:
        if not isinstance(f, dict) or "name" not in f:
            continue
        m = re.search(r"(sub-\d+)", f["name"])
        if m:
            ids.add(m.group(1))
    return sorted(ids)


def resolve_vo_root_folder_id(root_id: str) -> str:
    """
    If root_id is the parent (DataLiteracyProject), find child folder named
    'Preprocessed_VisualOddball'. If root_id already points to that folder, return as-is.
    """
    svc = _drive_service()
    meta = svc.files().get(fileId=root_id, fields="id,name,mimeType", supportsAllDrives=True).execute()
    if meta.get("name", "") == "Preprocessed_VisualOddball":
        return root_id

    children = drive_list_children(root_id)
    for c in children:
        if c.get("mimeType") == "application/vnd.google-apps.folder" and c.get("name") == "Preprocessed_VisualOddball":
            return c["id"]

    return root_id


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


def find_any_set(files_index, folder, pattern="*.set", participant_id=None):
    """
    Drive-backed find_any_set. Exactly like your local version:
      - participant_id -> finds participant_id*<suffix>
      - else -> finds *<suffix>
    Downloads .set + paired .fdt to /tmp and returns local path string.
    """
    suffix = pattern.replace("*", "")
    if suffix == "":
        suffix = ".set"

    valid = [f for f in files_index if isinstance(f, dict) and "name" in f and "id" in f]

    if participant_id:
        candidates = [f for f in valid if f["name"].startswith(str(participant_id)) and f["name"].endswith(suffix)]
    else:
        candidates = [f for f in valid if f["name"].endswith(suffix)]

    candidates.sort(key=lambda x: x["name"])
    if not candidates:
        return None

    hit = candidates[0]
    set_name = hit["name"]

    local_set = CACHE_ROOT / folder / set_name
    _download(hit["id"], local_set)

    if set_name.lower().endswith(".set"):
        fdt_name = set_name[:-4] + ".fdt"
        fdt_meta = next((x for x in valid if x["name"] == fdt_name), None)
        if fdt_meta:
            local_fdt = CACHE_ROOT / folder / fdt_name
            _download(fdt_meta["id"], local_fdt)

    return str(local_set)


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


def preica_view(files_index: Optional[List[Dict]] = None) -> None:
    st.header("Pre-ICA Metrics")

    excel_path: Optional[str] = None
    if LOCAL_PREICA_XLSX.exists():
        excel_path = str(LOCAL_PREICA_XLSX)
        st.caption("Using local COCOA_preICAextremeloss.xlsx")
    else:
        if files_index is not None:
            xlsx = _find_by_name(files_index, PREICA_XLSX_NAME)
            if xlsx:
                local = CACHE_ROOT / "03_preICA" / xlsx["name"]
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


# ============================================================
# VO QC helpers
# ============================================================
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


def _cache_size_mb() -> float:
    total = 0
    for p in CACHE_ROOT.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total / (1024 * 1024)


def overview_view(files_index: List[Dict], root_meta: Dict, vo_folder_id: str) -> None:
    st.header("Overview")

    participants = participants_from_index(files_index)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Root folder", root_meta.get("name", ""))
    c2.metric("Participants", f"{len(participants):,}")
    c3.metric("Indexed files", f"{len(files_index):,}")
    c4.metric("Cache (MB)", f"{_cache_size_mb():.1f}")

    ext_counts: Dict[str, int] = {}
    for f in files_index:
        if not isinstance(f, dict) or "name" not in f:
            continue
        name = f["name"]
        ext = name.split(".")[-1].lower() if "." in name else "(none)"
        ext_counts[ext] = ext_counts.get(ext, 0) + 1

    df_ext = pd.DataFrame(
        [{"ext": k, "count": v} for k, v in sorted(ext_counts.items(), key=lambda x: -x[1])]
    ).head(12)

    st.subheader("Dataset composition (by extension)")
    chart = (
        alt.Chart(df_ext)
        .mark_bar()
        .encode(
            x=alt.X("count:Q", title="Count"),
            y=alt.Y("ext:N", sort="-x", title="Extension"),
            tooltip=["ext", "count"],
        )
        .properties(height=340)
    )
    st.altair_chart(chart, use_container_width=True)

    st.subheader("Expected folder structure (Visual Oddball)")
    st.markdown(
        """
- `01_raw` → `*.set` + `*.fdt`
- `02_preprocessed` → `*.set` + `*.fdt`
- `03_preICA` → `COCOA_preICAextremeloss.xlsx`
- `05_ICLabel` → `*_ICclassifications.xlsx`
- `06_postICA` → `*postICA.set`
- `07_epoched` → `*epoched.set`
- `08_AR` → `*autoAR.set`
"""
    )
    st.caption(f"Resolved VisualOddball folder id: {vo_folder_id}")


def _compute_psd_db_one(raw_obj, channel: str, fmin=0.5, fmax=45.0):
    ch = channel if channel in raw_obj.ch_names else raw_obj.ch_names[0]
    spec = raw_obj.compute_psd(fmin=fmin, fmax=fmax, picks=ch, verbose="ERROR")
    freqs = spec.freqs
    psd_db = 10 * np.log10(spec.get_data().squeeze())
    return freqs, psd_db, ch


def comparison_view(files_index: List[Dict]) -> None:
    st.header("Comparison Lab")

    participants_all = participants_from_index(files_index)
    if not participants_all:
        st.warning("No participants found in Drive index.")
        return

    chosen = st.multiselect(
        "Participants",
        participants_all,
        default=participants_all[:6],
        help="Select multiple participants to compare metrics.",
    )
    if len(chosen) < 2:
        st.info("Select at least 2 participants.")
        return

    tabs = st.tabs(["PSD Overlay", "Bandpower", "PreICA Heatmap", "ICLabel Compare", "P3b Feature Compare"])

    with tabs[0]:
        st.subheader("PSD overlay across participants")
        stage = st.selectbox("Stage", ["01_raw", "02_preprocessed", "06_postICA"], index=1)
        channel = st.selectbox("Channel", ["Pz", "Cz", "Fz", "Oz"], index=0)
        fmin, fmax = st.slider("Frequency range (Hz)", 0.1, 60.0, (0.5, 45.0), 0.1)

        run = st.button("Run PSD overlay", type="primary", key="run_psd_overlay")
        if run:
            curves = []
            pattern = {"01_raw": "*raw.set", "02_preprocessed": "*preprocessed.set", "06_postICA": "*postICA.set"}[stage]
            for pid in chosen:
                path = find_any_set(files_index, stage, pattern, pid)
                if not path:
                    continue
                try:
                    raw = mne.io.read_raw_eeglab(path, preload=False, verbose="ERROR")
                    freqs, psd_db, _ = _compute_psd_db_one(raw, channel, fmin=fmin, fmax=fmax)
                    curves.append((pid, freqs, psd_db))
                except Exception:
                    continue

            if not curves:
                st.warning("No PSD curves could be computed.")
            else:
                apply_plot_style()
                fig, ax = plt.subplots(figsize=(10, 4))
                for pid, freqs, psd_db in curves:
                    ax.plot(freqs, psd_db, label=pid)
                ax.set_title(f"PSD overlay ({stage}) @ {channel}")
                ax.set_xlabel("Frequency (Hz)")
                ax.set_ylabel("PSD (dB)")
                if len(curves) <= 10:
                    ax.legend()
                st.pyplot(fig)
                plt.close(fig)

    with tabs[1]:
        st.subheader("Bandpower comparison (mean PSD in bands)")
        stage = st.selectbox("Stage for bandpower", ["01_raw", "02_preprocessed"], index=1, key="bp_stage")
        channel = st.selectbox("Channel for bandpower", ["Pz", "Cz", "Fz", "Oz"], index=0, key="bp_ch")
        band_names = st.multiselect("Bands", list(BANDS.keys()), default=["alpha (8–13 Hz)", "beta (13–30 Hz)"])

        run = st.button("Run bandpower", type="primary", key="run_bandpower")
        if run:
            rows = []
            pattern = "*preprocessed.set" if stage == "02_preprocessed" else "*raw.set"
            for pid in chosen:
                path = find_any_set(files_index, stage, pattern, pid)
                if not path:
                    rows.append({"participant": pid, "status": "missing"})
                    continue
                try:
                    raw = mne.io.read_raw_eeglab(path, preload=False, verbose="ERROR")
                    freqs, psd_db, used_ch = _compute_psd_db_one(raw, channel, fmin=0.5, fmax=45.0)
                    row = {"participant": pid, "status": "ok", "channel_used": used_ch}
                    for bn in band_names:
                        lo, hi = BANDS[bn]
                        mask = (freqs >= lo) & (freqs <= hi)
                        row[bn] = float(np.nanmean(psd_db[mask])) if np.any(mask) else np.nan
                    rows.append(row)
                except Exception:
                    rows.append({"participant": pid, "status": "error"})

            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)

            ok = df[df["status"] == "ok"].set_index("participant")
            if not ok.empty and band_names:
                apply_plot_style()
                fig, ax = plt.subplots(figsize=(11, 4))
                ok[band_names].plot(kind="bar", ax=ax)
                ax.set_ylabel("Mean PSD (dB)")
                ax.set_title(f"Bandpower ({stage}) @ {channel}")
                st.pyplot(fig)
                plt.close(fig)

    with tabs[2]:
        st.subheader("Pre-ICA extreme-loss heatmap (participant × channel)")
        xlsx = _find_by_name(files_index, PREICA_XLSX_NAME)
        if not xlsx:
            st.warning("COCOA_preICAextremeloss.xlsx not found in Drive index.")
        else:
            local_xlsx = CACHE_ROOT / "03_preICA" / xlsx["name"]
            _download(xlsx["id"], local_xlsx)
            df = pd.read_excel(local_xlsx)
            df["participant"] = df["ID"].astype(str).str.extract(r"^(sub-\d+)")
            df = df[df["participant"].isin(chosen)]
            if df.empty:
                st.warning("No loss rows for selected participants.")
            else:
                ch_cols = [c for c in df.columns if c not in ["ID", "participant"]]
                pivot = df.groupby("participant")[ch_cols].mean()
                long = pivot.reset_index().melt(id_vars="participant", var_name="channel", value_name="loss")

                heat = (
                    alt.Chart(long)
                    .mark_rect()
                    .encode(
                        x=alt.X("channel:N", title="Channel"),
                        y=alt.Y("participant:N", title="Participant"),
                        color=alt.Color("loss:Q", title="Extreme loss"),
                        tooltip=["participant", "channel", alt.Tooltip("loss:Q", format=".3f")],
                    )
                    .properties(height=min(600, 28 * len(pivot.index)))
                )
                st.altair_chart(heat, use_container_width=True)

    with tabs[3]:
        st.subheader("ICLabel composition (stacked)")
        cols = ["Brain", "Muscle", "Eye", "Heart", "Line_Noise", "Channel_Noise", "Other"]
        rows = []
        for pid in chosen:
            ic = find_first_match(files_index, contains=pid, endswith="_ICclassifications.xlsx")
            if not ic:
                continue
            local_ic = CACHE_ROOT / "05_ICLabel" / ic["name"]
            _download(ic["id"], local_ic)
            try:
                df = pd.read_excel(local_ic)
                if not set(cols).issubset(df.columns):
                    continue
                m = df[cols].mean()
                rows.append({"participant": pid, **{c: float(m[c]) for c in cols}})
            except Exception:
                continue

        if not rows:
            st.warning("No ICLabel files readable for the selected participants.")
        else:
            out = pd.DataFrame(rows).set_index("participant")
            st.dataframe(out, use_container_width=True)

            apply_plot_style()
            fig, ax = plt.subplots(figsize=(11, 4))
            out.plot(kind="bar", stacked=True, ax=ax)
            ax.set_ylabel("Mean probability (%)")
            ax.set_title("ICLabel class probability composition")
            st.pyplot(fig)
            plt.close(fig)

    with tabs[4]:
        st.subheader("P3b feature comparisons")
        csv = _find_by_name(files_index, FEATURES_CSV_NAME)
        if not csv:
            st.warning("Feature CSV not found in Drive index.")
        else:
            local_csv = CACHE_ROOT / csv["name"]
            _download(csv["id"], local_csv)
            df = pd.read_csv(local_csv)

            pid_col = next((c for c in ["participant", "participant_id", "subject_id", "sub_id", "id"] if c in df.columns), None)
            if not pid_col:
                st.warning("Could not detect participant id column in CSV.")
                st.write(list(df.columns))
            else:
                df[pid_col] = df[pid_col].astype(str).str.strip()
                df = df[df[pid_col].isin(chosen)]
                numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                if not numeric_cols:
                    st.warning("No numeric feature columns found.")
                else:
                    feat = st.selectbox("Feature", numeric_cols, index=0)

                    apply_plot_style()
                    fig, ax = plt.subplots(figsize=(10, 4))
                    df.boxplot(column=feat, by=pid_col, ax=ax)
                    ax.set_title(f"{feat} by participant")
                    ax.set_ylabel(feat)
                    plt.suptitle("")
                    st.pyplot(fig)
                    plt.close(fig)

                    summary = df.groupby(pid_col)[feat].agg(["count", "mean", "std", "min", "max"]).reset_index()
                    st.dataframe(summary, use_container_width=True)


# ============================================================
# Visual Oddball QC (keep your working style)
# ============================================================
def vo_qc_view(files_index: List[Dict]) -> None:
    st.header("Visual Oddball QC (Google Drive)")

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

    # [1] Raw
    st.header("[1] Raw Continuous Dataset")
    path = find_any_set(files_index, "01_raw", "*raw.set", pid)
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

    # [2] Preprocessed
    st.header("[2] Preprocessed Dataset")
    path = find_any_set(files_index, "02_preprocessed", "*preprocessed.set", pid)
    if not path:
        st.warning("Preprocessed file not found.")
    else:
        try:
            preproc = mne.io.read_raw_eeglab(path, preload=True, verbose="ERROR")
            st.write(f"**File:** {os.path.basename(path)}")
            plot_segment_st(preproc, "Preprocessed Segment", "Pz")

            fig = preproc.compute_psd(fmin=0.1, fmax=45.0, verbose="ERROR").plot(show=False)
            st.pyplot(fig)
            st.write("Expected: 1/f shape and alpha peak (8–12 Hz).")
        except Exception as e:
            st.error("MNE could not open preprocessed file.")
            st.exception(e)

    # [3] Pre-ICA loss
    st.header("[3] Pre ICA Bad Channel Detection")
    xlsx = _find_by_name(files_index, PREICA_XLSX_NAME)
    if not xlsx:
        st.warning("Loss table not found.")
    else:
        local = CACHE_ROOT / "03_preICA" / xlsx["name"]
        _download(xlsx["id"], local)
        df = pd.read_excel(local)
        subj = df[df["ID"].astype(str).str.contains(pid)]
        if subj.empty:
            st.write("No loss data for this subject.")
        else:
            ch_cols = [c for c in df.columns if c != "ID"]
            values = subj[ch_cols].values.flatten().astype(float)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(range(len(ch_cols)), values)
            ax.set_ylabel("% Extreme Artifact")
            ax.set_title(f"Artifact Loss per Channel: {pid}")
            st.pyplot(fig)
            plt.close(fig)

    # [4] ICLabel
    st.header("[4] ICLabel Classification")
    ic_xlsx = find_first_match(files_index, contains=pid, endswith="_ICclassifications.xlsx")
    if not ic_xlsx:
        st.warning("ICLabel file not found for this participant.")
    else:
        local_ic = CACHE_ROOT / "05_ICLabel" / ic_xlsx["name"]
        _download(ic_xlsx["id"], local_ic)
        df = pd.read_excel(local_ic)
        cols = ["Brain", "Muscle", "Eye", "Heart", "Line_Noise", "Channel_Noise", "Other"]
        if set(cols).issubset(df.columns):
            mean_probs = df[cols].mean()
            fig, ax = plt.subplots()
            mean_probs.plot(kind="bar", ax=ax)
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning("ICLabel columns mismatch.")
            st.write(list(df.columns))

    # [5] Epoched ERP
    st.header("[5] Epoched Data & ERP")
    path = (
        find_any_set(files_index, "08_AR", "*autoAR.set", pid)
        or find_any_set(files_index, "07_epoched", "*epoched.set", pid)
    )
    if not path:
        st.warning("Epoched file not found.")
    else:
        try:
            epochs = mne.io.read_epochs_eeglab(path, verbose="ERROR")
            st.write(f"**File:** {os.path.basename(path)}")
            if not epochs.event_id:
                st.warning("No event_id found in this epochs file.")
            else:
                keys = list(epochs.event_id.keys())
                target_code = next((k for k in keys if "11" in str(k)), keys[0])
                evoked = epochs[target_code].average()
                fig = evoked.plot(picks="Pz" if "Pz" in evoked.ch_names else None, show=False)
                st.pyplot(fig)
                st.write(f"ERP for condition: **{target_code}**. Expected: P3b deflection 300–500ms.")
        except Exception as e:
            st.error("MNE could not open or plot epoched/ERP data.")
            st.exception(e)

    # [6] Post ICA (EOG)
    st.header("[6] Post ICA & Corrected EOG")
    path = find_any_set(files_index, "06_postICA", "*postICA.set", pid)
    if not path:
        st.warning("Post ICA file not found.")
    else:
        try:
            raw = mne.io.read_raw_eeglab(path, preload=True, verbose="ERROR")
            st.write(f"**File:** {os.path.basename(path)}")

            found_any = False
            for ch in ["CVEOGR", "CHEOG"]:
                if ch in raw.ch_names:
                    found_any = True
                    st.write(f"**Channel:** {ch}")
                    plot_segment_st(raw, f"Corrected {ch}", ch)

            if not found_any:
                st.info("EOG channels (CVEOGR/CHEOG) not found in this recording.")

        except Exception as e:
            st.error("MNE could not open postICA file.")
            st.exception(e)


# ============================================================
# Sidebar tools
# ============================================================
def sidebar_tools() -> None:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Tools")

    if st.sidebar.button("Clear local cache (/tmp)", help="Deletes downloaded files under /tmp/cocoa_cache"):
        shutil.rmtree(CACHE_ROOT, ignore_errors=True)
        CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        st.sidebar.success("Cache cleared.")

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
    root_meta = drive_smoke_test(root_folder_id)

    vo_folder_id = resolve_vo_root_folder_id(root_folder_id)

    # If user passed parent folder, resolve VO folder
    if vo_folder_id == root_folder_id:
        children = drive_list_children(root_folder_id)
        child_names = {c.get("name") for c in children}
        if "Preprocessed_VisualOddball" in child_names:
            vo_folder_id = next(c["id"] for c in children if c.get("name") == "Preprocessed_VisualOddball")

    files_index = drive_index_recursive(vo_folder_id)

    sidebar_tools()

    view = st.sidebar.selectbox(
        "Select view",
        ["Overview", "Visual Oddball QC", "Comparison Lab", "Pre-ICA Metrics"],
        key="main_view"
    )

    if view == "Overview":
        overview_view(files_index, root_meta, vo_folder_id)
    elif view == "Visual Oddball QC":
        vo_qc_view(files_index=files_index)
    elif view == "Comparison Lab":
        comparison_view(files_index)
    else:
        preica_view(files_index=files_index)


if __name__ == "__main__":
    main()
