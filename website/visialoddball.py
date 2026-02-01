import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import mne

# -----------------------------------------
# You already have these implemented:
# PROJECT
# find_any_set(folder, pattern, p_id)
# plot_segment_st(raw, title, channel)
# qc_1_raw ... qc_7_final_features
# -----------------------------------------


# ========= Utility helpers =========

def list_participants_from_sets() -> list[str]:
    """Derive participant ids by scanning .set filenames across the pipeline."""
    folders = ["01_raw", "02_preprocessed", "03_preICA", "04_ICAweighted", "06_postICA", "07_epoched", "08_AR"]
    ids = set()
    for fol in folders:
        base = os.path.join(PROJECT, fol)
        if not os.path.isdir(base):
            continue
        for p in glob.glob(os.path.join(base, "*.set")):
            bn = os.path.basename(p)
            # expects filenames like sub-001_task-visualoddball_...
            if "sub-" in bn:
                ids.add(bn.split("_")[0])  # sub-001
    return sorted(ids)


def safe_read_raw(path: str):
    """Read raw, raise real error to UI."""
    try:
        return mne.io.read_raw_eeglab(path, preload=True, verbose="ERROR")
    except Exception as e:
        st.error(f"MNE could not open: {os.path.basename(path)}")
        st.exception(e)
        return None


def compute_channel_psd_db(raw, channel="Pz", fmin=0.5, fmax=45.0):
    """Compute PSD for one channel and return (freqs, psd_db)."""
    if raw is None:
        return None
    ch = channel if channel in raw.ch_names else raw.ch_names[0]
    try:
        spec = raw.compute_psd(fmin=fmin, fmax=fmax, picks=ch, verbose="ERROR")
        freqs = spec.freqs
        psd_db = 10 * np.log10(spec.get_data().squeeze())
        return freqs, psd_db, ch
    except Exception as e:
        st.error("PSD computation failed.")
        st.exception(e)
        return None


def bandpower_from_psd(freqs, psd_db, band):
    lo, hi = band
    mask = (freqs >= lo) & (freqs <= hi)
    if not np.any(mask):
        return np.nan
    return float(np.nanmean(psd_db[mask]))


# ========= Comparison Lab plots =========

def compare_psd_overlay(participants: list[str], stage: str, channel: str, fmin: float, fmax: float):
    """Overlay PSDs across participants for a given pipeline stage."""
    suffix_map = {
        "01_raw": "*raw.set",
        "02_preprocessed": "*preprocessed.set",
        "06_postICA": "*postICA.set",
    }
    pattern = suffix_map[stage]

    curves = []
    for pid in participants:
        path = find_any_set(stage, pattern, pid)
        if not path:
            continue
        raw = safe_read_raw(path)
        res = compute_channel_psd_db(raw, channel=channel, fmin=fmin, fmax=fmax)
        if res:
            freqs, psd_db, used_ch = res
            curves.append((pid, freqs, psd_db, used_ch))

    if not curves:
        st.warning("No PSD data found for selected participants.")
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    for pid, freqs, psd_db, used_ch in curves:
        ax.plot(freqs, psd_db, label=pid)
    ax.set_title(f"PSD overlay ({stage}) @ {channel}")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (dB)")
    if len(curves) <= 10:
        ax.legend()
    st.pyplot(fig)
    plt.close(fig)


def compare_bandpower_bars(participants: list[str], stage: str, channel: str):
    """Bandpower-style comparison using mean PSD(dB) in common bands."""
    bands = {
        "delta (1–4)": (1, 4),
        "theta (4–8)": (4, 8),
        "alpha (8–13)": (8, 13),
        "beta (13–30)": (13, 30),
        "gamma (30–45)": (30, 45),
    }

    pattern = "*preprocessed.set" if stage == "02_preprocessed" else "*raw.set"
    rows = []
    for pid in participants:
        path = find_any_set(stage, pattern, pid)
        if not path:
            rows.append({"participant": pid, "status": "missing"})
            continue
        raw = safe_read_raw(path)
        res = compute_channel_psd_db(raw, channel=channel, fmin=0.5, fmax=45.0)
        if not res:
            rows.append({"participant": pid, "status": "error"})
            continue
        freqs, psd_db, used_ch = res
        row = {"participant": pid, "channel_used": used_ch, "status": "ok"}
        for name, band in bands.items():
            row[name] = bandpower_from_psd(freqs, psd_db, band)
        rows.append(row)

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        st.warning("No successful bandpower results.")
        return

    fig, ax = plt.subplots(figsize=(11, 4))
    ok.set_index("participant")[list(bands.keys())].plot(kind="bar", ax=ax)
    ax.set_ylabel("Mean PSD (dB)")
    ax.set_title(f"Bandpower comparison ({stage}) @ {channel}")
    st.pyplot(fig)
    plt.close(fig)


def compare_preica_heatmap(participants: list[str]):
    """Pre-ICA extreme loss heatmap participant x channel."""
    loss_xlsx = os.path.join(PROJECT, "03_preICA", "COCOA_preICAextremeloss.xlsx")
    if not os.path.exists(loss_xlsx):
        st.warning("Pre-ICA loss excel not found.")
        return

    df = pd.read_excel(loss_xlsx)
    df["participant"] = df["ID"].astype(str).str.extract(r"^(sub-\d+)")
    df = df[df["participant"].isin(participants)].copy()

    ch_cols = [c for c in df.columns if c not in ["ID", "participant"]]
    if df.empty:
        st.warning("No rows for selected participants.")
        return

    pivot = df.groupby("participant")[ch_cols].mean()

    # Altair heatmap (clean)
    long = pivot.reset_index().melt(id_vars="participant", var_name="channel", value_name="loss")
    import altair as alt
    heat = (
        alt.Chart(long)
        .mark_rect()
        .encode(
            x=alt.X("channel:N", title="Channel"),
            y=alt.Y("participant:N", title="Participant"),
            color=alt.Color("loss:Q", title="Extreme loss"),
            tooltip=["participant", "channel", alt.Tooltip("loss:Q", format=".3f")],
        )
        .properties(height=min(600, 30 * len(pivot.index)))
    )
    st.altair_chart(heat, use_container_width=True)


def compare_iclabel(participants: list[str]):
    """ICLabel stacked bar compare across participants (mean probs)."""
    ic_dir = os.path.join(PROJECT, "05_ICLabel")
    cols = ["Brain", "Muscle", "Eye", "Heart", "Line_Noise", "Channel_Noise", "Other"]

    rows = []
    for pid in participants:
        files = glob.glob(os.path.join(ic_dir, f"{pid}*_ICclassifications.xlsx"))
        if not files:
            continue
        df = pd.read_excel(files[0])
        if not set(cols).issubset(df.columns):
            continue
        m = df[cols].mean()
        row = {"participant": pid}
        row.update({c: float(m[c]) for c in cols})
        rows.append(row)

    if not rows:
        st.warning("No ICLabel files found (or columns mismatch) for selected participants.")
        return

    out = pd.DataFrame(rows).set_index("participant")
    st.dataframe(out, use_container_width=True)

    fig, ax = plt.subplots(figsize=(11, 4))
    out.plot(kind="bar", stacked=True, ax=ax)
    ax.set_ylabel("Mean probability (%)")
    ax.set_title("ICLabel composition by participant")
    st.pyplot(fig)
    plt.close(fig)


def compare_p3b_features(participants: list[str]):
    """Compare P3b features from trial_features CSV."""
    feat_csv = os.path.join(PROJECT, "COCOA_VO_P3b_trial_features_with_meta.csv")
    if not os.path.exists(feat_csv):
        st.warning("Trial features CSV not found.")
        return

    df = pd.read_csv(feat_csv)
    # best guess: participant id column often called participant_id or subject_id; fallback to 'participant'
    pid_col = None
    for c in ["participant", "participant_id", "subject_id", "sub_id", "id"]:
        if c in df.columns:
            pid_col = c
            break
    if not pid_col:
        st.warning("Could not find participant column in CSV.")
        st.write("Columns:", list(df.columns))
        return

    df[pid_col] = df[pid_col].astype(str).str.strip()
    df = df[df[pid_col].isin(participants)].copy()

    if df.empty:
        st.warning("No rows for selected participants in features CSV.")
        return

    feature = st.selectbox("Feature to compare", options=[c for c in df.columns if df[c].dtype != object], index=0)
    st.write(f"Comparing feature: **{feature}**")

    # boxplot
    fig, ax = plt.subplots(figsize=(10, 4))
    df.boxplot(column=feature, by=pid_col, ax=ax)
    ax.set_title(f"{feature} by participant")
    ax.set_ylabel(feature)
    plt.suptitle("")
    st.pyplot(fig)
    plt.close(fig)

    # summary table
    summary = df.groupby(pid_col)[feature].agg(["count", "mean", "std", "min", "max"]).reset_index()
    st.dataframe(summary, use_container_width=True)


# =========================
# Main App
# =========================
def main():
    st.set_page_config(layout="wide", page_title="COCOA QC Suite")
    st.title("COCOA QC Suite")

    # Navigation
    nav = st.sidebar.radio("Mode", ["Single Participant QC", "Comparison Lab", "Run Full QC (1 subject)"], index=2)

    # Participants
    participants = list_participants_from_sets()
    if not participants:
        st.error("No participants found. Check PROJECT path and folder structure.")
        st.stop()

    if nav == "Single Participant QC":
        pid = st.sidebar.selectbox("Participant", participants, index=0)
        st.sidebar.markdown("---")
        st.sidebar.write("Run individual QC steps:")
        step = st.sidebar.selectbox("QC step", [
            "1 Raw", "2 Preprocessed", "3 PreICA Loss", "4 ICLabel", "5 PostICA", "6 Epoched ERP", "7 Final Features"
        ])

        if step == "1 Raw":
            qc_1_raw(pid)
        elif step == "2 Preprocessed":
            qc_2_preprocessed(pid)
        elif step == "3 PreICA Loss":
            qc_3_preICA_loss(pid)
        elif step == "4 ICLabel":
            qc_4_ICLabel(pid)
        elif step == "5 PostICA":
            qc_5_postICA(pid)
        elif step == "6 Epoched ERP":
            qc_6_epoched(pid)
        else:
            qc_7_final_features()

    elif nav == "Run Full QC (1 subject)":
        pid = st.sidebar.selectbox("Participant", participants, index=0)
        st.sidebar.info("Runs all QC steps in order.")
        qc_1_raw(pid)
        qc_2_preprocessed(pid)
        qc_3_preICA_loss(pid)
        qc_4_ICLabel(pid)
        qc_5_postICA(pid)
        qc_6_epoched(pid)
        qc_7_final_features()

    else:
        st.sidebar.subheader("Comparison settings")
        chosen = st.sidebar.multiselect("Participants", participants, default=participants[:4])
        if len(chosen) < 2:
            st.info("Select at least 2 participants.")
            return

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "PSD Overlay", "Bandpower", "PreICA Heatmap", "ICLabel Compare", "P3b Feature Compare"
        ])

        with tab1:
            stage = st.selectbox("Stage", ["01_raw", "02_preprocessed", "06_postICA"], index=1)
            channel = st.selectbox("Channel", ["Pz", "Cz", "Fz", "Oz"], index=0)
            fmin, fmax = st.slider("Frequency range", 0.1, 60.0, (0.5, 45.0), 0.1)
            compare_psd_overlay(chosen, stage, channel, fmin, fmax)

        with tab2:
            stage = st.selectbox("Stage for bandpower", ["01_raw", "02_preprocessed"], index=1)
            channel = st.selectbox("Channel for bandpower", ["Pz", "Cz", "Fz", "Oz"], index=0)
            compare_bandpower_bars(chosen, stage, channel)

        with tab3:
            compare_preica_heatmap(chosen)

        with tab4:
            compare_iclabel(chosen)

        with tab5:
            compare_p3b_features(chosen)


if __name__ == "__main__":
    main()
