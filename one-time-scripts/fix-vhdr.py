#!/usr/bin/env python3
import os
import re

# Root directory containing all the subject folders
ROOT = "/home/kourosh/Downloads/ds005863"

# Regular expressions to match lines in .vhdr files
datafile_pattern = re.compile(r"^DataFile=.*$", re.IGNORECASE)
markerfile_pattern = re.compile(r"^MarkerFile=.*$", re.IGNORECASE)

# Traverse all subjects (sub-001, sub-002, ...)
for subj_dir in sorted(os.listdir(ROOT)):
    subj_path = os.path.join(ROOT, subj_dir)
    eeg_path = os.path.join(subj_path, "eeg")

    # Only proceed if the directory has an eeg subfolder
    if not os.path.isdir(eeg_path):
        continue

    print(f"Processing {subj_dir}...")

    # Find all .vhdr files inside this subject's eeg folder
    for fname in os.listdir(eeg_path):
        if fname.endswith(".vhdr"):
            vhdr_path = os.path.join(eeg_path, fname)

            # Extract the task and base name, e.g., sub-001_task-flanker_eeg.vhdr
            base_name = fname.replace(".vhdr", "")
            eeg_file = f"{base_name}.eeg"
            vmrk_file = f"{base_name}.vmrk"

            # Read the header file
            with open(vhdr_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            # Replace DataFile and MarkerFile lines
            new_lines = []
            for line in lines:
                if datafile_pattern.match(line.strip()):
                    new_lines.append(f"DataFile={eeg_file}\n")
                elif markerfile_pattern.match(line.strip()):
                    new_lines.append(f"MarkerFile={vmrk_file}\n")
                else:
                    new_lines.append(line)

            # Write the updated header file
            with open(vhdr_path, "w", encoding="utf-8") as f:
                f.writelines(new_lines)

            print(f"  Fixed: {fname}")

print("✅ All .vhdr files processed successfully.")
