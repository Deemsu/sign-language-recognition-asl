"""
run_project.py — Project Launcher Menu
Run this file to access all parts of the project.
"""

import os
import subprocess
import sys

SCRIPTS = {
    "1": ("1_Data_Collector.py",      "Collect training data (webcam)"),
    "2": ("1b_Kaggle_Extractor.py",   "Extract landmarks from Kaggle images"),
    "3": ("2_Augment_Data.py",        "Augment data (noise, rotation, flip...)"),
    "4": ("3_Train_v2.py",            "Train the model"),
    "5": ("4_Real_Time_v2.py",        "Real-time recognition + Word Builder"),
    "6": ("webcam_wordbuilder_v2.py", "Dedicated Word Builder"),
}

def check_model():
    exists = os.path.exists("sign_model.pkl")
    status = "Found" if exists else "Not found — run training first"
    return exists, status

def check_data():
    data_dir = "sign_data"
    if not os.path.exists(data_dir):
        return False, "sign_data folder missing"
    csvs = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    if not csvs:
        return False, "No CSV files found"
    return True, f"{len(csvs)} CSV file(s) found"

while True:
    os.system('cls' if os.name == 'nt' else 'clear')
    print("╔══════════════════════════════════════════════╗")
    print("║      Sign Language Recognition Project       ║")
    print("╠══════════════════════════════════════════════╣")

    _, model_status = check_model()
    _, data_status  = check_data()
    print(f"║  Model : {model_status:<36}║")
    print(f"║  Data  : {data_status:<36}║")
    print("╠══════════════════════════════════════════════╣")
    print("║  --- Data Collection ---                     ║")
    print("║  [1] ✎ Collect training data (webcam)        ║")
    print("║  [2] ✎ Extract landmarks from Kaggle images  ║")
    print("║      (run 1 and/or 2, then continue)         ║")
    print("╠══════════════════════════════════════════════╣")
    print("║  --- Training Pipeline ---                   ║")

    for key, (script, desc) in list(SCRIPTS.items())[2:]:
        exists = "✓" if os.path.exists(script) else "✗"
        print(f"║  [{key}] {exists} {desc:<38}║")

    print("║  [0] Exit                                    ║")
    print("╚══════════════════════════════════════════════╝")

    choice = input("\nChoice: ").strip()

    if choice == "0":
        print("Goodbye!")
        break
    elif choice in SCRIPTS:
        script, desc = SCRIPTS[choice]
        if not os.path.exists(script):
            print(f"[ERROR] File '{script}' not found!")
            input("Press Enter to continue...")
            continue
        print(f"\n  Running: {desc}")
        subprocess.run([sys.executable, script])
        input("\nPress Enter to return to menu...")
    else:
        print("Invalid choice.")
        input("Press Enter to continue...")