#!/usr/bin/env bash

# List of target objects
targets=("LZK12"
  "IRAS03301+3111"
  "NAME_IRAS_04108+2803B"
  "[BHS98]_MHO_2"
  "FP_Tau"
  "NAME_IRAS_04181+2654B"
  "IRAS_04181+2654"
  "T_Tau_N"
  "DG_Tau"
  "GV_Tau"
  "HK_Tau"
  "Haro6-13"
  "IRAS04295+2251"
  "Haro6-28"
  "Haro6-33"
  "UY_Aur_A"
  "IRAS04489+3042"
  "V347_Aur"
  "IRAS04591-0856"
  "[MB91]_54"
  "2MASS_J05574918-1406080"
  "DoAr_25"
  "EM*_SR24_S"
  "[GY92]_235"
  "WLY2-42"
  "Elia2-32"
  "Elia2-33"
  "YLW58"
  "DoAr43"
  "WLY_2-63"
  "[EC92]_92"
  "[TS84]_IRS5"
  "IRAS_19247+2238"
)


for obj in "${targets[@]}"; do
  echo "Processing $obj"

  # Name used by queryDB.py internally
  fs_obj="${obj//_/}"

  phot_file="$fs_obj/${fs_obj}_phot.dat"

  # Skip if already exists
  if [[ -f "$phot_file" ]]; then
    echo "Skipping $obj (already exists)"
    continue
  fi

  # Run query
  python3 queryDB.py --obj="$obj" --rad=3s --closest CLOSEST

  # Wait until file appears (safety)
  while [[ ! -f "$phot_file" ]]; do
    sleep 1
  done

  # Plot
  python3 inspectSED.py --phot="$phot_file" --savePlt=True

done
