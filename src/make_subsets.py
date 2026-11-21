#!/usr/bin/env python3
"""Derived UKB female target file used in the supplementary analyses: participants aged >=70 years
(UKfemale_for_DA_aligned_fu2_age70.csv). Run after make_fu2.py. The femur-BMD subset
(UK*_for_DA_aligned_fu2_hipBMD.csv) replaces HA_THD with the mean of left and right total-femur BMD at the imaging
visit (UKB fields 23212 and 23291) and keeps participants with a measured value; it requires those fields in the extract."""
import os, pandas as pd
D = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
d = pd.read_csv(os.path.join(D, "UKfemale_for_DA_aligned_fu2.csv"))
s = d[d.HA_AGE >= 70]; s.to_csv(os.path.join(D, "UKfemale_for_DA_aligned_fu2_age70.csv"), index=False)
print("age >= 70:", len(s), "participants,", int(s.HA_HIPFX.sum()), "hip fractures")
