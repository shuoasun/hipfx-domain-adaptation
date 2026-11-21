"""Build follow-up-restricted UKB target files (design A): keep participants whose Instance-2 visit
is >=2 years before the end of inpatient records (2023-03-31). Row-aligned with the *_aligned.csv files.
Filter depends only on visit date, not on outcome or features."""
import os, pandas as pd
os.chdir(os.environ.get("DATA_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")))
CUT = pd.Timestamp("2023-03-31"); MIN_YEARS = 2.0
for s in ["female", "male"]:
    h = pd.read_csv(f"UK_{s}_incident10y_HAready_FINAL.csv", parse_dates=["index_date"])
    a = pd.read_csv(f"UK{s}_for_DA_aligned.csv")
    assert len(h) == len(a) and (h.HA_HIPFX.values == a.HA_HIPFX.values).all()
    keep = ((CUT - h.index_date).dt.days / 365.25 >= MIN_YEARS).values
    out = a[keep].reset_index(drop=True)
    out.to_csv(f"UK{s}_for_DA_aligned_fu2.csv", index=False)
    print(s, len(a), "->", len(out), "cases", int(out.HA_HIPFX.sum()),
          "BMD miss", out[["HA_FND","HA_LSD","HA_THD"]].isna().mean().round(3).to_dict())
