import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def occ() -> pd.DataFrame:
    """Small synthetic occupation table with the columns the engine needs."""
    rng = np.random.default_rng(0)
    majors = ["11", "13", "15", "29", "35", "41", "43", "47", "51", "53"]
    rows = []
    for i in range(200):
        mg = majors[i % len(majors)]
        e = rng.random()
        rows.append({"soc": f"{mg}-{1000 + i:04d}", "title": f"Occupation {i}", "emp_2022": int(rng.integers(1e4, 5e5)),
                     "exposure_eloundou": e, "pctl_eloundou": e, "automation_share": rng.uniform(0.2, 0.7)})
    return pd.DataFrame(rows)
