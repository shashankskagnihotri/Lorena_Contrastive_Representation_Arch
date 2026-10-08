import numpy as np
import pytest
from sparse_contrast.metrics import ClassificationCounts,aggregate_cells

def test_sample_weighting_and_coverage():
    c=ClassificationCounts(); c.add([0,1],[0,0],2.); c.add([1],[1],4.)
    r=c.result(); assert r['correct']==2 and r['total']==3 and r['loss']==2
    rows=[dict(corruption='a',severity=s,error=v,total=10000) for s,v in [(1,.1),(2,.3)]]
    assert aggregate_cells(rows,[('a',1),('a',2)])['mCE_raw']==pytest.approx(.2)
    with pytest.raises(ValueError): aggregate_cells(rows[:1],[('a',1),('a',2)])
    with pytest.raises(ValueError): aggregate_cells(rows+rows,[('a',1),('a',2)])
