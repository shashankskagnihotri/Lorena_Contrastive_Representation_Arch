from __future__ import annotations
import numpy as np

class ClassificationCounts:
    def __init__(self): self.total=0; self.correct=0; self.loss_sum=0.; self.confusion=np.zeros((10,10),dtype=np.int64)
    def add(self, labels, predictions, loss_sum):
        y=np.asarray(labels,dtype=np.int64); p=np.asarray(predictions,dtype=np.int64)
        if y.shape!=p.shape or y.ndim!=1 or np.any((y<0)|(y>9)|(p<0)|(p>9)): raise ValueError('Invalid predictions/labels')
        if not np.isfinite(loss_sum): raise ValueError('Nonfinite loss')
        self.total+=len(y); self.correct+=int((y==p).sum()); self.loss_sum+=float(loss_sum)
        self.confusion+=np.bincount(y*10+p,minlength=100).reshape(10,10)
    def result(self):
        if not self.total: raise ValueError('No samples')
        return dict(correct=self.correct,total=self.total,loss=self.loss_sum/self.total,accuracy=self.correct/self.total,error=1-self.correct/self.total,confusion=self.confusion.tolist(),per_class_accuracy=np.divide(np.diag(self.confusion),self.confusion.sum(1),out=np.zeros(10,dtype=float),where=self.confusion.sum(1)>0).tolist())

def aggregate_cells(rows,expected_cells):
    keys=[(r['corruption'],str(r['severity'])) for r in rows]
    expected={(c,str(s)) for c,s in expected_cells}
    if len(keys)!=len(set(keys)) or set(keys)!=expected: raise ValueError('Missing/duplicate/unexpected corruption cells')
    for r in rows:
        if r['total']!=10000: raise ValueError('Incomplete cell')
    per={c:float(np.mean([r['error'] for r in rows if r['corruption']==c])) for c in sorted({r['corruption'] for r in rows})}
    return dict(per_corruption_error=per,mCE_raw=float(np.mean(list(per.values()))),expected_cells=len(expected),completed_cells=len(rows),total=sum(r['total'] for r in rows))
