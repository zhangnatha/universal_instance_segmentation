#!/usr/bin/env python3
"""使用 compare_json 的精确掩码 IoU 匹配器遍历得分阈值。"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

from instance_segmentation.evaluation.compare_json import (  # noqa: E402
    collect_json_files,
    load_ground_truth,
    load_predictions,
    match_class_instances,
)

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--ground-truth-dir', required=True); p.add_argument('--prediction-dir', required=True)
    p.add_argument('--output', required=True); p.add_argument('--classes', nargs='+', required=True)
    p.add_argument('--iou-threshold', type=float, default=.5)
    p.add_argument('--thresholds', nargs='*', type=float, default=[i/20 for i in range(1,20)])
    a=p.parse_args(argv); gt=collect_json_files(Path(a.ground_truth_dir),False); pr=collect_json_files(Path(a.prediction_dir),True)
    stats=defaultdict(lambda:[0,0,0])
    for n,key in enumerate(sorted(gt),1):
        _, gis, (h,w)=load_ground_truth(gt[key],a.classes)
        pis=load_predictions(pr[key],a.classes,h,w) if key in pr else []
        for c in a.classes:
            cg=[x for x in gis if x['class_name']==c]; cp=[x for x in pis if x['class_name']==c]
            for t in a.thresholds:
                chosen=[x for x in cp if x['score']>=t]
                m,ug,up=match_class_instances(cg,chosen,a.iou_threshold)
                s=stats[(c,t)]; s[0]+=len(m); s[1]+=len(up); s[2]+=len(ug)
        if n%250==0: print(f'{n}/{len(gt)}',flush=True)
    rows=[]
    for c in a.classes:
        for t in a.thresholds:
            tp,fp,fn=stats[(c,t)]; precision=tp/(tp+fp) if tp+fp else 0; recall=tp/(tp+fn) if tp+fn else 0
            f1=2*precision*recall/(precision+recall) if precision+recall else 0
            rows.append(dict(class_name=c,threshold=t,tp=tp,false_positive=fp,missed=fn,precision=precision,recall=recall,f1=f1))
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    for c in a.classes:
        best=max((r for r in rows if r['class_name']==c),key=lambda r:r['f1']); print(c,best)

if __name__=='__main__': main()
