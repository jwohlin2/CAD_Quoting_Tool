# tools/diag_geom_dump.py
import csv, json, re, sys, math
from pathlib import Path
from collections import Counter, defaultdict
from typing import Optional, Tuple, List

# ----------- small shared helpers -----------
_NUM = r'(?:\d+(?:\.\d+)?|\d+\s*-\s*\d+\/\d+|\d+\/\d+)'
RE_THK = re.compile(r'\b(?:THK|T(?:H(?:I(?:C(?:K(?:NESS)?)?)?)?)?)\s*[:=]?\s*(?P<t>'+_NUM+r')(?:\s*(?P<u>in(?:ch(?:es)?)?|"))?\b', re.I)

def _to_float(token: str) -> Optional[float]:
    s = token.strip().replace('"','')
    m = re.match(r'^(\d+)\s*-\s*(\d+)\/(\d+)$', s)
    if m:
        return int(m.group(1)) + int(m.group(2))/int(m.group(3))
    m = re.match(r'^(\d+)\/(\d+)$', s)
    if m:
        return int(m.group(1))/int(m.group(2))
    try:
        return float(s)
    except: return None

# ----------- DXF scan -----------
def scan_dxf(dxf_path: str, text_csv: Optional[str]=None, text_jsonl: Optional[str]=None):
    import ezdxf
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    out_dir = Path(dxf_path).parent

    # 1) Entity histogram by layer
    by_layer = defaultdict(Counter)
    for e in msp:
        by_layer[e.dxf.layer or ""].update([e.dxftype()])

    with (out_dir / "layers_entities.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["layer","entity_type","count"])
        for L, cnt in sorted(by_layer.items()):
            for kind, c in sorted(cnt.items()):
                w.writerow([L, kind, c])

    # 2) AABB per layer (with INSERT recursion)
    def aabb_over_entities(ents):
        xmin=ymin=zmin= float("inf")
        xmax=ymax=zmax= -float("inf")
        found=False
        def upd(x,y,z=0.0):
            nonlocal xmin,ymin,zmin,xmax,ymax,zmax,found
            xmin=min(xmin,x); ymin=min(ymin,y); zmin=min(zmin,z)
            xmax=max(xmax,x); ymax=max(ymax,y); zmax=max(zmax,z)
            found=True
        def approx_arc(cx,cy,r,sa,ea,steps=72):
            s=math.radians(sa); e=math.radians(ea); 
            if e<s: e+=2*math.pi
            for i in range(steps+1):
                t=s+(e-s)*i/steps
                yield cx+r*math.cos(t), cy+r*math.sin(t)
        def handle(ent, depth=0):
            kind=ent.dxftype()
            try:
                if kind=="INSERT":
                    if depth>=3: return
                    for ve in ent.virtual_entities():
                        handle(ve, depth+1)
                    return
                if kind=="LINE":
                    sx,sy,sz=map(float,ent.dxf.start); ex,ey,ez=map(float,ent.dxf.end)
                    upd(sx,sy,sz); upd(ex,ey,ez); return
                if kind in ("LWPOLYLINE","POLYLINE"):
                    try:
                        for v in ent.vertices():
                            x,y,z=float(v.dxf.location.x),float(v.dxf.location.y),float(v.dxf.location.z); upd(x,y,z)
                    except:
                        for x,y,*_ in ent.get_points("xy"): upd(float(x),float(y),0.0)
                    return
                if kind=="CIRCLE":
                    cx,cy,cz=map(float,ent.dxf.center); r=float(ent.dxf.radius)
                    upd(cx-r,cy,cz); upd(cx+r,cy,cz); upd(cx,cy-r,cz); upd(cx,cy+r,cz); return
                if kind=="ARC":
                    cx,cy,cz=map(float,ent.dxf.center); r=float(ent.dxf.radius)
                    sa=float(ent.dxf.start_angle); ea=float(ent.dxf.end_angle)
                    for x,y in approx_arc(cx,cy,r,sa,ea): upd(x,y,cz)
                    return
                if kind=="ELLIPSE":
                    try:
                        for p in ent.flattening(deviation=0.01, segments=128): upd(float(p.x),float(p.y),float(p.z))
                    except:
                        pass
                    return
                if kind=="SPLINE":
                    try:
                        for p in ent.approximate(128): upd(float(p.x),float(p.y),float(p.z))
                    except: pass
                    return
                if kind in ("SOLID","TRACE"):
                    for vx,vy,vz in ent.wcs_vertices(): upd(float(vx),float(vy),float(vz))
                    return
            except:
                return
        for e in ents:
            handle(e,0)
        if not found:
            return None,None,None
        dx=xmax-xmin; dy=ymax-ymin; dz=zmax-zmin
        return (dx if dx>1e-6 else None, dy if dy>1e-6 else None, dz if dz>1e-6 else None)

    with (out_dir / "aabb_by_layer.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["layer","dx","dy","dz"])
        for L in sorted(by_layer.keys()):
            ents=[e for e in msp if (e.dxf.layer or "")==L]
            dx,dy,dz=aabb_over_entities(ents)
            w.writerow([L, dx, dy, dz])

    # 3) Dimensions: linear/aligned & ordinate
    lin_rows=[]
    ord_rows=[]
    for dim in msp.query("DIMENSION"):
        try:
            dt=int(dim.dxf.dimtype)
        except: 
            dt=0

        base = dt & 7         # 0=linear(rotated), 1=aligned, 6=ordinate
        layer = dim.dxf.layer or ""
        # measurement
        val=None
        try:
            val=float(dim.get_measurement())
        except:
            txt=(dim.dxf.text or "").strip()
            if txt and txt!="<>":
                m=re.search(r"([-+]?\d+(?:\.\d+)?)", txt)
                if m:
                    try: val=float(m.group(1))
                    except: pass
        if base in (0,1):   # linear/aligned
            angle = getattr(dim.dxf, "angle", None)
            orient=None
            if angle is not None:
                a=abs(float(angle))%180.0
                orient="H" if (a<=45.0 or a>=135.0) else "V"
            else:
                try:
                    p1=dim.dxf.defpoint; p2=getattr(dim.dxf,"defpoint2",None)
                    if p2 is not None:
                        dx=abs(p2[0]-p1[0]); dy=abs(p2[1]-p1[1])
                        orient="H" if dx>=dy else "V"
                except: pass
            p1=tuple(getattr(dim.dxf,"defpoint",(None,None,None)))
            p2=tuple(getattr(dim.dxf,"defpoint2",(None,None,None)))
            lin_rows.append([layer, val, angle, orient, p1, p2])
        elif (dt & 64):     # ordinate
            axis=getattr(dim.dxf,"azin",None)  # 0=X, 1=Y (often)
            ord_rows.append([layer, axis, val])

    with (out_dir / "dims_linear.csv").open("w", newline="", encoding="utf-8") as fh:
        w=csv.writer(fh); w.writerow(["layer","value","angle_deg","orient","defpoint","defpoint2"])
        for r in lin_rows: w.writerow(r)

    with (out_dir / "dims_ordinate.csv").open("w", newline="", encoding="utf-8") as fh:
        w=csv.writer(fh); w.writerow(["layer","axis_hint(0=X,1=Y)","value"])
        for r in ord_rows: w.writerow(r)

    # 4) Optional: thickness hits from text
    thk_rows=[]
    if text_csv or text_jsonl:
        lines=[]
        if text_csv:
            with open(text_csv,"r",encoding="utf-8",newline="") as fh:
                rdr=csv.reader(fh)
                for row in rdr:
                    if len(row)>=4: lines.append(row[3])
        if text_jsonl:
            with open(text_jsonl,"r",encoding="utf-8") as fh:
                for line in fh:
                    line=line.strip()
                    if not line: continue
                    try:
                        obj=json.loads(line)
                        t=obj.get("text") or obj.get("Text") or ""
                        if t: lines.append(str(t))
                    except: pass
        for s in lines:
            m=RE_THK.search(s)
            if m:
                thk=_to_float(m.group("t"))
                if thk is not None:
                    thk_rows.append([s.strip(), thk])
        with (out_dir / "thickness_hits.csv").open("w", newline="", encoding="utf-8") as fh:
            w=csv.writer(fh); w.writerow(["line","thickness_in"])
            for r in thk_rows: w.writerow(r)

    # 5) Small picker summary for quick glance
    summary={
        "layers": sorted(by_layer.keys()),
        "counts": {L: dict(cnt) for L, cnt in by_layer.items()},
        "thickness_hits": len(thk_rows),
    }
    (out_dir / "picker_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[diag] wrote: {out_dir / 'layers_entities.csv'}")
    print(f"[diag] wrote: {out_dir / 'aabb_by_layer.csv'}")
    print(f"[diag] wrote: {out_dir / 'dims_linear.csv'}")
    print(f"[diag] wrote: {out_dir / 'dims_ordinate.csv'}")
    if text_csv or text_jsonl:
        print(f"[diag] wrote: {out_dir / 'thickness_hits.csv'}")
    print(f"[diag] wrote: {out_dir / 'picker_summary.json'}")

def main(argv=None):
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--dxf", required=True)
    ap.add_argument("--csv")
    ap.add_argument("--jsonl")
    args=ap.parse_args(argv)
    scan_dxf(args.dxf, args.csv, args.jsonl)

if __name__=="__main__":
    sys.exit(main(sys.argv[1:]))
