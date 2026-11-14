"""Contact sheet tinh (1 file index.html, co zoom anh) cho N box co score nghi ngo THAP nhat, moi anh 1 box.

    python -m tools.t14_lowest_sheet [--run out/poc] [--input fixtures/t14_auto_annotation] [--out out/site] [-n 30]

Khong doc ground truth. Deploy: cd <out> roi `wrangler pages deploy . --project-name t14-review`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.common.contact_sheet import render_contact_sheet

ZOOM = """
<style>
.imgs img{cursor:zoom-in} #lb{position:fixed;inset:0;z-index:9;background:#fff;display:none;flex-direction:column}
#lb.on{display:flex} #lbv{flex:1;overflow:auto;background:#eee;cursor:grab} #lbv img{display:block;max-width:none;image-rendering:auto}
#lbb{display:flex;gap:8px;align-items:center;padding:8px 16px;border-bottom:1px solid #ddd}
</style>
<div id="lb"><div id="lbb"><b id="lbz"></b><button id="zo">&minus;</button><button id="zi">+</button>
<button id="zf">V&#7915;a khung</button><button id="zc">&#272;&oacute;ng (Esc)</button></div><div id="lbv"><img id="lbi"></div></div>
<script>
(function(){const lb=document.getElementById('lb'),v=document.getElementById('lbv'),im=document.getElementById('lbi');let z=1,fw=0;
function set(n){z=Math.min(8,Math.max(.25,n));im.style.width=Math.round(fw*z)+'px';document.getElementById('lbz').textContent=Math.round(z*100)+'%';}
document.addEventListener('click',e=>{const t=e.target;if(t.tagName==='IMG'&&t.closest('.imgs')){im.src=t.src;lb.classList.add('on');
document.body.style.overflow='hidden';fw=v.clientWidth;set(1);v.scrollTo(0,0);}});
const close=()=>{lb.classList.remove('on');document.body.style.overflow='';};
document.getElementById('zc').onclick=close;document.getElementById('zi').onclick=()=>set(z*1.5);
document.getElementById('zo').onclick=()=>set(z/1.5);document.getElementById('zf').onclick=()=>set(1);
document.addEventListener('keydown',e=>{if(!lb.classList.contains('on'))return;if(e.key==='Escape')close();else if(e.key==='+'||e.key==='=')set(z*1.5);else if(e.key==='-')set(z/1.5);});
v.addEventListener('dblclick',()=>set(z<2?3:1));let p=null;
v.addEventListener('pointerdown',e=>{p={x:e.clientX,y:e.clientY,l:v.scrollLeft,t:v.scrollTop};v.setPointerCapture(e.pointerId);});
v.addEventListener('pointermove',e=>{if(p){v.scrollLeft=p.l-(e.clientX-p.x);v.scrollTop=p.t-(e.clientY-p.y);}});
v.addEventListener('pointerup',()=>{p=null;});})();
</script>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="out/poc")
    ap.add_argument("--input", default="fixtures/t14_auto_annotation")
    ap.add_argument("--out", default="out/site")
    ap.add_argument("-n", type=int, default=30)
    ap.add_argument("--lowest", action="store_true", help="dung N box score thap nhat thay vi review_queue.csv")
    a = ap.parse_args()
    b = pd.read_csv(Path(a.run) / "boxes_all.csv")
    if a.lowest:
        low = b[b["kept"]].sort_values(["score", "box_id"], kind="stable").drop_duplicates("image_id").head(a.n)
        q = pd.DataFrame({"rank": range(1, len(low) + 1), "item_id": low["box_id"].values, "score": low["score"].round(3).values,
                          "reason": low["reason"].values, "suggested_action": "keep (check)", "slice": low["class"].values,
                          "evidence_path": low["path"].values})
        selected = low
        title = f"T14 review: {len(q)} box score thap nhat (do = box can duyet, xam = box khac)"
    else:
        q = pd.read_csv(Path(a.run) / "review_queue.csv", dtype=str).fillna("").head(a.n)
        b_idx = b.set_index("box_id", drop=False)
        selected = b_idx.loc[q["item_id"].tolist()]
        title = f"T14 review: {len(q)} box nghi ngo cao nhat (do = box can duyet, xam = box khac)"
    boxes = {}
    for r in selected.itertuples():
        others = b[(b.image_id == r.image_id) & (b.box_id != r.box_id)]
        boxes[r.box_id] = [{"x": o.x, "y": o.y, "w": o.w, "h": o.h, "label": o._8, "color": "gray"} for o in others.itertuples()] + \
                          [{"x": r.x, "y": r.y, "w": r.w, "h": r.h, "label": f"{r._8} {r.conf:.2f}", "color": "red"}]
    out = render_contact_sheet(q, Path(a.out) / "index.html", image_root=a.input, boxes_by_item=boxes, thumb_width=640, title=title)
    out.write_text(out.read_text(encoding="utf-8").replace("</body>", ZOOM + "</body>"), encoding="utf-8")
    print(f"{len(q)} items -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
