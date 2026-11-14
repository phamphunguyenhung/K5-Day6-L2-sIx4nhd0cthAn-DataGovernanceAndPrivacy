"""T14 task F helper: draw a box with the mouse, get the `override_value` string for an `edit` decision.

    python -m tools.t14_box_picker            # -> out/poc/box_picker.html (offline, open in a browser)

The reviewer drags a rectangle over the enlarged image; the page converts it back to ORIGINAL image pixels and
prints `class,x,y,w,h` (copy button) to paste into the review web's "Sửa…" field. Red = model box under review,
gray = other model boxes in the image. The tool only measures what the reviewer draws; it never suggests a box.
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

PAGE = """<!doctype html><html lang="vi"><head><meta charset="utf-8"><title>T14 box picker</title>
<style>
body{font:14px system-ui,sans-serif;margin:16px;background:#fafafa;color:#222}
.card{background:#fff;border:1px solid #ddd;border-radius:8px;padding:12px;margin:0 0 16px}
canvas{cursor:crosshair;border:1px solid #999;image-rendering:pixelated;max-width:100%}
code{background:#eef;padding:4px 6px;border-radius:4px;font-size:15px}
.meta{color:#555;margin:4px 0 8px}
</style></head><body>
<h1>T14 box picker</h1>
<p>Kéo chuột trên ảnh để vẽ khung đúng (vẽ lại nếu chưa ưng). Đỏ = box đang review, xám = box khác.
Chuỗi bên dưới là toạ độ theo pixel ảnh gốc: bấm <b>Copy</b> rồi dán vào ô "Sửa…" trên web.
Chỉ đổi class mà box đã đúng thì chọn class rồi bấm <b>Chỉ class</b>.</p>
<label><input type="checkbox" id="others" checked> hiện box xám</label>
<div id="list"></div>
<script>
const DATA = __DATA__, CLASSES = __CLASSES__, S = 3;
const showOthers = document.getElementById("others");
const cards = [];
function rect(ctx, b, color, width) { ctx.strokeStyle = color; ctx.lineWidth = width;
  ctx.strokeRect(b.x * S, b.y * S, b.w * S, b.h * S); }
for (const it of DATA) {
  const card = document.createElement("div"); card.className = "card";
  const t = it.target;
  card.innerHTML = `<b>#${it.rank} ${it.item_id}</b>
    <div class="meta">model: ${t.class} · conf ${t.conf} · box ${t.x},${t.y},${t.w},${t.h} · ${it.reason}</div>`;
  const cv = document.createElement("canvas"); cv.width = it.width * S; cv.height = it.height * S;
  const sel = document.createElement("select");
  for (const c of CLASSES) sel.add(new Option(c, c, c === t.class, c === t.class));
  const out = document.createElement("code"); out.textContent = "(chưa vẽ)";
  const copy = document.createElement("button"); copy.textContent = "Copy";
  const only = document.createElement("button"); only.textContent = "Chỉ class";
  const row = document.createElement("p"); row.append("class: ", sel, " ", out, " ", copy, " ", only);
  card.append(cv, row); document.getElementById("list").append(card);
  const img = new Image(); img.src = it.src;
  let drawn = null, start = null;
  const ctx = cv.getContext("2d"); ctx.imageSmoothingEnabled = false;
  function paint() {
    ctx.drawImage(img, 0, 0, cv.width, cv.height);
    if (showOthers.checked) for (const o of it.others) rect(ctx, o, "#bbb", 1);
    rect(ctx, t, "red", 2);
    if (drawn) rect(ctx, drawn, "#00c853", 2);
  }
  function show() {
    if (!drawn) return;
    const r = (v) => Math.round(v * 10) / 10;
    out.textContent = `${sel.value},${r(drawn.x)},${r(drawn.y)},${r(drawn.w)},${r(drawn.h)}`;
  }
  const pos = (e) => { const b = cv.getBoundingClientRect();
    return { x: Math.min(it.width, Math.max(0, (e.clientX - b.left) / b.width * it.width)),
             y: Math.min(it.height, Math.max(0, (e.clientY - b.top) / b.height * it.height)) }; };
  cv.onpointerdown = (e) => { start = pos(e); cv.setPointerCapture(e.pointerId); };
  cv.onpointermove = (e) => { if (!start) return; const p = pos(e);
    drawn = { x: Math.min(start.x, p.x), y: Math.min(start.y, p.y), w: Math.abs(p.x - start.x), h: Math.abs(p.y - start.y) };
    paint(); show(); };
  cv.onpointerup = () => { start = null; if (drawn && (drawn.w < 1 || drawn.h < 1)) { drawn = null; out.textContent = "(khung quá nhỏ, vẽ lại)"; paint(); } };
  sel.onchange = show;
  copy.onclick = () => navigator.clipboard.writeText(out.textContent);
  only.onclick = () => { drawn = null; paint(); out.textContent = sel.value; };
  img.onload = paint; cards.push(paint);
}
showOthers.onchange = () => cards.forEach((p) => p());
</script></body></html>"""


def build(queue_json: Path, out_html: Path) -> Path:
    q = json.loads(queue_json.read_text(encoding="utf-8"))
    root = queue_json.parent
    items = []
    for it in q["items"]:
        img = (root / it["image"]).read_bytes()
        items.append({k: it[k] for k in ("rank", "item_id", "reason", "width", "height", "target", "others")}
                     | {"src": "data:image/jpeg;base64," + base64.b64encode(img).decode()})
    html = PAGE.replace("__DATA__", json.dumps(items)).replace("__CLASSES__", json.dumps(q["classes"]))
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
    return out_html


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default="web/public/queue.json")
    ap.add_argument("--out", default="out/poc/box_picker.html")
    a = ap.parse_args(argv)
    print(f"wrote {build(Path(a.queue), Path(a.out))}")


if __name__ == "__main__":
    main()
