import os
import html
import pandas as pd

from features import ROOT

OUTPUT_PATH = os.path.join(ROOT, "outputs", "mexico_gp_2026_tahmin.html")

TEMPLATE = """<!doctype html>
<html lang="tr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mexico GP Olasılıkları</title>
<style>
:root {
  color-scheme: light;
  --surface-0: #f5f5f2;
  --surface-1: #fcfcfb;
  --text-primary: #0b0b0b;
  --text-secondary: #52514e;
  --text-muted: #77766f;
  --rule: #e4e3de;
  --seq-hi: #104281;
  --accent: #2a78d6;
  --range: #9e9d97;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --surface-0: #111110;
    --surface-1: #1a1a19;
    --text-primary: #f0efec;
    --text-secondary: #c3c2b7;
    --text-muted: #8f8e86;
    --rule: #2e2e2b;
    --seq-hi: #6da7ec;
    --accent: #3987e5;
    --range: #6b6a64;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface-0: #111110;
  --surface-1: #1a1a19;
  --text-primary: #f0efec;
  --text-secondary: #c3c2b7;
  --text-muted: #8f8e86;
  --rule: #2e2e2b;
  --seq-hi: #6da7ec;
  --accent: #3987e5;
  --range: #6b6a64;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--surface-0); color: var(--text-primary);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
}
main { max-width: 1400px; margin: 0 auto; padding: 32px 16px 48px; }
main > * { min-width: 0; }
h1, .sub { overflow-wrap: anywhere; }
h1 { font-size: 26px; margin: 0 0 6px; letter-spacing: -0.01em; }
.sub { color: var(--text-secondary); margin: 0 0 24px; max-width: 760px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(160px, 100%), 1fr)); gap: 12px; margin-bottom: 24px; }
.tile { background: var(--surface-1); border: 1px solid var(--rule); border-radius: 10px; padding: 14px 16px; }
.tile .k { color: var(--text-secondary); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }
.tile .v { font-size: 28px; font-weight: 650; font-variant-numeric: tabular-nums; margin-top: 2px; }
.tile .d { color: var(--text-muted); font-size: 13px; }
.card { background: var(--surface-1); border: 1px solid var(--rule); border-radius: 12px; padding: 16px; }
.card h2 { font-size: 16px; margin: 0 0 4px; }
.card .note { color: var(--text-secondary); font-size: 13px; margin: 0 0 12px; }
.legend { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 20px; color: var(--text-secondary); font-size: 12px; margin-bottom: 12px; }
.ramp { width: 140px; height: 10px; border-radius: 3px;
  background: linear-gradient(90deg, var(--surface-1), color-mix(in oklab, var(--seq-hi) 50%, var(--surface-1)), var(--seq-hi));
  border: 1px solid var(--rule); }
.scroll { overflow-x: auto; -webkit-overflow-scrolling: touch; }
table { border-collapse: separate; border-spacing: 2px; font-variant-numeric: tabular-nums; }
th { font-weight: 600; color: var(--text-secondary); font-size: 12px; padding: 4px 6px; white-space: nowrap; }
th.pos { width: 28px; text-align: center; padding: 4px 0; }
td { padding: 0 6px; height: 30px; white-space: nowrap; }
td.rank { color: var(--text-muted); text-align: right; }
td.drv { font-weight: 650; }
td.team { color: var(--text-secondary); font-size: 13px; max-width: 140px; overflow: hidden; text-overflow: ellipsis; }
td.grid, td.exp, td.pct { text-align: right; }
td.exp { font-weight: 600; }
td.pct { color: var(--text-secondary); }
td.cell { width: 28px; min-width: 28px; padding: 0; text-align: center; font-size: 11px; border-radius: 4px; cursor: default; }
td.cell:hover, td.cell:focus { outline: 2px solid var(--text-primary); outline-offset: -1px; }
td.cell.hot { color: #ffffff; }
:root[data-theme="dark"] td.cell.hot { color: #0b0b0b; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) td.cell.hot { color: #0b0b0b; } }
td.cell.diag { box-shadow: inset 0 -2px 0 var(--text-muted); }
td.range { padding: 0 8px; }
svg .rng { stroke: var(--range); stroke-width: 2; stroke-linecap: round; }
svg .dot { fill: var(--accent); stroke: var(--surface-1); stroke-width: 2; }
svg .gridmark { fill: none; stroke: var(--text-secondary); stroke-width: 1.5; }
svg .axis { stroke: var(--rule); stroke-width: 1; }
.key-dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; background: var(--accent); vertical-align: -1px; }
.key-rng { display: inline-block; width: 22px; height: 2px; background: var(--range); vertical-align: 3px; }
.key-grid { display: inline-block; width: 8px; height: 8px; border: 1.5px solid var(--text-secondary); transform: rotate(45deg); vertical-align: 0; }
.key-diag { display: inline-block; width: 14px; height: 10px; box-shadow: inset 0 -2px 0 var(--text-muted); border: 1px solid var(--rule); vertical-align: -1px; }
#tip { position: fixed; pointer-events: none; z-index: 10; background: var(--surface-1); color: var(--text-primary);
  border: 1px solid var(--rule); border-radius: 8px; padding: 8px 10px; font-size: 13px; box-shadow: 0 4px 16px rgba(0,0,0,.18);
  opacity: 0; transition: opacity .08s; max-width: 240px; }
#tip b { font-weight: 650; }
#tip .m { color: var(--text-secondary); }
.method { color: var(--text-secondary); font-size: 13px; margin-top: 24px; max-width: 820px; }
.method h3 { color: var(--text-primary); font-size: 14px; margin: 0 0 6px; }
.method p { margin: 0 0 8px; }
</style>
</head>
<body>
<main>
  <h1>Mexico City GP 2026: Pozisyon Olasılıkları</h1>
  <p class="sub">Her pilotun her pozisyonda bitirme ihtimali, __NSIMS__ yarış simülasyonundan. Grid kaynağı: <b>__GRIDSRC__</b>.</p>

  <div class="tiles">__TILES__</div>

  <section class="card">
    <h2>Pilot × bitiş pozisyonu</h2>
    <p class="note">Satırlar ortalama (beklenen) pozisyona göre sıralı. Koyu hücre = yüksek ihtimal. Hücrenin üzerine gelince detay görünür.</p>
    <div class="legend">
      <span>0%</span><span class="ramp" aria-hidden="true"></span><span>__MAXP__</span>
      <span><span class="key-dot"></span> ortalama pozisyon</span>
      <span><span class="key-rng"></span> %10–%90 aralığı</span>
      <span><span class="key-grid"></span> grid</span>
      <span><span class="key-diag"></span> grid pozisyonundaki hücre</span>
    </div>
    <div class="scroll">
      <table>
        <thead><tr>
          <th></th><th style="text-align:left">Pilot</th><th style="text-align:left">Takım</th>
          <th>Grid</th><th>Ort. poz.</th><th style="text-align:left">Dağılım (1 → __N__)</th>
          <th>Galibiyet</th><th>Podyum</th><th>Puan</th>
          __POSHEAD__
        </tr></thead>
        <tbody>__ROWS__</tbody>
      </table>
    </div>
  </section>

  <section class="method">
    <h3>Nasıl hesaplandı?</h3>
    <p>LightGBM sıralama modeli (2022–2026 yarışları) her pilota bir skor veriyor. Özellikler: grid, son 3/8 yarış formu, sıralama temposu, takım formu, DNF oranı ve bu pistteki geçmiş. Skor, doğrulama setinde seçilen ağırlıkla grid pozisyonuyla harmanlanıyor.</p>
    <p>Skorlar Plackett-Luce modeliyle olasılığa çevriliyor: pozisyonlar 1.'den başlayarak sırayla dolduruluyor, ön sıralarda belirsizlik düşük, arkaya doğru yüksek. Belirsizlik miktarı 2025–2026'da modelin görmediği yarışlara göre kalibre edildi, sonra __NSIMS__ yarış simüle edildi.</p>
    <p><b>Dürüst not:</b> 2026 testinde model, "sonuç = grid sırası" tahminiyle aynı oranda kazananı bildi ve sıralamayı biraz daha iyi tuttu, ama olasılık kalitesinde saf gridin biraz gerisinde kaldı. Bu olasılıkları kesin tahmin değil, makul bir belirsizlik aralığı olarak okuyun. Arka sıralar için puan ihtimalleri muhtemelen biraz iyimser.</p>
  </section>
</main>
<div id="tip" role="tooltip"></div>
<script>
(function () {
  var tip = document.getElementById("tip");
  function show(e) {
    var c = e.target.closest("td.cell");
    if (!c) { tip.style.opacity = 0; return; }
    tip.innerHTML = "<b>" + c.dataset.d + "</b> <span class='m'>" + c.dataset.t + "</span><br>" +
      "P" + c.dataset.p + " bitirme ihtimali: <b>" + c.dataset.v + "</b><br>" +
      "<span class='m'>Grid: " + c.dataset.g + " · Ort. poz.: " + c.dataset.e + "</span>";
    tip.style.opacity = 1;
    var r = c.getBoundingClientRect(), w = tip.offsetWidth, h = tip.offsetHeight;
    var x = Math.min(window.innerWidth - w - 8, Math.max(8, r.left + r.width / 2 - w / 2));
    var y = r.top - h - 8; if (y < 8) y = r.bottom + 8;
    tip.style.left = x + "px"; tip.style.top = y + "px";
  }
  document.addEventListener("mouseover", show);
  document.addEventListener("focusin", show);
  document.addEventListener("scroll", function () { tip.style.opacity = 0; }, true);
})();
</script>
</body>
</html>
"""


def _pct(p):
    return f"{p * 100:.1f}%" if p >= 0.001 else "<0.1%"


def _range_svg(row, n, width=150, height=22):
    def x(pos):
        return 6 + (pos - 1) / (n - 1) * (width - 12)
    mid = height / 2
    gx = x(row["grid_position"])
    return (
        f'<svg width="{width}" height="{height}" role="img" '
        f'aria-label="Ortalama {row["expected_position"]:.1f}, aralık {row["p10_position"]:.0f}-{row["p90_position"]:.0f}">'
        f'<line class="axis" x1="6" x2="{width - 6}" y1="{mid}" y2="{mid}"/>'
        f'<line class="rng" x1="{x(row["p10_position"]):.1f}" x2="{x(row["p90_position"]):.1f}" y1="{mid}" y2="{mid}"/>'
        f'<rect class="gridmark" x="{gx - 3.5:.1f}" y="{mid - 3.5:.1f}" width="7" height="7" '
        f'transform="rotate(45 {gx:.1f} {mid:.1f})"/>'
        f'<circle class="dot" cx="{x(row["expected_position"]):.1f}" cy="{mid}" r="5"/>'
        f'</svg>'
    )


def build_html(df_race, grid_source, output_path=OUTPUT_PATH, n_sims=20000):
    df = df_race.sort_values("expected_position").reset_index(drop=True)
    n = len(df)
    pos_cols = [f"p_pos_{p}" for p in range(1, n + 1)]
    max_p = float(df[pos_cols].to_numpy().max())
    # Renk ölçeği %40'ta doyuyor: favorinin %80'i orta grubun farklarını silmesin
    color_cap = min(max_p, 0.40)

    rows = []
    for i, r in df.iterrows():
        drv, team = html.escape(str(r["driver_id"])), html.escape(str(r["team"]))
        grid = int(r["grid_position"])
        cells = []
        for p in range(1, n + 1):
            v = float(r[f"p_pos_{p}"])
            # Tek renk (sıralı) ölçek: yüzey -> koyu mavi. Karekök, küçük farkları görünür kılar
            mix = (min(v, color_cap) / color_cap) ** 0.6 * 100 if color_cap > 0 else 0
            cls = "cell" + (" hot" if mix > 58 else "") + (" diag" if p == grid else "")
            label = f"{v * 100:.0f}" if v >= 0.10 else ""  # seçici etiket: sadece ≥%10
            cells.append(
                f'<td class="{cls}" tabindex="0" style="background:color-mix(in oklab, var(--seq-hi) {mix:.0f}%, var(--surface-1))" '
                f'data-d="{drv}" data-t="{team}" data-p="{p}" data-v="{_pct(v)}" data-g="{grid}" '
                f'data-e="{r["expected_position"]:.1f}">{label}</td>'
            )
        rows.append(
            f'<tr><td class="rank">{i + 1}</td><td class="drv">{drv}</td><td class="team" title="{team}">{team}</td>'
            f'<td class="grid">{grid}</td><td class="exp">{r["expected_position"]:.1f}</td>'
            f'<td class="range">{_range_svg(r, n)}</td>'
            f'<td class="pct">{_pct(r["p_win"])}</td><td class="pct">{_pct(r["p_podium"])}</td>'
            f'<td class="pct">{_pct(r["p_points"])}</td>{"".join(cells)}</tr>'
        )

    fav = df.sort_values("p_win", ascending=False).head(3)
    tiles = "".join(
        f'<div class="tile"><div class="k">Galibiyet ihtimali #{k + 1}</div>'
        f'<div class="v">{_pct(r["p_win"])}</div>'
        f'<div class="d">{html.escape(str(r["driver_id"]))} · {html.escape(str(r["team"]))} · Grid {int(r["grid_position"])}</div></div>'
        for k, (_, r) in enumerate(fav.iterrows())
    )
    pod = df.sort_values("p_podium", ascending=False).iloc[0]
    tiles += (f'<div class="tile"><div class="k">En yüksek podyum ihtimali</div>'
              f'<div class="v">{_pct(pod["p_podium"])}</div>'
              f'<div class="d">{html.escape(str(pod["driver_id"]))} · {html.escape(str(pod["team"]))}</div></div>')

    page = (TEMPLATE
            .replace("__TILES__", tiles)
            .replace("__ROWS__", "\n".join(rows))
            .replace("__POSHEAD__", "".join(f'<th class="pos">{p}</th>' for p in range(1, n + 1)))
            .replace("__N__", str(n))
            .replace("__MAXP__", ("≥" if max_p > color_cap else "") + _pct(color_cap))
            .replace("__NSIMS__", f"{n_sims:,}".replace(",", "."))
            .replace("__GRIDSRC__", html.escape(os.path.basename(grid_source) if os.sep in grid_source else grid_source)))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"Görselleştirme kaydedildi: {output_path}")
    return output_path


if __name__ == "__main__":
    # Son tahmin CSV'sinden yeniden üret
    csv_path = os.path.join(ROOT, "outputs", "mexico_gp_2026_probabilities.csv")
    build_html(pd.read_csv(csv_path), "son tahmin dosyası")
