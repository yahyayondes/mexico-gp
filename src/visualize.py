import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.lines import Line2D

from features import ROOT

OUTPUT_PATH = os.path.join(ROOT, "outputs", "mexico_gp_2026_tahmin.png")

# Tek renkli (sıralı) mavi skala: düşük ihtimal açık, yüksek ihtimal koyu
BLUES = LinearSegmentedColormap.from_list(
    "seq_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#104281"]
)
ACCENT = "#2a78d6"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
MUTED = "#9e9d97"
RULE = "#e4e3de"
SURFACE = "#fcfcfb"


def _pct(p):
    return f"{p * 100:.1f}%" if p >= 0.001 else "<0.1%"


def plot_probabilities(df_race, grid_source, output_path=OUTPUT_PATH, n_sims=20000, show=False):
    df = df_race.sort_values("expected_position").reset_index(drop=True)
    n = len(df)
    pos_cols = [f"p_pos_{p}" for p in range(1, n + 1)]
    prob = df[pos_cols].to_numpy()
    y = np.arange(n)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": TEXT,
                         "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2, "ytick.color": TEXT})

    fig = plt.figure(figsize=(20, 11), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 3, width_ratios=[3.6, 3.4, 9.5], wspace=0.04,
                          left=0.12, right=0.975, top=0.83, bottom=0.14)
    ax_rng, ax_tbl, ax_hm = (fig.add_subplot(gs[i]) for i in range(3))

    # 1) Ortalama pozisyon + %10-%90 aralığı + grid
    ax_rng.set_facecolor(SURFACE)
    ax_rng.hlines(y, df["p10_position"], df["p90_position"], color=MUTED, lw=2, capstyle="round", zorder=1)
    ax_rng.scatter(df["grid_position"], y, marker="D", s=34, facecolor="none", edgecolor=TEXT_2, lw=1.3, zorder=2)
    ax_rng.scatter(df["expected_position"], y, s=70, color=ACCENT, edgecolor=SURFACE, lw=2, zorder=3)
    ax_rng.set_xlim(0.3, n + 0.7)
    ax_rng.set_ylim(n - 0.5, -0.5)
    ax_rng.set_yticks(y, [f"{i + 1:>2}  {d}   {t}" for i, (d, t) in enumerate(zip(df["driver_id"], df["team"]))])
    ax_rng.set_xticks([1, 5, 10, 15, 20])
    ax_rng.set_xlabel("Bitiş pozisyonu")
    ax_rng.grid(axis="x", color=RULE, lw=0.8)
    ax_rng.set_axisbelow(True)
    ax_rng.tick_params(axis="y", length=0, pad=6)
    for s in ax_rng.spines.values():
        s.set_visible(False)
    ax_rng.set_title("Ortalama pozisyon", loc="left", fontsize=11, color=TEXT, fontweight="bold")

    # 2) Ortalama pozisyon + Galibiyet / Podyum / Puan ihtimalleri (metin tablo)
    ax_tbl.set_facecolor(SURFACE)
    ax_tbl.set_xlim(0, 4)
    ax_tbl.set_ylim(n - 0.5, -0.5)
    ax_tbl.axis("off")
    for yi, e in zip(y, df["expected_position"]):
        ax_tbl.text(0.75, yi, f"{e:.1f}", ha="right", va="center", fontsize=10, fontweight="bold")
    ax_tbl.text(0.75, -1.05, "Ort. poz.", ha="right", va="center", color=TEXT_2, fontsize=10, fontweight="bold")
    for j, (col, name) in enumerate([("p_win", "Galibiyet"), ("p_podium", "Podyum"), ("p_points", "Puan")], start=1):
        ax_tbl.text(j + 0.85, -1.05, name, ha="right", va="center", color=TEXT_2, fontsize=10, fontweight="bold")
        for yi, v in zip(y, df[col]):
            ax_tbl.text(j + 0.85, yi, _pct(v), ha="right", va="center", fontsize=10, color=TEXT_2)

    # 3) Isı haritası: pilot x pozisyon. Skala %40'ta doyuyor ki favorinin yüksek
    # ihtimali orta grubun farklarını silmesin; gamma küçük farkları görünür kılar
    color_cap = min(prob.max(), 0.40)
    norm = PowerNorm(gamma=0.6, vmin=0, vmax=color_cap)
    im = ax_hm.imshow(np.minimum(prob, color_cap), cmap=BLUES, norm=norm, aspect="auto")
    # Hücreler arası yüzey boşluğu
    ax_hm.set_xticks(np.arange(-0.5, n), minor=True)
    ax_hm.set_yticks(np.arange(-0.5, n), minor=True)
    ax_hm.grid(which="minor", color=SURFACE, lw=2)
    ax_hm.tick_params(which="minor", length=0)
    ax_hm.set_xticks(np.arange(n), [str(p) for p in range(1, n + 1)])
    ax_hm.xaxis.tick_top()
    ax_hm.set_yticks([])
    ax_hm.tick_params(axis="x", length=0)
    for s in ax_hm.spines.values():
        s.set_visible(False)

    for i in range(n):
        # Seçici etiket: sadece ≥%10 olan hücrelerde sayı
        for p in range(n):
            v = prob[i, p]
            if v >= 0.10:
                dark = norm(min(v, color_cap)) > 0.55
                ax_hm.text(p, i, f"{v * 100:.0f}", ha="center", va="center", fontsize=9,
                           color="white" if dark else TEXT, fontweight="bold")
        # Pilotun grid pozisyonundaki hücrenin altını çiz
        g = int(df.loc[i, "grid_position"]) - 1
        ax_hm.plot([g - 0.42, g + 0.42], [i + 0.40, i + 0.40], color=TEXT_2, lw=1.6, solid_capstyle="butt")

    # Renk çubuğu ayrı eksende: ısı haritasından yer çalıp satır hizasını bozmasın
    hm_box = ax_hm.get_position()
    cax = fig.add_axes([hm_box.x0, 0.075, hm_box.width, 0.014])
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    ticks = [t for t in [0, 0.05, 0.10, 0.20, 0.30, 0.40] if t <= color_cap + 1e-9]
    cbar.set_ticks(ticks, labels=[f"{t * 100:.0f}%" + ("+" if t == ticks[-1] and prob.max() > color_cap else "")
                                  for t in ticks])
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0, labelsize=9, colors=TEXT_2)
    cbar.set_label("Bu pozisyonda bitirme ihtimali (hücredeki sayı: %, sadece ≥%10)", color=TEXT_2, fontsize=9)

    # Başlık, açıklama, lejant
    fav = df.sort_values("p_win", ascending=False).iloc[0]
    fig.text(0.12, 0.955, "Mexico City GP 2026: Pozisyon Olasılıkları", fontsize=20, fontweight="bold")
    fig.text(0.12, 0.922,
             f"{n_sims:,}".replace(",", ".") + f" yarış simülasyonu  ·  Grid kaynağı: {grid_source}  ·  "
             f"Favori: {fav['driver_id']} ({_pct(fav['p_win'])} galibiyet)",
             fontsize=11, color=TEXT_2)
    handles = [
        Line2D([], [], marker="o", ls="", color=ACCENT, markersize=8, label="ortalama pozisyon"),
        Line2D([], [], color=MUTED, lw=2, label="%10–%90 aralığı"),
        Line2D([], [], marker="D", ls="", markerfacecolor="none", markeredgecolor=TEXT_2, markersize=6, label="grid"),
        Line2D([], [], color=TEXT_2, lw=1.6, label="ısı haritasında grid hücresi"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.115, 0.912), ncol=4, frameon=False,
               fontsize=10, labelcolor=TEXT_2)
    fig.text(0.12, 0.018,
             "Not: 2026 testinde model, 'sonuç = grid sırası' tahminiyle aynı oranda kazananı bildi ve sıralamayı biraz daha "
             "iyi tuttu; olasılık kalitesinde saf gridin biraz gerisinde. Kesin tahmin değil, belirsizlik aralığı olarak okuyun.",
             fontsize=9, color=TEXT_2)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, facecolor=SURFACE)
    print(f"Görselleştirme kaydedildi: {output_path}")
    if show:
        plt.show()
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    # Son tahmin CSV'sinden yeniden çiz (pencerede göstermek için: python src/visualize.py --show)
    csv_path = os.path.join(ROOT, "outputs", "mexico_gp_2026_probabilities.csv")
    plot_probabilities(pd.read_csv(csv_path), "son tahmin dosyası", show="--show" in sys.argv)
