import os
import pickle
import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import spearmanr

from features import FEATURES, PROCESSED_PATH, ROOT

MODEL_PATH = os.path.join(ROOT, "models", "race_ranker.pkl")

# label_gain: varsayılan 2^etiket -> kazanan ~2 milyon kat ağırlık alıyordu.
# Doğrusal kazanç tüm sıralamayı önemsiyor (iki seçenek doğrulama setinde karşılaştırılıyor)
LABEL_GAINS = {"linear": list(range(32)), "exponential": [2**i - 1 for i in range(32)]}
N_ESTIMATORS = [50, 100, 200, 400]
BLEND_WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]


def make_model(label_gain, n_estimators):
    params = dict(
        objective="lambdarank",
        n_estimators=n_estimators,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        label_gain=label_gain,
        lambdarank_truncation_level=22,  # tüm sıralama önemli, sadece ilk sıralar değil
        random_state=42,
        verbose=-1,
    )
    return lgb.LGBMRanker(**params)


def blend_scores(model_pred, grid_position, scales, grid_weight):
    """
    Model skoru ile grid skorunu (-log grid) harmanlar. Tek başına model kazananı bilmede
    gridin gerisinde kalıyordu; ağırlık doğrulama setinde seçiliyor.
    """
    model_part = np.asarray(model_pred) / scales["model"]
    grid_part = -np.log(np.asarray(grid_position, dtype=float)) / scales["grid"]
    return (1 - grid_weight) * model_part + grid_weight * grid_part


def fit_scales(model, df):
    # İki skoru aynı ölçeğe getirmek için eğitim verisindeki standart sapmalar
    return {"model": float(np.std(model.predict(df[FEATURES]))),
            "grid": float(np.std(np.log(df["grid_position"])))}


def groups_of(df):
    # LightGBM grupları satır sırasına göre okur -> df race_id'ye göre sıralı olmalı
    return df.groupby("race_id", sort=False).size().to_numpy()


def evaluate(df, score_col):
    """Yarış bazında: kazanan doğruluğu, ortalama pozisyon hatası, Spearman korelasyonu."""
    df = df.copy()
    df["pred_rank"] = df.groupby("race_id")[score_col].rank(ascending=False, method="first")
    top1 = (df[df["finish_position"] == 1]["pred_rank"] == 1).mean()
    mae = np.mean(np.abs(df["finish_position"] - df["pred_rank"]))
    rho = df.groupby("race_id").apply(
        lambda g: spearmanr(g["finish_position"], g["pred_rank"])[0], include_groups=False
    ).mean()
    return {"top1": top1, "mae": mae, "spearman": rho}


def stage_temperatures(n, params):
    """
    Pozisyona bağlı sıcaklık: T_k = exp(a + b * k / (n-1)), k = 0 (birinci seçimi) ... n-1.
    Tek sıcaklık kaotik orta grubu yüzünden ön sıraları fazla düzleştiriyordu
    (pole'den kazanma %69 iken %24 tahmin ediyordu). Ön sıralar daha belirleyici.
    """
    a, b = params
    return np.exp(a + b * np.arange(n) / max(n - 1, 1))


def plackett_luce_nll(df, score_col, params):
    """Gerçek bitiş sırasının, skorlar/T_k ile Plackett-Luce altında negatif log-olabilirliği."""
    nll = 0.0
    for _, g in df.groupby("race_id"):
        s = g.sort_values("finish_position")[score_col].to_numpy()
        temps = stage_temperatures(len(s), params)
        for i in range(len(s) - 1):
            x = s[i:] / temps[i]
            nll -= x[0] - logsumexp(x)
    return nll


def fit_temperature(df, score_col):
    # Simülasyondaki belirsizlik miktarını gerçek sonuçlara göre ayarlıyoruz (maksimum olabilirlik)
    res = minimize(lambda p: plackett_luce_nll(df, score_col, p), x0=[0.0, 0.0], method="Nelder-Mead")
    return [float(v) for v in res.x]


def nll_per_pick(df, score_col, params):
    picks = sum(len(g) - 1 for _, g in df.groupby("race_id"))
    return plackett_luce_nll(df, score_col, params) / picks


def print_metrics(name, m):
    print(f"{name:<22} Galibiyet: %{m['top1'] * 100:5.1f} | MAE: {m['mae']:.2f} | Spearman: {m['spearman']:.3f}")


def train_model(input_path=PROCESSED_PATH, model_output_path=MODEL_PATH):
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    df = pd.read_csv(input_path)
    df = df[df["finish_position"].notna()].sort_values(["race_id", "grid_position"]).reset_index(drop=True)
    df["target_score"] = df["target_score"].astype(int)
    df["grid_score"] = -df["grid_position"]  # Referans: "grid sırası = sonuç"

    # Time-Aware Split (Zaman Boyutunda Bölme)
    # Train: 2022-2024 | Validation (erken durdurma + seçim): 2025 | Test (yeni kurallar): 2026
    train_df = df[df["year"] <= 2024]
    val_df = df[df["year"] == 2025].copy()
    test_df = df[df["year"] == 2026].copy()
    print(f"Train: {len(train_df)} satır, {train_df['race_id'].nunique()} yarış")
    print(f"Val:   {len(val_df)} satır, {val_df['race_id'].nunique()} yarış")
    print(f"Test:  {len(test_df)} satır, {test_df['race_id'].nunique()} yarış")

    # Ayar seçimi doğrulama setinde (2025): olasılık kalitesi = Plackett-Luce log-kaybı
    # (Erken durdurma ndcg@1'e bakıp 1-2 iterasyonda kesiyordu, o yüzden sabit iterasyon adayları)
    best = None
    for gain_name, gain in LABEL_GAINS.items():
        for n_iter in N_ESTIMATORS:
            model = make_model(gain, n_iter)
            model.fit(train_df[FEATURES], train_df["target_score"], group=groups_of(train_df))
            val_df["pred_score"] = model.predict(val_df[FEATURES])
            t = fit_temperature(val_df, "pred_score")
            nll = nll_per_pick(val_df, "pred_score", t)
            print(f"Val [{gain_name:<11} it={n_iter:<3}] log-kayıp: {nll:.4f}")
            if best is None or nll < best[0]:
                best = (nll, gain_name, n_iter, model)
    _, gain_name, n_iter, model = best
    scales = fit_scales(model, train_df)

    # Grid harmanlama ağırlığını da doğrulama setinde seç
    best = None
    for w in BLEND_WEIGHTS:
        val_df["pred_score"] = blend_scores(model.predict(val_df[FEATURES]), val_df["grid_position"], scales, w)
        t = fit_temperature(val_df, "pred_score")
        nll = nll_per_pick(val_df, "pred_score", t)
        print(f"Val [grid ağırlığı={w:.2f}] log-kayıp: {nll:.4f}")
        if best is None or nll < best[0]:
            best = (nll, w, t)
    _, grid_weight, t_model = best

    # Test seti (2026) değerlendirmesi: model hiç görmedi, seçimde de kullanılmadı
    val_df["pred_score"] = blend_scores(model.predict(val_df[FEATURES]), val_df["grid_position"], scales, grid_weight)
    test_df["pred_score"] = blend_scores(model.predict(test_df[FEATURES]), test_df["grid_position"], scales, grid_weight)
    t_grid = fit_temperature(val_df, "grid_score")
    print("\n==================================================")
    print(f" TEST (2026) - label_gain={gain_name}, n_estimators={n_iter}, grid ağırlığı={grid_weight}")
    print_metrics("Model", evaluate(test_df, "pred_score"))
    print_metrics("Referans (grid sırası)", evaluate(test_df, "grid_score"))
    print(f"{'Olasılık log-kaybı':<22} Model: {nll_per_pick(test_df, 'pred_score', t_model):.4f} | "
          f"Grid: {nll_per_pick(test_df, 'grid_score', t_grid):.4f} (düşük = iyi)")
    print("==================================================")

    # Olasılık kalibrasyonu: örneklem dışı tahminlerle (2025 + 2026)
    temperature = fit_temperature(pd.concat([val_df, test_df]), "pred_score")
    temps = stage_temperatures(22, temperature)
    print(f"Plackett-Luce sıcaklığı: 1. seçim {temps[0]:.3f} -> 22. seçim {temps[-1]:.3f}")

    # Nihai model: tüm veriyle (2026 dahil) aynı ayarlarla yeniden eğit
    final_model = make_model(LABEL_GAINS[gain_name], n_iter)
    final_model.fit(df[FEATURES], df["target_score"], group=groups_of(df))

    # Modeli Kaydet
    bundle = {"model": final_model, "features": FEATURES, "temperature": temperature,
              "scales": fit_scales(final_model, df), "grid_weight": grid_weight,
              "label_gain": gain_name, "n_estimators": n_iter}
    with open(model_output_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"Model Kaydedildi: {model_output_path}")

    importances = pd.Series(final_model.booster_.feature_importance("gain"), index=FEATURES)
    print("\n Features önemleri (gain, %):")
    print((importances / importances.sum() * 100).sort_values(ascending=False).round(1).to_string())

    return bundle


if __name__ == "__main__":
    train_model()
