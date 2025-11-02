# bullying_detector.py
import re
import os
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from datetime import timedelta

# ===== 1) léxico básico de toxicidade (edite/amplie) =====
BAD_WORDS = {
    # PT comuns (adicione variações/masculino-feminino, etc.)
    "burro","idiota","otario","otário","imbecil","lixo","merda","bosta",
    "nojento","escroto","ridiculo","ridículo","inutil","inútil","vagabundo",
    "feio","gordo","bicha","viadinho","racista","macaco","favelado","retardado",
    "puta","piranha","corno","corna","cornao","cornão","cornao",
    # EN
    "stupid","idiot","dumb","trash","loser","bitch","asshole","moron","ugly","fat",
}
BAD_PATTERNS = [
    r"\b(cala a boca|ninguém te perguntou|ningu[eé]m gosta de você)\b",
    r"\b(vai se f\*?|\bse f\*?)\b",
    r"\b(n[aã]o presta|m[eé]rda de pessoa)\b",
]
BAD_RE = re.compile(r"(" + "|".join(map(re.escape, BAD_WORDS)) + r")", re.IGNORECASE)
BAD_PATTERNS_RE = re.compile("|".join(BAD_PATTERNS), re.IGNORECASE)

# ===== 2) helpers =====
def normalize_text(s: str) -> str:
    s = s or ""
    s = re.sub(r"https?://\S+|www\.\S+", " ", s, flags=re.I)
    s = s.replace("\n", " ")
    return s.strip()

def toxicity_score(text: str) -> float:
    """Heurística 0..1 baseada em léxico."""
    if not text: 
        return 0.0
    t = normalize_text(text.lower())
    if not t:
        return 0.0
    hit_words = BAD_RE.findall(t)
    hit_patts = BAD_PATTERNS_RE.findall(t)
    # Peso simples: palavras (0.15 cada), padrões (0.35 cada), clamp em 1.0
    score = 0.15*len(hit_words) + 0.35*len(hit_patts)
    return float(min(1.0, score))

def build_aliases(users, extra_aliases=None):
    """Cria dicionário de aliases para detectar menções por nome/apelido."""
    aliases = defaultdict(set)
    for u in users:
        u_str = str(u)
        parts = re.split(r"[\s._-]+", u_str)
        for p in parts:
            if len(p) >= 3:
                aliases[u].add(p.lower())
        # também nickname entre parênteses
        m = re.findall(r"\(([^)]+)\)", u_str)
        for nick in m:
            if len(nick) >= 3:
                aliases[u].add(nick.lower())
    if extra_aliases:
        for u, arr in extra_aliases.items():
            for a in arr:
                if len(a) >= 2:
                    aliases[u].add(a.lower())
    return {u: sorted(v) for u, v in aliases.items()}

def find_targets(text: str, alias_map) -> set:
    """Retorna conjunto de usuários mencionados no texto (por alias)."""
    t = " " + normalize_text(text.lower()) + " "
    targets = set()
    for user, aliases in alias_map.items():
        for a in aliases:
            # requer bordas para evitar falsos positivos
            if re.search(rf"[\s@#]{re.escape(a)}[\s?!,.:;)]", t):
                targets.add(user)
                break
    return targets

# ===== 3) Núcleo de detecção =====
def detect_bullying(df: pd.DataFrame,
                    window_minutes: int = 60,
                    min_events: int = 3,
                    min_authors: int = 2,
                    toxic_threshold: float = 0.35):
    """
    Gera episódios suspeitos em janelas de T minutos quando:
      - mensagens com toxicidade >= limiar,
      - direcionadas ao mesmo alvo (por menção),
      - repetição >= min_events,
      - por >= min_authors distintos (pile-on).

    Retorna:
      episodes_df (linhas: alvo, t0..t1, n_msgs, n_authors, bullying_score, autores, exemplos)
      edges_dir (DataFrame with from->to contagem e média de toxicidade)
    """
    df = df.copy().sort_values("datetime").reset_index(drop=True)
    users = sorted(df["user"].dropna().unique())
    alias_map = build_aliases(users)

    # escore de toxicidade e alvos
    df["tox"] = df["message"].astype(str).map(toxicity_score)
    df["targets"] = df["message"].astype(str).map(lambda s: list(find_targets(s, alias_map)))
    # mantemos apenas msgs “negativas e direcionadas”
    neg = df[(df["tox"] >= toxic_threshold) & (df["targets"].map(len) > 0)].copy()
    if neg.empty:
        return pd.DataFrame(), pd.DataFrame()

    # arestas direcionais autor -> alvo (para rede de agressões)
    rows = []
    for _, r in neg.iterrows():
        for tgt in r["targets"]:
            rows.append((r["user"], tgt, r["datetime"], r["tox"], r["message"]))
    edges = pd.DataFrame(rows, columns=["from_user","to_user","datetime","tox","message"])
    edges_dir = (edges
                 .groupby(["from_user","to_user"])
                 .agg(count=("message","size"),
                      mean_tox=("tox","mean"))
                 .reset_index()
                 .sort_values("count", ascending=False))

    # varre janelas deslizantes por alvo
    win = timedelta(minutes=window_minutes)
    episodes = []
    for tgt, sub in edges.groupby("to_user"):
        sub = sub.sort_values("datetime").reset_index(drop=True)
        i = 0
        while i < len(sub):
            t0 = sub.loc[i, "datetime"]
            inwin = sub[(sub["datetime"] >= t0) & (sub["datetime"] < t0 + win)]
            n_msgs = len(inwin)
            authors = set(inwin["from_user"])
            if n_msgs >= min_events and len(authors) >= min_authors:
                t1 = inwin["datetime"].max()
                # bullying_score: repetição * pile-on * toxicidade média
                score = n_msgs * (1 + 0.3*(len(authors)-1)) * float(inwin["tox"].mean())
                ex_msgs = inwin.head(3)["message"].tolist()
                episodes.append({
                    "target": tgt,
                    "t_start": t0,
                    "t_end": t1,
                    "window_min": window_minutes,
                    "n_msgs": n_msgs,
                    "n_authors": len(authors),
                    "bullying_score": round(score, 3),
                    "authors": ", ".join(sorted(authors)),
                    "sample_msgs": " | ".join(ex_msgs)
                })
                # avança o ponteiro para fora da janela atual (evita duplicar)
                # pula até a primeira fora do intervalo
                j = inwin.index.max() + 1
                i = max(i+1, j)
            else:
                i += 1

    episodes_df = pd.DataFrame(episodes).sort_values(["bullying_score","n_msgs"], ascending=False)
    return episodes_df, edges_dir

# ===== 4) plots úteis =====
def plot_bullying_network(edges_dir: pd.DataFrame, out_png: str, top=20):
    if edges_dir.empty: 
        return
    # gráfico simples de cordas seria ideal, mas aqui usamos barra “from->to” textual
    top_df = edges_dir.head(top).copy()
    top_df["label"] = top_df["from_user"] + " ➜ " + top_df["to_user"]
    ax = top_df.plot(kind="barh", x="label", y="count", figsize=(10, 6), legend=False)
    ax.set_xlabel("Ocorrências negativas direcionadas")
    ax.set_ylabel("from ➜ to")
    ax.set_title("Principais interações negativas (heurística)")
    for i, v in enumerate(top_df["count"]):
        ax.text(v + 0.2, i, str(int(v)))
    import matplotlib.pyplot as plt
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
