#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analisador de conversas do WhatsApp (.txt) com:
- total de mensagens por usuário
- total de mensagens por usuário após 22h
- média de mensagens por dia por usuário
- total de mensagens com palavrões por usuário
- total de mensagens de mídia por usuário
- wordcloud por usuário e geral
- tempo médio de resposta por par de usuários (direcional e não-direcional)
- gráficos diversos (barras, heatmap de atividade, série diária, heatmap de resposta)

Uso:
    python whatsapp_analyzer_wc_reply.py caminho/Conversa.txt --tz America/Sao_Paulo
"""

import argparse
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
from bullying_detector import detect_bullying, plot_bullying_network

# -------------------- 1) PARSE DO WHATSAPP --------------------

DATE_PATTERNS = [
    r"^\[(?P<date>\d{1,2}/\d{1,2}/\d{2,4}),\s*(?P<time>\d{1,2}:\d{2}(?:\s*[APMapm]{2})?)\]\s*(?P<user>[^:]+):\s*(?P<msg>.*)$",
    r"^(?P<date>\d{1,2}/\d{1,2}/\d{2,4}),\s*(?P<time>\d{1,2}:\d{2}(?:\s*[APMapm]{2})?)\s*-\s*(?P<user>[^:]+):\s*(?P<msg>.*)$",
    r"^(?P<date>\d{1,2}/\d{1,2}/\d{2,4})\s+(?P<time>\d{1,2}:\d{2}(?:\s*[APMapm]{2})?)\s*-\s*(?P<user>[^:]+):\s*(?P<msg>.*)$",
]
DATE_RE_LIST = [re.compile(p) for p in DATE_PATTERNS]

def detect_line(line: str) -> Optional[re.Match]:
    for rx in DATE_RE_LIST:
        m = rx.match(line)
        if m:
            return m
    return None

def to_datetime(date_str: str, time_str: str) -> Optional[datetime]:
    date_formats = ["%d/%m/%Y", "%d/%m/%y", "%m/%d/%Y", "%m/%d/%y"]
    time_formats = ["%H:%M", "%I:%M %p", "%I:%M%p"]
    for df in date_formats:
        for tf in time_formats:
            for sep in [", ", " ", ""]:
                fmt = df + sep + tf
                try:
                    return datetime.strptime(f"{date_str}{sep}{time_str.upper()}", fmt)
                except ValueError:
                    pass
    return None

DELETED_MESSAGE_PATTERNS = [
    "Mensagem apagada"
]
DELETED_MESSAGE_PATTERNS_RE = re.compile("|".join(DELETED_MESSAGE_PATTERNS), re.IGNORECASE)

MEDIA_PATTERNS = [
    r"<Media omitted>", r"image omitted", r"video omitted", r"(arquivo anexado)",
    r"Arquivo de mídia oculto", r"arquivo de mídia oculto", r"<Mídia oculta>",
    r"imagem omitida", r"vídeo omitido", r"sticker omitido", r"figurinha omitida",
    r"Mensagem de áudio", r"mensagem de áudio", r"áudio omitido",
    r"🖼️|🎥|🎙️|📎"
]
MEDIA_RE = re.compile("|".join(MEDIA_PATTERNS), re.IGNORECASE)

PROFANITY = {
    "porra","caralho","merda","bosta","puta","punheta","desgraça",
    "c*ralho","p*rra","m*rda",
    "fuck","shit","asshole","bitch","bastard","crap","dick","piss","bollocks"
}
PROFANITY_RE = re.compile(r"\b(" + "|".join(map(re.escape, PROFANITY)) + r")\b", re.IGNORECASE)

def parse_chat(path: str) -> pd.DataFrame:
    messages: List[Tuple[datetime, str, str]] = []
    cur_user, cur_dt, cur_msg_lines = None, None, []

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            m = detect_line(line)
            if m:
                if cur_dt is not None and cur_user is not None:
                    messages.append((cur_dt, cur_user, "\n".join(cur_msg_lines).strip()))
                    cur_msg_lines = []
                dt = to_datetime(m.group("date").strip(), m.group("time").strip())
                if dt is None:
                    if cur_dt is not None:
                        cur_msg_lines.append(line)
                        continue
                    else:
                        continue
                cur_dt = dt
                cur_user = m.group("user").strip()
                cur_msg_lines = [m.group("msg").strip()]
            else:
                if cur_dt is not None:
                    cur_msg_lines.append(line)

    if cur_dt is not None and cur_user is not None and cur_msg_lines:
        messages.append((cur_dt, cur_user, "\n".join(cur_msg_lines).strip()))

    if not messages:
        raise RuntimeError("Nenhuma mensagem reconhecida. Verifique o formato do arquivo.")

    df = pd.DataFrame(messages, columns=["datetime", "user", "message"])
    df["date"] = df["datetime"].dt.date
    df["time"] = df["datetime"].dt.time
    df["hour"] = df["datetime"].dt.hour
    df["is_media"] = df["message"].apply(lambda s: bool(MEDIA_RE.search(s)))
    df["has_profanity"] = df["message"].apply(lambda s: bool(PROFANITY_RE.search(s)))
    df["is_deleted"] = df["message"].apply(lambda s: bool(DELETED_MESSAGE_PATTERNS_RE.search(s)))

    return df

# -------------------- 2) MÉTRICAS BÁSICAS --------------------

def build_metrics(df: pd.DataFrame) -> dict:
    metrics = {}
    metrics["total_por_usuario"] = df.groupby("user").size().sort_values(ascending=False)
    metrics["total_pos_22_por_usuario"] = df[df["hour"] >= 22].groupby("user").size().sort_values(ascending=False)
    msgs_per_day_user = df.groupby(["user","date"]).size().rename("count").reset_index()
    metrics["media_por_dia_por_usuario"] = msgs_per_day_user.groupby("user")["count"].mean().sort_values(ascending=False)
    metrics["total_profanidade_por_usuario"] = df[df["has_profanity"]].groupby("user").size().sort_values(ascending=False)
    metrics["total_midias_por_usuario"] = df[df["is_media"]].groupby("user").size().sort_values(ascending=False)
    metrics["total_deleted_por_usuario"] = df[df["is_deleted"]].groupby("user").size().sort_values(ascending=False)
    return metrics

# -------------------- 3) PLOTS BÁSICOS --------------------

def save_bar(series: pd.Series, title: str, fname: str, xlabel="Usuário", ylabel="Quantidade"):
    if series.empty:
        return
    plt.figure(figsize=(10, 5))
    series.plot(kind="bar")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()

def save_heatmap_activity(df: pd.DataFrame, fname: str):
    if df.empty:
        return
    tmp = df.copy()
    tmp["dow"] = tmp["datetime"].dt.dayofweek
    pivot = tmp.pivot_table(index="dow", columns="hour", values="message",
                            aggfunc="count", fill_value=0).reindex(index=[0,1,2,3,4,5,6])
    plt.figure(figsize=(12, 4))
    plt.imshow(pivot, aspect="auto")
    plt.title("Atividade por dia da semana x hora")
    plt.yticks(range(7), ["Seg","Ter","Qua","Qui","Sex","Sáb","Dom"])
    plt.xticks(range(24), range(24))
    plt.colorbar(label="Qtde de mensagens")
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()

def save_daily_timeseries(df: pd.DataFrame, out_png: str):
    per_day = df.groupby("date").size()
    plt.figure(figsize=(12, 4))
    per_day.plot()
    plt.title("Mensagens por dia (grupo)")
    plt.xlabel("Data")
    plt.ylabel("Mensagens")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

# -------------------- 4) WORDCLOUD --------------------

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+", re.UNICODE)

PT_STOP = {
    "de","da","do","das","dos","e","a","o","os","as","que","pra","para","com","um","uma","na","no",
    "em","se","é","já","mais","mas","porque","por","só","sim","não","tá","to","tô","vai","vou","aí",
    "lá","aqui","tem","tão","tudo","nada","essa","esse","isso","é","né","kkk","kkkk","kk","rs","vc","vcs"
}

def normalize_text_for_cloud(s: str) -> str:
    s = s.lower()
    s = URL_RE.sub(" ", s)
    s = MENTION_RE.sub(" ", s)
    s = re.sub(r"[\d\W_]+", " ", s, flags=re.UNICODE)
    return s

def build_wordclouds(df: pd.DataFrame, outdir: str, width=1600, height=900, max_words=300):
    wc_dir = os.path.join(outdir, "wordclouds")
    os.makedirs(wc_dir, exist_ok=True)

    stop = STOPWORDS.union(PT_STOP)
    text_all = " ".join(
        normalize_text_for_cloud(m)
        for m in df.loc[~df["is_media"], "message"].astype(str).tolist()
    )
    if text_all.strip():
        wc = WordCloud(width=width, height=height, stopwords=stop, background_color="white").generate(text_all)
        wc.to_file(os.path.join(wc_dir, "wordcloud_geral.png"))

    for user, sub in df[~df["is_media"]].groupby("user"):
        text = " ".join(normalize_text_for_cloud(x) for x in sub["message"].astype(str))
        if not text.strip():
            continue
        wc = WordCloud(width=width, height=height, stopwords=stop, background_color="white").generate(text)
        safe_user = re.sub(r"[^a-zA-Z0-9_-]+", "_", user)
        wc.to_file(os.path.join(wc_dir, f"wordcloud_{safe_user}.png"))

# -------------------- 5) TEMPO DE RESPOSTA --------------------

def compute_reply_times(df: pd.DataFrame, max_gap_minutes: int = 12*60):
    """
    Calcula tempos de resposta entre mensagens consecutivas de usuários diferentes.
    - max_gap_minutes: descarta gaps muito grandes (ex: madrugada/semana seguinte).

    Retorna:
      replies_df: linhas com (from_user, to_user, delta_sec, datetime_from, datetime_to)
      mean_dir: média por direção (from->to)
      mean_undir: média por par sem direção (frozenset)
    """
    df = df.sort_values("datetime").reset_index(drop=True)
    rows = []
    max_gap = timedelta(minutes=max_gap_minutes)

    for i in range(len(df) - 1):
        u1, u2 = df.at[i, "user"], df.at[i+1, "user"]
        if u1 == u2:
            continue
        t1, t2 = df.at[i, "datetime"], df.at[i+1, "datetime"]
        delta = t2 - t1
        if delta <= timedelta(0) or delta > max_gap:
            continue
        rows.append({
            "from_user": u1,
            "to_user": u2,
            "delta_sec": delta.total_seconds(),
            "datetime_from": t1,
            "datetime_to": t2
        })

    replies_df = pd.DataFrame(rows)
    if replies_df.empty:
        mean_dir = pd.DataFrame(columns=["from_user","to_user","mean_reply_sec","count"])
        mean_undir = pd.DataFrame(columns=["pair","mean_reply_sec","count"])
        return replies_df, mean_dir, mean_undir

    mean_dir = replies_df.groupby(["from_user","to_user"])["delta_sec"].agg(["mean","count"]).reset_index()
    mean_dir.rename(columns={"mean":"mean_reply_sec"}, inplace=True)

    replies_df["pair"] = replies_df.apply(lambda r: tuple(sorted([r["from_user"], r["to_user"]])), axis=1)
    mean_undir = replies_df.groupby("pair")["delta_sec"].agg(["mean","count"]).reset_index()
    mean_undir.rename(columns={"mean":"mean_reply_sec"}, inplace=True)

    return replies_df, mean_dir, mean_undir

def format_seconds_to_hms(s: float) -> str:
    s = int(round(s))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    if h > 0:
        return f"{h}h {m}m {sec}s"
    return f"{m}m {sec}s"

def save_reply_heatmap(mean_dir: pd.DataFrame, out_png: str, top_users: int = 10):
    """Cria um heatmap (from -> to) com mean_reply_sec para os principais usuários por volume de interações."""
    if mean_dir.empty:
        return

    # Seleciona top usuários por 'count' agregado
    counts = mean_dir.groupby("from_user")["count"].sum().add(
             mean_dir.groupby("to_user")["count"].sum(), fill_value=0).sort_values(ascending=False)
    top = counts.head(top_users).index.tolist()

    data = mean_dir[mean_dir["from_user"].isin(top) & mean_dir["to_user"].isin(top)]
    if data.empty:
        return

    users = sorted(set(top))
    matrix = np.full((len(users), len(users)), np.nan)

    pos = {u:i for i,u in enumerate(users)}
    for _, row in data.iterrows():
        i, j = pos[row["from_user"]], pos[row["to_user"]]
        matrix[i, j] = row["mean_reply_sec"] / 60.0  # minutos

    plt.figure(figsize=(1.0 + 0.6*len(users), 1.0 + 0.6*len(users)))
    im = plt.imshow(matrix, cmap="viridis")
    plt.title("Tempo médio de resposta (min) — from ➜ to")
    plt.xticks(range(len(users)), users, rotation=45, ha="right")
    plt.yticks(range(len(users)), users)
    plt.colorbar(im, label="minutos")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

# -------------------- 6) MAIN --------------------

def main():
    parser = argparse.ArgumentParser(description="Analisador WhatsApp com Wordcloud e tempos de resposta")
    parser.add_argument("input", help="Caminho do arquivo .txt exportado do WhatsApp")
    parser.add_argument("--tz", default=None, help="Timezone (ex.: America/Sao_Paulo)")
    parser.add_argument("--outdir", default="out", help="Diretório de saída")
    parser.add_argument("--max_reply_gap_min", type=int, default=12*60,
                        help="Gap máximo (min) para considerar uma resposta (default=720)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = parse_chat(args.input)

    if args.tz:
        try:
            import pytz
            tz = pytz.timezone(args.tz)
            df["datetime"] = df["datetime"].apply(lambda d: tz.localize(d))
            df["date"] = df["datetime"].dt.date
            df["time"] = df["datetime"].dt.time
            df["hour"] = df["datetime"].dt.hour
        except Exception as e:
            print(f"Aviso: não foi possível aplicar timezone ({e}). Prosseguindo sem TZ…", file=sys.stderr)

    # Métricas básicas
    metrics = build_metrics(df)

    # Wordclouds
    build_wordclouds(df, args.outdir)

    # Tempos de resposta
    replies_df, mean_dir, mean_undir = compute_reply_times(df, max_gap_minutes=args.max_reply_gap_min)

    # Salva CSVs
    df.to_csv(os.path.join(args.outdir, "mensagens_raw.csv"), index=False)
    for k, s in metrics.items():
        s.to_csv(os.path.join(args.outdir, f"{k}.csv"))
    replies_df.to_csv(os.path.join(args.outdir, "replies_direcionais_raw.csv"), index=False)
    mean_dir.to_csv(os.path.join(args.outdir, "tempo_medio_resposta_direcional.csv"), index=False)
    mean_undir.to_csv(os.path.join(args.outdir, "tempo_medio_resposta_pares.csv"), index=False)

    # Plots básicos
    save_bar(metrics["total_por_usuario"], "Total de mensagens por usuário",
             os.path.join(args.outdir, "total_por_usuario.png"))
    save_bar(metrics["total_pos_22_por_usuario"], "Total após 22h por usuário",
             os.path.join(args.outdir, "total_pos_22_por_usuario.png"))
    save_bar(metrics["media_por_dia_por_usuario"], "Média mensagens/dia por usuário",
             os.path.join(args.outdir, "media_por_dia_por_usuario.png"), ylabel="Média por dia")
    save_bar(metrics["total_profanidade_por_usuario"], "Palavrões por usuário",
             os.path.join(args.outdir, "total_profanidade_por_usuario.png"))
    save_bar(metrics["total_midias_por_usuario"], "Mídias por usuário",
             os.path.join(args.outdir, "total_midias_por_usuario.png"))
    save_heatmap_activity(df, os.path.join(args.outdir, "heatmap_atividade.png"))
    save_daily_timeseries(df, os.path.join(args.outdir, "serie_diaria_grupo.png"))

    # Heatmap de tempos de resposta (principais usuários)
    save_reply_heatmap(mean_dir, os.path.join(args.outdir, "heatmap_tempo_resposta.png"), top_users=10)

    # Resumo no console
    print("\n=== RESUMO ===")
    if not mean_dir.empty:
        top5 = mean_dir.sort_values("mean_reply_sec").head(5)
        print("\nTop 5 respostas mais rápidas (direcional):")
        for _, r in top5.iterrows():
            print(f"  {r['from_user']} ➜ {r['to_user']}: {format_seconds_to_hms(r['mean_reply_sec'])} (n={int(r['count'])})")
    else:
        print("\nNão foi possível calcular tempos de resposta (dados insuficientes).")

    print("\nWordclouds salvas em ./out/wordclouds/")
    print(f"Arquivos gerados em: ./{args.outdir}\n")

def main_bullying(df: pd.DataFrame):    
    episodes_df, edges_dir = detect_bullying(
        df,
        window_minutes=60,   # janela para “repetição”
        min_events=3,        # pelo menos 3 msgs
        min_authors=2,       # de 2+ autores (pile-on)
        toxic_threshold=0.35  # ajuste de sensibilidade
    )

    outdir = "out"
    episodes_csv = os.path.join(outdir, "bullying_episodios_suspeitos.csv")
    edges_csv = os.path.join(outdir, "bullying_interacoes_direcionais.csv")
    episodes_df.to_csv(episodes_csv, index=False)
    edges_dir.to_csv(edges_csv, index=False)

    # gráfico rápido dos top ataques direcionais
    plot_bullying_network(edges_dir, os.path.join(
        outdir, "bullying_top_interacoes.png"))
    print(f"\nPossíveis episódios suspeitos: {episodes_csv}")
    print(f"Interações direcionais negativas: {edges_csv}")


if __name__ == "__main__":
    main()
