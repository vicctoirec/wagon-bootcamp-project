# ------------------------------------------------------------------
# Charge le csv original "20250609_17k_lyrics_eng_fr.csv"
# Nettoie la colonne 'lyrics_clean'
# Charge le csv des embeddings "embedded_17klyrics.csv"
# Encode dynamiquement les requêtes utilisateur SBERT (nomic-ai/nomic-embed-text-v2-moe)
# Calcule la similarité cosinus entre les requêtes et les embeddings et renvoie les 50 meilleurs titres
# ------------------------------------------------------------------

from pathlib import Path
import pandas as pd
import numpy as np
import torch
# import argparse
from sentence_transformers import SentenceTransformer, util
from zeroshots_function.zeroshot_pipeline import preprocess_lyrics
from ai_spotify_lyrics.params import *

# ----------------------- PARAMÈTRES --------------------------------
EMBD_CSV = Path(DATA_CSV_17k_EMBED)
RAW_CSV = Path(DATA_CSV_17k) # mêmes index !
MODEL_NAME = "nomic-ai/nomic-embed-text-v2-moe"  # SBERT model
BATCH_SIZE = 32  # Batch size for encoding
TOP_K = 50  # Number of top matches to return
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# -------------------------------------------------------------------

model = SentenceTransformer(MODEL_NAME, device=DEVICE, trust_remote_code=True)

def build_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build embeddings for the lyrics in the DataFrame using SBERT.

    Args:
        df (pd.DataFrame): DataFrame containing lyrics.

    Returns:
        pd.DataFrame: New DataFrame with embeddings.
    """
    # Initialize the SBERT model
    model_sbert = SentenceTransformer(MODEL_NAME, device=DEVICE)

    # Encode the lyrics
    print("🔹 Encoding lyrics…")
    embs = model_sbert.encode(
        df["lyrics_clean"].tolist(),
        batch_size=BATCH_SIZE,
        show_progress_bar=True)

    # `embs` -> ndarray (N, 768) float32
    cols = [f"embedding_{i}" for i in range(embs.shape[1])]
    emb_df = pd.DataFrame(embs, columns=cols, index=df.index)

    return emb_df


def ensure_embeddings() -> pd.DataFrame:
    """Charge l'embedding CSV s'il existe, sinon le crée et le sauvegarde."""

    if EMBD_CSV.exists():
        print("✔️  Embeddings CSV trouvé ; chargement…")
        return pd.read_csv(EMBD_CSV, index_col=0)

    else:
        print("⚠️  Embeddings CSV introuvable : génération en cours.")
        df_raw = pd.read_csv(RAW_CSV)
        df_raw = df_raw.dropna(subset=["lyrics_clean"])
        df_raw["lyrics_clean"] = df_raw["lyrics_clean"].apply(preprocess_lyrics)

        emb_df = build_embeddings(df_raw)
        emb_df.to_csv(EMBD_CSV)

        print(f"✔️ Embeddings sauvegardés → {EMBD_CSV}")
        return emb_df


# --------------- FONCTION PRINCIPALE -------------------

def get_top_k(user_input: str, k: int = TOP_K) -> pd.DataFrame:
    """Renvoie un DataFrame top-k (artist, track_title_clean, score)."""

    # 1. Métadonnées
    df_meta = pd.read_csv(RAW_CSV)

    required_cols = {"artist", "track_title_clean"}
    if not required_cols.issubset(df_meta.columns):
        raise ValueError(f"Le CSV doit contenir les colonnes : {required_cols}")

    df_meta = df_meta.loc[:,["artist", "track_title_clean"]].loc[:,["artist", "track_title_clean"]]

    # 2. Embeddings (N, 768)
    emb_df = ensure_embeddings()
    emb_np = emb_df.to_numpy(dtype=np.float32)

    # 3. Vérification de l'input utilisateur
    if not user_input or not isinstance(user_input, str):
        print("⚠️  Input utilisateur vide ou invalide.")
        return pd.DataFrame(columns=["artist", "track_title_clean", "score"])

    # 4. Modèle SBERT identique pour l’input
    # model = SentenceTransformer(MODEL_NAME, device=DEVICE, trust_remote_code=True)
    user_vec = model.encode(user_input,device=DEVICE)

    # 5. Cosine similarity
    scores = util.cos_sim(user_vec, emb_np)[0]

    # Gérer le cas où k est supérieur au nombre de lignes
    k_safe = min(k, scores.shape[0])
    top_scores = scores.topk(k_safe).indices
    top_scores = top_scores.tolist()


    # 6. Résultat sous forme de DataFrame
    top_df = df_meta.iloc[top_scores].copy()
    top_df['score'] = scores[top_scores].tolist()

    return top_df.reset_index(drop=True)
