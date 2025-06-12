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
from torch.nn.functional import normalize
from sentence_transformers import SentenceTransformer, util
from zeroshots_function.zeroshot_pipeline import preprocess_lyrics, get_zeroshot_score

# ----------------------- PARAMÈTRES --------------------------------
EMBD_CSV  = Path("../raw_data/embedded_17klyrics.csv")
RAW_CSV = Path('../raw_data/data_17k_lyrics.csv')
MODEL_NAME = "nomic-ai/nomic-embed-text-v2-moe"  # SBERT model
BATCH_SIZE = 32  # Batch size for encoding
TOP_K = 50  # Number of top matches to return
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# -------------------------------------------------------------------


def build_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build embeddings for the lyrics in the DataFrame using SBERT.

    Args:
        df (pd.DataFrame): DataFrame containing lyrics.

    Returns:
        pd.DataFrame: New DataFrame with embeddings.
    """
    # Initialize the SBERT model
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)

    # Encode the lyrics
    print("🔹 Encoding lyrics…")
    embs = model.encode(
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


def get_top_k(user_input: str, k=TOP_K):
    """
    Trouve les k meilleurs titres correspondant à l'input utilisateur.

    Args:
        user_input (str): Input utilisateur pour la recherche.
        k (int): Nombre de résultats à retourner.

    Returns:
        pd.DataFrame: DataFrame contenant les artistes, titres et scores des correspondances."""

 # 1. Métadonnées
    df_meta = pd.read_csv(RAW_CSV)

    required_cols = {"artist", "track_title_clean"}
    if not required_cols.issubset(df_meta.columns):
        raise ValueError(f"Le CSV doit contenir les colonnes : {required_cols}")

    df_meta = df_meta.loc[:,["artist", "track_title_clean"]].loc[:,["artist", "track_title_clean"]]
    n_rows = len(df_meta)

    # 2. Embeddings (N, 768)
    emb_df = ensure_embeddings()
    emb_np = emb_df.to_numpy(dtype=np.float32)
    emb_t = torch.tensor(emb_np, device=DEVICE)
    emb_t = normalize(emb_t, dim=1)  # Normalisation des embeddings

    # 3. Vérification de l'input utilisateur
    if not user_input or not isinstance(user_input, str):
        print("⚠️  Input utilisateur vide ou invalide.")
        return pd.DataFrame(columns=["artist", "track_title_clean", "score"])

    # 4. Modèle SBERT identique pour l’input
    model = SentenceTransformer(MODEL_NAME, device=DEVICE, trust_remote_code=True)
    user_vec = model.encode(user_input,device=DEVICE, convert_to_tensor=True, normalize_embeddings=True)

    # 5. Cosine similarity
    scores = util.cos_sim(user_vec, emb_t)[0]

    # Gérer le cas où k est supérieur au nombre de lignes
    k_safe = min(k, n_rows)
    top_scores = scores.topk(k_safe).indices.cpu().numpy()


    # 6. Résultat sous forme de DataFrame
    top_df = df_meta.iloc[top_scores].copy()
    top_df['score'] = scores[top_scores].cpu().numpy()

    return top_df.reset_index(drop=True)

# --------------- FONCTION PRINCIPALE -------------------

def refine_top_k(user_input: str, threshold : float =0.8, k_recall : int =100, k_final : int =10):
    """
    Refine the top-k results by applying a zero-shot classification model.

    Parameters:
    - user_input (str): The user's input text.
    - k_recall (int): The number of top results to recall.
    - k_final (int): The number of final results to return.

    Returns:
    - pd.DataFrame: A DataFrame containing the refined top-k results.
    """

    # 1 ─ SBERT recall --------------------------------------------------------
    data = get_top_k(user_input, k=k_recall)
    if data.empty:
        return data

    full = pd.read_csv(RAW_CSV, usecols=["artist", "track_title_clean", "lyrics_clean"])
    data = data.merge(full, on=["artist", "track_title_clean"], how="left")

    # 2 ─ ZS score -----------------------------------------------------------
    data["zs_score"] = data["lyrics_clean"].apply(
        lambda txt: get_zeroshot_score(txt, user_input)
    )

    # 3 ─ Tri final ----------------------------------------------------------
    data = data[data["zs_score"] >= threshold]
    top = (data.sort_values("zs_score", ascending=False)
                .head(k_final)
                .reset_index(drop=True))

    return top[["artist", "track_title_clean", "zs_score"]]
