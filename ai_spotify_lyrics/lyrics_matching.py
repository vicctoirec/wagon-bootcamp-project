# ------------------------------------------------------------------
# Charge le csv original "20250609_17k_lyrics_eng_fr.csv"
# Nettoie la colonne 'lyrics_clean'
# Charge le csv des embeddings "embedded_17klyrics.csv"
# Encode dynamiquement les requêtes utilisateur SBERT (nomic-ai/nomic-embed-text-v2-moe)
# Calcule la similarité cosinus entre les requêtes et les embeddings et renvoie les 50 meilleurs titres
# ------------------------------------------------------------------

from pathlib import Path
import pandas as pd, time
import numpy as np
import torch
from tqdm.auto import tqdm
from torch.nn.functional import normalize
from sentence_transformers import SentenceTransformer, util
from ai_spotify_lyrics.zeroshot_pipeline import ZeroShotLyrics, preprocess_lyrics

# ----------------------- PARAMÈTRES --------------------------------
BATCH_SIZE = 32  # Batch size for encoding
TOP_K = 50  # Number of top matches to return
# -------------------------------------------------------------------


class LyricsMatching:

    def __init__(self, model_path: str, model_name: str, embed_csv: str, raw_csv: str):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embed_csv = Path(embed_csv)
        self.raw_csv = Path(raw_csv)

        if Path(model_path).exists():
            self.model = SentenceTransformer(model_path, device=self.device)
        else:
            self.model = SentenceTransformer(model_name, device=self.device)
            self.model.save(model_path)


    def build_embeddings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build embeddings for the lyrics in the DataFrame using SBERT.

        Args:
            df (pd.DataFrame): DataFrame containing lyrics.

        Returns:
            pd.DataFrame: New DataFrame with embeddings.
        """

        # Encode the lyrics
        print("🔹 Encoding lyrics…")
        embs = self.model.encode(
            df["lyrics_clean"].tolist(),
            batch_size=BATCH_SIZE,
            show_progress_bar=True)

        # `embs` -> ndarray (N, 768) float32
        cols = [f"embedding_{i}" for i in range(embs.shape[1])]
        emb_df = pd.DataFrame(embs, columns=cols, index=df.index)

        return emb_df


    def ensure_embeddings(self) -> pd.DataFrame:
        """Charge l'embedding CSV s'il existe, sinon le crée et le sauvegarde."""

        if self.embed_csv.exists():
            print("✔️  Embeddings CSV trouvé ; chargement…")
            return pd.read_csv(self.embed_csv, index_col=0)

        else:
            print("⚠️  Embeddings CSV introuvable : génération en cours.")
            df_raw = pd.read_csv(self.raw_csv)
            df_raw = df_raw.dropna(subset=["lyrics_clean"])
            df_raw["lyrics_clean"] = df_raw["lyrics_clean"].apply(preprocess_lyrics)

            emb_df = self.build_embeddings(df_raw)
            emb_df.to_csv(self.embed_csv)

            print(f"✔️ Embeddings sauvegardés → {self.embed_csv}")
            return emb_df


    def get_top_k(self, user_input: str, k=TOP_K):
        """
        Trouve les k meilleurs titres correspondant à l'input utilisateur.

        Args:
            user_input (str): Input utilisateur pour la recherche.
            k (int): Nombre de résultats à retourner.

        Returns:
            pd.DataFrame: DataFrame contenant les artistes, titres et scores des correspondances."""

    # 1. Métadonnées
        df_meta = pd.read_csv(self.raw_csv)

        required_cols = {"artist", "track_title_clean"}
        if not required_cols.issubset(df_meta.columns):
            raise ValueError(f"Le CSV doit contenir les colonnes : {required_cols}")

        df_meta = df_meta.loc[:,["artist", "track_title_clean"]].loc[:,["artist", "track_title_clean"]]
        n_rows = len(df_meta)

        # 2. Embeddings (N, 768)
        emb_df = self.ensure_embeddings()
        emb_np = emb_df.to_numpy(dtype=np.float32)
        emb_t = torch.tensor(emb_np, device=self.device)
        emb_t = normalize(emb_t, dim=1)

        # 3. Vérification de l'input utilisateur
        if not user_input or not isinstance(user_input, str):
            print("⚠️  Input utilisateur vide ou invalide.")
            return pd.DataFrame(columns=["artist", "track_title_clean", "score"])

        # 4. Modèle SBERT identique pour l’input
        user_vec = self.model.encode(user_input,device=self.device, convert_to_tensor=True, normalize_embeddings=True)

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
    def refine_top_k(self,
                    enriched_input: str,
                    zeroshot_model: ZeroShotLyrics,
                    threshold : float =0.2,
                    k_recall : int =50,
                    k_final : int =10,
                    verbose : bool = True) -> pd.DataFrame:
        """
        Refine the top-k results by applying a zero-shot classification model.

        Parameters:
        - user_input (str): The user's input text.
        - k_recall (int): The number of top results to recall.
        - k_final (int): The number of final results to return.

        Returns:
        - pd.DataFrame: A DataFrame containing the refined top-k results.
        """

        t0 = time.perf_counter()

        # 1 ─ SBERT recall ---------------------------------------------------------
        data = self.get_top_k(enriched_input, k=k_recall)
        if data.empty:
            return data

        # Ajout des lyrics (pour Zero-Shot)
        full = pd.read_csv(self.raw_csv, usecols=["artist", "track_title_clean", "lyrics_clean"])
        data = data.merge(full, on=["artist", "track_title_clean"], how="left")
        print(data.shape)
        # 2 ─ ZS score -------------------------------------------------------------
        if verbose:
            print("⏳ Recherche des meilleurs matching titles..")
        tqdm_bar = tqdm(total=len(data), desc="Chargement de la playlist...", unit="song")
        zs_scores = []
        for txt in data["lyrics_clean"]:
            zs = zeroshot_model.get_zeroshot_score(txt, enriched_input)
            zs_scores.append(zs)
            tqdm_bar.update(1)
        tqdm_bar.close()
        data["zs_score"] = zs_scores

        # 3 ─ Tri final ------------------------------------------------------------
        data = data[data["zs_score"] >= threshold]
        top = (data.sort_values("zs_score", ascending=False)
                    .head(k_final)
                    .reset_index(drop=True))

        if verbose:
            dt = time.perf_counter() - t0
            print(f"🎉 Votre playlist est prête ! (temps total : {dt:,.1f} s)")

        return top[["artist", "track_title_clean", "zs_score"]]
