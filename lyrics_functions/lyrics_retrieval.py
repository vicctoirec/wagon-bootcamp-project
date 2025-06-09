from cleaning import clean_text, clean_title
import time
import requests

def enrich_df_with_lyrics(df):
    lyrics = []

    for i, row in df.iterrows():
        artist = row['artist_name']
        raw_title = row['track_name']
        title = clean_title(raw_title)

        print(f"→ {i+1}/{len(df)} : {artist} - {title}")
        url = f"https://api.lyrics.ovh/v1/{artist}/{title}"

        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                raw_lyrics = response.json().get("lyrics", "")
                lyrics.append(clean_text(raw_lyrics))
                print("  ↳ Paroles trouvées")
            else:
                lyrics.append(None)
                print(f"  ↳ Paroles non trouvées (code {response.status_code})")
        except Exception as e:
            print(f"[!] Erreur : {e}")
            lyrics.append(None)

        time.sleep(0.1)

    df['lyrics']
