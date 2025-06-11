from cleaning import clean_text, clean_title
import time
import requests
import random
import pandas as pd

def get_lyrics(artist, title):
    url = f"https://api.lyrics.ovh/v1/{artist}/{title}"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.json().get('lyrics', None)
        else:
            print(f"Erreur HTTP {response.status_code} pour {artist} - {title}")
            return None
    except Exception as e:
        print(f"Exception pour {artist} - {title}: {e}")
        return None


def update_lyrics_column(df):
    for i, row in df.iterrows():
        if pd.notna(row.get('lyrics')) and str(row['lyrics']).strip() != '':
            continue

        artist = row['artist_name']
        title = row['title_clean']

        print(f"→ {i+1}/{len(df)} : {artist} - {title}")
        raw_lyrics = get_lyrics(artist, title)
        cleaned_lyrics = clean_text(raw_lyrics)
        print("  ↳", "Paroles trouvées" if cleaned_lyrics else "Paroles introuvables")

        if cleaned_lyrics:
            df.at[i, 'lyrics'] = cleaned_lyrics

        time.sleep(random.uniform(0.1, 0.3))  # Pause aléatoire

    return df
