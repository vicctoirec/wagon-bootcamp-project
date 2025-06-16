
import pandas as pd

from ai_spotify_lyrics.params import DATA_CSV_INIT


def initialize_dummy_model():
    """
    returns a dummy model with a predict method returning 3 random artist-songs
    """

    class DummyModel():
        """
        DummyModel class
        with a predict method
        """

        def __init__(self, csv_path: str):
            """
            load data from a CSV file
            """
            self.data = pd.read_csv(csv_path)

        def predict(self, input: str, nb_songs: int=3):
            return self.data.sample(nb_songs)[['artist', 'track_name']].values.tolist()


    model = DummyModel(DATA_CSV_INIT)

    return model




if __name__ == "__main__":
    model = initialize_dummy_model()
    predictions = model.predict('', 5)
    print(predictions)
