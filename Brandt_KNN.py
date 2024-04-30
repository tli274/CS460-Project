import json
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Load the Spotify Million Playlist dataset from JSON
def load_dataset(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Extract features from playlists and tracks
def extract_features(data):
    playlists = []
    track_features = []

    for playlist in data['playlists']:
        playlists.append(playlist['name'])
        tracks = playlist['tracks']
        for track in tracks:
            track_features.append([
                track['track_name'],
                track['artist_name'],
                track['album_name']
            ])
    
    return playlists, track_features

# Perform KNN to find similar tracks
def knn_recommendation(seed_track, track_features, k=5):
    knn = NearestNeighbors(n_neighbors=k, algorithm='ball_tree')
    knn.fit(track_features)
    distances, indices = knn.kneighbors([seed_track])
    return distances, indices

if __name__ == "__main__":
    # Load the dataset
    dataset_path = 'spotify_million_playlist.json'
    dataset = load_dataset(dataset_path)

    # Extract features
    playlists, track_features = extract_features(dataset)

    # Example seed track
    seed_track = ['Seed Track Name', 'Seed Artist Name', 'Seed Album Name']

    # Perform KNN recommendation
    distances, indices = knn_recommendation(seed_track, track_features)

    # Print recommended tracks
    print("Recommended Tracks:")
    for i in indices[0]:
        print(track_features[i])

