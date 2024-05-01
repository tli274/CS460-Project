
import os
import numpy as np
import pandas as pd
import json
import torch
from torch import nn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Define a function to load a JSON file into a DataFrame
def load_json_to_df(file_path: str) -> pd.DataFrame:
    '''
    Loads JSON files into a pandas DataFrame

    Parameters:
    - file_path (str): File path of JSON file to load into DataFrame

    Returns:
    - DataFrame: DataFrame with tracks data from file_path
    '''
    with open(file_path, 'r') as file:
        data = json.load(file)
    tracks = []
    for playlist in data['playlists']:
        pid = playlist['pid']
        for track in playlist['tracks']:
            track['playlist_id'] = pid  # Add playlist ID to each track
            tracks.append(track)
    return pd.DataFrame(tracks)

# Define the data directory
data_dir = 'data/.'

# List to store DataFrames
data_partitions = []

# Loop through each JSON file in the directory
for filename in os.listdir(data_dir):
    if filename.endswith('.json'):
        file_path = os.path.join(data_dir, filename)
        data_partition = load_json_to_df(file_path)
        data_partitions.append(data_partition)

# Concatenate all DataFrames into a single DataFrame
data = pd.concat(data_partitions, ignore_index=True)

# Create a new DataFrame with only the relevant features
df = data[['playlist_id', 'pos', 'artist_name', 'album_name', 'track_name', 'duration_ms']]

# Encode categorical features
label_encoders = {}
def encode_feature(feature):
    global label_encoders
    
    if feature.name in label_encoders:
        label_encoder = label_encoders[feature.name]
    else:
        label_encoder = LabelEncoder()
        label_encoder.fit(feature)
        label_encoders[feature.name] = label_encoder
    
    encoded_feature = label_encoder.transform(feature)
    encoded_feature = encoded_feature.reshape(-1, 1)
    
    return encoded_feature

# Encode each categorical feature separately
encoded_artist_name = encode_feature(df['artist_name'])
encoded_album_name = encode_feature(df['album_name'])
encoded_track_name = encode_feature(df['track_name'])

# Concatenate encoded features horizontally
encoded_features = np.hstack((encoded_artist_name, encoded_album_name, encoded_track_name))

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(encoded_features, df['pos'], test_size=0.2, random_state=42)

# Convert the data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# Define the RNN model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out[:, -1, :])
        return out
    
    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)

# Define the model parameters
input_size = encoded_features.shape[1]  
hidden_size = 128  
output_size = 1  

# Instantiate the model
model = RNN(input_size, hidden_size, output_size)

# Define loss function and optimizer
criterion = nn.MSELoss()  
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  
    optimizer.zero_grad()  
    outputs = model(X_train_tensor)  
    loss = criterion(outputs.squeeze(), y_train_tensor)  
    loss.backward()  
    optimizer.step()  
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
model.eval() 
with torch.no_grad():
    outputs = model(X_test_tensor)  
    test_loss = criterion(outputs.squeeze(), y_test_tensor)  
    print(f'Test Loss: {test_loss.item():.4f}')  
