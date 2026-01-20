python
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import folium

# Load the dataset
data = pd.read_csv('Traffic_Accident_2025.csv')

# Preprocess the data
data['Accident_Date'] = pd.to_datetime(data['Accident_Date'])
data['Hour'] = data['Accident_Time'].apply(lambda x: int(x.split(':')[0]))
data['Weather_Condition'] = data['Weather_Condition'].fillna('Unknown')
data = pd.get_dummies(data, columns=['Weather_Condition'], drop_first=True)

# Clustering for accident hotspots
coordinates = data[['Latitude', 'Longitude']]
kmeans = KMeans(n_clusters=10, random_state=42).fit(coordinates)
data['Cluster'] = kmeans.labels_

# Visualization of hotspots
map_hotspots = folium.Map(location=[24.4539, 54.3773], zoom_start=10)
for _, row in data.iterrows():
    folium.CircleMarker(
        location=(row['Latitude'], row['Longitude']),
        radius=2,
        color='red',
        fill=True
    ).add_to(map_hotspots)
map_hotspots.save('hotspots_map.html')

# Train a predictive model
X = data[['Hour', 'Cluster'] + list(data.columns[data.columns.str.startswith('Weather_Condition_')])]
y = data['Accident_Severity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict accident severity
predictions = model.predict(X_test)
print("Model Accuracy:", model.score(X_test, y_test))
