import pandas as pd

# Load dataset
df = pd.read_csv('restaurant.csv')
print(df.columns.tolist())
# Drop rows with missing coordinates
df = df.dropna(subset=['Latitude', 'Longitude'])
df.head()
import folium
import plotly.express as px
#1. Visualize restaurant locations on a map
map_center = [df['Latitude'].mean(), df['Longitude'].mean()]
restaurant_map = folium.Map(location=map_center, zoom_start=12)

for _, row in df.iterrows():
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=f"{row['Restaurant Name']} ({row['Aggregate rating']})",
        tooltip=row['Cuisines']
    ).add_to(restaurant_map)

restaurant_map.save('restaurant_map.html')
# 2. Group by city/locality and analyze concentration

grouped = df.groupby(['City', 'Locality']).agg({
    'Restaurant Name': 'count',
    'Aggregate rating': 'mean',
    'Price range': 'mean'
}).reset_index().rename(columns={'Restaurant Name': 'restaurant_count'})
#3. Plot concentration of restaurants

fig = px.bar(grouped.sort_values('restaurant_count', ascending=False),
             x='Locality', y='restaurant_count',
             color='City', title='Restaurant Concentration by Locality')
fig.write_html('restaurant_concentration.html')
# 4. Cuisine distribution by city

cuisine_city = df.groupby(['City', 'Cuisines']).size().reset_index(name='count')
fig2 = px.sunburst(cuisine_city, path=['City', 'Cuisines'], values='count',
                   title='Cuisine Distribution by City')
fig2.write_html('cuisine_distribution.html')
# 5. Insights

top_localities = grouped.sort_values('restaurant_count', ascending=False).head(5)
print("Top 5 localities with highest restaurant density:")
print(top_localities[['City', 'Locality', 'restaurant_count']])

print("\nAverage ratings and price ranges by city:")
print(grouped.groupby('City')[['Aggregate rating', 'Price range']].mean().round(2))

