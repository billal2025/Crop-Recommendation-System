import pandas as pd
import random

# Define possible values
crops = ["Corn", "Wheat", "Rice", "Potato", "Tomato"]
problems = ["Pest infestation", "Nutrient deficiency", "Drought", "Fungal disease", "Weed overgrowth"]
seasons = ["Spring", "Summer", "Autumn", "Winter"]
locations = ["Region A", "Region B", "Region C", "Region D"]
weather_conditions = ["Hot", "Humid", "Dry", "Rainy"]
soil_types = ["Clay", "Sandy", "Loamy", "Silty", "Peaty", "Chalky", "Saline", "Laterite", "Black"]
pest_types = ["Aphids", "Cutworms", "Whiteflies", "Spider Mites", "Root-Knot Nematodes",
              "Armyworms", "Stink Bugs", "Thrips", "Cabbage Loopers", "Fruit Flies"]

recommendations = {
    "Pest infestation": "Use appropriate pesticide XYZ based on pest type",
    "Nutrient deficiency": "Apply nitrogen-based fertilizer",
    "Drought": "Increase irrigation frequency",
    "Fungal disease": "Use antifungal spray ABC",
    "Weed overgrowth": "Manual weeding or herbicide application"
}

# Generate random fake data
data = []
num_samples = 200

for _ in range(num_samples):
    crop = random.choice(crops)
    problem = random.choice(problems)
    season = random.choice(seasons)
    location = random.choice(locations)
    weather = random.choice(weather_conditions)
    soil = random.choice(soil_types)
    pest = random.choice(pest_types) if problem == "Pest infestation" else "None"
    recommendation = recommendations[problem]
    data.append([crop, problem, season, location, weather, soil, pest, recommendation])

# Create a DataFrame
df = pd.DataFrame(data, columns=["Crop", "Problem", "Season", "Location", "Weather",
                                  "Soil Type", "Pest Type", "Recommendation"])

# Save to CSV
df.to_csv("enhanced_synthetic_crop_data.csv", index=False)
print("Enhanced data generated and saved.")
