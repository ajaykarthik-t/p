# dataset.py

import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta
import json

# Create directory for data if it doesn't exist
os.makedirs('data', exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# List of departments
departments = [
    "Education", "Health", "Transport", "Taxation", 
    "Housing", "Energy", "Water", "Agriculture", "Infrastructure"
]

# Function to generate random names
def generate_name():
    first_names = ["John", "Mary", "James", "Patricia", "Robert", "Jennifer", "Michael", "Linda", 
                  "William", "Elizabeth", "David", "Susan", "Richard", "Jessica", "Joseph", "Sarah"]
    last_names = ["Smith", "Johnson", "Williams", "Jones", "Brown", "Davis", "Miller", "Wilson",
                 "Moore", "Taylor", "Anderson", "Thomas", "Jackson", "White", "Harris", "Martin"]
    return f"{random.choice(first_names)} {random.choice(last_names)}"

# Function to generate random phone numbers
def generate_phone():
    return f"+1-{random.randint(200, 999)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}"

# Function to generate random addresses
def generate_address():
    street_numbers = list(range(1, 200))
    street_names = ["Main", "Oak", "Pine", "Maple", "Cedar", "Elm", "Washington", "Park", 
                   "Lake", "Hill", "River", "Spring", "Garden", "Sunset", "Meadow", "Forest"]
    street_types = ["St", "Ave", "Blvd", "Ln", "Dr", "Way", "Pl", "Ct"]
    cities = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia", 
              "San Antonio", "San Diego", "Dallas", "San Jose", "Austin", "Jacksonville"]
    states = ["NY", "CA", "IL", "TX", "AZ", "PA", "FL", "OH", "GA", "NC", "MI", "NJ"]
    zip_codes = [f"{random.randint(10000, 99999)}" for _ in range(20)]
    
    street = f"{random.choice(street_numbers)} {random.choice(street_names)} {random.choice(street_types)}"
    city_state_zip = f"{random.choice(cities)}, {random.choice(states)} {random.choice(zip_codes)}"
    
    return f"{street}, {city_state_zip}"

# Example petition templates for each department
department_templates = {
    "Education": [
        "I would like to request improvement in the school curriculum for {subject}. The current material is outdated and doesn't reflect modern standards.",
        "Our local school {school_name} is facing severe shortage of qualified teachers for {subject}. We need immediate intervention.",
        "The school bus service in our area is unreliable. Children often arrive late to {school_name} and miss important classes.",
        "I'm concerned about the lack of special education resources at {school_name}. My child needs additional support but the school doesn't provide adequate facilities.",
        "The school infrastructure at {school_name} is deteriorating. The classrooms are overcrowded and facilities are in poor condition."
    ],
    "Health": [
        "I've been waiting for {waiting_time} months for my appointment at {hospital_name}. This delay is affecting my health condition severely.",
        "The local clinic in our area lacks basic medical equipment. We need to travel {distance} miles to access proper healthcare.",
        "There's a shortage of {medicine_name} in all pharmacies in our district. This medication is essential for many patients with chronic conditions.",
        "The sanitation around {location} is poor and might lead to disease outbreaks. The garbage collection service is inadequate.",
        "We need more mental health professionals in our community. The waiting list for counseling services is over {waiting_time} months long."
    ],
    "Transport": [
        "The public bus service on route {route_number} is consistently late, causing many workers to miss their shifts.",
        "The road condition on {street_name} is dangerous with numerous potholes that have caused multiple accidents.",
        "We need traffic lights at the intersection of {street_name} and {cross_street}. It's become a hazardous spot for pedestrians.",
        "The train service between {location_1} and {location_2} has reduced its frequency, causing significant inconvenience to commuters.",
        "Parking facilities near {location} are insufficient. Local businesses are suffering as customers can't find places to park."
    ],
    "Taxation": [
        "I believe there's an error in my property tax assessment. My property value is overstated by approximately {percentage}%.",
        "The new tax on {item} is adversely affecting small businesses in our community. Many are struggling to maintain profitability.",
        "I submitted my tax refund application {time_period} months ago but have not received any response or refund.",
        "The local business tax rate is disproportionately high compared to neighboring districts, discouraging new businesses from setting up here.",
        "I need clarification on the recent changes to the tax code regarding {specific_tax}. The guidelines published are confusing."
    ],
    "Housing": [
        "Our apartment building at {address} has been without proper heating for {time_period} weeks. The landlord is not responding to our complaints.",
        "Housing prices in {location} have become unaffordable for local residents. We need more affordable housing options.",
        "The public housing waitlist is over {waiting_time} years long. Many families are living in inadequate conditions while waiting.",
        "A construction project near {location} is not following building codes and is causing danger to nearby residents.",
        "Our neighborhood lacks adequate housing for seniors. We need accessible and affordable options for our aging population."
    ],
    "Energy": [
        "We've been experiencing frequent power outages in {location}, especially during {weather_condition}. This is affecting both households and businesses.",
        "The electricity rates have increased by {percentage}% in the last year without any improvement in service quality.",
        "Our community is interested in renewable energy options, but the current regulations make it difficult to install solar panels.",
        "The street lights on {street_name} have been non-functional for {time_period} months, creating safety concerns at night.",
        "The energy company is not providing accurate billing. Many residents are receiving inflated bills without explanation."
    ],
    "Water": [
        "The water quality in {location} has deteriorated. It has a strange {characteristic} and might be unsafe for consumption.",
        "We've been experiencing low water pressure in {location} for {time_period} weeks. This is affecting daily activities.",
        "The water bills have increased by {percentage}% without any prior notice or explanation from the water department.",
        "There's a major water leak on {street_name} that has been unaddressed for {time_period} days, wasting a significant amount of water.",
        "During heavy rain, our area at {location} experiences flooding due to poor drainage systems."
    ],
    "Agriculture": [
        "The recent {weather_condition} has damaged crops in our region. Farmers need immediate financial assistance to recover from the losses.",
        "Agricultural subsidies for {crop_type} farmers have been reduced, making it difficult to maintain profitable operations.",
        "Irrigation facilities in {location} are outdated and insufficient. Farmers are unable to properly water their crops during dry seasons.",
        "We need better market access for local farmers. Many are unable to sell their produce at fair prices due to lack of connectivity.",
        "The quality of agricultural supplies provided by the government program is substandard. Seeds for {crop_type} have poor germination rates."
    ],
    "Infrastructure": [
        "The bridge on {street_name} is in dangerous condition. Several cracks have appeared, and it needs immediate repair.",
        "Our community lacks proper sidewalks, making it unsafe for pedestrians, especially children walking to {location}.",
        "The public library building in {location} needs renovation. Its facilities are outdated and insufficient for community needs.",
        "Internet connectivity in our rural area is extremely poor. We need better infrastructure to support remote work and education.",
        "The public park at {location} is neglected and unsafe. The playground equipment is broken, and the area is poorly lit at night."
    ]
}

# Variables to fill in the templates
fill_variables = {
    "subject": ["mathematics", "science", "history", "physical education", "arts", "English literature", "computer science"],
    "school_name": ["Lincoln High School", "Washington Elementary", "Jefferson Middle School", "Roosevelt Academy", "Kennedy Public School"],
    "waiting_time": [3, 4, 6, 8, 12, 18],
    "hospital_name": ["City General Hospital", "Memorial Medical Center", "Community Health Clinic", "University Hospital", "Regional Medical Center"],
    "distance": [10, 15, 20, 30, 45, 60],
    "medicine_name": ["insulin", "antibiotics", "hypertension medication", "pain relievers", "asthma inhalers", "diabetes medication"],
    "location": ["Downtown", "West End", "North Side", "South Community", "East Village", "Central District", "Riverside", "Hillcrest"],
    "route_number": [101, 202, 303, 404, 505, 606, 707, 808],
    "street_name": ["Main Street", "Oak Avenue", "Pine Boulevard", "Maple Lane", "Cedar Drive", "Elm Way", "Washington Place", "Park Court"],
    "cross_street": ["1st Avenue", "2nd Street", "3rd Boulevard", "4th Lane", "5th Drive", "6th Way", "7th Place", "8th Road"],
    "location_1": ["Downtown", "West End", "North Side", "South Community", "East Village"],
    "location_2": ["Central District", "Riverside", "Hillcrest", "University Area", "Industrial Zone"],
    "percentage": [10, 15, 20, 25, 30, 40, 50],
    "time_period": [2, 3, 4, 6, 8, 12],
    "specific_tax": ["capital gains", "property", "sales", "income", "business", "inheritance", "luxury goods"],
    "address": ["123 Pine St", "456 Oak Ave", "789 Maple Blvd", "321 Cedar Ln", "654 Elm Dr"],
    "waiting_time": [1, 2, 3, 5, 7, 10],
    "weather_condition": ["winter", "summer", "rainy season", "storm", "heat wave", "cold snap"],
    "characteristic": ["odor", "color", "taste", "sediment", "cloudiness"],
    "crop_type": ["corn", "wheat", "soybean", "rice", "cotton", "vegetables", "fruits"],
}

# Function to fill template with random values
def fill_template(template):
    for key, values in fill_variables.items():
        if "{" + key + "}" in template:
            template = template.replace("{" + key + "}", str(random.choice(values)))
    return template

# Function to generate dataset with synthetic petition data
def generate_dataset(num_petitions=500):
    petitions = []
    
    # Generate submission dates in the last 90 days
    current_date = datetime.now()
    submission_dates = [(current_date - timedelta(days=random.randint(0, 90))).strftime("%Y-%m-%d %H:%M:%S") 
                        for _ in range(num_petitions)]
    
    for i in range(num_petitions):
        # Select department and random template for that department
        department = random.choice(departments)
        template = random.choice(department_templates[department])
        
        # Fill template with random values
        description = fill_template(template)
        
        # Determine urgency based on content and random factor
        # Words that might indicate urgency
        urgency_indicators = ["immediately", "urgent", "emergency", "critical", "severe", "dangerous", "safety", "hazard", "life-threatening"]
        
        # Check if any urgency indicators are in the description
        has_urgency_words = any(word in description.lower() for word in urgency_indicators)
        
        # 30% chance of high urgency if urgency words are present, 10% chance otherwise
        urgency_probability = 0.3 if has_urgency_words else 0.1
        urgency = "High" if random.random() < urgency_probability else "Normal"
        
        # Generate petition data
        petition = {
            'id': f"{i+1:08d}",
            'name': generate_name(),
            'phone': generate_phone(),
            'address': generate_address(),
            'description': description,
            'department': department,
            'urgency': urgency,
            'status': random.choice(["In Progress", "Resolved"]),
            'date_submitted': submission_dates[i],
            'submitted_by': f"user_{random.randint(1, 100)}"
        }
        
        petitions.append(petition)
    
    # Convert to DataFrame
    df = pd.DataFrame(petitions)
    
    # Save to CSV
    df.to_csv('data/training_petitions.csv', index=False)
    
    # Create a JSON file with petition templates and keywords for each department
    department_keywords = {}
    for dept, templates in department_templates.items():
        # Extract key words from templates
        all_words = ' '.join(templates).lower()
        words = [word for word in all_words.split() if len(word) > 4]  # Only consider words longer than 4 characters
        # Count frequency
        from collections import Counter
        word_counts = Counter(words)
        # Get top 10 keywords
        keywords = [word for word, _ in word_counts.most_common(10)]
        department_keywords[dept] = keywords
    
    # Save keywords to JSON
    with open('data/department_keywords.json', 'w') as f:
        json.dump(department_keywords, f, indent=4)
    
    print(f"Generated {num_petitions} synthetic petitions and saved to data/training_petitions.csv")
    print(f"Generated department keywords and saved to data/department_keywords.json")
    
    return df

if __name__ == "__main__":
    # Generate dataset with 500 synthetic petitions
    df = generate_dataset(500)
    
    # Print sample
    print("\nSample petitions:")
    print(df.sample(5)[['description', 'department', 'urgency']])