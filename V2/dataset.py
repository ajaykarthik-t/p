import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_petition_dataset(num_samples=1000):
    """
    Generate a synthetic dataset for petitions categorized by urgency level
    """
    
    categories = {
        "Public Safety": {"urgency": "High", "reasoning": "Direct impact on human life & security (crime, fire hazards, accidents)."},
        "Healthcare": {"urgency": "High", "reasoning": "Critical for life-saving interventions, emergency care, and hospital facilities."},
        "Transportation": {"urgency": "High", "reasoning": "Affects daily life, road safety, and emergency access."},
        "Infrastructure": {"urgency": "Medium", "reasoning": "Roads, bridges, public utilities—important but not immediate life threats."},
        "Environment": {"urgency": "Medium", "reasoning": "Climate concerns, pollution, and waste management impact long-term health."},
        "Housing & Shelter": {"urgency": "Medium", "reasoning": "Essential for quality of life, homelessness prevention, and affordable housing."},
        "Education": {"urgency": "Medium", "reasoning": "Long-term impact; crucial but not urgent like health/safety."},
        "Community Services": {"urgency": "Low", "reasoning": "Recreational spaces, libraries, cultural programs—not immediate needs."},
        "Economic Development": {"urgency": "Low", "reasoning": "Business support and employment issues are important but less urgent."}
    }
    
    issues = [
        "road maintenance", "public transportation", "waste management", "park facilities", "street lighting",
        "traffic signals", "sidewalk repairs", "bike lanes", "public safety", "noise pollution",
        "air quality", "water supply", "school facilities", "healthcare access", "senior services",
        "youth programs", "affordable housing", "small business support", "parking facilities", "community centers"
    ]
    
    locations = [
        "downtown", "north district", "south district", "west side", "east side", "central area",
        "suburban area", "business district", "residential area", "industrial zone"
    ]
    
    actions = [
        "improve", "implement", "upgrade", "maintain", "develop", "establish",
        "enhance", "renovate", "expand", "create"
    ]
    
    reasons = [
        "for better quality of life", "to ensure public safety", "for community development", "to meet growing demands",
        "to address resident concerns", "for environmental protection", "to support local economy",
        "for sustainable development", "to improve accessibility", "for future generations"
    ]
    
    data = []
    start_date = datetime.now() - timedelta(days=365)
    
    for _ in range(num_samples):
        issue = np.random.choice(issues)
        location = np.random.choice(locations)
        action = np.random.choice(actions)
        reason = np.random.choice(reasons)
        
        petition_text = f"Request to {action} {issue} in the {location} {reason}."
        category = np.random.choice(list(categories.keys()))
        urgency = categories[category]["urgency"]
        reasoning = categories[category]["reasoning"]
        
        submission_date = start_date + timedelta(days=np.random.randint(0, 365))
        days_since_submission = (datetime.now() - submission_date).days
        
        if days_since_submission > 30:
            status = np.random.choice(["Resolved", "Rejected", "In Progress"], p=[0.6, 0.1, 0.3])
        else:
            status = np.random.choice(["Under Review", "In Progress"], p=[0.7, 0.3])
        
        first_names = ["John", "Jane", "Michael", "Sarah", "David", "Lisa", "Robert", "Emily"]
        last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis"]
        
        name = f"{np.random.choice(first_names)} {np.random.choice(last_names)}"
        email = f"{name.lower().replace(' ', '.')}@example.com"
        
        data.append({
            "name": name,
            "email": email,
            "petition": petition_text,
            "category": category,
            "urgency_level": urgency,
            "reasoning": reasoning,
            "status": status,
            "created_at": submission_date,
            "votes": np.random.randint(0, 1000)
        })
    
    df = pd.DataFrame(data)
    df.to_csv("petition_data_updated.csv", index=False)
    print(f"Generated {num_samples} petition records and saved to petition_data_updated.csv")
    
    return df

if __name__ == "__main__":
    df = generate_petition_dataset(1000)
    print(df.head())
