# dataset_generator.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_petition_dataset(num_samples=1000):
    """
    Generate a synthetic dataset for petitions with realistic content
    """
    
    # Define base components for generating petitions
    issues = [
        "road maintenance",
        "public transportation",
        "waste management",
        "park facilities",
        "street lighting",
        "traffic signals",
        "sidewalk repairs",
        "bike lanes",
        "public safety",
        "noise pollution",
        "air quality",
        "water supply",
        "school facilities",
        "healthcare access",
        "senior services",
        "youth programs",
        "affordable housing",
        "small business support",
        "parking facilities",
        "community centers"
    ]
    
    locations = [
        "downtown",
        "north district",
        "south district",
        "west side",
        "east side",
        "central area",
        "suburban area",
        "business district",
        "residential area",
        "industrial zone"
    ]
    
    actions = [
        "improve",
        "implement",
        "upgrade",
        "maintain",
        "develop",
        "establish",
        "enhance",
        "renovate",
        "expand",
        "create"
    ]
    
    reasons = [
        "for better quality of life",
        "to ensure public safety",
        "for community development",
        "to meet growing demands",
        "to address resident concerns",
        "for environmental protection",
        "to support local economy",
        "for sustainable development",
        "to improve accessibility",
        "for future generations"
    ]
    
    # Define categories and their weights
    categories = {
        "Infrastructure": 0.25,
        "Transportation": 0.15,
        "Environment": 0.12,
        "Public Safety": 0.13,
        "Education": 0.10,
        "Healthcare": 0.10,
        "Community Services": 0.08,
        "Economic Development": 0.07
    }
    
    # Define departments
    departments = {
        "Infrastructure": ["Public Works", "City Planning"],
        "Transportation": ["Transportation", "Traffic Management"],
        "Environment": ["Environmental Protection", "Parks and Recreation"],
        "Public Safety": ["Public Safety", "Emergency Services"],
        "Education": ["Education", "Youth Services"],
        "Healthcare": ["Health Services", "Public Health"],
        "Community Services": ["Community Development", "Social Services"],
        "Economic Development": ["Economic Development", "Business Relations"]
    }
    
    # Generate synthetic data
    data = []
    start_date = datetime.now() - timedelta(days=365)
    
    for _ in range(num_samples):
        # Generate petition text
        issue = np.random.choice(issues)
        location = np.random.choice(locations)
        action = np.random.choice(actions)
        reason = np.random.choice(reasons)
        
        petition_text = f"Request to {action} {issue} in the {location} {reason}."
        
        # Select category based on weights
        category = np.random.choice(
            list(categories.keys()),
            p=list(categories.values())
        )
        
        # Select department based on category
        department = np.random.choice(departments[category])
        
        # Generate status based on date
        submission_date = start_date + timedelta(
            days=np.random.randint(0, 365)
        )
        days_since_submission = (datetime.now() - submission_date).days
        
        if days_since_submission > 30:
            status = np.random.choice(
                ["Resolved", "Rejected", "In Progress"],
                p=[0.6, 0.1, 0.3]
            )
        else:
            status = np.random.choice(
                ["Under Review", "In Progress"],
                p=[0.7, 0.3]
            )
        
        # Determine priority
        if "safety" in petition_text.lower() or "emergency" in petition_text.lower():
            priority = "High"
        elif "improve" in petition_text.lower() or "upgrade" in petition_text.lower():
            priority = "Medium"
        else:
            priority = "Low"
        
        # Generate petitioner details
        first_names = ["John", "Jane", "Michael", "Sarah", "David", "Lisa", "Robert", "Emily"]
        last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis"]
        
        name = f"{np.random.choice(first_names)} {np.random.choice(last_names)}"
        email = f"{name.lower().replace(' ', '.')}@example.com"
        
        # Add to dataset
        data.append({
            "name": name,
            "email": email,
            "petition": petition_text,
            "category": category,
            "department": department,
            "status": status,
            "priority": priority,
            "created_at": submission_date,
            "votes": np.random.randint(0, 1000)
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv("petition_data.csv", index=False)
    print(f"Generated {num_samples} petition records and saved to petition_data.csv")
    
    return df

if __name__ == "__main__":
    # Generate dataset with 1000 samples
    df = generate_petition_dataset(1000)
    
    # Print sample statistics
    print("\nDataset Statistics:")
    print("-" * 50)
    print(f"Total number of petitions: {len(df)}")
    print("\nCategory Distribution:")
    print(df['category'].value_counts(normalize=True))
    print("\nPriority Distribution:")
    print(df['priority'].value_counts(normalize=True))
    print("\nStatus Distribution:")
    print(df['status'].value_counts(normalize=True))