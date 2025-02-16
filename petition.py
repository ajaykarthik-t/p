import random

def generate_test_petition():
    # Templates for different parts of petitions
    issues = ["traffic signal", "playground", "street light", "sidewalk", "park facility"]
    locations = ["Main Street", "Central Park", "Oak Avenue", "Downtown", "West District"]
    conditions = ["broken", "damaged", "outdated", "insufficient", "malfunctioning"]
    
    # High priority templates
    high_template = [
        f"EMERGENCY: Safety hazard - {random.choice(conditions)} {random.choice(issues)} at {random.choice(locations)}",
        f"Urgent safety concern regarding {random.choice(conditions)} {random.choice(issues)} near {random.choice(locations)}",
    ]
    
    # Medium priority templates
    medium_template = [
        f"Improve the {random.choice(issues)} conditions at {random.choice(locations)}",
        f"Upgrade needed for {random.choice(conditions)} {random.choice(issues)} in {random.choice(locations)}",
    ]
    
    # Low priority templates
    low_template = [
        f"Request for new {random.choice(issues)} installation at {random.choice(locations)}",
        f"Suggestion to add more {random.choice(issues)} in {random.choice(locations)}",
    ]
    
    # Generate a random petition
    priority = random.choice(['high', 'medium', 'low'])
    if priority == 'high':
        petition = random.choice(high_template)
    elif priority == 'medium':
        petition = random.choice(medium_template)
    else:
        petition = random.choice(low_template)
        
    return petition

# Generate 5 random petitions
for i in range(5):
    print(f"\nPetition {i+1}:")
    print(generate_test_petition())
    print("-" * 50)