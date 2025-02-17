import csv
import random

# Define categories and urgency levels
categories = [
    "Public Safety", "Healthcare", "Transportation", "Infrastructure",
    "Environment", "Housing & Shelter", "Education", "Community Services",
    "Economic Development"
]

urgency_levels = ["High", "Medium", "Low"]

# Define sample reasoning and petitions
reasoning_templates = {
    "Public Safety": "Direct impact on human life & security (crime, fire hazards, accidents).",
    "Healthcare": "Critical for life-saving interventions, emergency care, and hospital facilities.",
    "Transportation": "Affects daily life, road safety, and emergency access.",
    "Infrastructure": "Roads, bridges, public utilities—important but not immediate life threats.",
    "Environment": "Climate concerns, pollution, and waste management impact long-term health.",
    "Housing & Shelter": "Essential for quality of life, homelessness prevention, and affordable housing.",
    "Education": "Long-term impact; crucial but not urgent like health/safety.",
    "Community Services": "Recreational spaces, libraries, cultural programs—not immediate needs.",
    "Economic Development": "Business support and employment issues are important but less urgent."
}

petition_templates = {
    "Public Safety": "We request increased police patrols and better street lighting to reduce crime.",
    "Healthcare": "We urge the government to allocate more funds for emergency healthcare services.",
    "Transportation": "We demand better road maintenance and improved public transportation.",
    "Infrastructure": "We request funding for bridge repairs and water supply improvements.",
    "Environment": "We call for stricter regulations on industrial pollution.",
    "Housing & Shelter": "We request the construction of more affordable housing units.",
    "Education": "We demand better school infrastructure and updated learning materials.",
    "Community Services": "We seek funding to build a new community library and cultural center.",
    "Economic Development": "We propose tax incentives for small businesses."
}

# Generate dataset
data = []
for _ in range(1000):
    category = random.choice(categories)
    urgency = random.choice(urgency_levels)
    reasoning = reasoning_templates[category]
    petition = petition_templates[category]
    data.append([category, urgency, reasoning, petition])

# Write to CSV file
csv_filename = "category_urgency_dataset.csv"
with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Category", "Urgency Level", "Reasoning", "Sample Petition"])  # Header
    writer.writerows(data)

print(f"CSV file '{csv_filename}' created successfully!")
