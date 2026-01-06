import pandas as pd
import numpy as np
import random
from datetime import datetime

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_house_price_dataset(num_samples=800):
    """
    Generate realistic Indian house price dataset
    """
    
    # City definitions with base price per sqft
    cities_tier1 = ['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 'Pune', 'Kolkata']
    cities_tier2 = ['Ahmedabad', 'Chandigarh', 'Jaipur', 'Lucknow', 'Kochi', 'Bhubaneswar', 'Indore']
    premium_locations = ['Mumbai_Bandra', 'Delhi_VasantVihar', 'Bangalore_Whitefield', 
                        'Hyderabad_Gachibowli', 'Gurgaon_DLF', 'Noida_Sector150']
    
    # Base price per sqft in ₹ (2024 realistic rates)
    base_rate_per_sqft = {
        'Mumbai': 18000, 'Mumbai_Bandra': 35000,
        'Delhi': 14000, 'Delhi_VasantVihar': 28000,
        'Bangalore': 12000, 'Bangalore_Whitefield': 18000,
        'Hyderabad': 10000, 'Hyderabad_Gachibowli': 16000,
        'Chennai': 9500,
        'Pune': 10000,
        'Kolkata': 9000,
        'Ahmedabad': 8000,
        'Chandigarh': 8500,
        'Jaipur': 7500,
        'Lucknow': 7000,
        'Kochi': 8000,
        'Bhubaneswar': 7500,
        'Indore': 7000,
        'Gurgaon_DLF': 22000,
        'Noida_Sector150': 16000
    }
    
    # Property types
    property_types = ['apartment', 'independent_house', 'villa']
    
    # Furnishing status
    furnishing_statuses = ['furnished', 'semi-furnished', 'unfurnished']
    
    # Initialize lists for storing data
    data = {
        'location': [],
        'sqft': [],
        'beds': [],
        'baths': [],
        'parking': [],
        'furnishing_status': [],
        'property_type': [],
        'age_of_property': [],
        'price_in_rupees': []
    }
    
    # Generate samples
    for i in range(num_samples):
        # 40% Tier 1, 30% Tier 2, 30% Premium
        if i < 0.4 * num_samples:
            location = random.choice(cities_tier1)
        elif i < 0.7 * num_samples:
            location = random.choice(cities_tier2)
        else:
            location = random.choice(premium_locations)
        
        # Generate correlated features
        # Base sqft based on property type probabilities
        property_type = random.choices(property_types, weights=[0.7, 0.2, 0.1])[0]
        
        if property_type == 'apartment':
            sqft = random.randint(600, 2500)
            beds = random.choices([1, 2, 3, 4], weights=[0.2, 0.5, 0.25, 0.05])[0]
        elif property_type == 'independent_house':
            sqft = random.randint(1500, 5000)
            beds = random.choices([2, 3, 4, 5], weights=[0.1, 0.4, 0.4, 0.1])[0]
        else:  # villa
            sqft = random.randint(2000, 6000)
            beds = random.choices([3, 4, 5, 6], weights=[0.2, 0.4, 0.3, 0.1])[0]
        
        # Bathrooms (usually 1-2 less than or equal to bedrooms)
        baths = max(1, beds - random.randint(0, 1))
        
        # Parking spaces (correlated with sqft and property type)
        if sqft < 1000:
            parking = random.choices([0, 1], weights=[0.3, 0.7])[0]
        elif sqft < 2000:
            parking = random.choices([1, 2], weights=[0.6, 0.4])[0]
        else:
            parking = random.choices([1, 2, 3], weights=[0.3, 0.5, 0.2])[0]
        
        # Furnishing status
        furnishing = random.choices(furnishing_statuses, weights=[0.3, 0.4, 0.3])[0]
        
        # Age of property (0-30 years, weighted towards newer properties)
        age = random.choices(range(0, 31), weights=[0.15] + [0.85/30]*30)[0]
        
        # Calculate price
        base_rate = base_rate_per_sqft[location]
        
        # Base price
        base_price = sqft * base_rate
        
        # Property type premium
        if property_type == 'villa':
            base_price *= 1.35
        elif property_type == 'independent_house':
            base_price *= 1.25
        
        # Furnishing premium
        if furnishing == 'furnished':
            base_price *= 1.12
        elif furnishing == 'semi-furnished':
            base_price *= 1.05
        
        # Age depreciation (-1.5% per year, max 30%)
        age_discount = min(0.3, age * 0.015)
        base_price *= (1 - age_discount)
        
        # Parking premium (₹300,000 per parking)
        base_price += parking * 300000
        
        # Bedroom premium (₹500,000 per bedroom after 2nd)
        if beds > 2:
            base_price += (beds - 2) * 500000
        
        # Location premium for premium areas (already in base_rate)
        # Add some random variation (±10%)
        random_variation = random.uniform(0.9, 1.1)
        final_price = int(base_price * random_variation)
        
        # Ensure minimum price
        final_price = max(4000000, final_price)  # Minimum ₹40 lakhs
        
        # Add to data
        data['location'].append(location)
        data['sqft'].append(sqft)
        data['beds'].append(beds)
        data['baths'].append(baths)
        data['parking'].append(parking)
        data['furnishing_status'].append(furnishing)
        data['property_type'].append(property_type)
        data['age_of_property'].append(age)
        data['price_in_rupees'].append(final_price)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df

def save_dataset(df, filename='indian_housing_prices.csv'):
    """
    Save dataset to CSV file
    """
    df.to_csv(filename, index=False)
    print(f"✅ Dataset saved as '{filename}'")
    print(f"📊 Dataset shape: {df.shape}")
    print(f"💰 Price range: ₹{df['price_in_rupees'].min():,} to ₹{df['price_in_rupees'].max():,}")
    print(f"🏙️ Locations: {df['location'].nunique()} unique locations")
    return filename

def analyze_dataset(df):
    """
    Print basic dataset statistics
    """
    print("\n" + "="*50)
    print("📈 DATASET STATISTICS")
    print("="*50)
    
    print(f"\nTotal records: {len(df)}")
    
    print(f"\n📍 Location Distribution:")
    loc_counts = df['location'].value_counts()
    for loc, count in loc_counts.head(10).items():
        print(f"  {loc}: {count} properties ({count/len(df)*100:.1f}%)")
    
    print(f"\n🏠 Property Type Distribution:")
    for prop_type, count in df['property_type'].value_counts().items():
        print(f"  {prop_type}: {count} ({count/len(df)*100:.1f}%)")
    
    print(f"\n🛋️ Furnishing Status:")
    for furn, count in df['furnishing_status'].value_counts().items():
        print(f"  {furn}: {count} ({count/len(df)*100:.1f}%)")
    
    print(f"\n💰 Price Statistics:")
    print(f"  Minimum: ₹{df['price_in_rupees'].min():,}")
    print(f"  Maximum: ₹{df['price_in_rupees'].max():,}")
    print(f"  Average: ₹{df['price_in_rupees'].mean():,.0f}")
    print(f"  Median:  ₹{df['price_in_rupees'].median():,.0f}")
    
    print(f"\n📏 Size Statistics:")
    print(f"  Average sqft: {df['sqft'].mean():.0f}")
    print(f"  Average bedrooms: {df['beds'].mean():.1f}")
    print(f"  Average age: {df['age_of_property'].mean():.1f} years")

# Main execution
if __name__ == "__main__":
    print("🏠 Generating Indian House Price Dataset...")
    print("="*50)
    
    # Generate dataset with 800 samples
    df = generate_house_price_dataset(num_samples=800)
    
    # Save to CSV
    filename = save_dataset(df, 'indian_housing_prices.csv')
    
    # Show statistics
    analyze_dataset(df)
    
    print("\n" + "="*50)
    print("✅ Dataset generation complete!")
    print("📁 File saved as: indian_housing_prices.csv")
    print("="*50)
    
    # Display first few rows
    print("\n📋 Sample data (first 5 rows):")
    print(df.head().to_string())