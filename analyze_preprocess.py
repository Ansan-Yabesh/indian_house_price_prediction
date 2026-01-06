"""
House Price Dataset - Analysis & Preprocessing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class HousePriceAnalyzer:
    """
    Comprehensive class for analyzing and preprocessing house price data
    """
    
    def __init__(self, file_path='indian_housing_prices.csv'):
        """
        Initialize the analyzer with dataset path
        """
        self.file_path = file_path
        self.df = None
        self.numerical_cols = None
        self.categorical_cols = None
        
    def load_data(self):
        """
        Load and display basic information about the dataset
        """
        print("="*60)
        print("📊 LOADING DATASET")
        print("="*60)
        
        # Load the dataset
        self.df = pd.read_csv(self.file_path)
        
        # Display basic information
        print(f"\n✅ Dataset loaded successfully!")
        print(f"📁 File: {self.file_path}")
        print(f"📈 Shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
        
        # Display first few rows
        print("\n" + "="*60)
        print("🔍 FIRST 5 ROWS OF DATASET:")
        print("="*60)
        print(self.df.head().to_string())
        
        return self.df
    
    def explore_data(self):
        """
        Perform comprehensive data exploration
        """
        print("\n" + "="*60)
        print("🔍 EXPLORATORY DATA ANALYSIS (EDA)")
        print("="*60)
        
        # Dataset information
        print("\n📋 DATASET INFORMATION:")
        print("-"*40)
        self.df.info()
        
        # Statistical summary
        print("\n📊 STATISTICAL SUMMARY:")
        print("-"*40)
        print(self.df.describe().T)
        
        # Check for missing values
        print("\n🔍 MISSING VALUES CHECK:")
        print("-"*40)
        missing_values = self.df.isnull().sum()
        if missing_values.sum() == 0:
            print("✅ No missing values found!")
        else:
            print("❌ Missing values found:")
            print(missing_values[missing_values > 0])
        
        # Check for duplicates
        print("\n🔍 DUPLICATE ROWS CHECK:")
        print("-"*40)
        duplicates = self.df.duplicated().sum()
        if duplicates == 0:
            print("✅ No duplicate rows found!")
        else:
            print(f"❌ Found {duplicates} duplicate rows")
            self.df = self.df.drop_duplicates()
            print(f"✅ Duplicates removed. New shape: {self.df.shape}")
        
        # Identify column types
        self.numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
        print("\n📊 COLUMN TYPES:")
        print("-"*40)
        print(f"📐 Numerical columns ({len(self.numerical_cols)}): {self.numerical_cols}")
        print(f"🏷️  Categorical columns ({len(self.categorical_cols)}): {self.categorical_cols}")
        
    def visualize_distributions(self):
        """
        Create visualizations for data distributions
        """
        print("\n" + "="*60)
        print("📈 DATA VISUALIZATIONS")
        print("="*60)
        
        # Create subplots
        fig, axes = plt.subplots(3, 3, figsize=(18, 12))
        fig.suptitle('Feature Distributions', fontsize=16, fontweight='bold')
        
        # 1. Price distribution (target variable)
        axes[0, 0].hist(self.df['price_in_rupees'] / 100000, bins=30, 
                       edgecolor='black', color='skyblue', alpha=0.7)
        axes[0, 0].set_xlabel('Price (Lakhs ₹)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('House Price Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Square feet distribution
        axes[0, 1].hist(self.df['sqft'], bins=30, edgecolor='black', 
                       color='lightcoral', alpha=0.7)
        axes[0, 1].set_xlabel('Area (sqft)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Area Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Property age distribution
        axes[0, 2].hist(self.df['age_of_property'], bins=20, edgecolor='black',
                       color='lightgreen', alpha=0.7)
        axes[0, 2].set_xlabel('Age (years)')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Property Age Distribution')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Bedrooms count
        bed_counts = self.df['beds'].value_counts().sort_index()
        axes[1, 0].bar(bed_counts.index, bed_counts.values, 
                      color='gold', alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Number of Bedrooms')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Bedrooms Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Property type distribution
        prop_counts = self.df['property_type'].value_counts()
        axes[1, 1].bar(range(len(prop_counts)), prop_counts.values, 
                      color='violet', alpha=0.7, edgecolor='black')
        axes[1, 1].set_xticks(range(len(prop_counts)))
        axes[1, 1].set_xticklabels(prop_counts.index, rotation=45)
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Property Type Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Location distribution (top 10)
        loc_counts = self.df['location'].value_counts().head(10)
        axes[1, 2].bar(range(len(loc_counts)), loc_counts.values,
                      color='orange', alpha=0.7, edgecolor='black')
        axes[1, 2].set_xticks(range(len(loc_counts)))
        axes[1, 2].set_xticklabels(loc_counts.index, rotation=45, ha='right')
        axes[1, 2].set_ylabel('Count')
        axes[1, 2].set_title('Top 10 Locations')
        axes[1, 2].grid(True, alpha=0.3)
        
        # 7. Furnishing status
        furn_counts = self.df['furnishing_status'].value_counts()
        axes[2, 0].pie(furn_counts.values, labels=furn_counts.index, 
                      autopct='%1.1f%%', colors=['lightblue', 'lightgreen', 'lightcoral'])
        axes[2, 0].set_title('Furnishing Status')
        
        # 8. Parking spaces
        park_counts = self.df['parking'].value_counts().sort_index()
        axes[2, 1].bar(park_counts.index, park_counts.values,
                      color='cyan', alpha=0.7, edgecolor='black')
        axes[2, 1].set_xlabel('Parking Spaces')
        axes[2, 1].set_ylabel('Count')
        axes[2, 1].set_title('Parking Spaces Distribution')
        axes[2, 1].grid(True, alpha=0.3)
        
        # 9. Correlation heatmap
        corr_matrix = self.df[self.numerical_cols].corr()
        im = axes[2, 2].imshow(corr_matrix, cmap='coolwarm', aspect='auto')
        axes[2, 2].set_title('Feature Correlation Heatmap')
        axes[2, 2].set_xticks(range(len(self.numerical_cols)))
        axes[2, 2].set_yticks(range(len(self.numerical_cols)))
        axes[2, 2].set_xticklabels(self.numerical_cols, rotation=45, ha='right')
        axes[2, 2].set_yticklabels(self.numerical_cols)
        
        # Add correlation values
        for i in range(len(self.numerical_cols)):
            for j in range(len(self.numerical_cols)):
                text = axes[2, 2].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                      ha="center", va="center", 
                                      color="white" if abs(corr_matrix.iloc[i, j]) > 0.5 else "black",
                                      fontsize=8)
        
        plt.colorbar(im, ax=axes[2, 2])
        
        plt.tight_layout()
        plt.savefig('eda_visualizations.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualizations saved as 'eda_visualizations.png'")
    
    def analyze_relationships(self):
        """
        Analyze relationships between features and target variable
        """
        print("\n" + "="*60)
        print("📊 FEATURE-TARGET RELATIONSHIPS")
        print("="*60)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Price vs Area
        axes[0, 0].scatter(self.df['sqft'], self.df['price_in_rupees'] / 100000,
                          alpha=0.5, c='blue', edgecolors='black', linewidth=0.5)
        axes[0, 0].set_xlabel('Area (sqft)')
        axes[0, 0].set_ylabel('Price (Lakhs ₹)')
        axes[0, 0].set_title('Price vs Area')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(self.df['sqft'], self.df['price_in_rupees'] / 100000, 1)
        p = np.poly1d(z)
        axes[0, 0].plot(self.df['sqft'].sort_values(), 
                       p(self.df['sqft'].sort_values()), 
                       "r--", alpha=0.8, linewidth=2)
        
        # 2. Price vs Bedrooms (boxplot)
        data = [self.df[self.df['beds'] == i]['price_in_rupees'] / 100000 
                for i in sorted(self.df['beds'].unique())]
        axes[0, 1].boxplot(data, labels=sorted(self.df['beds'].unique()))
        axes[0, 1].set_xlabel('Number of Bedrooms')
        axes[0, 1].set_ylabel('Price (Lakhs ₹)')
        axes[0, 1].set_title('Price by Number of Bedrooms')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Price by Property Type
        prop_types = self.df['property_type'].unique()
        prop_data = [self.df[self.df['property_type'] == pt]['price_in_rupees'] / 100000 
                    for pt in prop_types]
        axes[1, 0].boxplot(prop_data, labels=prop_types)
        axes[1, 0].set_xlabel('Property Type')
        axes[1, 0].set_ylabel('Price (Lakhs ₹)')
        axes[1, 0].set_title('Price by Property Type')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Average Price by Location (top 8)
        top_locations = self.df['location'].value_counts().head(8).index
        location_prices = self.df[self.df['location'].isin(top_locations)]
        avg_price_by_loc = location_prices.groupby('location')['price_in_rupees'].mean() / 100000
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(avg_price_by_loc)))
        bars = axes[1, 1].bar(range(len(avg_price_by_loc)), avg_price_by_loc.values, 
                            color=colors, edgecolor='black')
        axes[1, 1].set_xlabel('Location')
        axes[1, 1].set_ylabel('Average Price (Lakhs ₹)')
        axes[1, 1].set_title('Average Price by Location (Top 8)')
        axes[1, 1].set_xticks(range(len(avg_price_by_loc)))
        axes[1, 1].set_xticklabels(avg_price_by_loc.index, rotation=45, ha='right')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, avg_price_by_loc.values):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                          f'₹{value:.1f}L', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('feature_relationships.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Relationship analysis saved as 'feature_relationships.png'")
        
    def detect_outliers(self):
        """
        Detect and handle outliers in numerical features
        """
        print("\n" + "="*60)
        print("🔍 OUTLIER DETECTION")
        print("="*60)
        
        numerical_features = ['sqft', 'price_in_rupees', 'age_of_property']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, feature in enumerate(numerical_features):
            # Calculate IQR
            Q1 = self.df[feature].quantile(0.25)
            Q3 = self.df[feature].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier boundaries
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Count outliers
            outliers = self.df[(self.df[feature] < lower_bound) | (self.df[feature] > upper_bound)]
            
            print(f"\n📊 {feature.upper()}:")
            print(f"   Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
            print(f"   Lower bound: {lower_bound:.2f}, Upper bound: {upper_bound:.2f}")
            print(f"   Outliers detected: {len(outliers)} ({len(outliers)/len(self.df)*100:.1f}%)")
            
            # Create boxplot
            axes[idx].boxplot(self.df[feature], vert=True, patch_artist=True)
            axes[idx].set_title(f'Boxplot of {feature}')
            axes[idx].set_ylabel(feature)
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outlier_detection.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\n✅ Outlier detection completed!")
        return self.df
    
    def preprocess_data(self):
        """
        Preprocess the data for machine learning
        """
        print("\n" + "="*60)
        print("⚙️  DATA PREPROCESSING")
        print("="*60)
        
        # Create a copy for preprocessing
        df_processed = self.df.copy()
        
        # 1. Handle categorical variables
        print("\n1. Encoding categorical variables...")
        
        # Label encode for ordinal categories
        furnishing_order = {'unfurnished': 0, 'semi-furnished': 1, 'furnished': 2}
        df_processed['furnishing_status_encoded'] = df_processed['furnishing_status'].map(furnishing_order)
        
        # One-hot encode location and property_type
        df_processed = pd.get_dummies(df_processed, 
                                     columns=['location', 'property_type'],
                                     drop_first=True)
        
        # 2. Feature engineering
        print("2. Creating new features...")
        
        # Price per sqft
        df_processed['price_per_sqft'] = df_processed['price_in_rupees'] / df_processed['sqft']
        
        # Total rooms
        df_processed['total_rooms'] = df_processed['beds'] + df_processed['baths']
        
        # Property age category
        df_processed['age_category'] = pd.cut(df_processed['age_of_property'],
                                             bins=[-1, 5, 10, 20, 100],
                                             labels=['new', 'recent', 'old', 'very_old'])
        df_processed = pd.get_dummies(df_processed, columns=['age_category'], drop_first=True)
        
        # 3. Remove original categorical columns (keep encoded ones)
        columns_to_drop = ['furnishing_status']
        for col in columns_to_drop:
            if col in df_processed.columns:
                df_processed = df_processed.drop(col, axis=1)
        
        # 4. Handle missing values (if any)
        print("3. Checking for missing values...")
        missing_values = df_processed.isnull().sum().sum()
        if missing_values > 0:
            print(f"   Found {missing_values} missing values. Filling with median/mode...")
            for col in df_processed.columns:
                if df_processed[col].dtype in ['int64', 'float64']:
                    df_processed[col].fillna(df_processed[col].median(), inplace=True)
                else:
                    df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
        else:
            print("   ✅ No missing values found!")
        
        # 5. Feature scaling (prepare for ML)
        print("4. Preparing features for ML modeling...")
        
        # Identify numerical columns (excluding target)
        numerical_features = ['sqft', 'beds', 'baths', 'parking', 'age_of_property',
                             'furnishing_status_encoded', 'price_per_sqft', 'total_rooms']
        
        # Create scaler instance
        scaler = StandardScaler()
        
        # Scale numerical features
        df_scaled = df_processed.copy()
        df_scaled[numerical_features] = scaler.fit_transform(df_scaled[numerical_features])
        
        # Display processed data info
        print("\n✅ Preprocessing completed!")
        print(f"📊 Original shape: {self.df.shape}")
        print(f"📊 Processed shape: {df_processed.shape}")
        
        # Show processed columns
        print(f"\n📋 Processed columns ({len(df_processed.columns)} total):")
        for i, col in enumerate(df_processed.columns.tolist(), 1):
            print(f"   {i:2d}. {col}")
        
        return df_processed, df_scaled, scaler
    
    def save_processed_data(self, df_processed, df_scaled, scaler):
        """
        Save processed data and preprocessing objects
        """
        # Save processed data
        df_processed.to_csv('processed_housing_data.csv', index=False)
        df_scaled.to_csv('scaled_housing_data.csv', index=False)
        
        # Save scaler for future use
        import joblib
        joblib.dump(scaler, 'scaler.pkl')
        
        print("\n" + "="*60)
        print("💾 SAVING PROCESSED DATA")
        print("="*60)
        print("✅ Processed data saved as 'processed_housing_data.csv'")
        print("✅ Scaled data saved as 'scaled_housing_data.csv'")
        print("✅ Scaler saved as 'scaler.pkl'")
        
        # Show sample of processed data
        print("\n🔍 Sample of processed data (first 3 rows):")
        print(df_processed.head(3).T)

def main():
    """
    Main execution function
    """
    print("🏠 HOUSE PRICE DATASET - ANALYSIS & PREPROCESSING")
    print("="*60)
    
    # Initialize analyzer
    analyzer = HousePriceAnalyzer('indian_housing_prices.csv')
    
    try:
        # 1. Load data
        df = analyzer.load_data()
        
        # 2. Explore data
        analyzer.explore_data()
        
        # 3. Visualize distributions
        analyzer.visualize_distributions()
        
        # 4. Analyze relationships
        analyzer.analyze_relationships()
        
        # 5. Detect outliers
        analyzer.detect_outliers()
        
        # 6. Preprocess data
        df_processed, df_scaled, scaler = analyzer.preprocess_data()
        
        # 7. Save processed data
        analyzer.save_processed_data(df_processed, df_scaled, scaler)
        
        print("\n" + "="*60)
        print("✅ ANALYSIS & PREPROCESSING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\n🎯 Next steps:")
        print("   1. Use 'processed_housing_data.csv' for ML modeling")
        print("   2. Use 'scaler.pkl' to scale new data")
        print("   3. Check the generated PNG files for EDA insights")
        
    except FileNotFoundError:
        print(f"\n❌ Error: File 'indian_housing_prices.csv' not found!")
        print("Please ensure the dataset file exists in the current directory.")
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")

if __name__ == "__main__":
    main()