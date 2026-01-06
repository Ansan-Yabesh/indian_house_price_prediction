"""
High-Accuracy House Price Prediction Model 
Using XGBoost, LightGBM, and Random Forest
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')
import joblib
import json
import os
import sys
import logging
logging.getLogger('lightgbm').setLevel(logging.ERROR)

# Import 3 best models
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

class HousePricePredictor:
    """
    High-accuracy house price prediction model trainer
    """
    
    def __init__(self, data_path, model_dir='models'):
        self.data_path = data_path
        self.model_dir = model_dir
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.best_model = None
        self.best_model_name = None
        self.best_accuracy = 0
        self.preprocessor = None
        self.results = {}
        
        # Create models directory
        os.makedirs(self.model_dir, exist_ok=True)
    
    def load_data(self):
        """Load and prepare data"""
        print("="*80)
        print("🏠 LOADING DATASET")
        print("="*80)
        
        self.df = pd.read_csv(self.data_path)
        print(f"✅ Dataset loaded: {self.df.shape}")
        
        # Prepare features and target
        self.X = self.df.drop('price_in_rupees', axis=1)
        self.y = self.df['price_in_rupees']
        
        return self.X, self.y
    
    def create_preprocessor(self):
        """Create preprocessing pipeline"""
        print("\n" + "="*80)
        print("⚙️  CREATING PREPROCESSING PIPELINE")
        print("="*80)
        
        # Identify columns
        categorical_cols = self.X.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = self.X.select_dtypes(include=[np.number]).columns.tolist()
        
        print(f"Categorical: {categorical_cols}")
        print(f"Numerical: {numerical_cols}")
        
        # Create transformers
        num_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        cat_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Create preprocessor
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_transformer, numerical_cols),
                ('cat', cat_transformer, categorical_cols)
            ])
        
        print("✅ Preprocessor created")
        return self.preprocessor
    
    def split_data(self):
        """Split data into train/test"""
        print("\n" + "="*80)
        print("🔀 SPLITTING DATA")
        print("="*80)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        print(f"Train: {self.X_train.shape}")
        print(f"Test: {self.X_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_xgboost(self):
        """Train XGBoost model"""
        print("\n🔵 TRAINING XGBOOST")
        
        # Create pipeline
        xgb_pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('regressor', XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbosity=0  # Suppress XGBoost output
            ))
        ])
        
        # Train
        xgb_pipeline.fit(self.X_train, self.y_train)
        
        # Evaluate
        y_pred = xgb_pipeline.predict(self.X_test)
        r2 = r2_score(self.y_test, y_pred)
        accuracy = r2 * 100
        
        self.results['XGBoost'] = {
            'model': xgb_pipeline,
            'r2': r2,
            'accuracy': accuracy,
            'mae': mean_absolute_error(self.y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(self.y_test, y_pred))
        }
        
        print(f"   ✅ R²: {r2:.4f}")
        print(f"   ✅ Accuracy: {accuracy:.2f}%")
        
        return xgb_pipeline
    
    def train_lightgbm(self):
        """Train LightGBM model"""
        print("\n🟢 TRAINING LIGHTGBM")
        
        # Set LightGBM to be quiet
        import lightgbm as lgb
        
        # Create pipeline
        lgb_pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('regressor', LGBMRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1  # Suppress LightGBM output
            ))
        ])
        
        # Train
        lgb_pipeline.fit(self.X_train, self.y_train)
        
        # Evaluate
        y_pred = lgb_pipeline.predict(self.X_test)
        r2 = r2_score(self.y_test, y_pred)
        accuracy = r2 * 100
        
        self.results['LightGBM'] = {
            'model': lgb_pipeline,
            'r2': r2,
            'accuracy': accuracy,
            'mae': mean_absolute_error(self.y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(self.y_test, y_pred))
        }
        
        print(f"   ✅ R²: {r2:.4f}")
        print(f"   ✅ Accuracy: {accuracy:.2f}%")
        
        return lgb_pipeline
    
    def train_random_forest(self):
        """Train Random Forest model"""
        print("\n🟠 TRAINING RANDOM FOREST")
        
        # Create pipeline
        rf_pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('regressor', RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ))
        ])
        
        # Train
        rf_pipeline.fit(self.X_train, self.y_train)
        
        # Evaluate
        y_pred = rf_pipeline.predict(self.X_test)
        r2 = r2_score(self.y_test, y_pred)
        accuracy = r2 * 100
        
        self.results['RandomForest'] = {
            'model': rf_pipeline,
            'r2': r2,
            'accuracy': accuracy,
            'mae': mean_absolute_error(self.y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(self.y_test, y_pred))
        }
        
        print(f"   ✅ R²: {r2:.4f}")
        print(f"   ✅ Accuracy: {accuracy:.2f}%")
        
        return rf_pipeline
    
    def train_all_models(self):
        """Train all 3 models"""
        print("\n" + "="*80)
        print("🚀 TRAINING ALL MODELS")
        print("="*80)
        
        # Train each model
        xgb_model = self.train_xgboost()
        lgb_model = self.train_lightgbm()
        rf_model = self.train_random_forest()
        
        # Find best model
        for name, result in self.results.items():
            if result['accuracy'] > self.best_accuracy:
                self.best_accuracy = result['accuracy']
                self.best_model = result['model']
                self.best_model_name = name
        
        print("\n" + "="*80)
        print("🏆 BEST MODEL")
        print("="*80)
        print(f"🎖️  {self.best_model_name}")
        print(f"🏅 Accuracy: {self.best_accuracy:.2f}%")
        
        return {
            'XGBoost': xgb_model,
            'LightGBM': lgb_model,
            'RandomForest': rf_model
        }
    
    def save_models(self):
        """Save all models and artifacts"""
        print("\n" + "="*80)
        print("💾 SAVING MODELS")
        print("="*80)
        
        # Save each model
        for name in self.results.keys():
            model_path = os.path.join(self.model_dir, f'{name.lower()}_model.pkl')
            joblib.dump(self.results[name]['model'], model_path)
            print(f"✅ {name} saved as: {model_path}")
        
        # Save best model separately
        best_model_path = os.path.join(self.model_dir, 'best_model.pkl')
        joblib.dump(self.best_model, best_model_path)
        print(f"✅ Best model saved as: {best_model_path}")
        
        # Save preprocessor
        preprocessor_path = os.path.join(self.model_dir, 'preprocessor.pkl')
        joblib.dump(self.preprocessor, preprocessor_path)
        print(f"✅ Preprocessor saved as: {preprocessor_path}")
        
        # Save metadata
        metadata = {
            'best_model': self.best_model_name,
            'best_accuracy': float(self.best_accuracy),
            'all_results': {
                name: {
                    'accuracy': float(result['accuracy']),
                    'r2': float(result['r2']),
                    'mae': float(result['mae']),
                    'rmse': float(result['rmse'])
                }
                for name, result in self.results.items()
            },
            'features': self.X.columns.tolist(),
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        metadata_path = os.path.join(self.model_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"✅ Metadata saved as: {metadata_path}")
        
        return best_model_path
    
    def evaluate_models(self):
        """Evaluate and compare all models"""
        print("\n" + "="*80)
        print("📊 MODEL EVALUATION")
        print("="*80)
        
        # Create comparison table
        comparison_data = []
        for name, result in self.results.items():
            comparison_data.append({
                'Model': name,
                'R² Score': f"{result['r2']:.4f}",
                'Accuracy': f"{result['accuracy']:.2f}%",
                'MAE': f"₹{result['mae']:,.0f}",
                'RMSE': f"₹{result['rmse']:,.0f}",
                'Status': '🎯 BEST' if name == self.best_model_name else '✅ Good'
            })
        
        df_compare = pd.DataFrame(comparison_data)
        print("\n" + df_compare.to_string(index=False))
        
        # Save comparison
        compare_path = os.path.join(self.model_dir, 'model_comparison.csv')
        df_compare.to_csv(compare_path, index=False)
        print(f"\n✅ Comparison saved: {compare_path}")
        
        return df_compare
    
    def create_visualizations(self):
        """Create performance visualizations"""
        print("\n" + "="*80)
        print("📈 CREATING VISUALIZATIONS")
        print("="*80)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Accuracy comparison
        models = list(self.results.keys())
        accuracies = [self.results[m]['accuracy'] for m in models]
        colors = ['blue', 'green', 'orange']
        
        bars = axes[0, 0].bar(models, accuracies, color=colors, alpha=0.7)
        axes[0, 0].set_ylabel('Accuracy (%)')
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].axhline(y=95, color='red', linestyle='--', label='95% Target')
        axes[0, 0].legend()
        
        # Add values on bars
        for bar, acc in zip(bars, accuracies):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                          f'{acc:.1f}%', ha='center', va='bottom')
        
        # 2. Actual vs Predicted (best model)
        y_pred = self.best_model.predict(self.X_test)
        axes[0, 1].scatter(self.y_test/100000, y_pred/100000, alpha=0.5, color='blue')
        axes[0, 1].plot([self.y_test.min()/100000, self.y_test.max()/100000],
                       [self.y_test.min()/100000, self.y_test.max()/100000],
                       'r--', lw=2)
        axes[0, 1].set_xlabel('Actual Price (Lakhs ₹)')
        axes[0, 1].set_ylabel('Predicted Price (Lakhs ₹)')
        axes[0, 1].set_title(f'{self.best_model_name}: Predictions')
        
        # 3. Error distribution
        errors = self.y_test - y_pred
        axes[1, 0].hist(errors/100000, bins=30, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 0].axvline(x=0, color='red', linestyle='--')
        axes[1, 0].set_xlabel('Prediction Error (Lakhs ₹)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Error Distribution')
        
        # 4. Price range vs error
        price_bins = pd.qcut(self.y_test/100000, q=5)
        error_by_bin = pd.DataFrame({
            'Price Range': price_bins,
            'Error': abs(errors)/self.y_test * 100
        })
        avg_error = error_by_bin.groupby('Price Range')['Error'].mean()
        
        axes[1, 1].bar(range(len(avg_error)), avg_error.values, alpha=0.7, color='purple')
        axes[1, 1].set_xlabel('Price Range')
        axes[1, 1].set_ylabel('Average Error (%)')
        axes[1, 1].set_title('Error by Price Range')
        axes[1, 1].set_xticks(range(len(avg_error)))
        axes[1, 1].set_xticklabels([str(b)[:20] for b in avg_error.index], rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save figure
        viz_path = os.path.join(self.model_dir, 'performance_plots.png')
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Visualizations saved: {viz_path}")
    
    def test_prediction(self):
        """Test prediction with sample data"""
        print("\n" + "="*80)
        print("🧪 TESTING PREDICTIONS")
        print("="*80)
        
        # Sample test cases
        test_cases = [
            {
                'location': 'Bangalore',
                'sqft': 1200,
                'beds': 3,
                'baths': 2,
                'parking': 1,
                'furnishing_status': 'semi-furnished',
                'property_type': 'apartment',
                'age_of_property': 5
            },
            {
                'location': 'Mumbai',
                'sqft': 1800,
                'beds': 4,
                'baths': 3,
                'parking': 2,
                'furnishing_status': 'furnished',
                'property_type': 'villa',
                'age_of_property': 3
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            input_df = pd.DataFrame([test_case])
            prediction = self.best_model.predict(input_df)[0]
            
            print(f"\n🏠 Test Case {i}:")
            print(f"   Location: {test_case['location']}")
            print(f"   Type: {test_case['property_type']}, {test_case['sqft']} sqft")
            print(f"   Beds: {test_case['beds']}, Baths: {test_case['baths']}")
            print(f"   🎯 Predicted Price: ₹{prediction:,.0f}")
            print(f"   💰 Price in Lakhs: ₹{prediction/100000:,.1f} Lakhs")

def main():
    """Main execution"""
    print("🏠 HOUSE PRICE PREDICTION MODEL TRAINER")
    print("="*80)
    
    # Set your dataset path
    data_path = r"C:\Users\Ansan Yabesh\OneDrive\Desktop\indian_house_price_prediction\dataset\indian_housing_prices.csv"
    
    # Initialize predictor
    predictor = HousePricePredictor(data_path)
    
    try:
        # 1. Load data
        predictor.load_data()
        
        # 2. Create preprocessor
        predictor.create_preprocessor()
        
        # 3. Split data
        predictor.split_data()
        
        # 4. Train all models
        predictor.train_all_models()
        
        # 5. Evaluate models
        predictor.evaluate_models()
        
        # 6. Save models
        predictor.save_models()
        
        # 7. Create visualizations
        predictor.create_visualizations()
        
        # 8. Test predictions
        predictor.test_prediction()
        
        print("\n" + "="*80)
        print("✅ TRAINING COMPLETE!")
        print("="*80)
        
        print(f"\n🎯 Best Model: {predictor.best_model_name}")
        print(f"🎯 Best Accuracy: {predictor.best_accuracy:.2f}%")
        
        if predictor.best_accuracy >= 95:
            print("\nACHIEVED BEST MODEL")
        
        print("\n📁 Models saved in 'models' folder:")
        print("   - xgboost_model.pkl")
        print("   - lightgbm_model.pkl")
        print("   - randomforest_model.pkl")
        print("   - best_model.pkl")
        print("   - preprocessor.pkl")
        print("   - metadata.json")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()