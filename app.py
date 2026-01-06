import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import json
import os

# Page configuration
st.set_page_config(
    page_title="RealEstate Pro | AI Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 2rem;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .header-title {
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: white;
    }
    
    .header-subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
        font-weight: 300;
        color: rgba(255,255,255,0.9);
    }
    
    /* Card styling */
    .prediction-card {
        background: white;
        padding: 2.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.08);
        margin: 1.5rem 0;
        border: 1px solid #eef2f7;
    }
    
    .metric-card {
        background: white;
        padding: 1.8rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        text-align: center;
        border-top: 4px solid #1e3c72;
        height: 100%;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border: none;
        padding: 14px 36px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        width: 100%;
        margin: 1rem 0;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 7px 20px rgba(30, 60, 114, 0.4);
    }
    
    /* Price display */
    .price-display {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin: 1.5rem 0;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        border-radius: 10px;
    }
    
    /* Loading spinner */
    .stSpinner > div {
        border-color: #1e3c72 transparent transparent transparent !important;
    }
    
    /* Input field styling */
    .stSelectbox, .stNumberInput, .stSlider {
        margin-bottom: 1.2rem;
    }
    
    /* Full width containers */
    .full-width {
        width: 100%;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Section headers */
    .section-header {
        color: #2d3748;
        font-size: 1.8rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #eef2f7;
    }
</style>
""", unsafe_allow_html=True)

class RealEstatePredictor:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.metadata = None
        self.load_models()
    
    def load_models(self):
        """Load ML models and metadata"""
        try:
            model_dir = 'models'
            
            if not os.path.exists(model_dir):
                st.error(f"Models directory not found at: {os.path.abspath(model_dir)}")
                return False
            
            best_model_path = os.path.join(model_dir, 'best_model.pkl')
            if not os.path.exists(best_model_path):
                st.error(f"best_model.pkl not found at: {best_model_path}")
                return False
            
            try:
                self.model = joblib.load(best_model_path)
                st.sidebar.success("✅ Prediction model loaded successfully")
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                return False
            
            preprocessor_path = os.path.join(model_dir, 'preprocessor.pkl')
            if os.path.exists(preprocessor_path):
                try:
                    self.preprocessor = joblib.load(preprocessor_path)
                except:
                    pass
            
            metadata_path = os.path.join(model_dir, 'metadata.json')
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        self.metadata = json.load(f)
                except:
                    pass
            
            return True
            
        except Exception as e:
            st.error(f"Error in model loading process: {str(e)}")
            return False
    
    def predict_price(self, input_data):
        """Make prediction using loaded model"""
        try:
            input_df = pd.DataFrame([input_data])
            prediction = self.model.predict(input_df)[0]
            return float(prediction)
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return None

def create_input_form():
    """Create professional input form"""
    st.markdown("""
    <div class="prediction-card">
        <div class="section-header">📋 Property Details</div>
    """, unsafe_allow_html=True)
    
    # Create two columns for input fields
    col1, col2 = st.columns(2)
    
    with col1:
        location = st.selectbox(
            "📍 Location",
            ["Bangalore", "Mumbai", "Delhi", "Hyderabad", "Chennai", "Pune", "Kolkata", 
             "Ahmedabad", "Jaipur", "Chandigarh", "Lucknow", "Kochi"],
            help="Select the city where property is located",
            key="location"
        )
        
        sqft = st.number_input(
            "📏 Built-up Area (sqft)",
            min_value=400,
            max_value=10000,
            value=1200,
            step=50,
            help="Total built-up area in square feet",
            key="sqft"
        )
        
        beds = st.selectbox(
            "🛏️ Bedrooms",
            [1, 2, 3, 4, 5, 6],
            help="Number of bedrooms",
            key="beds"
        )
        
        property_type = st.selectbox(
            "🏠 Property Type",
            ["apartment", "independent_house", "villa"],
            help="Type of property",
            key="property_type"
        )
    
    with col2:
        baths = st.selectbox(
            "🚿 Bathrooms",
            [1, 2, 3, 4, 5],
            help="Number of bathrooms",
            key="baths"
        )
        
        parking = st.selectbox(
            "🚗 Parking Spaces",
            [0, 1, 2, 3],
            help="Number of dedicated parking spaces",
            key="parking"
        )
        
        furnishing = st.selectbox(
            "🛋️ Furnishing Status",
            ["unfurnished", "semi-furnished", "furnished"],
            help="Furnishing condition of property",
            key="furnishing"
        )
        
        age = st.slider(
            "📅 Property Age (years)",
            min_value=0,
            max_value=50,
            value=5,
            help="Age of property since construction",
            key="age"
        )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    return {
        'location': location,
        'sqft': int(sqft),
        'beds': int(beds),
        'baths': int(baths),
        'parking': int(parking),
        'furnishing_status': furnishing,
        'property_type': property_type,
        'age_of_property': int(age)
    }

def format_price(price):
    """Format price in Indian numbering system"""
    price = float(price)
    
    if price >= 10000000:
        crores = price / 10000000
        return f"₹{crores:.2f} Crores"
    elif price >= 100000:
        lakhs = price / 100000
        return f"₹{lakhs:.2f} Lakhs"
    else:
        return f"₹{price:,.0f}"

def create_price_display(price):
    """Create animated price display"""
    st.markdown(f"""
    <div class="prediction-card" style="text-align: center;">
        <h3 style="color: #4a5568; margin-bottom: 1rem;">🏠 Predicted Market Value</h3>
        <div class="price-display">{format_price(price)}</div>
        <p style="color: #718096; font-size: 1.1rem; margin-top: 0.5rem;">
            💰 Estimated market value based on current trends and property specifications
        </p>
    </div>
    """, unsafe_allow_html=True)

def create_metrics_display(price, input_data):
    """Create metrics cards"""
    st.markdown("""
    <div class="prediction-card">
        <div class="section-header">📊 Property Metrics</div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        price_per_sqft = price / input_data['sqft'] if input_data['sqft'] > 0 else 0
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: #4a5568; margin-bottom: 0.8rem;">📐 Price per sqft</h4>
            <h2 style="color: #1e3c72; margin: 0; font-size: 1.8rem;">
                ₹{price_per_sqft:,.0f}
            </h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: #4a5568; margin-bottom: 0.8rem;">📏 Property Size</h4>
            <h2 style="color: #1e3c72; margin: 0; font-size: 1.8rem;">
                {input_data['sqft']} sqft
            </h2>
            <p style="color: #718096; font-size: 0.9rem; margin: 0.5rem 0 0 0;">
                {input_data['beds']} Beds • {input_data['baths']} Baths
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: #4a5568; margin-bottom: 0.8rem;">🏷️ Property Type</h4>
            <h2 style="color: #1e3c72; margin: 0; font-size: 1.6rem;">
                {input_data['property_type'].replace('_', ' ').title()}
            </h2>
            <p style="color: #718096; font-size: 0.9rem; margin: 0.5rem 0 0 0;">
                {input_data['furnishing_status'].replace('_', ' ').title()}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: #4a5568; margin-bottom: 0.8rem;">📍 Location</h4>
            <h2 style="color: #1e3c72; margin: 0; font-size: 1.6rem;">
                {input_data['location']}
            </h2>
            <p style="color: #718096; font-size: 0.9rem; margin: 0.5rem 0 0 0;">
                {input_data['age_of_property']} years old
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

def create_market_analysis(input_data, predicted_price):
    """Create market analysis visualizations"""
    st.markdown("""
    <div class="prediction-card">
        <div class="section-header">📈 Market Analysis</div>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Price Comparison", "Investment Insights"])
    
    with tab1:
        # Create comparison chart
        fig = go.Figure()
        
        locations = ["Bangalore", "Mumbai", "Delhi", "Hyderabad", "Chennai", input_data['location']]
        avg_prices_per_sqft = [12000, 25000, 15000, 9000, 8500, predicted_price / input_data['sqft']]
        
        comparable_prices = [price * input_data['sqft'] for price in avg_prices_per_sqft]
        
        colors = ['#1e3c72', '#2a5298', '#3a6bc7', '#4a85f0', '#5a9fff', '#38a169']
        
        fig.add_trace(go.Bar(
            x=locations,
            y=comparable_prices,
            name="Property Value",
            marker_color=colors,
            text=[format_price(p) for p in comparable_prices],
            textposition='outside'
        ))
        
        fig.update_layout(
            title="Price Comparison Across Major Cities",
            xaxis_title="Location",
            yaxis_title="Price (₹)",
            showlegend=False,
            hovermode='x unified',
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=500,
            margin=dict(t=50, b=100, l=50, r=50)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🏠 Property Specifications
            
            **Location Analysis:**
            - **City:** {}
            - **Market Segment:** {}
            
            **Property Details:**
            - **Total Area:** {} sqft
            - **Configuration:** {} Beds, {} Baths
            - **Parking:** {} dedicated spaces
            - **Property Age:** {} years
            - **Furnishing:** {}
            - **Property Type:** {}
            """.format(
                input_data['location'],
                "Premium" if input_data['location'] in ['Mumbai', 'Bangalore', 'Delhi'] else "Standard",
                input_data['sqft'],
                input_data['beds'],
                input_data['baths'],
                input_data['parking'],
                input_data['age_of_property'],
                input_data['furnishing_status'].title(),
                input_data['property_type'].title()
            ))
        
        with col2:
            monthly_rent = predicted_price * 0.0004
            annual_yield = (monthly_rent * 12 / predicted_price) * 100
            
            st.markdown("""
            ### 💰 Investment Analysis
            
            **Valuation Summary:**
            - **Current Market Value:** {}
            - **Price per sqft:** ₹{:,}
            
            **Rental Potential:**
            - **Estimated Monthly Rent:** ₹{:,}
            - **Annual Rental Yield:** {:.1f}%
            - **Expected Appreciation:** 8-12% annually
            
            **Market Position:**
            - **Demand Level:** {}
            - **Investment Rating:** {}
            """.format(
                format_price(predicted_price),
                int(predicted_price / input_data['sqft']),
                int(monthly_rent),
                annual_yield,
                "High" if input_data['location'] in ['Mumbai', 'Bangalore'] else "Moderate",
                "Good" if annual_yield > 3 else "Average"
            ))
    
    st.markdown("</div>", unsafe_allow_html=True)

def create_recommendations(input_data, predicted_price):
    """Create investment recommendations"""
    st.markdown("""
    <div class="prediction-card">
        <div class="section-header">💡 Recommendations & Insights</div>
    """, unsafe_allow_html=True)
    
    recommendations = []
    
    if input_data['age_of_property'] > 10:
        recommendations.append({
            'icon': '🔄',
            'title': 'Modernization Potential',
            'description': 'Consider strategic upgrades to increase property value by 15-20%',
            'priority': 'High'
        })
    
    if input_data['parking'] < 2:
        recommendations.append({
            'icon': '🚗',
            'title': 'Parking Enhancement',
            'description': 'Additional parking space can increase value by 5-8%',
            'priority': 'Medium'
        })
    
    if input_data['furnishing_status'] == 'unfurnished':
        recommendations.append({
            'icon': '🛋️',
            'title': 'Furnishing Upgrade',
            'description': 'Furnished properties command 8-12% premium in the market',
            'priority': 'Medium'
        })
    
    if input_data['location'] in ['Mumbai', 'Bangalore', 'Delhi']:
        recommendations.append({
            'icon': '📈',
            'title': 'Prime Market Location',
            'description': 'Properties in metro cities show strong long-term appreciation',
            'priority': 'High'
        })
    
    if recommendations:
        for rec in recommendations:
            priority_color = {
                'High': '#e53e3e',
                'Medium': '#ed8936',
                'Low': '#38a169'
            }[rec['priority']]
            
            st.markdown(f"""
            <div style="
                background: #f7fafc;
                padding: 1.2rem;
                border-radius: 10px;
                border-left: 4px solid {priority_color};
                margin-bottom: 1rem;
            ">
                <div style="display: flex; align-items: start; gap: 1rem;">
                    <span style="font-size: 1.8rem;">{rec['icon']}</span>
                    <div style="flex: 1;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <h4 style="margin: 0; color: #2d3748; font-size: 1.2rem;">{rec['title']}</h4>
                            <span style="
                                background: {priority_color}20;
                                color: {priority_color};
                                padding: 4px 12px;
                                border-radius: 12px;
                                font-size: 0.85rem;
                                font-weight: 600;
                            ">
                                {rec['priority']} Priority
                            </span>
                        </div>
                        <p style="color: #718096; margin: 0.8rem 0 0 0; font-size: 1rem;">{rec['description']}</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("""
        **Property Assessment:** This property has optimal specifications for its segment. 
        No major improvements recommended at this time. Consider regular maintenance 
        to preserve value.
        """)
    
    st.markdown("</div>", unsafe_allow_html=True)

def create_footer():
    """Create professional footer"""
    st.markdown("""
    <div style="
        margin-top: 3rem;
        padding-top: 2rem;
        border-top: 1px solid #e2e8f0;
        text-align: center;
        color: #718096;
        font-size: 0.9rem;
    ">
        <p>
            <strong>RealEstate Pro AI</strong> | 
            Professional Property Valuation System
        </p>
        <p style="font-size: 0.8rem; margin-top: 0.5rem;">
            ℹ️ Valuation based on market trends and property specifications. 
            Actual prices may vary based on market conditions and negotiations.
        </p>
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Header Section
    st.markdown("""
    <div class="header-container">
        <div class="header-title">RealEstate Pro AI</div>
        <div class="header-subtitle">
            Advanced Property Valuation System for Indian Real Estate Market
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="padding: 1.5rem;">
            <h3 style="color: #2d3748; margin-bottom: 1rem;">⚙️ System Status</h3>
        </div>
        """, unsafe_allow_html=True)
        
        predictor = RealEstatePredictor()
        
        if predictor.model is None:
            st.error("⚠️ Models not loaded")
            st.info("Please ensure models are trained and available in 'models' folder")
            return
        
        st.markdown("""
        <div style="
            background: #f7fafc;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
        ">
            <p style="margin: 0.2rem 0; color: #4a5568;">
                <strong>✅ System Status:</strong> Operational
            </p>
            <p style="margin: 0.2rem 0; color: #4a5568;">
                <strong>🔧 Model Type:</strong> Machine Learning
            </p>
            <p style="margin: 0.2rem 0; color: #4a5568;">
                <strong>📊 Coverage:</strong> 12+ Indian Cities
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="
            background: #1e3c7210;
            padding: 1rem;
            border-radius: 10px;
            margin-top: 2rem;
            border: 1px solid #1e3c7230;
        ">
            <h4 style="color: #1e3c72; margin-bottom: 0.5rem;">💡 About This System</h4>
            <p style="color: #4a5568; font-size: 0.9rem; margin: 0;">
                Professional property valuation system using advanced 
                machine learning algorithms to analyze market trends 
                and property specifications.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content - Single column full width
    st.markdown('<div class="full-width">', unsafe_allow_html=True)
    
    # Input form
    input_data = create_input_form()
    
    # Predict button - centered
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🔮 Predict Market Price", type="primary", use_container_width=True)
    
    if predict_button:
        with st.spinner("Analyzing property details and market trends..."):
            predicted_price = predictor.predict_price(input_data)
            
            if predicted_price:
                # Display results in sequence
                create_price_display(predicted_price)
                create_metrics_display(predicted_price, input_data)
                create_market_analysis(input_data, predicted_price)
                create_recommendations(input_data, predicted_price)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    create_footer()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        st.info("Please check system configuration and try again.")
