import findspark
findspark.init()

import builtins
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import os
import warnings
warnings.filterwarnings('ignore')

from pyspark.sql import SparkSession # type: ignore
from pyspark.sql.functions import * # type: ignore
from pyspark.ml.classification import LogisticRegressionModel, RandomForestClassificationModel, GBTClassificationModel # type: ignore
from pyspark.ml.feature import VectorAssembler, StandardScalerModel # type: ignore
from pyspark.sql import Row # type: ignore

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS
# ============================================
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .fraud-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    .legit-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .risk-high {
        background-color: #ff6b6b;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.25rem 0;
        font-weight: bold;
    }
    .risk-medium {
        background-color: #ffd93d;
        color: #333;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.25rem 0;
        font-weight: bold;
    }
    .risk-low {
        background-color: #6bcf7f;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.25rem 0;
        font-weight: bold;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Add this new diagnostic function to your app
def model_behavior_analysis(models, config, assembler, scaler):
    """Analyze why different models give different predictions"""
    st.header("🔬 Model Behavior Analysis")
    
    if not models:
        st.error("No models available!")
        return
    
    spark = init_spark()
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")
    
    # Test multiple scenarios
    test_scenarios = {
        "Normal Transaction": {
            "amount": 50.0,
            "hour": 14.0,
            "day_of_week": 3.0,
            "customer_fraud_rate": 0.01,
            "merchant_fraud_rate": 0.01,
            "customer_transaction_count": 100.0,
            "customer_fraud_count": 0.0,
            "merchant_fraud_volatility": 0.005,
            "merchant": "grocery"
        },
        "Slightly Suspicious": {
            "amount": 500.0,
            "hour": 22.0,
            "day_of_week": 6.0,
            "customer_fraud_rate": 0.05,
            "merchant_fraud_rate": 0.03,
            "customer_transaction_count": 10.0,
            "customer_fraud_count": 1.0,
            "merchant_fraud_volatility": 0.02,
            "merchant": "electronics"
        },
        "Highly Suspicious": {
            "amount": 5000.0,
            "hour": 3.0,
            "day_of_week": 7.0,
            "customer_fraud_rate": 0.3,
            "merchant_fraud_rate": 0.05,
            "customer_transaction_count": 3.0,
            "customer_fraud_count": 2.0,
            "merchant_fraud_volatility": 0.04,
            "merchant": "jewelry"
        },
        "Your Test Case": {
            "amount": 150.0,
            "hour": 14.0,
            "day_of_week": 4.0,
            "customer_fraud_rate": 0.02,
            "merchant_fraud_rate": 0.03,
            "customer_transaction_count": 10.0,
            "customer_fraud_count": 0.0,
            "merchant_fraud_volatility": 0.01,
            "merchant": "electronics"
        }
    }
    
    for scenario_name, params in test_scenarios.items():
        st.subheader(f"📊 Scenario: {scenario_name}")
        
        # Create feature vector
        df = spark.createDataFrame([Row(
            amount=params["amount"],
            hour=params["hour"],
            day_of_week=params["day_of_week"],
            customer_fraud_rate=params["customer_fraud_rate"],
            merchant_fraud_rate=params["merchant_fraud_rate"],
            customer_transaction_count=params["customer_transaction_count"],
            customer_fraud_count=params["customer_fraud_count"],
            merchant_fraud_volatility=params["merchant_fraud_volatility"]
        )])
        
        # Apply pipeline
        if assembler:
            df = assembler.transform(df)
        if scaler:
            df = scaler.transform(df)
        if 'features_scaled' in df.columns:
            df = df.withColumnRenamed('features_scaled', 'features')
        
        # Get scaled feature values
        features = df.select("features").first()[0]
        if hasattr(features, 'toArray'):
            scaled_values = features.toArray()
            st.write("**Scaled Feature Values:**")
            feature_names = [
                "amount", "hour", "day_of_week", "customer_fraud_rate",
                "merchant_fraud_rate", "customer_transaction_count",
                "customer_fraud_count", "merchant_fraud_volatility"
            ]
            for name, value in zip(feature_names, scaled_values):
                st.write(f"  • {name}: {value:.4f}")
        
        # Test each model
        results = []
        for model_name, model in models.items():
            try:
                predictions = model.transform(df)
                pred = int(predictions.select("prediction").first()[0])
                
                prob = 0.5
                if "probability" in predictions.columns:
                    prob_vec = predictions.select("probability").first()[0]
                    if hasattr(prob_vec, 'toArray'):
                        prob = float(prob_vec.toArray()[1])
                
                results.append({
                    "Model": model_name,
                    "Prediction": "🚨 FRAUD" if pred == 1 else "✅ LEGIT",
                    "Fraud Probability": f"{prob:.1%}",
                    "Raw Prob": prob
                })
                
                st.write(f"**{model_name}:** {'🚨 FRAUD' if pred == 1 else '✅ LEGIT'} (Confidence: {prob:.1%})")
                
            except Exception as e:
                st.error(f"Error with {model_name}: {str(e)[:100]}")
        
        st.markdown("---")
    
    # Explanation section
    st.header("📚 Why Models Behave Differently")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Logistic Regression
        **Tends to predict MORE frauds**
        
        🔹 **Linear model** - Makes decisions based on weighted sum of features
        🔹 **More sensitive** to individual feature values
        🔹 **Lower threshold** for fraud detection
        🔹 **Better recall** (catches more fraud)
        
        **Why it flags more:**
        - Even small suspicious patterns trigger it
        - Linear combination can exceed threshold easily
        - Weighted features add up quickly
        """)
    
    with col2:
        st.markdown("""
        ### Tree-Based Models
        **Tends to predict FEWER frauds**
        
        🔹 **Non-linear** - Makes decisions through multiple splits
        🔹 **More conservative** - Requires multiple conditions
        🔹 **Higher precision** (fewer false alarms)
        🔹 **Better at finding complex patterns**
        
        **Why it's more conservative:**
        - Needs multiple conditions to be true
        - Splits on specific thresholds
        - Won't flag unless multiple red flags present
        """)
    
    # Check training performance
    if config and os.path.exists("models/model_info.json"):
        with open("models/model_info.json", "r") as f:
            model_info = json.load(f)
        
        if model_info.get('metrics'):
            st.subheader("📊 Training Performance Comparison")
            metrics_df = pd.DataFrame(model_info['metrics']).T
            
            st.dataframe(metrics_df)
            
            # Create comparison chart
            fig = go.Figure()
            for metric in ['F1-Score', 'Precision', 'Recall', 'AUC-ROC']:
                if metric in metrics_df.columns:
                    fig.add_trace(go.Bar(
                        name=metric,
                        x=metrics_df.index,
                        y=metrics_df[metric],
                        text=metrics_df[metric].round(3)
                    ))
            
            fig.update_layout(
                title="Model Performance Metrics",
                barmode='group',
                yaxis_range=[0, 1]
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Explain the trade-off
            st.info("""
            **Understanding the Trade-off:**
            
            - **Logistic Regression** typically has **higher recall** (catches more actual fraud)
            - **Tree-based models** typically have **higher precision** (fewer false alarms)
            
            In production:
            - Use **Logistic Regression** for initial screening (catch everything)
            - Use **Tree models** for high-confidence decisions (reduce false alarms)
            - Combine both for best results!
            """)

# Add this mini-analyzer to understand your specific case
def analyze_specific_case():
    st.header("🎯 Analyzing Your Test Transaction")
    
    st.write("""
    ### Transaction Details:
    - **Amount:** $150 (moderate)
    - **Time:** 2 PM (normal business hours)  
    - **Merchant:** Electronics (slightly elevated risk)
    - **Customer history:** Clean (no prior fraud)
    
    ### Why Logistic Regression says FRAUD:
    1. Electronics category has elevated risk weight
    2. Linear model adds up all signals
    3. Even moderate amount + electronics might trigger threshold
    
    ### Why Trees say LEGIT:
    1. Trees split on specific conditions
    2. $150 might be below a key split point (e.g., $200)
    3. Business hours (14h) might be in a "safe" branch
    4. No prior fraud history keeps it in legitimate path
    
    ### Which is right?
    **Both could be right!** This is a borderline case:
    - LR is being **cautious** (flagging for review)
    - Trees are being **precise** (not enough red flags)
    
    **Best practice:** Use ensemble - if ANY model flags it, review manually!
    """)
    
    st.success("""
    ### Recommendation:
    For production fraud detection, use a **voting ensemble**:
    - If 2/3 models say fraud → Automatic block
    - If 1/3 models say fraud → Manual review
    - If 0/3 models say fraud → Approve
    """)

# ============================================
# SPARK SESSION
# ============================================
@st.cache_resource
def init_spark():
    """Initialize Spark session with Windows-specific configurations"""
    import sys
    import os
    
    # CRITICAL: Set Python executable for Spark workers
    python_exec = sys.executable
    os.environ['PYSPARK_PYTHON'] = python_exec
    os.environ['PYSPARK_DRIVER_PYTHON'] = python_exec
    
    spark = SparkSession.builder \
        .appName("FraudDetectionApp") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "2g") \
        .config("spark.driver.maxResultSize", "2g") \
        .config("spark.python.worker.reuse", "true") \
        .config("spark.python.worker.memory", "512m") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
        .config("spark.sql.execution.pyspark.udf.faulthandler.enabled", "false") \
        .config("spark.python.profile", "false") \
        .master("local[1]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    
    # Test if Spark works
    try:
        test_df = spark.range(1)
        test_df.collect()
        print("✓ Spark initialized successfully")
    except Exception as e:
        print(f"⚠️ Spark test failed: {e}")
    
    return spark

# ============================================
# MODEL LOADING - WITH TREE MODEL WORKAROUND
# ============================================
@st.cache_resource
def load_models():
    """Load all trained models with tree model workaround"""
    spark = init_spark()
    models = {}
    model_info = None
    config = None
    merchants = []
    assembler = None
    scaler = None
    load_errors = []
    
    # Check for best_model folder first (this is your best performing model)
    if os.path.exists("best_model"):
        st.info("📁 Found 'best_model' directory")
        try:
            # Try different model types
            for ModelClass, model_type in [
                (GBTClassificationModel, "Gradient Boosted Trees"),
                (RandomForestClassificationModel, "Random Forest"),
                (LogisticRegressionModel, "Logistic Regression")
            ]:
                try:
                    best_model = ModelClass.load("best_model")
                    models[f"Best Model - {model_type}"] = best_model
                    st.success(f"✅ Loaded best model as {model_type}")
                    break
                except:
                    continue
        except Exception as e:
            load_errors.append(f"Could not load best_model: {str(e)[:100]}")
    
    # Load models from models directory
    if not os.path.exists("models"):
        st.error("❌ 'models' folder not found!")
        return models, model_info, config, merchants, assembler, scaler, load_errors
    
    # Load model info
    info_path = "models/model_info.json"
    if os.path.exists(info_path):
        try:
            with open(info_path, "r") as f:
                model_info = json.load(f)
                st.info("📊 Loaded model performance data")
        except Exception as e:
            load_errors.append(f"Could not load model_info.json: {str(e)[:100]}")
    
    # Load preprocessing config
    config_path = "models/preprocessing_config.json"
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
                st.info(f"⚙️ Loaded config with {len(config.get('feature_columns', []))} features")
        except Exception as e:
            load_errors.append(f"Could not load preprocessing_config: {str(e)[:100]}")
    
    # Load merchant categories
    merchants_path = "models/merchant_categories.json"
    if os.path.exists(merchants_path):
        try:
            with open(merchants_path, "r") as f:
                merchants = json.load(f)
                st.info(f"🏪 Loaded {len(merchants)} merchant categories")
        except Exception as e:
            load_errors.append(f"Could not load merchant_categories: {str(e)[:100]}")
    
    # Load individual models
    model_paths = {
        "Logistic Regression": ("models/logistic_regression", LogisticRegressionModel),
        "Random Forest": ("models/random_forest", RandomForestClassificationModel),
        "Gradient Boosted Trees": ("models/gradient_boosted_trees", GBTClassificationModel),
    }
    
    for name, (path, model_class) in model_paths.items():
        if os.path.exists(path):
            try:
                model = model_class.load(path)
                models[name] = model
                st.success(f"✅ Loaded {name}")
            except Exception as e:
                error_msg = str(e)
                if "treeID" in error_msg or "UNRESOLVED_COLUMN" in error_msg:
                    load_errors.append(f"{name}: Tree model compatibility issue")
                else:
                    load_errors.append(f"{name}: {error_msg[:100]}")
                st.warning(f"⚠️ Could not load {name}")
    
    # Load assembler - check both locations
    for assembler_path in ["models/assembler", "best_model/assembler"]:
        if os.path.exists(assembler_path):
            try:
                assembler = VectorAssembler.load(assembler_path)
                st.success(f"✅ Loaded Vector Assembler from {assembler_path}")
                break
            except Exception as e:
                load_errors.append(f"Could not load assembler from {assembler_path}: {str(e)[:100]}")
    
    # Load scaler - check both locations
    for scaler_path in ["models/scaler", "best_model/scaler"]:
        if os.path.exists(scaler_path):
            try:
                scaler = StandardScalerModel.load(scaler_path)
                st.success(f"✅ Loaded Standard Scaler from {scaler_path}")
                break
            except Exception as e:
                load_errors.append(f"Could not load scaler from {scaler_path}: {str(e)[:100]}")
    
    return models, model_info, config, merchants, assembler, scaler, load_errors

# ============================================
# FEATURE IMPORTANCE EXTRACTION
# ============================================
@st.cache_resource
def get_feature_importance(_models, feature_columns):  # Note the underscore before 'models'
    """Extract feature importance from tree-based models"""
    importance_dict = {}
    
    for name, model in _models.items():  # Use _models here
        if hasattr(model, 'featureImportances'):
            try:
                importances = model.featureImportances.toArray().tolist()
                # Make sure we have the right number of features
                if len(importances) == len(feature_columns):
                    importance_dict[name] = dict(zip(feature_columns, importances))
                else:
                    # Handle case where feature count doesn't match
                    importance_dict[name] = dict(zip([f"feature_{i}" for i in range(len(importances))], importances))
            except Exception as e:
                st.warning(f"Could not extract feature importance for {name}: {str(e)[:100]}")
        else:
            # Logistic regression doesn't have featureImportances attribute
            importance_dict[name] = {}
    
    return importance_dict

# ============================================
# PREDICTION FUNCTION - FIXED PROBABILITY EXTRACTION
# ============================================
def predict_transaction(model, amount, merchant, hour, day_of_week,
                       customer_fraud_rate=0.02, merchant_fraud_rate=None,
                       customer_txn_count=10, customer_fraud_count=0,
                       merchant_volatility=0.01, assembler=None, scaler=None, 
                       config=None, debug=False):
    """Make fraud prediction with proper probability extraction - Arrow safe"""
    
    merchant_risk = {
        "electronics": 0.03, "travel": 0.04, "entertainment": 0.025,
        "clothing": 0.02, "grocery": 0.008, "restaurant": 0.01,
        "home": 0.015, "health": 0.012, "automotive": 0.02,
        "jewelry": 0.05, "other": 0.015
    }
    
    if merchant_fraud_rate is None:
        merchant_fraud_rate = merchant_risk.get(merchant.lower(), 0.015)
    
    spark = init_spark()
    
    # Disable Arrow for this operation
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")
    
    try:
        # Create DataFrame with raw features
        df = spark.createDataFrame([Row(
            amount=float(amount),
            hour=float(hour),
            day_of_week=float(day_of_week),
            customer_fraud_rate=float(customer_fraud_rate),
            merchant_fraud_rate=float(merchant_fraud_rate),
            customer_transaction_count=float(customer_txn_count),
            customer_fraud_count=float(customer_fraud_count),
            merchant_fraud_volatility=float(merchant_volatility)
        )])
        
        if debug:
            st.write("**Input features:**")
            for col_name in df.columns:
                st.write(f"  {col_name}: {df.first()[col_name]}")
        
        # Apply assembler to create features_raw
        if assembler is not None:
            try:
                df = assembler.transform(df)
                if debug:
                    st.success("✓ Assembler applied")
            except Exception as e:
                if debug:
                    st.error(f"❌ Assembler error: {str(e)[:200]}")
                return 0, 0.0
            
            # Apply scaler to create features_scaled
            if scaler is not None:
                try:
                    df = scaler.transform(df)
                    if debug:
                        st.success("✓ Scaler applied")
                except Exception as e:
                    if debug:
                        st.warning(f"⚠️ Scaler error: {str(e)[:100]}")
        
        # CRITICAL: Rename features_scaled to features (what model expects)
        if 'features_scaled' in df.columns:
            df = df.withColumnRenamed('features_scaled', 'features')
            if debug:
                st.success("✓ Renamed 'features_scaled' to 'features'")
        elif 'features' not in df.columns:
            if debug:
                st.error("❌ Neither 'features_scaled' nor 'features' found")
                st.write("Available columns:", df.columns)
            return 0, 0.0
        
        # Final validation
        if 'features' not in df.columns:
            if debug:
                st.error("❌ 'features' column not found")
            return 0, 0.0
        
        if debug:
            # Show features safely
            features_val = df.select("features").first()[0]
            st.write(f"Features type: {type(features_val)}")
            if hasattr(features_val, 'toArray'):
                st.write(f"Features array: {features_val.toArray()}")
        
        # Transform using the model
        result = model.transform(df)
        
        if debug:
            st.success(f"✓ Model transform successful")
        
        # Check for prediction column
        if "prediction" not in result.columns:
            if debug:
                st.error("❌ 'prediction' column missing")
            return 0, 0.0
        
        # Extract prediction
        try:
            result_row = result.select("prediction").first()
            
            if result_row is None:
                if debug:
                    st.error("❌ Empty prediction result")
                return 0, 0.0
            
            prediction = int(result_row[0])
            
            if debug:
                st.write(f"**Prediction: {prediction}**")
            
        except Exception as e:
            if debug:
                st.error(f"❌ Failed to extract prediction: {str(e)[:150]}")
            return 0, 0.0
        
        # Extract probability
        probability = 0.5  # Default
        
        if "probability" in result.columns:
            try:
                prob_result = result.select("probability").first()
                
                if prob_result is not None:
                    prob_vector = prob_result[0]
                    
                    # Handle DenseVector
                    if hasattr(prob_vector, 'toArray'):
                        prob_array = prob_vector.toArray()
                        if len(prob_array) >= 2:
                            # [prob_class_0, prob_class_1]
                            probability = float(prob_array[1])
                        elif len(prob_array) == 1:
                            probability = float(prob_array[0])
                    # Handle array-like objects
                    elif hasattr(prob_vector, '__getitem__'):
                        if len(prob_vector) >= 2:
                            probability = float(prob_vector[1])
                        elif len(prob_vector) == 1:
                            probability = float(prob_vector[0])
                    else:
                        probability = float(prob_vector)
                    
                    if debug:
                        st.write(f"**Fraud probability: {probability:.4f} ({probability:.1%})**")
                
            except Exception as e:
                if debug:
                    st.write(f"⚠️ Could not extract probability: {str(e)[:100]}")
        
        if debug:
            if prediction == 1 and probability > 0.7:
                st.error(f"🚨 HIGH RISK: Fraud probability {probability:.1%}")
            elif prediction == 1:
                st.warning(f"⚠️ SUSPICIOUS: Fraud probability {probability:.1%}")
            else:
                st.success(f"✅ LEGITIMATE: Safe probability {1-probability:.1%}")
        
        return prediction, probability
    
    except Exception as e:
        if debug:
            st.error(f"❌ Prediction failed: {str(e)[:150]}")
            import traceback
            st.code(traceback.format_exc())
        return 0, 0.0

# ============================================
# SUPER DETAILED DIAGNOSTIC FUNCTION
# ============================================
def deep_diagnostic(models, config, assembler, scaler):
    """Deep diagnostic to find the exact issue - Arrow safe version"""
    st.header("🔍 DEEP DIAGNOSTIC MODE")
    
    if not models:
        st.error("No models loaded!")
        return
    
    spark = init_spark()
    
    # Disable Arrow for this session
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")
    
    # Test with a simple transaction
    test_amount = 150.0
    test_hour = 14.0
    test_day = 4.0
    test_cust_fraud = 0.02
    test_merc_fraud = 0.03
    test_cust_txn = 10.0
    test_cust_fraud_count = 0.0
    test_merc_vol = 0.01
    
    st.subheader("Step 1: Create Raw Data")
    
    # Create the raw dataframe
    raw_df = spark.createDataFrame([Row(
        amount=test_amount,
        hour=test_hour,
        day_of_week=test_day,
        customer_fraud_rate=test_cust_fraud,
        merchant_fraud_rate=test_merc_fraud,
        customer_transaction_count=test_cust_txn,
        customer_fraud_count=test_cust_fraud_count,
        merchant_fraud_volatility=test_merc_vol
    )])
    
    # Show raw data safely (without Arrow)
    st.write("Raw DataFrame (collecting first row):")
    try:
        raw_row = raw_df.first()
        for col_name in raw_df.columns:
            st.write(f"  {col_name}: {raw_row[col_name]}")
    except Exception as e:
        st.error(f"Error showing raw data: {e}")
    
    # Step 2: Check assembler
    st.subheader("Step 2: Check Assembler")
    
    if assembler:
        st.write(f"Assembler input columns: {assembler.getInputCols()}")
        
        try:
            assembled_df = assembler.transform(raw_df)
            st.success("✓ Assembler transform successful")
            st.write(f"Output columns: {assembled_df.columns}")
            
            # Show features_raw safely
            features_raw = assembled_df.select("features_raw").first()[0]
            st.write(f"features_raw type: {type(features_raw)}")
            if hasattr(features_raw, 'toArray'):
                st.write(f"features_raw values: {features_raw.toArray()}")
        except Exception as e:
            st.error(f"❌ Assembler error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            return
    else:
        st.error("No assembler available!")
        return
    
    # Step 3: Check Scaler
    st.subheader("Step 3: Check Scaler")
    
    if scaler:
        try:
            scaled_df = scaler.transform(assembled_df)
            st.success("✓ Scaler transform successful")
            st.write(f"Output columns: {scaled_df.columns}")
            
            # Show features_scaled safely
            if 'features_scaled' in scaled_df.columns:
                features_scaled = scaled_df.select("features_scaled").first()[0]
                st.write(f"features_scaled type: {type(features_scaled)}")
                if hasattr(features_scaled, 'toArray'):
                    st.write(f"features_scaled values: {features_scaled.toArray()}")
            else:
                st.error("❌ No 'features_scaled' column after scaler!")
                
        except Exception as e:
            st.error(f"❌ Scaler error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            return
    else:
        st.error("No scaler available!")
        return
    
    # Step 4: Prepare for model (rename features_scaled to features)
    st.subheader("Step 4: Prepare for Model")
    
    if 'features_scaled' in scaled_df.columns:
        model_df = scaled_df.withColumnRenamed('features_scaled', 'features')
        st.success("✓ Renamed 'features_scaled' to 'features'")
        st.write(f"Columns: {model_df.columns}")
        
        # Verify features column exists
        features_check = model_df.select("features").first()[0]
        st.write(f"Features type: {type(features_check)}")
        if hasattr(features_check, 'toArray'):
            st.write(f"Features values: {features_check.toArray()}")
    else:
        st.error("❌ Cannot prepare for model - missing features_scaled")
        return
    
    # Step 5: Test each model
    st.subheader("Step 5: Test Models")
    
    for model_name, model in models.items():
        st.write(f"\n--- Testing {model_name} ---")
        
        try:
            # Get model info
            st.write(f"Model type: {type(model).__name__}")
            
            # Try to predict
            predictions = model.transform(model_df)
            st.success(f"✓ Transform successful")
            st.write(f"Prediction columns: {predictions.columns}")
            
            # Get prediction safely
            pred_row = predictions.select("prediction").first()
            prediction_value = int(pred_row[0])
            st.write(f"**Prediction: {prediction_value} ({'🚨 FRAUD' if prediction_value == 1 else '✅ LEGITIMATE'})**")
            
            # Get probability safely
            if "probability" in predictions.columns:
                prob_row = predictions.select("probability").first()
                prob_vector = prob_row[0]
                st.write(f"Probability type: {type(prob_vector)}")
                
                if hasattr(prob_vector, 'toArray'):
                    prob_array = prob_vector.toArray()
                    st.write(f"Probability array: {prob_array}")
                    st.write(f"**Fraud probability: {prob_array[1]:.4f} ({prob_array[1]:.1%})**")
                    
                    # Visual indicator
                    if prob_array[1] > 0.8:
                        st.error(f"🔴 High confidence fraud: {prob_array[1]:.1%}")
                    elif prob_array[1] > 0.5:
                        st.warning(f"🟡 Medium confidence fraud: {prob_array[1]:.1%}")
                    else:
                        st.success(f"🟢 Low risk: {prob_array[1]:.1%}")
                else:
                    st.write(f"Probability value: {prob_vector}")
            
        except Exception as e:
            st.error(f"❌ Error with {model_name}: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

# ============================================
# DATA LOADING FOR EDA (UPDATED FOR YOUR COLUMNS)
# ============================================
@st.cache_data
def load_eda_data(limit=50000):
    """Load data for exploratory analysis"""
    spark = init_spark()
    
    # Check for parquet folder (your structure)
    parquet_path = "processed_fraud_data_balanced.parquet"
    
    if os.path.exists(parquet_path):
        try:
            df = spark.read.parquet(parquet_path)
            pdf = df.limit(limit).toPandas()
            st.success(f"✅ Loaded {len(pdf):,} transactions")
            
            # Show column info in sidebar for debugging
            with st.sidebar.expander("🔍 Available Data Fields"):
                st.write("**Columns in dataset:**")
                for col in pdf.columns:
                    st.caption(f"• {col}")
            
            return pdf
        except Exception as e:
            st.error(f"Could not load parquet: {e}")
            return pd.DataFrame()
    else:
        st.warning("No data files found. Please run training first.")
        return pd.DataFrame()

# ============================================
# RISK ASSESSMENT
# ============================================
def assess_risk(amount, merchant, hour, customer_fraud_count, prediction, probability):
    """Calculate risk factors"""
    risk_factors = []
    
    if amount > 5000:
        risk_factors.append(("💰 Exceptionally high amount (>$5000)", "high"))
    elif amount > 2000:
        risk_factors.append(("💰 Very high amount (>$2000)", "high"))
    elif amount > 1000:
        risk_factors.append(("💰 High amount (>$1000)", "medium"))
    
    high_risk_merchants = ["electronics", "travel", "jewelry"]
    if merchant.lower() in high_risk_merchants:
        risk_factors.append(("🏪 High-risk merchant category", "high"))
    
    if hour < 6 or hour > 22:
        risk_factors.append(("⏰ Unusual transaction time", "medium"))
    
    if customer_fraud_count > 0:
        risk_factors.append((f"👤 Customer has {customer_fraud_count} previous frauds", "high"))
    
    if prediction == 1 and probability > 0.7:
        risk_factors.append(("🤖 High model confidence in fraud", "high"))
    
    return risk_factors

# ============================================
# SINGLE PREDICTION PAGE
# ============================================
def single_prediction_page(models, model_info, merchants, config, assembler, scaler):
    st.header("🎯 Real-time Fraud Prediction")
    
    if not models:
        st.error("❌ No models available. Check the System Status in the sidebar for details.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💰 Transaction Details")
        amount = st.number_input("Transaction Amount ($)", min_value=0.01, value=150.00, step=25.0)
        
        default_merchants = ["electronics", "clothing", "grocery", "travel", "entertainment", 
                             "restaurant", "home", "health", "automotive", "jewelry", "other"]
        merchant_options = merchants if merchants else default_merchants
        merchant = st.selectbox("Merchant Category", merchant_options)
        
        date = st.date_input("Transaction Date", datetime.now())
        time = st.time_input("Transaction Time", datetime.now().time())
        
        hour = time.hour
        day_of_week = date.weekday() + 1
    
    with col2:
        st.subheader("🤖 Model Selection")
        model_names = list(models.keys())
        selected_model_name = st.selectbox("Select Model", model_names)
        selected_model = models[selected_model_name]
        
        if model_info and model_info.get('metrics'):
            metrics = model_info['metrics'].get(selected_model_name, {})
            if metrics:
                st.metric("F1-Score", f"{metrics.get('F1-Score', 0):.3f}")
                st.metric("AUC-ROC", f"{metrics.get('AUC-ROC', 0):.3f}")
        
        with st.expander("🔧 Customer History (Optional)"):
            customer_fraud_rate = st.slider("Historical Fraud Rate", 0.0, 0.5, 0.02, 0.005)
            customer_txn_count = st.number_input("Total Transactions", 1, 1000, 10)
            customer_fraud_count = st.number_input("Previous Fraud Count", 0, 100, 0)
            merchant_volatility = st.slider("Merchant Volatility", 0.0, 0.5, 0.01, 0.01)
    
    col1, col2 = st.columns(2)
    with col1:
        predict_btn = st.button("🚀 Predict", use_container_width=True)
    with col2:
        debug_mode = st.checkbox("🔧 Debug Mode", value=False)
    
    if predict_btn:
        with st.spinner("⏳ Processing prediction..."):
            prediction, probability = predict_transaction(
                selected_model, amount, merchant, hour, day_of_week,
                customer_fraud_rate=customer_fraud_rate,
                customer_txn_count=customer_txn_count,
                customer_fraud_count=customer_fraud_count,
                merchant_volatility=merchant_volatility,
                assembler=assembler,
                scaler=scaler,
                config=config,
                debug=debug_mode
            )
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if prediction == 1:
                st.markdown("""
                <div class="fraud-card">
                    <h2>🚨 FRAUD DETECTED</h2>
                    <p>High Risk Transaction</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="legit-card">
                    <h2>✅ LEGITIMATE</h2>
                    <p>Safe Transaction</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            # Display actual confidence, not fake 50%
            st.metric("Confidence Level", f"{probability:.1%}")
        
        with col3:
            st.metric("Model Used", selected_model_name)
        
        # Risk assessment
        risk_factors = assess_risk(amount, merchant, hour, customer_fraud_count, prediction, probability)
        
        if risk_factors:
            st.subheader("⚠️ Risk Factors")
            for factor, level in risk_factors:
                st.markdown(f'<div class="risk-{level}">{factor}</div>', unsafe_allow_html=True)

# ============================================
# BATCH UPLOAD PAGE (FIXED VISUALIZATIONS)
# ============================================
def batch_upload_page(models, config, assembler, scaler):
    st.header("📁 Batch Transaction Analysis")
    
    if not models:
        st.error("No models available.")
        return
    
    with st.expander("📋 CSV Format Requirements"):
        st.markdown("""
        Your CSV file should contain these columns:
        - **customer_id** (int) - Customer identifier
        - **transaction_id** (int) - Transaction identifier  
        - **transaction_timestamp** (datetime) - When transaction occurred
        - **amount** (float) - Transaction amount
        - **merchant_category** (string) - Type of merchant (electronics, travel, etc.)
        
        **Note:** The system will automatically extract hour and day_of_week from transaction_timestamp
        """)
        
        sample = pd.DataFrame({
            'customer_id': [1001, 1002, 1003],
            'transaction_id': [50001, 50002, 50003],
            'transaction_timestamp': ['2024-01-15 14:30:00', '2024-01-15 22:15:00', '2024-01-16 09:45:00'],
            'amount': [150.00, 2500.00, 45.50],
            'merchant_category': ['electronics', 'travel', 'grocery']
        })
        st.dataframe(sample)
        st.download_button(
            "Download Sample Template", 
            sample.to_csv(index=False), 
            "batch_template.csv"
        )
    
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read the uploaded CSV
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} transactions")
            
            # Display basic data info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                if 'amount' in df.columns:
                    st.metric("Total Amount", f"${df['amount'].sum():,.2f}")
            
            # Show data preview
            st.subheader("📊 Data Preview")
            st.dataframe(df.head())
            
            # Check required columns
            required_cols = ['amount', 'merchant_category', 'transaction_timestamp']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                st.info("Please ensure your CSV has the correct format. Download the template above.")
                return
            
            # Model selection
            model_name = st.selectbox("Select Model for Batch Prediction", list(models.keys()))
            model = models[model_name]
            
            # Show optional settings
            show_detailed = st.checkbox("Show detailed analysis", value=False)
            confidence_threshold = st.slider("Confidence Threshold for Alerts", 0.5, 1.0, 0.8, 0.05)
            
            if st.button("🚀 Run Batch Analysis", type="primary", use_container_width=True):
                with st.spinner(f"Processing {len(df)} transactions..."):
                    results = []
                    fraud_count = 0
                    high_confidence_frauds = 0
                    
                    # Create progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for idx, row in df.iterrows():
                        try:
                            # Extract timestamp features
                            timestamp = pd.to_datetime(row['transaction_timestamp'])
                            hour = timestamp.hour
                            day_of_week = timestamp.dayofweek + 1  # Monday=1, Sunday=7
                            
                            # Get merchant category
                            merchant = str(row['merchant_category']).lower()
                            
                            # Make prediction
                            pred, prob = predict_transaction(
                                model,
                                float(row['amount']),
                                merchant,
                                hour,
                                day_of_week,
                                assembler=assembler,
                                scaler=scaler,
                                config=config
                            )
                            
                            # Determine risk level
                            if pred == 1:
                                fraud_count += 1
                                if prob >= confidence_threshold:
                                    high_confidence_frauds += 1
                            
                            # Determine risk level for display
                            if pred == 1:
                                if prob >= 0.9:
                                    risk_level = "CRITICAL"
                                    risk_emoji = "🔴"
                                elif prob >= 0.7:
                                    risk_level = "HIGH"
                                    risk_emoji = "🟠"
                                elif prob >= 0.5:
                                    risk_level = "MEDIUM"
                                    risk_emoji = "🟡"
                                else:
                                    risk_level = "LOW"
                                    risk_emoji = "🔵"
                            else:
                                risk_level = "LEGIT"
                                risk_emoji = "✅"
                            
                            # Store results
                            result = {
                                'transaction_id': row.get('transaction_id', idx),
                                'customer_id': row.get('customer_id', 'N/A'),
                                'amount': float(row['amount']),
                                'merchant_category': merchant.title(),
                                'transaction_time': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                                'hour': hour,
                                'day_of_week': timestamp.strftime('%A'),
                                'prediction': 'FRAUD' if pred == 1 else 'LEGITIMATE',
                                'confidence': prob,
                                'risk_level': risk_level,
                                'risk_emoji': risk_emoji
                            }
                            
                            results.append(result)
                            
                            # Update progress
                            progress_bar.progress((idx + 1) / len(df))
                            status_text.text(f"Processed {idx + 1}/{len(df)} transactions... | Frauds found: {fraud_count}")
                            
                        except Exception as e:
                            st.warning(f"Error processing row {idx}: {str(e)}")
                            continue
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Create results dataframe
                    results_df = pd.DataFrame(results)
                    
                    if results_df.empty:
                        st.error("No transactions were processed successfully!")
                        return
                    
                    # Display summary statistics
                    st.markdown("---")
                    st.subheader("📊 Batch Analysis Summary")
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("Total Transactions", len(results_df))
                    with col2:
                        st.metric("🚨 Fraud Detected", fraud_count)
                    with col3:
                        fraud_rate = (fraud_count / len(results_df)) * 100 if len(results_df) > 0 else 0
                        st.metric("Fraud Rate", f"{fraud_rate:.1f}%")
                    with col4:
                        alert_rate = (high_confidence_frauds / len(results_df)) * 100 if len(results_df) > 0 else 0
                        st.metric(f"⚠️ High Risk", f"{alert_rate:.1f}%")
                    with col5:
                        avg_confidence = results_df['confidence'].mean() * 100 if len(results_df) > 0 else 0
                        st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
                    
                    # Show detailed results table
                    st.subheader("📈 Detailed Results")
                    
                    # Prepare display dataframe
                    display_df = results_df.copy()
                    display_df['amount'] = display_df['amount'].apply(lambda x: f"${x:,.2f}")
                    display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.1%}")
                    display_df['prediction'] = display_df['prediction'].apply(
                        lambda x: '🚨 FRAUD' if x == 'FRAUD' else '✅ LEGITIMATE'
                    )
                    
                    st.dataframe(display_df, use_container_width=True)
                    
                    # Only show charts if there are fraudulent transactions
                    fraud_df = results_df[results_df['prediction'] == 'FRAUD']
                    
                    if not fraud_df.empty:
                        st.subheader("📊 Fraud Analytics")
                        
                        # Create two columns for charts
                        chart_col1, chart_col2 = st.columns(2)
                        
                        with chart_col1:
                            # Fraud by hour
                            try:
                                fraud_by_hour = fraud_df.groupby('hour').size().reset_index(name='count')
                                all_by_hour = results_df.groupby('hour').size().reset_index(name='total')
                                
                                # Merge and calculate rates
                                hourly_rates = all_by_hour.merge(fraud_by_hour, on='hour', how='left')
                                hourly_rates['count'] = hourly_rates['count'].fillna(0)
                                hourly_rates['fraud_rate'] = (hourly_rates['count'] / hourly_rates['total']) * 100
                                
                                fig = px.bar(
                                    hourly_rates, x='hour', y='fraud_rate',
                                    title='Fraud Rate by Hour of Day',
                                    labels={'hour': 'Hour (0-23)', 'fraud_rate': 'Fraud Rate (%)'},
                                    color='fraud_rate',
                                    color_continuous_scale='Reds',
                                    text=hourly_rates['fraud_rate'].round(1).astype(str) + '%'
                                )
                                fig.update_traces(textposition='outside')
                                fig.update_layout(height=400)
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.warning(f"Could not create hourly chart: {str(e)[:100]}")
                        
                        with chart_col2:
                            # Fraud by merchant category
                            try:
                                fraud_by_merchant = fraud_df.groupby('merchant_category').size().reset_index(name='count')
                                all_by_merchant = results_df.groupby('merchant_category').size().reset_index(name='total')
                                
                                merchant_rates = all_by_merchant.merge(fraud_by_merchant, on='merchant_category', how='left')
                                merchant_rates['count'] = merchant_rates['count'].fillna(0)
                                merchant_rates['fraud_rate'] = (merchant_rates['count'] / merchant_rates['total']) * 100
                                merchant_rates = merchant_rates.sort_values('fraud_rate', ascending=False).head(10)
                                
                                fig = px.bar(
                                    merchant_rates, x='merchant_category', y='fraud_rate',
                                    title='Fraud Rate by Merchant Category',
                                    labels={'merchant_category': 'Merchant Category', 'fraud_rate': 'Fraud Rate (%)'},
                                    color='fraud_rate',
                                    color_continuous_scale='Reds',
                                    text=merchant_rates['fraud_rate'].round(1).astype(str) + '%'
                                )
                                fig.update_traces(textposition='outside')
                                fig.update_layout(xaxis_tickangle=-45, height=400)
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.warning(f"Could not create merchant chart: {str(e)[:100]}")
                        
                        # Risk level distribution
                        st.subheader("🎯 Risk Level Distribution")
                        try:
                            risk_counts = fraud_df['risk_level'].value_counts().reset_index()
                            risk_counts.columns = ['Risk Level', 'Count']
                            
                            risk_colors = {
                                'CRITICAL': '#ff0000',
                                'HIGH': '#ff8c00',
                                'MEDIUM': '#ffd700',
                                'LOW': '#4169e1'
                            }
                            
                            fig = go.Figure(data=[go.Pie(
                                labels=risk_counts['Risk Level'],
                                values=risk_counts['Count'],
                                marker_colors=[risk_colors.get(risk, '#cccccc') for risk in risk_counts['Risk Level']],
                                hole=0.3,
                                textinfo='label+percent',
                                textposition='auto'
                            )])
                            fig.update_layout(title='Fraud Transaction Risk Level Distribution', height=450)
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Could not create risk distribution chart: {str(e)[:100]}")
                        
                        # Show high-risk transactions (above threshold)
                        high_risk_frauds = fraud_df[fraud_df['confidence'] >= confidence_threshold]
                        
                        if not high_risk_frauds.empty:
                            st.subheader(f"⚠️ High-Risk Transactions (Confidence ≥ {confidence_threshold:.0%})")
                            st.warning(f"Found {len(high_risk_frauds)} high-risk transactions requiring immediate attention")
                            
                            high_risk_display = high_risk_frauds.copy()
                            high_risk_display['amount'] = high_risk_display['amount'].apply(lambda x: f"${x:,.2f}")
                            high_risk_display['confidence'] = high_risk_display['confidence'].apply(lambda x: f"{x:.1%}")
                            
                            st.dataframe(high_risk_display[['transaction_id', 'customer_id', 'amount', 
                                                           'merchant_category', 'confidence', 'risk_level']], 
                                        use_container_width=True)
                            
                            st.error("🚨 These transactions have been flagged as high-risk and should be reviewed immediately!")
                        else:
                            st.success(f"✅ No transactions above the {confidence_threshold:.0%} confidence threshold")
                        
                        # Show suspicious patterns if detailed analysis is enabled
                        if show_detailed and not fraud_df.empty:
                            st.subheader("🔍 Suspicious Pattern Detection")
                            
                            # Find customers with multiple frauds
                            fraud_customers = fraud_df['customer_id'].value_counts()
                            repeat_customers = fraud_customers[fraud_customers > 1]
                            
                            if not repeat_customers.empty:
                                st.warning("🚨 **Repeat Offenders Detected**")
                                st.write("Customers with multiple fraudulent transactions:")
                                for customer, count in repeat_customers.head(5).items():
                                    st.write(f"- Customer {customer}: {count} fraudulent transactions")
                            
                            # Find high-value frauds
                            high_value_frauds = fraud_df[fraud_df['amount'] > 1000]
                            if not high_value_frauds.empty:
                                st.warning("💰 **High-Value Frauds Detected**")
                                st.write(f"Found {len(high_value_frauds)} fraudulent transactions over $1,000")
                                high_value_display = high_value_frauds[['transaction_id', 'customer_id', 'amount', 'merchant_category']].copy()
                                high_value_display['amount'] = high_value_display['amount'].apply(lambda x: f"${x:,.2f}")
                                st.dataframe(high_value_display)
                            
                            # Late night fraud analysis
                            late_night_frauds = fraud_df[fraud_df['hour'].isin([0, 1, 2, 3, 4, 5, 22, 23])]
                            if not late_night_frauds.empty:
                                st.info("🌙 **Late Night Transactions**")
                                st.write(f"Found {len(late_night_frauds)} fraudulent transactions during unusual hours (22:00-6:00)")
                    else:
                        st.success("🎉 No fraudulent transactions detected in this batch!")
                    
                    # Download options
                    st.subheader("📥 Export Results")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        # Summary report
                        summary_report = results_df[['transaction_id', 'customer_id', 'amount', 
                                                    'merchant_category', 'prediction', 'confidence']].copy()
                        summary_report['amount'] = summary_report['amount'].apply(lambda x: f"${x:,.2f}")
                        summary_report['confidence'] = summary_report['confidence'].apply(lambda x: f"{x:.1%}")
                        csv_summary = summary_report.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Summary Report (CSV)",
                            data=csv_summary,
                            file_name="fraud_detection_summary.csv",
                            mime="text/csv"
                        )
                    
                    with col2:
                        # Full report with all details
                        full_report = results_df.copy()
                        full_report['amount'] = full_report['amount'].apply(lambda x: f"${x:,.2f}")
                        full_report['confidence'] = full_report['confidence'].apply(lambda x: f"{x:.1%}")
                        csv_full = full_report.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Full Report (CSV)",
                            data=csv_full,
                            file_name="fraud_detection_full_report.csv",
                            mime="text/csv"
                        )
                    
                    # High-risk transaction report
                    if 'fraud_df' in locals() and not fraud_df.empty:
                        high_risk_for_export = fraud_df[fraud_df['confidence'] >= confidence_threshold]
                        if not high_risk_for_export.empty:
                            high_risk_export = high_risk_for_export.copy()
                            high_risk_export['amount'] = high_risk_export['amount'].apply(lambda x: f"${x:,.2f}")
                            high_risk_export['confidence'] = high_risk_export['confidence'].apply(lambda x: f"{x:.1%}")
                            csv_high_risk = high_risk_export.to_csv(index=False)
                            st.download_button(
                                label="🚨 Download High-Risk Transactions (CSV)",
                                data=csv_high_risk,
                                file_name="high_risk_transactions.csv",
                                mime="text/csv"
                            )
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

# ============================================
# EDA PAGE (COMPLETELY FIXED - REMOVES DENSEVECTOR)
# ============================================
def eda_page():
    st.header("📊 Exploratory Data Analysis")
    
    with st.spinner("Loading data..."):
        df = load_eda_data()
    
    if df.empty:
        st.warning("""
        No training data available. 
        
        **Please ensure:**
        - `processed_fraud_data_balanced.parquet` folder exists
        - Run the training script first to generate the parquet file
        """)
        return
    
    # Remove non-serializable columns (DenseVector, etc.)
    columns_to_drop = []
    for col in df.columns:
        # Check if column contains DenseVector or other non-serializable objects
        if len(df) > 0:
            first_val = df[col].iloc[0] if len(df) > 0 else None
            if 'DenseVector' in str(type(first_val)) or 'SparseVector' in str(type(first_val)):
                columns_to_drop.append(col)
    
    if columns_to_drop:
        st.info(f"Removing non-displayable columns: {', '.join(columns_to_drop)}")
        df = df.drop(columns=columns_to_drop)
    
    # Also drop any columns that might cause issues
    problem_columns = ['features_scaled', 'features_raw', 'features']
    for col in problem_columns:
        if col in df.columns:
            df = df.drop(columns=[col])
            st.info(f"Removed '{col}' column for display purposes")
    
    # Create a copy to avoid modifying original
    df_copy = df.copy()
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💰 Amount Analysis", 
        "⏰ Time Patterns", 
        "🎯 Fraud Insights",
        "🏪 Merchant Analysis",
        "📊 Data Overview"
    ])
    
    with tab1:
        st.subheader("Transaction Amount Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Amount distribution histogram
            fig = px.histogram(
                df_copy, x='amount', nbins=50, 
                title='Distribution of Transaction Amounts',
                labels={'amount': 'Amount ($)', 'count': 'Number of Transactions'},
                color_discrete_sequence=['#667eea']
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Box plot for amount by fraud status
            fig = px.box(
                df_copy, y='amount', x='target',
                title='Amount Distribution by Transaction Type',
                labels={'amount': 'Amount ($)', 'target': 'Is Fraud'},
                color='target',
                color_discrete_map={0: '#4facfe', 1: '#f5576c'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Amount statistics
        col3, col4, col5, col6 = st.columns(4)
        with col3:
            st.metric("Mean Amount", f"${df_copy['amount'].mean():.2f}")
        with col4:
            st.metric("Median Amount", f"${df_copy['amount'].median():.2f}")
        with col5:
            st.metric("Max Amount", f"${df_copy['amount'].max():.2f}")
        with col6:
            st.metric("Min Amount", f"${df_copy['amount'].min():.2f}")
        
        # Amount by category
        if 'amount_category' in df_copy.columns:
            st.subheader("Amount Categories Distribution")
            amount_cat_counts = df_copy['amount_category'].value_counts().reset_index()
            amount_cat_counts.columns = ['category', 'count']
            fig = px.pie(
                amount_cat_counts, values='count', names='category',
                title='Transaction Volume by Amount Category',
                color_discrete_sequence=px.colors.sequential.Purples_r
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Time Pattern Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Hourly distribution
            hourly_counts = df_copy.groupby('hour').size().reset_index(name='count')
            fig = px.line(
                hourly_counts, x='hour', y='count',
                title='Transactions by Hour of Day',
                labels={'hour': 'Hour (0-23)', 'count': 'Number of Transactions'},
                markers=True,
                color_discrete_sequence=['#667eea']
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Day of week distribution
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            dow_counts = df_copy.groupby('day_of_week').size().reset_index(name='count')
            dow_counts['day_name'] = dow_counts['day_of_week'].map(lambda x: days[int(x)-1] if 1 <= x <= 7 else 'Unknown')
            
            fig = px.bar(
                dow_counts, x='day_name', y='count',
                title='Transactions by Day of Week',
                labels={'day_name': 'Day', 'count': 'Number of Transactions'},
                color_discrete_sequence=['#764ba2']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Fraud by hour
        st.subheader("Fraud Patterns by Hour")
        hourly_fraud = df_copy.groupby('hour')['target'].mean().reset_index(name='fraud_rate')
        fig = px.bar(
            hourly_fraud, x='hour', y='fraud_rate',
            title='Fraud Rate by Hour of Day',
            labels={'hour': 'Hour', 'fraud_rate': 'Fraud Rate'},
            color='fraud_rate',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Fraud by day of week
        dow_fraud = df_copy.groupby('day_of_week')['target'].mean().reset_index(name='fraud_rate')
        dow_fraud['day_name'] = dow_fraud['day_of_week'].map(lambda x: days[int(x)-1] if 1 <= x <= 7 else 'Unknown')
        fig = px.bar(
            dow_fraud, x='day_name', y='fraud_rate',
            title='Fraud Rate by Day of Week',
            labels={'day_name': 'Day', 'fraud_rate': 'Fraud Rate'},
            color='fraud_rate',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Fraud Insights")
        
        # Overall statistics
        fraud_rate = df_copy['target'].mean() * 100
        fraud_count = df_copy['target'].sum()
        legit_count = len(df_copy) - fraud_count
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Transactions", f"{len(df_copy):,}")
        with col2:
            st.metric("Fraudulent Transactions", f"{fraud_count:,}", delta=f"{fraud_rate:.1f}%")
        with col3:
            st.metric("Legitimate Transactions", f"{legit_count:,}", delta=f"{100-fraud_rate:.1f}%")
        with col4:
            ratio = legit_count/fraud_count if fraud_count > 0 else 0
            st.metric("Fraud Ratio", f"1:{ratio:.0f}")
        
        # Fraud by amount range - create temporary column for visualization only
        st.subheader("Fraud by Amount Range")
        temp_df = df_copy.copy()
        temp_df['amount_range'] = pd.cut(
            temp_df['amount'], 
            bins=[0, 100, 500, 1000, 5000, float('inf')], 
            labels=['<$100', '$100-500', '$500-1000', '$1000-5000', '>$5000']
        )
        fraud_by_amount = temp_df.groupby('amount_range', observed=True)['target'].mean().reset_index()
        fig = px.bar(
            fraud_by_amount, x='amount_range', y='target',
            title='Fraud Rate by Amount Range',
            labels={'amount_range': 'Amount Range', 'target': 'Fraud Rate'},
            color_discrete_sequence=['#f5576c']
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Customer risk profile
        st.subheader("Customer Risk Profile")
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                df_copy, x='customer_fraud_rate', nbins=50,
                title='Distribution of Customer Historical Fraud Rates',
                labels={'customer_fraud_rate': 'Customer Fraud Rate', 'count': 'Number of Customers'},
                color_discrete_sequence=['#00f2fe']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(
                df_copy, y='customer_fraud_rate', x='target',
                title='Customer Fraud Rate by Transaction Type',
                labels={'customer_fraud_rate': 'Customer Fraud Rate', 'target': 'Is Fraud'},
                color='target',
                color_discrete_map={0: '#4facfe', 1: '#f5576c'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Customer transaction patterns
        st.subheader("Customer Transaction Patterns")
        col1, col2 = st.columns(2)
        
        with col1:
            # Sample data for scatter plot
            sample_size = builtins.min(5000, len(df_copy))
            sampled_df = df_copy.sample(n=sample_size, random_state=42) if sample_size < len(df_copy) else df_copy
            
            fig = px.scatter(
                sampled_df, 
                x='customer_transaction_count', 
                y='customer_fraud_count',
                color='target',
                title='Customer Transactions vs Fraud Count',
                labels={
                    'customer_transaction_count': 'Total Transactions',
                    'customer_fraud_count': 'Previous Fraud Count',
                    'target': 'Is Fraud'
                },
                color_discrete_map={0: '#4facfe', 1: '#f5576c'},
                opacity=0.6
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Fraud rate by customer transaction count
            temp_df = df_copy.copy()
            temp_df['txn_bin'] = pd.cut(temp_df['customer_transaction_count'], bins=10)
            fraud_by_txn = temp_df.groupby('txn_bin', observed=True)['target'].mean().reset_index()
            # Convert bin to string for display
            fraud_by_txn['txn_bin_str'] = fraud_by_txn['txn_bin'].astype(str)
            fig = px.line(
                fraud_by_txn, x='txn_bin_str', y='target',
                title='Fraud Rate by Customer Transaction Volume',
                labels={'txn_bin_str': 'Transaction Count Range', 'target': 'Fraud Rate'},
                markers=True
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Merchant Analysis")
        
        # Top merchants by transaction count
        merchant_counts = df_copy['merchant_category'].value_counts().head(10).reset_index()
        merchant_counts.columns = ['merchant_category', 'count']
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                merchant_counts, x='merchant_category', y='count',
                title='Top 10 Merchant Categories by Transaction Volume',
                labels={'merchant_category': 'Merchant Category', 'count': 'Number of Transactions'},
                color_discrete_sequence=['#764ba2']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Fraud by merchant
        fraud_by_merchant = df_copy.groupby('merchant_category')['target'].mean().sort_values(ascending=False).head(10).reset_index()
        
        with col2:
            fig = px.bar(
                fraud_by_merchant, x='merchant_category', y='target',
                title='Top 10 High-Risk Merchant Categories',
                labels={'merchant_category': 'Merchant Category', 'target': 'Fraud Rate'},
                color='target',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Merchant fraud rate vs volatility
        st.subheader("Merchant Risk Profile")
        
        # Sample data for scatter plot (limit to avoid overcrowding)
        merchant_sample = df_copy.groupby('merchant_category').agg({
            'merchant_fraud_rate': 'first',
            'merchant_fraud_volatility': 'first',
            'target': 'mean'
        }).reset_index()
        
        fig = px.scatter(
            merchant_sample,
            x='merchant_fraud_rate',
            y='merchant_fraud_volatility',
            size='target',
            color='target',
            text='merchant_category',
            title='Merchant Risk Analysis: Fraud Rate vs Volatility',
            labels={
                'merchant_fraud_rate': 'Merchant Fraud Rate',
                'merchant_fraud_volatility': 'Merchant Volatility',
                'target': 'Observed Fraud Rate'
            },
            color_continuous_scale='Reds',
            size_max=40
        )
        fig.update_traces(textposition='top center')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.subheader("Dataset Overview")
        
        # Basic info
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Dataset Shape:**")
            st.write(f"📊 Rows: {len(df_copy):,}")
            st.write(f"📋 Columns: {len(df_copy.columns)}")
            
            st.write("**Target Variable Distribution:**")
            target_dist = df_copy['target'].value_counts().reset_index()
            target_dist.columns = ['Is Fraud', 'Count']
            target_dist['Is Fraud'] = target_dist['Is Fraud'].map({0: 'Legitimate', 1: 'Fraud'})
            st.dataframe(target_dist, use_container_width=True)
        
        with col2:
            st.write("**Missing Values:**")
            missing = df_copy.isnull().sum()
            missing = missing[missing > 0]
            if len(missing) > 0:
                st.dataframe(pd.DataFrame({'Missing Count': missing}), use_container_width=True)
            else:
                st.success("✅ No missing values found in the dataset!")
        
        # Feature columns
        st.write("**Feature Columns:**")
        feature_cols = ['amount', 'hour', 'day_of_week', 'customer_fraud_rate', 
                       'merchant_fraud_rate', 'customer_transaction_count', 
                       'customer_fraud_count', 'merchant_fraud_volatility']
        
        available_features = [col for col in feature_cols if col in df_copy.columns]
        st.write(f"✅ {len(available_features)} features available:")
        
        # Create a dataframe with feature statistics
        feature_stats = df_copy[available_features].describe().round(4)
        st.dataframe(feature_stats, use_container_width=True)
        
        # Correlation matrix - COMPLETELY FIXED VERSION
        st.subheader("Feature Correlation Matrix")
        
        try:
            # Select only numeric columns for correlation
            numeric_cols = available_features + ['target']
            # Ensure all columns are numeric
            correlation_data = df_copy[numeric_cols].select_dtypes(include=[np.number])
            
            if len(correlation_data.columns) > 1 and len(correlation_data) > 0:
                # Calculate correlation matrix
                corr_matrix = correlation_data.corr()
                
                # Create heatmap using plotly express (more stable)
                fig = px.imshow(
                    corr_matrix,
                    text_auto='.2f',
                    aspect='auto',
                    color_continuous_scale='RdBu',
                    title='Feature Correlation Heatmap',
                    labels=dict(color="Correlation"),
                    zmin=-1,
                    zmax=1
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough numeric columns for correlation matrix")
        except Exception as e:
            st.warning(f"Could not generate correlation matrix: {str(e)[:100]}")
            # Fallback: show simple correlation table
            try:
                numeric_cols = available_features + ['target']
                correlation_data = df_copy[numeric_cols].select_dtypes(include=[np.number])
                if len(correlation_data.columns) > 1:
                    corr_matrix = correlation_data.corr()
                    st.dataframe(corr_matrix.round(4), use_container_width=True)
            except:
                pass
        
        # Sample data - remove any remaining problematic columns
        st.subheader("Sample Data (first 20 rows)")
        display_df = df_copy.head(20).copy()
        
        # Convert any remaining non-serializable columns to strings
        for col in display_df.columns:
            try:
                # Try to see if column has DenseVector
                if len(display_df) > 0:
                    sample_val = display_df[col].iloc[0]
                    if 'DenseVector' in str(type(sample_val)) or 'Vector' in str(type(sample_val)):
                        display_df[col] = display_df[col].astype(str)
            except:
                pass
        
        st.dataframe(display_df, use_container_width=True)
        
        # Data types
        with st.expander("🔧 Column Data Types"):
            dtypes_df = pd.DataFrame(df_copy.dtypes.reset_index())
            dtypes_df.columns = ['Column Name', 'Data Type']
            st.dataframe(dtypes_df, use_container_width=True)

# ============================================
# MODEL COMPARISON PAGE (KEEP THIS ONE)
# ============================================
def model_comparison_page(models, config, assembler, scaler):
    st.header("📈 Model Performance Comparison")
    
    if not models:
        st.error("No models available.")
        return
    
    # Load model info to show metrics
    if os.path.exists("models/model_info.json"):
        with open("models/model_info.json", "r") as f:
            model_info = json.load(f)
        
        if model_info.get('metrics'):
            st.subheader("📊 Model Performance Metrics (from training)")
            metrics_df = pd.DataFrame(model_info['metrics']).T
            metrics_df = metrics_df.round(4)
            
            # Highlight the best model
            st.dataframe(metrics_df, use_container_width=True)
            
            # Create comparison chart
            fig = px.bar(
                metrics_df.reset_index().melt(id_vars='index', value_vars=['F1-Score', 'AUC-ROC']),
                x='index', y='value', color='variable',
                title='Model Performance Comparison',
                labels={'index': 'Model', 'value': 'Score', 'variable': 'Metric'},
                barmode='group',
                color_discrete_sequence=['#667eea', '#764ba2']
            )
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("🔮 Test Predictions on New Data")
    
    col1, col2 = st.columns(2)
    with col1:
        test_amount = st.number_input("Transaction Amount ($)", 0.01, 10000.0, 500.0, step=100.0)
        test_merchant = st.selectbox("Merchant Category", 
                                    ["electronics", "clothing", "grocery", "travel", "entertainment", 
                                     "restaurant", "home", "health", "automotive", "jewelry"])
    with col2:
        test_hour = st.slider("Hour of Day", 0, 23, 14)
        test_day = st.selectbox("Day of Week", 
                               ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        test_day_num = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(test_day) + 1
    
    if st.button("🚀 Compare All Models", type="primary", use_container_width=True):
        with st.spinner("Running predictions on all models..."):
            results = []
            for name, model in models.items():
                pred, prob = predict_transaction(
                    model, test_amount, test_merchant, test_hour, test_day_num,
                    assembler=assembler, scaler=scaler, config=config
                )
                results.append({
                    "Model": name,
                    "Prediction": "🚨 FRAUD" if pred == 1 else "✅ LEGITIMATE",
                    "Confidence": f"{prob:.1%}" if prob else "N/A",
                    "Confidence_Value": prob if prob else 0
                })
        
        # Display results
        results_df = pd.DataFrame(results)
        
        # Metrics row
        col1, col2, col3 = st.columns(3)
        with col1:
            fraud_models = len([r for r in results if "FRAUD" in r['Prediction']])
            st.metric("Models Predicting Fraud", f"{fraud_models}/{len(models)}")
        with col2:
            avg_confidence = results_df['Confidence_Value'].mean()
            st.metric("Average Confidence", f"{avg_confidence:.1%}")
        with col3:
            max_conf_model = results_df.loc[results_df['Confidence_Value'].idxmax(), 'Model']
            st.metric("Most Confident Model", max_conf_model)
        
        # Confidence comparison chart
        fig = px.bar(
            results_df, x='Model', y='Confidence_Value',
            title='Model Confidence Comparison',
            labels={'Model': 'Model', 'Confidence_Value': 'Confidence'},
            color='Prediction',
            color_discrete_map={'🚨 FRAUD': '#f5576c', '✅ LEGITIMATE': '#4facfe'}
        )
        fig.update_layout(showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed results
        st.subheader("🔍 Detailed Results")
        cols = st.columns(builtins.min(len(models), 3))
        for idx, result in enumerate(results):
            with cols[idx % len(cols)]:
                if "FRAUD" in result['Prediction']:
                    st.markdown(f"""
                    <div class="fraud-card" style="margin: 10px 0; padding: 1rem;">
                        <h4>{result['Model']}</h4>
                        <h3>{result['Prediction']}</h3>
                        <p>Confidence: {result['Confidence']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="legit-card" style="margin: 10px 0; padding: 1rem;">
                        <h4>{result['Model']}</h4>
                        <h3>{result['Prediction']}</h3>
                        <p>Confidence: {result['Confidence']}</p>
                    </div>
                    """, unsafe_allow_html=True)

# ============================================
# DIAGNOSTIC PAGE
# ============================================
def model_diagnostics_page(models, config, assembler, scaler):
    st.header("🔧 Model Diagnostics")
    
    if not models:
        st.error("No models available.")
        return
    
    st.subheader("📊 Feature Distribution in Training")
    
    # Load a sample of training data to show feature ranges
    try:
        df_sample = load_eda_data(limit=1000)
        if not df_sample.empty:
            feature_cols = config.get('feature_columns', [])
            available_features = [f for f in feature_cols if f in df_sample.columns]
            
            if available_features:
                st.write("**Feature Statistics from Training Data:**")
                st.dataframe(df_sample[available_features].describe())
    except:
        pass
    
    st.subheader("🧪 Test with Custom Values")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Test Case 1: Normal Transaction**")
        test_amount = 50.0
        test_merchant = "grocery"
        test_hour = 14
        
    with col2:
        st.write("**Test Case 2: Suspicious Transaction**")
        test_amount_high = 5000.0
        test_merchant_high = "electronics"
        test_hour_high = 3
    
    if st.button("Run Diagnostic Tests"):
        results_data = []
        
        for name, model in models.items():
            # Test normal transaction
            pred1, prob1 = predict_transaction(
                model, 50, "grocery", 14, 4,
                assembler=assembler, scaler=scaler, config=config
            )
            
            # Test suspicious transaction
            pred2, prob2 = predict_transaction(
                model, 5000, "electronics", 3, 4,
                customer_fraud_count=2,
                assembler=assembler, scaler=scaler, config=config
            )
            
            results_data.append({
                "Model": name,
                "Normal Pred": "FRAUD" if pred1 == 1 else "LEGIT",
                "Normal Conf": f"{prob1:.1%}",
                "Suspicious Pred": "FRAUD" if pred2 == 1 else "LEGIT",
                "Suspicious Conf": f"{prob2:.1%}"
            })
        
        st.dataframe(pd.DataFrame(results_data), use_container_width=True)
        
        # Warning if models always predict same class
        all_fraud = all(r["Suspicious Pred"] == "FRAUD" for r in results_data)
        all_legit = all(r["Normal Pred"] == "LEGIT" for r in results_data)
        
        if all_fraud and all_legit:
            st.success("✅ Models are distinguishing between normal and suspicious transactions")
        elif not all_fraud:
            st.warning("⚠️ Models are NOT flagging suspicious transactions as fraud!")
        elif not all_legit:
            st.warning("⚠️ Models are flagging normal transactions as fraud!")

# ============================================
# ABOUT PAGE
# ============================================
def about_page(model_info):
    st.header("ℹ️ About This Project")
    
    st.markdown("""
    ### 🛡️ Enterprise Fraud Detection System
    
    This system uses machine learning to detect fraudulent transactions in real-time.
    """)
    
    if model_info and model_info.get('metrics'):
        st.subheader("📊 Model Performance Summary")
        metrics_df = pd.DataFrame(model_info['metrics']).T
        metrics_df = metrics_df.round(4)
        st.dataframe(metrics_df, use_container_width=True)

# ============================================
# MAIN APP
# ============================================
def main():
    st.markdown("""
    <div class="main-header">
        <h1>🛡️ Advanced Fraud Detection System</h1>
        <p>Real-time transaction fraud detection powered by PySpark MLlib</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("## 🎯 Navigation")
        
        # In your sidebar, add this page option:
        page = st.radio(
            "Select Page",
            ["🎯 Single Prediction", "📁 Batch Upload", "📊 EDA & Insights", 
            "📈 Model Comparison", "🔧 Diagnostics", "🔍 Deep Diagnostic",
            "🔬 Model Behavior Analysis", "🎯 Case Analysis", "ℹ️ About"]
        )
        
        st.markdown("---")
        st.markdown("### 📊 System Status")
        
        with st.spinner("Loading models..."):
            models, model_info, config, merchants, assembler, scaler, load_errors = load_models()
        
        if models:
            st.success(f"✅ {len(models)} models loaded successfully")
            for name in models.keys():
                st.caption(f"• {name}")
        else:
            st.error("❌ No models loaded successfully")
        
        # Show feature importance - FIXED: proper progress bar usage
        if models and config:
            feature_columns = config.get('feature_columns', [])
            if feature_columns:
                importance_dict = get_feature_importance(models, feature_columns)
                
                with st.expander("📊 Feature Importance"):
                    for model_name, importance in importance_dict.items():
                        if importance:  # Only show if not empty
                            st.write(f"**{model_name}**")
                            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3]
                            for feat, imp in sorted_features:
                                if imp > 0:
                                    st.progress(builtins.min(imp, 1.0))  # Progress bar only
                                    st.caption(f"{feat}: {imp:.1%}")  # Text label below

        # Show any load errors in an expandable section
        if load_errors:
            with st.expander(f"⚠️ Load Issues ({len(load_errors)})"):
                for error in load_errors:
                    st.warning(f"• {error}")
    
    # Page routing
    if page == "🎯 Single Prediction":
        single_prediction_page(models, model_info, merchants, config, assembler, scaler)
    elif page == "📁 Batch Upload":
        batch_upload_page(models, config, assembler, scaler)
    elif page == "📊 EDA & Insights":
        eda_page()
    elif page == "📈 Model Comparison":
        model_comparison_page(models, config, assembler, scaler)
    elif page == "🔧 Diagnostics":
        model_diagnostics_page(models, config, assembler, scaler)
    elif page == "🔍 Deep Diagnostic":
        deep_diagnostic(models, config, assembler, scaler)
    elif page == "🔬 Model Behavior Analysis":
        model_behavior_analysis(models, config, assembler, scaler)
    elif page == "🎯 Case Analysis":
        analyze_specific_case()
    else:
        about_page(model_info)

# ============================================
# RUN APP
# ============================================
if __name__ == "__main__":
    main()
