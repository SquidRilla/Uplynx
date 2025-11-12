"""
Predictive Maintenance System - Complete Database Integration
Combines ML capabilities with PostgreSQL database operations
"""

import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import json

class DataAcquisition:
    """Enhanced data acquisition with direct PostgreSQL integration"""
    
    def __init__(self, db_config):
        self.db_config = db_config
        self.engine = self.create_connection()
        self.conn = None
        self.cursor = None
        self.connect_psycopg2()
    
    def create_connection(self):
        """Create SQLAlchemy engine for pandas operations"""
        try:
            engine = create_engine(
                f"postgresql://{self.db_config['user']}:{self.db_config['password']}"
                f"@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
            )
            print("✅ SQLAlchemy connection established")
            return engine
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            return None
    
    def connect_psycopg2(self):
        """Create psycopg2 connection for raw SQL"""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            print("✅ psycopg2 connection established")
        except Exception as e:
            print(f"❌ psycopg2 connection failed: {e}")
    
    def fetch_sensor_data(self, component_type, hours=720):
        """Fetch sensor data from the actual database tables"""
        table_mapping = {
            'battery': 'battery_data',
            'solar_panel': 'solar_panel_data',
            'generator': 'generator_data'
        }
        
        table_name = table_mapping.get(component_type)
        if not table_name:
            print(f"❌ Unknown component type: {component_type}")
            return pd.DataFrame()
        
        try:
            query = f"""
                SELECT * FROM {table_name} 
                WHERE timestamp >= NOW() - INTERVAL '{hours} hours'
                ORDER BY timestamp ASC
            """
            df = pd.read_sql(query, self.engine)
            print(f"📊 Fetched {len(df)} records for {component_type}")
            return df
        except Exception as e:
            print(f"❌ Error fetching {component_type} data: {e}")
            return pd.DataFrame()
    
    def fetch_maintenance_history(self):
        """Fetch maintenance schedule from database"""
        try:
            query = "SELECT * FROM maintenance_schedule ORDER BY scheduled_date DESC"
            df = pd.read_sql(query, self.engine)
            return df
        except Exception as e:
            print(f"❌ Error fetching maintenance history: {e}")
            return pd.DataFrame()
    
    def save_prediction(self, component_type, component_id, prediction_data):
        """Save ML predictions to database"""
        try:
            query = """
                INSERT INTO ml_predictions 
                (timestamp, component_type, component_id, predicted_health, 
                 predicted_rul, confidence_score, model_version, feature_values)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            self.cursor.execute(query, (
                datetime.now(),
                component_type,
                component_id,
                prediction_data.get('health_score'),
                prediction_data.get('rul'),
                prediction_data.get('confidence', 0.85),
                'v1.0',
                psycopg2.extras.Json(prediction_data.get('features', {}))
            ))
            self.conn.commit()
            print(f"💾 Saved prediction for {component_type}/{component_id}")
        except Exception as e:
            print(f"❌ Error saving prediction: {e}")
            self.conn.rollback()
    
    def save_health_history(self, component_type, component_id, health_score, rul_days):
        """Save to health_history table"""
        try:
            query = """
                INSERT INTO health_history 
                (timestamp, component_type, component_id, health_score, rul_days, prediction_confidence)
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            
            self.cursor.execute(query, (
                datetime.now(),
                component_type,
                component_id,
                health_score,
                rul_days,
                0.85
            ))
            self.conn.commit()
        except Exception as e:
            print(f"❌ Error saving health history: {e}")
            self.conn.rollback()
    
    def save_alert(self, alert_data):
        """Save alert to database"""
        try:
            query = """
                INSERT INTO alerts 
                (timestamp, component_type, component_id, severity, alert_type, message)
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            
            self.cursor.execute(query, (
                alert_data['timestamp'],
                alert_data['component_type'],
                alert_data['component_id'],
                alert_data['severity'],
                alert_data.get('alert_type', 'health_warning'),
                alert_data['message']
            ))
            self.conn.commit()
        except Exception as e:
            print(f"❌ Error saving alert: {e}")
            self.conn.rollback()

class DataPreprocessor:
    """Data preprocessing with enhanced time-series handling"""
    
    def __init__(self):
        self.scalers = {}
        self.imputation_values = {}
    
    def handle_missing_values(self, df, strategy='forward_fill'):
        """Handle missing values in sensor data"""
        df_clean = df.copy()
        
        # Remove duplicate timestamps
        df_clean = df_clean.drop_duplicates(subset=['timestamp'])
        
        # Set timestamp as index
        if 'timestamp' in df_clean.columns:
            df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'])
            df_clean = df_clean.set_index('timestamp').sort_index()
        
        # Handle missing values
        if strategy == 'forward_fill':
            df_clean = df_clean.ffill().bfill()
        elif strategy == 'interpolate':
            df_clean = df_clean.interpolate(method='time')
        
        # Store imputation values
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for column in numeric_cols:
            if df_clean[column].isna().any():
                self.imputation_values[column] = df_clean[column].median()
                df_clean[column].fillna(self.imputation_values[column], inplace=True)
        
        return df_clean.reset_index()
    
    def remove_outliers(self, df, method='IQR'):
        """Remove outliers using IQR method"""
        df_clean = df.copy()
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            Q1 = df_clean[column].quantile(0.25)
            Q3 = df_clean[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Clip outliers
            df_clean[column] = np.clip(df_clean[column], lower_bound, upper_bound)
        
        return df_clean
    
    def normalize_data(self, df, feature_columns, method='standard'):
        """Normalize sensor data"""
        df_normalized = df.copy()
        
        if method == 'standard':
            scaler = StandardScaler()
        else:
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        
        df_normalized[feature_columns] = scaler.fit_transform(df_normalized[feature_columns])
        self.scalers[method] = scaler
        
        return df_normalized

class FeatureEngineer:
    """Feature engineering for predictive maintenance"""
    
    def __init__(self):
        self.feature_config = {}
    
    def create_time_features(self, df, timestamp_col='timestamp'):
        """Create time-based features"""
        df_featured = df.copy()
        
        if timestamp_col in df_featured.columns:
            df_featured[timestamp_col] = pd.to_datetime(df_featured[timestamp_col])
            
            df_featured['hour'] = df_featured[timestamp_col].dt.hour
            df_featured['day_of_week'] = df_featured[timestamp_col].dt.dayofweek
            df_featured['month'] = df_featured[timestamp_col].dt.month
            df_featured['is_weekend'] = (df_featured[timestamp_col].dt.dayofweek >= 5).astype(int)
            
            # Cyclical encoding
            df_featured['hour_sin'] = np.sin(2 * np.pi * df_featured['hour']/24)
            df_featured['hour_cos'] = np.cos(2 * np.pi * df_featured['hour']/24)
        
        return df_featured
    
    def create_rolling_features(self, df, numeric_columns, windows=[6, 12]):
        """Create rolling window statistics"""
        df_rolled = df.copy()
        
        for column in numeric_columns[:5]:  # Limit to avoid too many features
            for window in windows:
                df_rolled[f'{column}_rolling_mean_{window}'] = df_rolled[column].rolling(window, min_periods=1).mean()
                df_rolled[f'{column}_rolling_std_{window}'] = df_rolled[column].rolling(window, min_periods=1).std().fillna(0)
        
        df_rolled = df_rolled.fillna(0)
        return df_rolled
    
    def create_component_specific_features(self, df, component_type):
        """Create component-specific features"""
        df_featured = df.copy()
        
        if component_type == 'battery':
            if 'voltage' in df.columns and 'current' in df.columns:
                df_featured['power'] = df_featured['voltage'] * df_featured['current']
            if 'soc' in df.columns:
                df_featured['soc_derivative'] = df_featured['soc'].diff().fillna(0)
        
        elif component_type == 'solar_panel':
            if 'power_output' in df.columns and 'irradiance' in df.columns:
                df_featured['efficiency_ratio'] = df_featured['power_output'] / (df_featured['irradiance'] + 1e-5)
            if 'temperature' in df.columns:
                df_featured['temp_loss'] = np.maximum(0, df_featured['temperature'] - 25) * 0.004
        
        elif component_type == 'generator':
            if 'vibration' in df.columns:
                df_featured['vibration_high'] = (df_featured['vibration'] > 7).astype(int)
        
        return df_featured

class MLModelTrainer:
    """ML model training and evaluation"""
    
    def __init__(self):
        self.models = {}
        self.model_performance = {}
    
    def prepare_features_target(self, df, target_column, feature_columns):
        """Prepare features and target"""
        available_features = [col for col in feature_columns if col in df.columns]
        
        X = df[available_features].copy()
        y = df[target_column].copy()
        
        X = X.fillna(0)
        y = y.fillna(y.median())
        
        return X, y
    
    def train_random_forest(self, X_train, y_train, **params):
        """Train Random Forest model"""
        default_params = {
            'n_estimators': 100,
            'max_depth': 15,
            'min_samples_split': 5,
            'random_state': 42,
            'n_jobs': -1
        }
        default_params.update(params)
        
        model = RandomForestRegressor(**default_params)
        model.fit(X_train, y_train)
        return model
    
    def train_gradient_boosting(self, X_train, y_train, **params):
        """Train Gradient Boosting model"""
        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'random_state': 42
        }
        default_params.update(params)
        
        model = GradientBoostingRegressor(**default_params)
        model.fit(X_train, y_train)
        return model
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate model performance"""
        y_pred = model.predict(X_test)
        
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred)
        }
        
        self.model_performance[model_name] = metrics
        return metrics

class PredictiveMaintenanceSystem:
    """Complete Predictive Maintenance System with Database Integration"""
    
    def __init__(self, db_config):
        self.db_config = db_config
        self.data_acquisition = DataAcquisition(db_config)
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.ml_trainer = MLModelTrainer()
        
        self.component_configs = {
            'battery': {
                'table': 'battery_data',
                'id_field': 'battery_id',
                'id_value': 'BAT001',
                'model_type': 'random_forest'
            },
            'solar_panel': {
                'table': 'solar_panel_data',
                'id_field': 'panel_id',
                'id_value': 'SP001',
                'model_type': 'gradient_boosting'
            },
            'generator': {
                'table': 'generator_data',
                'id_field': 'generator_id',
                'id_value': 'GEN001',
                'model_type': 'random_forest'
            }
        }
        
        self.models = {}
        self.scalers = {}
        self.feature_names = {}
        
        os.makedirs('models', exist_ok=True)
    
    def create_target_variable(self, df, component_type):
        """Create health score target variable"""
        if component_type == 'battery':
            health = 100.0
            if 'voltage' in df.columns:
                health -= np.where(df['voltage'] < 12, (12 - df['voltage']) * 10, 0)
            if 'temperature' in df.columns:
                health -= np.where(df['temperature'] > 40, (df['temperature'] - 40) * 2, 0)
            if 'internal_resistance' in df.columns:
                health -= np.where(df['internal_resistance'] > 0.08, 
                                 (df['internal_resistance'] - 0.08) * 200, 0)
            return np.clip(health, 0, 100)
        
        elif component_type == 'solar_panel':
            if 'performance_ratio' in df.columns:
                return df['performance_ratio'] * 100
            else:
                performance = 90.0
                if 'temperature' in df.columns:
                    performance -= (df['temperature'] - 25) * 0.4
                return np.clip(performance, 0, 100)
        
        elif component_type == 'generator':
            health = 100.0
            if 'oil_pressure' in df.columns:
                health -= np.where(df['oil_pressure'] < 40, (40 - df['oil_pressure']) * 2, 0)
            if 'coolant_temp' in df.columns:
                health -= np.where(df['coolant_temp'] > 90, (df['coolant_temp'] - 90) * 3, 0)
            if 'vibration' in df.columns:
                health -= np.where(df['vibration'] > 7, (df['vibration'] - 7) * 5, 0)
            return np.clip(health, 0, 100)
        
        return np.ones(len(df)) * 85.0
    
    def train_component_model(self, component_type, hours=720):
        """Train ML model for specific component"""
        print(f"\n🚀 Training ML model for {component_type}...")
        start_time = datetime.now()
        
        # Fetch data
        df = self.data_acquisition.fetch_sensor_data(component_type, hours)
        
        if df.empty or len(df) < 50:
            print(f"❌ Insufficient data ({len(df)} records)")
            return None
        
        # Preprocess
        df_clean = self.preprocessor.handle_missing_values(df)
        df_clean = self.preprocessor.remove_outliers(df_clean)
        
        # Feature engineering
        df_featured = self.feature_engineer.create_time_features(df_clean)
        numeric_cols = df_featured.select_dtypes(include=[np.number]).columns.tolist()
        df_featured = self.feature_engineer.create_rolling_features(df_featured, numeric_cols)
        df_featured = self.feature_engineer.create_component_specific_features(df_featured, component_type)
        
        # Create target
        df_featured['target'] = self.create_target_variable(df_featured, component_type)
        
        # Select features
        exclude_cols = ['timestamp', 'id', 'created_at', 'target',
                       self.component_configs[component_type]['id_field'],
                       'health_status', 'charge_status']
        
        feature_cols = [col for col in df_featured.columns 
                       if col not in exclude_cols and df_featured[col].dtype in [np.float64, np.int64]]
        
        X, y = self.ml_trainer.prepare_features_target(df_featured, 'target', feature_cols)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train
        model_type = self.component_configs[component_type]['model_type']
        if model_type == 'random_forest':
            model = self.ml_trainer.train_random_forest(X_train_scaled, y_train)
        else:
            model = self.ml_trainer.train_gradient_boosting(X_train_scaled, y_train)
        
        # Evaluate
        metrics = self.ml_trainer.evaluate_model(model, X_test_scaled, y_test, component_type)
        print(f"📈 Performance - MAE: {metrics['mae']:.2f}, RMSE: {metrics['rmse']:.2f}, R²: {metrics['r2']:.3f}")
        
        # Save
        joblib.dump(model, f'models/{component_type}_model.pkl')
        joblib.dump(scaler, f'models/{component_type}_scaler.pkl')
        
        self.models[component_type] = model
        self.scalers[component_type] = scaler
        self.feature_names[component_type] = feature_cols
        
        duration = (datetime.now() - start_time).total_seconds()
        print(f"✅ Training completed in {duration:.2f}s")
        
        return model, metrics
    
    def predict_health(self, component_type, sensor_data):
        """Predict component health"""
        if component_type not in self.models:
            try:
                self.models[component_type] = joblib.load(f'models/{component_type}_model.pkl')
                self.scalers[component_type] = joblib.load(f'models/{component_type}_scaler.pkl')
            except:
                print(f"❌ No trained model for {component_type}")
                return None
        
        # Process sensor data
        df = pd.DataFrame([sensor_data])
        df_featured = self.feature_engineer.create_component_specific_features(df, component_type)
        
        # Align features
        if component_type in self.feature_names:
            feature_cols = self.feature_names[component_type]
            for col in feature_cols:
                if col not in df_featured.columns:
                    df_featured[col] = 0
            X = df_featured[feature_cols]
        else:
            X = df_featured.select_dtypes(include=[np.number])
        
        # Predict
        X_scaled = self.scalers[component_type].transform(X)
        health_score = self.models[component_type].predict(X_scaled)[0]
        
        return float(health_score)
    
    def predict_rul(self, component_type, sensor_data, threshold=70):
        """Predict Remaining Useful Life"""
        health_score = self.predict_health(component_type, sensor_data)
        
        if health_score is None:
            return None
        
        degradation_rates = {
            'battery': 0.1,
            'solar_panel': 0.05,
            'generator': 0.15
        }
        
        rate = degradation_rates.get(component_type, 0.1)
        rul = (health_score - threshold) / rate if health_score > threshold else 0
        
        return max(float(rul), 0.0)
    
    def process_sensor_data(self, sensor_data):
        """Process sensor data and generate predictions"""
        predictions = {}
        alerts = []
        
        for component_key, data in sensor_data.items():
            # Map component names
            if 'solar' in component_key.lower():
                component_type = 'solar_panel'
            elif 'batter' in component_key.lower():
                component_type = 'battery'
            elif 'gen' in component_key.lower():
                component_type = 'generator'
            else:
                continue
            
            try:
                health = self.predict_health(component_type, data)
                rul = self.predict_rul(component_type, data)
                
                if health and rul:
                    predictions[component_key] = {
                        'health_score': health,
                        'rul': rul,
                        'timestamp': datetime.now()
                    }
                    
                    # Save to database
                    config = self.component_configs[component_type]
                    self.data_acquisition.save_prediction(
                        component_type, config['id_value'],
                        {'health_score': health, 'rul': rul, 'features': data}
                    )
                    
                    self.data_acquisition.save_health_history(
                        component_type, config['id_value'], health, rul
                    )
                    
                    # Generate alerts
                    if health < 70:
                        alert = {
                            'timestamp': datetime.now(),
                            'component_type': component_type,
                            'component_id': config['id_value'],
                            'severity': 'critical' if health < 60 else 'high',
                            'message': f'Low health score: {health:.1f}%'
                        }
                        alerts.append(alert)
                        self.data_acquisition.save_alert(alert)
                        
            except Exception as e:
                print(f"Error processing {component_key}: {e}")
        
        return predictions, alerts
    
    def generate_maintenance_schedule(self, predictions):
        """Generate maintenance schedule based on predictions"""
        schedule = {}
        
        for component, prediction in predictions.items():
            health_score = prediction['health_score']
            rul = prediction['rul']
            
            if health_score < 60 or rul < 7:
                priority = "CRITICAL"
                days = 1
            elif health_score < 70 or rul < 14:
                priority = "HIGH"
                days = 3
            elif health_score < 80 or rul < 30:
                priority = "MEDIUM"
                days = 7
            else:
                priority = "LOW"
                days = 30
            
            schedule[component] = {
                'priority': priority,
                'scheduled_date': datetime.now() + timedelta(days=days),
                'estimated_duration': 4,
                'required_parts': ['standard_kit']
            }
        
        return schedule


# Configuration
db_config = {
    'host': 'localhost',
    'port': 5432,
    'database': 'predictive_maintenance',
    'user': 'postgres',
    'password': '9090'
}

# Main execution
if __name__ == "__main__":
    print("="*70)
    print("PREDICTIVE MAINTENANCE SYSTEM")
    print("="*70)
    
    pm_system = PredictiveMaintenanceSystem(db_config)
    
    # Train models
    print("\n📚 Training Models...")
    for component in ['battery', 'solar_panel', 'generator']:
        pm_system.train_component_model(component, hours=720)
    
    # Make predictions
    print("\n🔮 Making Predictions...")
    sample_data = {
        'batteries': {
            'voltage': 12.8,
            'soc': 75.0,
            'temperature': 35.0,
            'internal_resistance': 0.05,
            'current': 25.0
        }
    }
    
    predictions, alerts = pm_system.process_sensor_data(sample_data)
    
    print("\n📊 Results:")
    for comp, pred in predictions.items():
        print(f"{comp}: Health={pred['health_score']:.1f}%, RUL={pred['rul']:.1f} days")
    
    print("\n✅ Complete!")