import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor, execute_batch
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
from decimal import Decimal


def convert_to_python_type(value):
    """Convert numpy/pandas types to Python native types"""
    if isinstance(value, (np.integer, np.int64, np.int32)):
        return int(value)
    elif isinstance(value, (np.floating, np.float64, np.float32)):
        return float(value)
    elif isinstance(value, Decimal):
        return float(value)
    elif pd.isna(value):
        return None
    return value


class MLDatabaseIntegration:
    """
    integration between ML Predictive Maintenance System and PostgreSQL Database
    """
    
    def __init__(self, db_config):
        self.db_config = db_config
        self.engine = None
        self.conn = None
        self.cursor = None
        self.connect()
        self.ensure_ml_tables()
    
    def connect(self):
        """Establish database connections"""
        try:
            # SQLAlchemy engine for pandas operations
            self.engine = create_engine(
                f"postgresql://{self.db_config['user']}:{self.db_config['password']}"
                f"@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
            )
            
            # psycopg2 connection for raw SQL operations
            self.conn = psycopg2.connect(**self.db_config)
            self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            
            print("✅ Database connections established")
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            raise
    
    def ensure_ml_tables(self):
        """Create ML-specific tables if they don't exist"""
        
        # ML Predictions table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS ml_predictions (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP NOT NULL,
                component_type VARCHAR(50) NOT NULL,
                component_id VARCHAR(50) NOT NULL,
                predicted_health NUMERIC(5,2),
                predicted_rul NUMERIC(8,1),
                confidence_score NUMERIC(4,3),
                model_version VARCHAR(50),
                feature_values JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Model Performance table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_performance (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP NOT NULL,
                component_type VARCHAR(50) NOT NULL,
                model_type VARCHAR(50) NOT NULL,
                mae NUMERIC(10,4),
                mse NUMERIC(10,4),
                rmse NUMERIC(10,4),
                r2_score NUMERIC(6,4),
                training_samples INTEGER,
                training_duration NUMERIC(8,2),
                model_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Feature Importance table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS feature_importance (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP NOT NULL,
                component_type VARCHAR(50) NOT NULL,
                feature_name VARCHAR(100) NOT NULL,
                importance_score NUMERIC(8,6),
                model_version VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # ML Training History table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS ml_training_history (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP NOT NULL,
                component_type VARCHAR(50) NOT NULL,
                training_start TIMESTAMP,
                training_end TIMESTAMP,
                samples_used INTEGER,
                features_used TEXT[],
                hyperparameters JSONB,
                status VARCHAR(20),
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.conn.commit()
        print("✅ ML tables verified/created")
    
    def fetch_training_data(self, component_type, hours=720):
        """Fetch historical sensor data for ML training"""
        
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
            
            # Convert all Decimal columns to float
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        df[col] = df[col].apply(lambda x: float(x) if isinstance(x, Decimal) else x)
                    except:
                        pass
            
            print(f"📊 Fetched {len(df)} records for {component_type} training")
            return df
            
        except Exception as e:
            print(f"❌ Error fetching training data: {e}")
            return pd.DataFrame()
    
    def fetch_latest_sensor_reading(self, component_type, component_id):
        """Fetch the most recent sensor reading for real-time prediction"""
        
        table_mapping = {
            'battery': 'battery_data',
            'solar_panel': 'solar_panel_data',
            'generator': 'generator_data'
        }
        
        id_column_mapping = {
            'battery': 'battery_id',
            'solar_panel': 'panel_id',
            'generator': 'generator_id'
        }
        
        table_name = table_mapping.get(component_type)
        id_column = id_column_mapping.get(component_type)
        
        if not table_name or not id_column:
            return None
        
        try:
            query = f"""
                SELECT * FROM {table_name}
                WHERE {id_column} = %s
                ORDER BY timestamp DESC
                LIMIT 1
            """
            
            self.cursor.execute(query, (component_id,))
            result = self.cursor.fetchone()
            
            if result:
                # Convert Decimals to floats
                result_dict = {}
                for key, value in dict(result).items():
                    result_dict[key] = convert_to_python_type(value)
                return result_dict
            return None
            
        except Exception as e:
            print(f"❌ Error fetching latest reading: {e}")
            return None
    
    def save_ml_prediction(self, component_type, component_id, prediction_data):
        """Save ML prediction to database"""
        try:
            query = """
                INSERT INTO ml_predictions 
                (timestamp, component_type, component_id, predicted_health, 
                 predicted_rul, confidence_score, model_version, feature_values)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            # Convert all numeric values to Python native types
            self.cursor.execute(query, (
                datetime.now(),
                component_type,
                component_id,
                convert_to_python_type(prediction_data.get('health_score')),
                convert_to_python_type(prediction_data.get('rul')),
                convert_to_python_type(prediction_data.get('confidence', 0.85)),
                prediction_data.get('model_version', 'v1.0'),
                psycopg2.extras.Json(prediction_data.get('features', {}))
            ))
            
            self.conn.commit()
            print(f"💾 Saved ML prediction for {component_type}/{component_id}")
            
        except Exception as e:
            print(f"❌ Error saving prediction: {e}")
            self.conn.rollback()
    
    def save_model_performance(self, component_type, model_type, metrics, training_info):
        """Save model performance metrics"""
        try:
            query = """
                INSERT INTO model_performance
                (timestamp, component_type, model_type, mae, mse, rmse, r2_score,
                 training_samples, training_duration, model_path)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            # Convert all metrics to Python native types
            self.cursor.execute(query, (
                datetime.now(),
                component_type,
                model_type,
                convert_to_python_type(metrics.get('mae')),
                convert_to_python_type(metrics.get('mse')),
                convert_to_python_type(metrics.get('rmse')),
                convert_to_python_type(metrics.get('r2')),
                int(training_info.get('samples')),
                convert_to_python_type(training_info.get('duration')),
                training_info.get('model_path')
            ))
            
            self.conn.commit()
            print(f"✅ Saved model performance for {component_type}")
            
        except Exception as e:
            print(f"❌ Error saving performance: {e}")
            self.conn.rollback()
    
    def save_feature_importance(self, component_type, feature_importance_dict, model_version='v1.0'):
        """Save feature importance scores"""
        try:
            records = [
                (
                    datetime.now(), 
                    component_type, 
                    feature_name, 
                    convert_to_python_type(importance),  # Convert numpy type
                    model_version
                )
                for feature_name, importance in feature_importance_dict.items()
            ]
            
            query = """
                INSERT INTO feature_importance
                (timestamp, component_type, feature_name, importance_score, model_version)
                VALUES (%s, %s, %s, %s, %s)
            """
            
            execute_batch(self.cursor, query, records)
            self.conn.commit()
            print(f"✅ Saved {len(records)} feature importance scores")
            
        except Exception as e:
            print(f"❌ Error saving feature importance: {e}")
            self.conn.rollback()
    
    def log_training_session(self, component_type, training_info):
        """Log ML training session details"""
        try:
            query = """
                INSERT INTO ml_training_history
                (timestamp, component_type, training_start, training_end,
                 samples_used, features_used, hyperparameters, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            self.cursor.execute(query, (
                datetime.now(),
                component_type,
                training_info.get('start_time'),
                training_info.get('end_time'),
                int(training_info.get('samples')),
                training_info.get('features'),
                psycopg2.extras.Json(training_info.get('hyperparameters', {})),
                training_info.get('status', 'success')
            ))
            
            self.conn.commit()
            
        except Exception as e:
            print(f"❌ Error logging training: {e}")
            self.conn.rollback()
    
    def get_prediction_history(self, component_type, component_id, days=30):
        """Get historical predictions for analysis"""
        try:
            query = """
                SELECT 
                    timestamp,
                    predicted_health,
                    predicted_rul,
                    confidence_score
                FROM ml_predictions
                WHERE component_type = %s
                  AND component_id = %s
                  AND timestamp >= NOW() - INTERVAL '%s days'
                ORDER BY timestamp DESC
            """
            
            df = pd.read_sql(query, self.engine, params=(component_type, component_id, days))
            return df
            
        except Exception as e:
            print(f"❌ Error fetching prediction history: {e}")
            return pd.DataFrame()
    
    def close(self):
        """Close database connections"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        if self.engine:
            self.engine.dispose()
        print("✅ Database connections closed")


class IntegratedPredictiveMaintenanceSystem:
    """
    Complete ML Predictive Maintenance System with Database Integration
    """
    
    def __init__(self, db_config):
        self.db_config = db_config
        self.db_integration = MLDatabaseIntegration(db_config)
        
        # Component configurations
        self.component_configs = {
            'battery': {
                'sensor_columns': ['voltage', 'soc', 'temperature', 'internal_resistance', 'current'],
                'target_column': 'health_score',
                'model_type': 'random_forest',
                'id_field': 'battery_id'
            },
            'solar_panel': {
                'sensor_columns': ['voltage', 'current', 'temperature', 'irradiance', 'power_output', 'efficiency'],
                'target_column': 'performance_ratio',
                'model_type': 'gradient_boosting',
                'id_field': 'panel_id'
            },
            'generator': {
                'sensor_columns': ['oil_pressure', 'coolant_temp', 'vibration', 'rpm', 'load_percentage'],
                'target_column': 'health_status_numeric',
                'model_type': 'random_forest',
                'id_field': 'generator_id'
            }
        }
        
        self.models = {}
        self.scalers = {}
        self.feature_names = {}
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
    
    def preprocess_data(self, df, component_type):
        """Preprocess sensor data for ML"""
        df_clean = df.copy()
        
        # Convert all Decimal columns to float
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                try:
                    df_clean[col] = df_clean[col].apply(lambda x: float(x) if isinstance(x, Decimal) else x)
                except:
                    pass
        
        # Remove duplicates
        if 'timestamp' in df_clean.columns:
            df_clean = df_clean.drop_duplicates(subset=['timestamp'])
            df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'])
            df_clean = df_clean.sort_values('timestamp')
        
        # Get numeric columns
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        
        # Handle missing values
        for col in numeric_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        # Remove outliers (IQR method)
        for col in numeric_cols:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df_clean[col] = df_clean[col].clip(lower, upper)
        
        return df_clean
    
    def engineer_features(self, df, component_type):
        """Create engineered features"""
        df_featured = df.copy()
        
        # Time-based features
        if 'timestamp' in df_featured.columns:
            df_featured['timestamp'] = pd.to_datetime(df_featured['timestamp'])
            df_featured['hour'] = df_featured['timestamp'].dt.hour
            df_featured['day_of_week'] = df_featured['timestamp'].dt.dayofweek
            df_featured['hour_sin'] = np.sin(2 * np.pi * df_featured['hour'] / 24)
            df_featured['hour_cos'] = np.cos(2 * np.pi * df_featured['hour'] / 24)
        
        # Get numeric columns
        numeric_cols = df_featured.select_dtypes(include=[np.number]).columns.tolist()
        
        # Rolling features (if enough data)
        if len(df_featured) > 12:
            for col in numeric_cols[:5]:  # Limit to avoid too many features
                if col not in ['hour', 'day_of_week', 'hour_sin', 'hour_cos']:
                    df_featured[f'{col}_rolling_mean_6'] = df_featured[col].rolling(6, min_periods=1).mean()
                    df_featured[f'{col}_rolling_std_6'] = df_featured[col].rolling(6, min_periods=1).std().fillna(0)
        
        # Component-specific features
        if component_type == 'battery':
            if 'voltage' in df_featured.columns and 'current' in df_featured.columns:
                df_featured['power'] = df_featured['voltage'] * df_featured['current']
            if 'soc' in df_featured.columns:
                df_featured['soc_change'] = df_featured['soc'].diff().fillna(0)
        
        elif component_type == 'solar_panel':
            if all(col in df_featured.columns for col in ['power_output', 'irradiance']):
                # Convert to float to handle Decimal types
                power = df_featured['power_output'].astype(float)
                irradiance = df_featured['irradiance'].astype(float)
                df_featured['efficiency_ratio'] = power / (irradiance + 1e-5)
            if 'temperature' in df_featured.columns:
                temp = df_featured['temperature'].astype(float)
                df_featured['temp_loss'] = np.maximum(0, temp - 25.0) * 0.004
        
        elif component_type == 'generator':
            if 'vibration' in df_featured.columns:
                vibration = df_featured['vibration'].astype(float)
                df_featured['vibration_high'] = (vibration > 7.0).astype(int)
        
        # Fill any remaining NaNs
        df_featured = df_featured.fillna(0)
        
        # Ensure all columns are numeric (convert any remaining object types)
        for col in df_featured.columns:
            if df_featured[col].dtype == 'object':
                try:
                    df_featured[col] = pd.to_numeric(df_featured[col], errors='coerce').fillna(0)
                except:
                    pass
        
        return df_featured
    
    def create_target_variable(self, df, component_type):
        """Create synthetic target variable for training"""
        
        if component_type == 'battery':
            # Health score based on battery parameters
            health = 100.0
            
            if 'voltage' in df.columns:
                voltage = df['voltage'].astype(float)
                health -= np.where(voltage < 12.0, (12.0 - voltage) * 10.0, 0.0)
            if 'temperature' in df.columns:
                temp = df['temperature'].astype(float)
                health -= np.where(temp > 40.0, (temp - 40.0) * 2.0, 0.0)
            if 'internal_resistance' in df.columns:
                resistance = df['internal_resistance'].astype(float)
                health -= np.where(resistance > 0.08, (resistance - 0.08) * 200.0, 0.0)
            if 'soc' in df.columns:
                soc = df['soc'].astype(float)
                health -= np.where(soc < 30.0, (30.0 - soc) * 0.5, 0.0)
            
            return np.clip(health, 0.0, 100.0)
        
        elif component_type == 'solar_panel':
            # Performance ratio as target
            if 'performance_ratio' in df.columns:
                return df['performance_ratio'].astype(float) * 100.0
            else:
                performance = 90.0
                if 'temperature' in df.columns:
                    temp = df['temperature'].astype(float)
                    performance -= (temp - 25.0) * 0.4
                if 'efficiency' in df.columns:
                    eff = df['efficiency'].astype(float)
                    performance = eff * 500.0  # Scale to 0-100
                return np.clip(performance, 0.0, 100.0)
        
        elif component_type == 'generator':
            # Health based on generator parameters
            health = 100.0
            
            if 'oil_pressure' in df.columns:
                pressure = df['oil_pressure'].astype(float)
                health -= np.where(pressure < 40.0, (40.0 - pressure) * 2.0, 0.0)
            if 'coolant_temp' in df.columns:
                temp = df['coolant_temp'].astype(float)
                health -= np.where(temp > 90.0, (temp - 90.0) * 3.0, 0.0)
            if 'vibration' in df.columns:
                vib = df['vibration'].astype(float)
                health -= np.where(vib > 7.0, (vib - 7.0) * 5.0, 0.0)
            
            return np.clip(health, 0.0, 100.0)
        
        return np.ones(len(df)) * 85.0  # Default
    
    def train_model(self, component_type, hours=720):
        """Train ML model for component using database data"""
        print(f"\n🚀 Training ML model for {component_type}...")
        start_time = datetime.now()
        
        # Fetch training data
        df = self.db_integration.fetch_training_data(component_type, hours)
        
        if df.empty or len(df) < 50:
            print(f"❌ Insufficient data for training ({len(df)} records)")
            return None
        
        # Preprocess
        df_clean = self.preprocess_data(df, component_type)
        df_featured = self.engineer_features(df_clean, component_type)
        
        # Create target variable
        df_featured['target'] = self.create_target_variable(df_featured, component_type)
        
        # Select features
        exclude_cols = ['timestamp', 'id', 'created_at', 'target', 
                       self.component_configs[component_type]['id_field'],
                       'health_status', 'charge_status']
        
        feature_cols = [col for col in df_featured.columns 
                       if col not in exclude_cols and df_featured[col].dtype in [np.float64, np.int64, np.float32, np.int32]]
        
        X = df_featured[feature_cols].astype(float)
        y = df_featured['target'].astype(float)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model_type = self.component_configs[component_type]['model_type']
        
        if model_type == 'random_forest':
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
        else:
            model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        
        metrics = {
            'mae': float(mean_absolute_error(y_test, y_pred)),
            'mse': float(mean_squared_error(y_test, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
            'r2': float(r2_score(y_test, y_pred))
        }
        
        print(f"📈 Model Performance:")
        print(f"   MAE: {metrics['mae']:.4f}")
        print(f"   RMSE: {metrics['rmse']:.4f}")
        print(f"   R²: {metrics['r2']:.4f}")
        
        # Save model and scaler
        model_path = f'models/{component_type}_model.pkl'
        scaler_path = f'models/{component_type}_scaler.pkl'
        
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        
        self.models[component_type] = model
        self.scalers[component_type] = scaler
        self.feature_names[component_type] = feature_cols
        
        # Save to database
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        training_info = {
            'start_time': start_time,
            'end_time': end_time,
            'samples': len(df),
            'features': feature_cols,
            'hyperparameters': {k: convert_to_python_type(v) for k, v in model.get_params().items()},
            'duration': duration,
            'model_path': model_path,
            'status': 'success'
        }
        
        self.db_integration.save_model_performance(
            component_type, model_type, metrics, training_info
        )
        
        self.db_integration.log_training_session(component_type, training_info)
        
        # Save feature importance
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(feature_cols, model.feature_importances_))
            self.db_integration.save_feature_importance(component_type, feature_importance)
        
        print(f"✅ Model training completed in {duration:.2f} seconds")
        
        return model, metrics
    
    def predict(self, component_type, component_id):
        """Make prediction for a component using latest sensor data"""
        
        # Load model if not in memory
        if component_type not in self.models:
            try:
                model_path = f'models/{component_type}_model.pkl'
                scaler_path = f'models/{component_type}_scaler.pkl'
                
                self.models[component_type] = joblib.load(model_path)
                self.scalers[component_type] = joblib.load(scaler_path)
                
            except Exception as e:
                print(f"❌ Failed to load model: {e}")
                return None
        
        # Fetch latest sensor reading
        sensor_data = self.db_integration.fetch_latest_sensor_reading(component_type, component_id)
        
        if not sensor_data:
            print(f"❌ No sensor data available for {component_type}/{component_id}")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame([sensor_data])
        
        # Feature engineering
        df_featured = self.engineer_features(df, component_type)
        
        # Get feature columns (must match training)
        if component_type in self.feature_names:
            feature_cols = self.feature_names[component_type]
        else:
            # Fallback: use all numeric columns except excluded ones
            exclude_cols = ['timestamp', 'id', 'created_at',
                           self.component_configs[component_type]['id_field'],
                           'health_status', 'charge_status']
            feature_cols = [col for col in df_featured.columns 
                           if col not in exclude_cols and df_featured[col].dtype in [np.float64, np.int64, np.float32, np.int32]]
        
        # Ensure all required features exist
        for col in feature_cols:
            if col not in df_featured.columns:
                df_featured[col] = 0.0
        
        X = df_featured[feature_cols].astype(float)
        
        # Scale and predict
        X_scaled = self.scalers[component_type].transform(X)
        health_score = float(self.models[component_type].predict(X_scaled)[0])
        
        # Calculate RUL
        rul = self.calculate_rul(component_type, health_score)
        
        # Prepare prediction data
        prediction_data = {
            'health_score': health_score,
            'rul': rul,
            'confidence': 0.85,
            'model_version': 'v1.0',
            'features': {k: convert_to_python_type(v) for k, v in sensor_data.items() 
                        if isinstance(v, (int, float, np.number, Decimal))}
        }
        
        # Save prediction
        self.db_integration.save_ml_prediction(component_type, component_id, prediction_data)
        
        print(f"✅ Prediction - Health: {health_score:.2f}%, RUL: {rul:.1f} days")
        
        return prediction_data
    
    def calculate_rul(self, component_type, current_health):
        """Calculate Remaining Useful Life"""
        # Degradation rates (% per day)
        degradation_rates = {
            'battery': 0.1,
            'solar_panel': 0.05,
            'generator': 0.15
        }
        
        rate = degradation_rates.get(component_type, 0.1)
        threshold = 70.0  # Minimum acceptable health
        
        if current_health <= threshold:
            return 0.0
        
        rul = (current_health - threshold) / rate
        return max(rul, 0.0)
    
    def run_full_prediction_cycle(self):
        """Run predictions for all components"""
        print("\n" + "="*70)
        print("RUNNING FULL PREDICTION CYCLE")
        print("="*70)
        
        component_ids = {
            'battery': 'BAT001',
            'solar_panel': 'SP001',
            'generator': 'GEN001'
        }
        
        results = {}
        
        for component_type, component_id in component_ids.items():
            print(f"\n📊 Processing {component_type} ({component_id})...")
            
            try:
                prediction = self.predict(component_type, component_id)
                if prediction:
                    results[component_type] = prediction
            except Exception as e:
                print(f"❌ Error predicting {component_type}: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n" + "="*70)
        print("PREDICTION CYCLE COMPLETE")
        print("="*70)
        
        return results
    
    def close(self):
        """Close all connections"""
        self.db_integration.close()


def main():
    """Main execution function"""
    print("="*70)
    print("ML PREDICTIVE MAINTENANCE SYSTEM - DATABASE INTEGRATION")
    print("="*70)
    
    # Database configuration
    db_config = {
        'host': 'localhost',
        'port': 5432,
        'database': 'predictive_maintenance',
        'user': 'postgres',
        'password': '9090'
    }
    
    try:
        # Initialize system
        pm_system = IntegratedPredictiveMaintenanceSystem(db_config)
        
        # Train models for all components
        print("\n" + "="*70)
        print("TRAINING ML MODELS")
        print("="*70)
        
        for component in ['battery', 'solar_panel', 'generator']:
            try:
                pm_system.train_model(component, hours=720)
            except Exception as e:
                print(f"❌ Error training {component}: {e}")
                import traceback
                traceback.print_exc()
        
        # Run prediction cycle
        results = pm_system.run_full_prediction_cycle()
        
        # Display results
        print("\n" + "="*70)
        print("PREDICTION RESULTS")
        print("="*70)
        
        for component, prediction in results.items():
            print(f"\n{component.upper()}:")
            print(f"  Health Score: {prediction['health_score']:.2f}%")
            print(f"  RUL: {prediction['rul']:.1f} days")
            print(f"  Confidence: {prediction['confidence']:.2f}")
        
        # Close connections
        pm_system.close()
        
        print("\n" + "="*70)
        print("SYSTEM EXECUTION COMPLETE")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ System Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()