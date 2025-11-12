from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_socketio import SocketIO, emit
import json
import threading
import time
from datetime import datetime, timedelta
import random
import numpy as np

from postgresql_database_creator import IntegratedPredictiveMaintenanceSystem
from ai_report_generator import AIReportGenerator

app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-in-production'
socketio = SocketIO(app, cors_allowed_origins="*")

# Database configuration for PostgreSQL
db_config = {
    'host': 'localhost',
    'port': '5432',
    'database': 'predictive_maintenance',
    'user': 'postgres',
    'password': '9090' 
}

# Initialize the integrated ML system
pm_system = None

def initialize_system():
    """Initialize ML system with error handling"""
    global pm_system
    try:
        print("🚀 Initializing Predictive Maintenance System...")
        pm_system = IntegratedPredictiveMaintenanceSystem(db_config)
        
        print("🚀 Training ML models on startup...")
        for component in ['battery', 'solar_panel', 'generator']:
            try:
                pm_system.train_model(component, hours=720)  # Last 30 days
                print(f"✅ {component} model trained successfully")
            except Exception as e:
                print(f"⚠️ Warning: {component} model training failed: {e}")
        
        print("✅ System initialized successfully")
        return True
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        return False


ai_reporter = None

def initialize_system():
    """Initialize ML system with error handling"""
    global pm_system, ai_reporter
    try:
        print("🚀 Initializing Predictive Maintenance System...")
        pm_system = IntegratedPredictiveMaintenanceSystem(db_config)
        
        # Initialize AI Reporter
        ai_reporter = AIReportGenerator(pm_system.db_integration)
        print("✅ AI Report Generator initialized")
        
        print("🚀 Training ML models on startup...")
        for component in ['battery', 'solar_panel', 'generator']:
            try:
                pm_system.train_model(component, hours=720)  # Last 30 days
                print(f"✅ {component} model trained successfully")
            except Exception as e:
                print(f"⚠️ Warning: {component} model training failed: {e}")
        
        print("✅ System initialized successfully")
        return True
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        return False

# Initialize on startup
initialize_system()

users = {'admin': 'password123', 'operator': 'op123'}

# Component name mapping helper
def map_component_name(frontend_name):
    """Map frontend component names to ML system names"""
    mapping = {
        'solar_panels': 'solar_panel',
        'batteries': 'battery',
        'generators': 'generator',
        'wind_turbines': 'wind_turbine'
    }
    return mapping.get(frontend_name, frontend_name)

def reverse_map_component_name(ml_name):
    """Map ML system names back to frontend names"""
    reverse_mapping = {
        'solar_panel': 'solar_panels',
        'battery': 'batteries',
        'generator': 'generators',
        'wind_turbine': 'wind_turbines'
    }
    return reverse_mapping.get(ml_name, ml_name)

# Component ID mapping
COMPONENT_IDS = {
    'solar_panel': 'SP001',
    'battery': 'BAT001',
    'generator': 'GEN001'
}

@app.route('/home')
def home():
    """Landing page for Uplynx"""
    return render_template('home.html')

@app.route('/')
def index():
    """Redirect to home page or dashboard based on authentication"""
    if 'username' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('home'))

@app.route('/login', methods=['POST'])
def login():
    """Handle AJAX login requests"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username in users and users[username] == password:
            session['username'] = username
            return jsonify({
                'success': True, 
                'message': 'Login successful!',
                'username': username,
                'redirect': url_for('dashboard')
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Invalid credentials'
            }), 401

@app.route('/logout')
def logout():
    """Handle logout"""
    session.pop('username', None)
    return redirect(url_for('home'))

@app.route('/api/check_auth')
def check_auth():
    """Check if user is authenticated"""
    if 'username' in session:
        return jsonify({
            'authenticated': True,
            'username': session['username']
        })
    return jsonify({'authenticated': False})

@app.route('/api/health_status')
def health_status():
    """Get current health status using REAL ML predictions from database"""
    if 'username' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    if not pm_system:
        return jsonify({'error': 'System not initialized'}), 503
    
    predictions = {}
    alerts = []
    
    # Get real predictions for each component
    for frontend_name, ml_component in [
        ('solar_panels', 'solar_panel'),
        ('batteries', 'battery'),
        ('generators', 'generator')
    ]:
        try:
            component_id = COMPONENT_IDS.get(ml_component)
            if not component_id:
                continue
            
            # Get REAL prediction from database
            prediction_data = pm_system.predict(ml_component, component_id)
            
            if prediction_data:
                predictions[frontend_name] = {
                    'health_score': round(prediction_data['health_score'], 1),
                    'rul': round(prediction_data['rul'], 1),
                    'timestamp': datetime.now().isoformat(),
                    'confidence': round(prediction_data.get('confidence', 0.85), 2)
                }
                
                # Generate alerts based on predictions
                component_alerts = generate_ml_alerts(
                    frontend_name, 
                    prediction_data['health_score'], 
                    prediction_data['rul']
                )
                alerts.extend(component_alerts)
            else:
                # Fallback if prediction fails
                predictions[frontend_name] = {
                    'health_score': 85.0,
                    'rul': 180.0,
                    'timestamp': datetime.now().isoformat(),
                    'confidence': 0.50,
                    'fallback': True
                }
                
        except Exception as e:
            print(f"❌ Error processing {frontend_name}: {e}")
            # Fallback prediction
            predictions[frontend_name] = {
                'health_score': 85.0,
                'rul': 180.0,
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    # Generate maintenance schedule
    schedule = generate_ml_maintenance_schedule(predictions)
    
    return jsonify({
        'predictions': predictions,
        'alerts': alerts,
        'schedule': schedule,
        'timestamp': datetime.now().isoformat(),
        'system_status': 'operational' if pm_system else 'degraded'
    })

@app.route('/api/sensor_data')
def get_sensor_data():
    """Get REAL sensor data from database"""
    if 'username' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    if not pm_system:
        return jsonify({'error': 'System not initialized'}), 503
    
    sensor_data = {}
    
    try:
        # Fetch sensor data for each component
        for ml_component, component_id in COMPONENT_IDS.items():
            latest_reading = pm_system.db_integration.fetch_latest_sensor_reading(
                ml_component, 
                component_id
            )
            
            if latest_reading:
                frontend_name = reverse_map_component_name(ml_component)
                sensor_data[frontend_name] = latest_reading
                
    except Exception as e:
        print(f"❌ Error fetching sensor data: {e}")
        return jsonify({'error': 'Failed to fetch sensor data'}), 500
    
    return jsonify(sensor_data)

@app.route('/api/ml_insights/<component_type>')
def get_ml_insights(component_type):
    """Get ML insights and feature importance for a component"""
    if 'username' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    if not pm_system:
        return jsonify({'error': 'System not initialized'}), 503
    
    try:
        ml_component = map_component_name(component_type)
        
        # Get feature importance from database
        query = """
            SELECT feature_name, importance_score
            FROM feature_importance
            WHERE component_type = %s
            ORDER BY importance_score DESC
            LIMIT 10
        """
        
        pm_system.db_integration.cursor.execute(query, (ml_component,))
        features = pm_system.db_integration.cursor.fetchall()
        
        # Get recent model performance
        query = """
            SELECT model_type, mae, rmse, r2_score, training_samples
            FROM model_performance
            WHERE component_type = %s
            ORDER BY timestamp DESC
            LIMIT 1
        """
        
        pm_system.db_integration.cursor.execute(query, (ml_component,))
        performance = pm_system.db_integration.cursor.fetchone()
        
        insights = {
            'component': component_type,
            'feature_importance': [
                {'name': f['feature_name'], 'importance': float(f['importance_score'])}
                for f in features
            ],
            'model_performance': {
                'model_type': performance['model_type'],
                'mae': float(performance['mae']),
                'rmse': float(performance['rmse']),
                'r2_score': float(performance['r2_score']),
                'training_samples': performance['training_samples']
            } if performance else None
        }
        
        return jsonify(insights)
        
    except Exception as e:
        print(f"❌ Error getting insights: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/retrain_model/<component_type>', methods=['POST'])
def retrain_model(component_type):
    """Retrain ML model for a specific component"""
    if 'username' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    if not pm_system:
        return jsonify({'error': 'System not initialized'}), 503
    
    try:
        ml_component = map_component_name(component_type)
        
        # Retrain model
        model, metrics = pm_system.train_model(ml_component, hours=720)
        
        if model and metrics:
            return jsonify({
                'status': 'success',
                'metrics': metrics,
                'message': f'{component_type} model retrained successfully'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Model training failed'
            }), 500
            
    except Exception as e:
        print(f"❌ Retraining error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/prediction_history/<component_type>')
def prediction_history(component_type):
    """Get historical predictions for a component"""
    if 'username' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    if not pm_system:
        return jsonify({'error': 'System not initialized'}), 503
    
    try:
        ml_component = map_component_name(component_type)
        component_id = COMPONENT_IDS.get(ml_component)
        
        if not component_id:
            return jsonify({'error': 'Unknown component'}), 400
        
        df = pm_system.db_integration.get_prediction_history(
            ml_component, 
            component_id, 
            days=30
        )
        
        history = df.to_dict('records')
        
        return jsonify({
            'component': component_type,
            'history': history
        })
        
    except Exception as e:
        print(f"❌ Error fetching history: {e}")
        return jsonify({'error': str(e)}), 500

def generate_ml_alerts(component_type, health_score, rul):
    """Generate alerts based on ML predictions"""
    alerts = []
    
    # Health-based alerts
    if health_score < 60:
        alerts.append({
            'component': component_type,
            'severity': 'critical',
            'message': f'Critical health alert: {health_score:.1f}% health score',
            'timestamp': datetime.now().isoformat(),
            'type': 'health',
            'action_required': 'Immediate inspection required'
        })
    elif health_score < 75:
        alerts.append({
            'component': component_type,
            'severity': 'high',
            'message': f'Health warning: {health_score:.1f}% health score',
            'timestamp': datetime.now().isoformat(),
            'type': 'health',
            'action_required': 'Schedule maintenance within 7 days'
        })
    
    # RUL-based alerts
    if rul < 7:
        alerts.append({
            'component': component_type,
            'severity': 'critical',
            'message': f'Critical RUL: {rul:.1f} days remaining',
            'timestamp': datetime.now().isoformat(),
            'type': 'rul',
            'action_required': 'Emergency maintenance required'
        })
    elif rul < 30:
        alerts.append({
            'component': component_type,
            'severity': 'high',
            'message': f'Low RUL warning: {rul:.1f} days remaining',
            'timestamp': datetime.now().isoformat(),
            'type': 'rul',
            'action_required': 'Plan maintenance soon'
        })
    
    return alerts

def generate_ml_maintenance_schedule(predictions):
    """Generate maintenance schedule based on ML predictions"""
    schedule = []
    
    for component, prediction in predictions.items():
        health_score = prediction['health_score']
        rul = prediction['rul']
        
        # Determine priority and schedule
        if health_score < 60 or rul < 7:
            priority = "CRITICAL"
            days_until = 1
        elif health_score < 70 or rul < 14:
            priority = "HIGH"
            days_until = 3
        elif health_score < 80 or rul < 30:
            priority = "MEDIUM"
            days_until = 7
        else:
            priority = "LOW"
            days_until = 30
        
        scheduled_date = (datetime.now() + timedelta(days=days_until)).strftime('%Y-%m-%d')
        
        schedule.append({
            'component': component,
            'priority': priority,
            'scheduled_date': scheduled_date,
            'health_score': health_score,
            'rul': rul,
            'status': 'scheduled',
            'type': 'ml_predicted',
            'confidence': prediction.get('confidence', 0.85)
        })
    
    # Sort by priority
    priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    schedule.sort(key=lambda x: priority_order.get(x['priority'], 99))
    
    return schedule

@app.route('/dashboard')
def dashboard():
    """Dashboard - requires login"""
    if 'username' not in session:
        return redirect(url_for('home'))
    return render_template('dashboard.html')

@app.route('/components')
def components():
    if 'username' not in session:
        return redirect(url_for('home'))
    return render_template('components.html')

@app.route('/maintenance')
def maintenance():
    if 'username' not in session:
        return redirect(url_for('home'))
    return render_template('maintenance.html')

@app.route('/reports')
def reports():
    if 'username' not in session:
        return redirect(url_for('home'))
    return render_template('reports.html')

@app.route('/ai_insights')
def ai_insights():
    if 'username' not in session:
        return redirect(url_for('home'))
    return render_template('ai_insights.html')

@app.route('/api/ai_reports/system_analysis')
def get_system_analysis():
    """Get comprehensive AI system analysis"""
    if 'username' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    if not ai_reporter:
        return jsonify({'error': 'AI system not initialized'}), 503
    
    try:
        analysis = ai_reporter.analyze_system_health()
        return jsonify(analysis)
    except Exception as e:
        print(f"❌ AI analysis error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ai_reports/component_analysis/<component_type>')
def get_component_analysis(component_type):
    """Get AI analysis for specific component"""
    if 'username' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    if not ai_reporter:
        return jsonify({'error': 'AI system not initialized'}), 503
    
    try:
        component_ids = {
            'solar_panels': 'SP001',
            'batteries': 'BAT001',
            'generators': 'GEN001'
        }
        
        component_id = component_ids.get(component_type)
        if not component_id:
            return jsonify({'error': 'Invalid component type'}), 400
        
        analysis = ai_reporter.analyze_component_health(component_type, component_id)
        return jsonify(analysis)
    except Exception as e:
        print(f"❌ Component analysis error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ai_reports/comprehensive')
def get_comprehensive_report():
    """Get comprehensive AI report"""
    if 'username' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    if not ai_reporter:
        return jsonify({'error': 'AI system not initialized'}), 503
    
    try:
        report = ai_reporter.generate_comprehensive_report()
        return jsonify(report)
    except Exception as e:
        print(f"❌ Comprehensive report error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ai_reports/maintenance_analysis')
def get_maintenance_analysis():
    """Get AI analysis of maintenance patterns"""
    if 'username' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    if not ai_reporter:
        return jsonify({'error': 'AI system not initialized'}), 503
    
    try:
        analysis = ai_reporter.analyze_maintenance_patterns()
        return jsonify(analysis)
    except Exception as e:
        print(f"❌ Maintenance analysis error: {e}")
        return jsonify({'error': str(e)}), 500

# Background thread for real-time updates
def background_thread():
    """Background thread for real-time predictions"""
    while True:
        try:
            if pm_system:
                # Run predictions for all components
                for frontend_name, ml_component in [
                    ('solar_panels', 'solar_panel'),
                    ('batteries', 'battery'),
                    ('generators', 'generator')
                ]:
                    try:
                        component_id = COMPONENT_IDS.get(ml_component)
                        prediction = pm_system.predict(ml_component, component_id)
                        
                        if prediction:
                            socketio.emit('health_update', {
                                'component': frontend_name,
                                'health_score': prediction['health_score'],
                                'rul': prediction['rul'],
                                'timestamp': datetime.now().isoformat()
                            })
                            
                            # Check for alerts
                            if prediction['health_score'] < 75 or prediction['rul'] < 30:
                                alerts = generate_ml_alerts(
                                    frontend_name,
                                    prediction['health_score'],
                                    prediction['rul']
                                )
                                for alert in alerts:
                                    socketio.emit('alert_triggered', alert)
                                    
                    except Exception as e:
                        print(f"❌ Background prediction error for {frontend_name}: {e}")
                        
        except Exception as e:
            print(f"❌ Background thread error: {e}")
        
        time.sleep(60)  # Update every minute

@socketio.on('connect')
def handle_connect():
    print('✅ Client connected')
    emit('connected', {
        'data': 'Connected to ML-Powered Predictive Maintenance System',
        'system_status': 'operational' if pm_system else 'degraded'
    })

@socketio.on('disconnect')
def handle_disconnect():
    print('❌ Client disconnected')

if __name__ == '__main__':
    # Start background thread
    thread = threading.Thread(target=background_thread)
    thread.daemon = True
    thread.start()
    
    print("\n" + "="*70)
    print("🚀 PREDICTIVE MAINTENANCE SYSTEM STARTING")
    print("="*70)
    print("📊 Dashboard: http://localhost:5000")
    print("👤 Login: admin / password123")
    print("="*70 + "\n")
    
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)