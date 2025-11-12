import psycopg2
from psycopg2.extras import execute_batch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

class PostgreSQLMaintenanceDB:
    """Create PostgreSQL database with 2 months of sensor data for predictive maintenance"""
    
    def __init__(self, db_config=None):
        """
        Initialize database connection
        
        db_config example:
        {
            'host': 'localhost',
            'port': 5432,
            'database': 'predictive_maintenance',
            'user': 'postgres',
            'password': 'your_password'
        }
        """
        if db_config is None:
            db_config = {
                'host': 'localhost',
                'port': 5432,
                'database': 'predictive_maintenance',
                'user': 'postgres',
                'password': 'postgres'
            }
        
        self.db_config = db_config
        self.num_days = 60  # 2 months
        self.start_date = datetime.now() - timedelta(days=self.num_days)
        np.random.seed(42)
        
        try:
            self.conn = psycopg2.connect(**db_config)
            self.conn.autocommit = False
            self.cursor = self.conn.cursor()
            print(f"✓ Connected to PostgreSQL database: {db_config['database']}")
        except psycopg2.OperationalError as e:
            print(f"Error connecting to database: {e}")
            print("\nTrying to create database...")
            self.create_database()
    
    def create_database(self):
        """Create the database if it doesn't exist"""
        temp_config = self.db_config.copy()
        db_name = temp_config.pop('database')
        temp_config['database'] = 'postgres'  # Connect to default database
        
        try:
            conn = psycopg2.connect(**temp_config)
            conn.autocommit = True
            cursor = conn.cursor()
            
            # Check if database exists
            cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = '{db_name}'")
            if not cursor.fetchone():
                cursor.execute(f'CREATE DATABASE {db_name}')
                print(f"✓ Database '{db_name}' created successfully")
            
            cursor.close()
            conn.close()
            
            # Now connect to the new database
            self.conn = psycopg2.connect(**self.db_config)
            self.conn.autocommit = False
            self.cursor = self.conn.cursor()
            print(f"✓ Connected to database: {db_name}")
            
        except Exception as e:
            print(f"Error creating database: {e}")
            raise
    
    def create_tables(self):
        """Create database tables for all components"""
        
        # Drop existing tables if they exist
        drop_tables = """
        DROP TABLE IF EXISTS health_history CASCADE;
        DROP TABLE IF EXISTS maintenance_schedule CASCADE;
        DROP TABLE IF EXISTS alerts CASCADE;
        DROP TABLE IF EXISTS generator_data CASCADE;
        DROP TABLE IF EXISTS battery_data CASCADE;
        DROP TABLE IF EXISTS solar_panel_data CASCADE;
        """
        self.cursor.execute(drop_tables)
        
        # Solar Panel Sensor Data Table
        self.cursor.execute('''
            CREATE TABLE solar_panel_data (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP NOT NULL,
                panel_id VARCHAR(50) NOT NULL,
                voltage NUMERIC(8,2) NOT NULL,
                current NUMERIC(8,2) NOT NULL,
                power_output NUMERIC(10,2) NOT NULL,
                temperature NUMERIC(6,2) NOT NULL,
                irradiance NUMERIC(8,2) NOT NULL,
                efficiency NUMERIC(6,4) NOT NULL,
                performance_ratio NUMERIC(5,3),
                health_status VARCHAR(20),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Battery Sensor Data Table
        self.cursor.execute('''
            CREATE TABLE battery_data (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP NOT NULL,
                battery_id VARCHAR(50) NOT NULL,
                voltage NUMERIC(6,3) NOT NULL,
                current NUMERIC(8,2) NOT NULL,
                soc NUMERIC(5,2) NOT NULL,
                temperature NUMERIC(6,2) NOT NULL,
                internal_resistance NUMERIC(6,4) NOT NULL,
                cycle_count INTEGER,
                capacity_ah NUMERIC(8,2),
                charge_status VARCHAR(20),
                health_status VARCHAR(20),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Generator Sensor Data Table
        self.cursor.execute('''
            CREATE TABLE generator_data (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP NOT NULL,
                generator_id VARCHAR(50) NOT NULL,
                oil_pressure NUMERIC(6,2) NOT NULL,
                coolant_temp NUMERIC(6,2) NOT NULL,
                vibration NUMERIC(6,3) NOT NULL,
                rpm NUMERIC(7,1) NOT NULL,
                load_percentage NUMERIC(5,2) NOT NULL,
                operating_hours INTEGER,
                fuel_level NUMERIC(5,2),
                runtime_hours NUMERIC(6,2),
                health_status VARCHAR(20),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Alerts Table
        self.cursor.execute('''
            CREATE TABLE alerts (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP NOT NULL,
                component_type VARCHAR(50) NOT NULL,
                component_id VARCHAR(50) NOT NULL,
                severity VARCHAR(20) NOT NULL,
                alert_type VARCHAR(50) NOT NULL,
                message TEXT NOT NULL,
                acknowledged BOOLEAN DEFAULT FALSE,
                resolved BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Maintenance Schedule Table
        self.cursor.execute('''
            CREATE TABLE maintenance_schedule (
                id SERIAL PRIMARY KEY,
                component_type VARCHAR(50) NOT NULL,
                component_id VARCHAR(50) NOT NULL,
                scheduled_date TIMESTAMP NOT NULL,
                priority VARCHAR(20) NOT NULL,
                task_description TEXT,
                estimated_duration NUMERIC(6,2),
                required_parts JSONB,
                status VARCHAR(20) DEFAULT 'scheduled',
                completed_date TIMESTAMP,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # System Health History Table
        self.cursor.execute('''
            CREATE TABLE health_history (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP NOT NULL,
                component_type VARCHAR(50) NOT NULL,
                component_id VARCHAR(50) NOT NULL,
                health_score NUMERIC(5,2) NOT NULL,
                rul_days NUMERIC(8,1) NOT NULL,
                prediction_confidence NUMERIC(4,3),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.conn.commit()
        print("✓ Database tables created successfully")
    
    def generate_solar_panel_data(self):
        """Generate and insert 2 months of solar panel data"""
        print("\nGenerating solar panel data...")
        records = []
        base_efficiency = 0.18
        
        for day in range(self.num_days):
            date = self.start_date + timedelta(days=day)
            
            # Gradual degradation over time
            degradation = 0.0001 * day
            current_efficiency = base_efficiency * (1 - degradation)
            
            # 6 readings per day (sunrise to sunset)
            for hour in [6, 9, 12, 15, 18, 20]:
                timestamp = date.replace(hour=hour, minute=np.random.randint(0, 60))
                
                # Solar irradiance varies by time of day
                if hour in [6, 20]:
                    irradiance = np.random.normal(300, 50)
                elif hour in [9, 18]:
                    irradiance = np.random.normal(650, 80)
                else:  # Peak hours 12, 15
                    irradiance = np.random.normal(900, 100)
                
                irradiance = max(0, irradiance)
                
                # Calculate power output
                panel_area = 16  # 16 square meters
                expected_power = irradiance * current_efficiency * panel_area
                power_output = expected_power * np.random.normal(1.0, 0.05)
                power_output = max(0, power_output)
                
                # Voltage proportional to irradiance
                voltage = 280 + (irradiance / 1000) * 80 + np.random.normal(0, 10)
                voltage = max(0, voltage)
                
                # Current from power and voltage
                current = (power_output / voltage) if voltage > 0 else 0
                
                # Temperature increases with irradiance
                ambient_temp = 25 + 10 * np.sin(2 * np.pi * (hour - 6) / 14)
                temperature = ambient_temp + (irradiance / 30) + np.random.normal(0, 3)
                
                # Performance ratio
                performance_ratio = (power_output / expected_power) if expected_power > 0 else 0
                
                # Health status based on performance
                if performance_ratio > 0.9:
                    health_status = 'excellent'
                elif performance_ratio > 0.8:
                    health_status = 'good'
                elif performance_ratio > 0.7:
                    health_status = 'fair'
                else:
                    health_status = 'poor'
                
                records.append((
                    timestamp,
                    'SP001',
                    round(voltage, 2),
                    round(current, 2),
                    round(power_output, 2),
                    round(temperature, 2),
                    round(irradiance, 2),
                    round(current_efficiency, 4),
                    round(performance_ratio, 3),
                    health_status
                ))
        
        execute_batch(self.cursor, '''
            INSERT INTO solar_panel_data 
            (timestamp, panel_id, voltage, current, power_output, temperature, 
             irradiance, efficiency, performance_ratio, health_status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ''', records)
        
        self.conn.commit()
        print(f"✓ Inserted {len(records)} solar panel records")
        return len(records)
    
    def generate_battery_data(self):
        """Generate and insert 2 months of battery data"""
        print("\nGenerating battery data...")
        records = []
        base_capacity = 200.0  # 200 Ah
        base_resistance = 0.03
        
        for day in range(self.num_days):
            date = self.start_date + timedelta(days=day)
            
            # Battery aging
            capacity_fade = 0.0002 * day
            current_capacity = base_capacity * (1 - capacity_fade)
            resistance_increase = 0.00005 * day
            
            # 8 readings per day (every 3 hours)
            for hour in range(0, 24, 3):
                timestamp = date.replace(hour=hour, minute=np.random.randint(0, 60))
                
                # SOC follows daily charge/discharge pattern
                if 0 <= hour < 6:
                    soc_base = 70 - (hour * 5)
                elif 6 <= hour < 12:
                    soc_base = 40 + ((hour - 6) * 8)
                elif 12 <= hour < 18:
                    soc_base = 88 - ((hour - 12) * 6)
                else:
                    soc_base = 52 + ((hour - 18) * 5)
                
                soc = soc_base + np.random.normal(0, 3)
                soc = np.clip(soc, 20, 95)
                
                # Voltage follows SOC (lead-acid battery curve)
                voltage = 11.8 + (soc / 100) * 2.5 + np.random.normal(0, 0.1)
                
                # Current varies with charge/discharge
                if 6 <= hour < 12 or 18 <= hour < 24:  # Charging
                    current = np.random.uniform(20, 50)
                    charge_status = 'charging'
                else:  # Discharging
                    current = -np.random.uniform(15, 40)
                    charge_status = 'discharging'
                
                # Internal resistance
                internal_resistance = base_resistance + resistance_increase + np.random.normal(0, 0.005)
                internal_resistance = max(0.01, internal_resistance)
                
                # Temperature varies with activity
                if abs(current) > 30:
                    temperature = 30 + np.random.normal(5, 2)
                else:
                    temperature = 25 + np.random.normal(0, 2)
                
                # Cycle count
                cycle_count = day
                
                # Health status
                if soc < 25:
                    health_status = 'critical'
                elif temperature > 40:
                    health_status = 'warning'
                elif internal_resistance > 0.08:
                    health_status = 'degraded'
                else:
                    health_status = 'normal'
                
                records.append((
                    timestamp,
                    'BAT001',
                    round(voltage, 3),
                    round(current, 2),
                    round(soc, 2),
                    round(temperature, 2),
                    round(internal_resistance, 4),
                    cycle_count,
                    round(current_capacity, 2),
                    charge_status,
                    health_status
                ))
        
        execute_batch(self.cursor, '''
            INSERT INTO battery_data 
            (timestamp, battery_id, voltage, current, soc, temperature, 
             internal_resistance, cycle_count, capacity_ah, charge_status, health_status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ''', records)
        
        self.conn.commit()
        print(f"✓ Inserted {len(records)} battery records")
        return len(records)
    
    def generate_generator_data(self):
        """Generate and insert 2 months of generator data"""
        print("\nGenerating generator data...")
        records = []
        
        # Generator runs intermittently (about 20% of days)
        operating_days = sorted(np.random.choice(self.num_days, 
                                                 size=int(self.num_days * 0.2), 
                                                 replace=False))
        
        total_hours = 500  # Starting operating hours
        
        for day_idx in operating_days:
            date = self.start_date + timedelta(days=day_idx)
            
            # Generator runs for 4-10 hours when activated
            runtime_hours = np.random.randint(4, 11)
            
            for hour_offset in range(runtime_hours):
                hour = 8 + hour_offset
                if hour >= 24:
                    hour = hour - 24
                    date = date + timedelta(days=1)
                
                timestamp = date.replace(hour=hour, minute=np.random.randint(0, 60))
                total_hours += 1
                
                # Wear increases with operating hours
                wear_factor = total_hours / 2000
                
                # Oil pressure degrades with wear
                oil_pressure = 55 - (wear_factor * 15) + np.random.normal(0, 3)
                oil_pressure = np.clip(oil_pressure, 25, 75)
                
                # Coolant temperature
                base_coolant_temp = 78 + (wear_factor * 10)
                coolant_temp = base_coolant_temp + np.random.normal(0, 4)
                
                # Add warm-up and cool-down periods
                if hour_offset < 2:
                    coolant_temp -= (2 - hour_offset) * 10
                
                # Vibration increases with wear
                vibration = 4.0 + (wear_factor * 3) + np.random.normal(0, 0.8)
                vibration = max(0, vibration)
                
                # RPM should be stable around 1800
                rpm = 1800 + np.random.normal(0, 25)
                
                # Load percentage
                load_percentage = np.random.uniform(50, 85)
                
                # Fuel level decreases during operation
                fuel_level = 100 - (hour_offset * 8) + np.random.uniform(-5, 5)
                fuel_level = np.clip(fuel_level, 10, 100)
                
                # Health status
                if oil_pressure < 35:
                    health_status = 'critical'
                elif coolant_temp > 90:
                    health_status = 'warning'
                elif vibration > 8:
                    health_status = 'degraded'
                else:
                    health_status = 'normal'
                
                records.append((
                    timestamp,
                    'GEN001',
                    round(oil_pressure, 2),
                    round(coolant_temp, 2),
                    round(vibration, 3),
                    round(rpm, 1),
                    round(load_percentage, 2),
                    total_hours,
                    round(fuel_level, 2),
                    round(hour_offset + 1, 2),
                    health_status
                ))
        
        execute_batch(self.cursor, '''
            INSERT INTO generator_data 
            (timestamp, generator_id, oil_pressure, coolant_temp, vibration, 
             rpm, load_percentage, operating_hours, fuel_level, runtime_hours, health_status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ''', records)
        
        self.conn.commit()
        print(f"✓ Inserted {len(records)} generator records")
        return len(records)
    
    def generate_alerts(self):
        """Generate sample alerts based on sensor anomalies"""
        print("\nGenerating alerts...")
        alerts = []
        
        # Low battery SOC alerts
        self.cursor.execute('''
            SELECT timestamp, battery_id, soc 
            FROM battery_data 
            WHERE soc < 25 
            ORDER BY timestamp
            LIMIT 10
        ''')
        
        for row in self.cursor.fetchall():
            alerts.append((
                row[0], 'batteries', row[1], 'high',
                'low_soc', f'Low State of Charge: {row[2]:.1f}%', False, False
            ))
        
        # High temperature alerts
        self.cursor.execute('''
            SELECT timestamp, battery_id, temperature 
            FROM battery_data 
            WHERE temperature > 38 
            ORDER BY timestamp
            LIMIT 10
        ''')
        
        for row in self.cursor.fetchall():
            alerts.append((
                row[0], 'batteries', row[1], 'critical',
                'high_temperature', f'Battery overheating: {row[2]:.1f}°C', False, False
            ))
        
        # Generator low oil pressure
        self.cursor.execute('''
            SELECT timestamp, generator_id, oil_pressure 
            FROM generator_data 
            WHERE oil_pressure < 40 
            ORDER BY timestamp
            LIMIT 10
        ''')
        
        for row in self.cursor.fetchall():
            alerts.append((
                row[0], 'generators', row[1], 'critical',
                'low_oil_pressure', f'Low oil pressure: {row[2]:.1f} PSI', False, False
            ))
        
        # Solar panel low performance
        self.cursor.execute('''
            SELECT timestamp, panel_id, performance_ratio 
            FROM solar_panel_data 
            WHERE performance_ratio < 0.75 
            ORDER BY timestamp
            LIMIT 10
        ''')
        
        for row in self.cursor.fetchall():
            alerts.append((
                row[0], 'solar_panels', row[1], 'medium',
                'low_performance', f'Low performance ratio: {row[2]:.2f}', False, False
            ))
        
        if alerts:
            execute_batch(self.cursor, '''
                INSERT INTO alerts 
                (timestamp, component_type, component_id, severity, 
                 alert_type, message, acknowledged, resolved)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ''', alerts)
            
            self.conn.commit()
            print(f"✓ Inserted {len(alerts)} alerts")
        
        return len(alerts)
    
    def generate_maintenance_schedule(self):
        """Generate sample maintenance schedule"""
        print("\nGenerating maintenance schedule...")
        
        schedule = [
            ('solar_panels', 'SP001', datetime.now() + timedelta(days=7),
             'MEDIUM', 'Panel cleaning and inspection', 3, 
             json.dumps(["cleaning_kit", "inspection_tools"]), 'scheduled'),
            
            ('batteries', 'BAT001', datetime.now() + timedelta(days=3),
             'HIGH', 'Battery health check and terminal maintenance', 2, 
             json.dumps(["terminal_cleaner", "multimeter"]), 'scheduled'),
            
            ('generators', 'GEN001', datetime.now() + timedelta(days=14),
             'MEDIUM', 'Oil change and filter replacement', 4, 
             json.dumps(["oil_filter", "engine_oil", "air_filter"]), 'scheduled'),
            
            ('solar_panels', 'SP001', datetime.now() - timedelta(days=30),
             'LOW', 'Routine panel inspection', 2, 
             json.dumps(["inspection_tools"]), 'completed'),
            
            ('batteries', 'BAT001', datetime.now() - timedelta(days=15),
             'MEDIUM', 'Battery capacity test', 3, 
             json.dumps(["load_tester"]), 'completed')
        ]
        
        execute_batch(self.cursor, '''
            INSERT INTO maintenance_schedule 
            (component_type, component_id, scheduled_date, priority, 
             task_description, estimated_duration, required_parts, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ''', schedule)
        
        self.conn.commit()
        print(f"✓ Inserted {len(schedule)} maintenance tasks")
        return len(schedule)
    
    def generate_health_history(self):
        """Generate component health history"""
        print("\nGenerating health history...")
        records = []
        
        for day in range(0, self.num_days, 7):  # Weekly snapshots
            date = self.start_date + timedelta(days=day)
            
            # Solar panel health
            solar_health = 95 - (day * 0.05) + np.random.normal(0, 2)
            solar_rul = 8000 - (day * 10) + np.random.normal(0, 100)
            records.append((
                date,
                'solar_panels', 'SP001',
                round(solar_health, 2),
                round(max(solar_rul, 0), 1),
                round(np.random.uniform(0.85, 0.95), 3)
            ))
            
            # Battery health
            battery_health = 90 - (day * 0.1) + np.random.normal(0, 3)
            battery_rul = 1200 - (day * 15) + np.random.normal(0, 50)
            records.append((
                date,
                'batteries', 'BAT001',
                round(battery_health, 2),
                round(max(battery_rul, 0), 1),
                round(np.random.uniform(0.80, 0.92), 3)
            ))
            
            # Generator health
            gen_health = 85 - (day * 0.08) + np.random.normal(0, 2.5)
            gen_rul = 800 - (day * 8) + np.random.normal(0, 40)
            records.append((
                date,
                'generators', 'GEN001',
                round(gen_health, 2),
                round(max(gen_rul, 0), 1),
                round(np.random.uniform(0.78, 0.90), 3)
            ))
        
        execute_batch(self.cursor, '''
            INSERT INTO health_history 
            (timestamp, component_type, component_id, health_score, 
             rul_days, prediction_confidence)
            VALUES (%s, %s, %s, %s, %s, %s)
        ''', records)
        
        self.conn.commit()
        print(f"✓ Inserted {len(records)} health history records")
        return len(records)
    
    def create_indexes(self):
        """Create indexes for better query performance"""
        print("\nCreating database indexes...")
        
        indexes = [
            'CREATE INDEX idx_solar_timestamp ON solar_panel_data(timestamp)',
            'CREATE INDEX idx_solar_panel_id ON solar_panel_data(panel_id)',
            'CREATE INDEX idx_solar_health ON solar_panel_data(health_status)',
            
            'CREATE INDEX idx_battery_timestamp ON battery_data(timestamp)',
            'CREATE INDEX idx_battery_id ON battery_data(battery_id)',
            'CREATE INDEX idx_battery_health ON battery_data(health_status)',
            
            'CREATE INDEX idx_generator_timestamp ON generator_data(timestamp)',
            'CREATE INDEX idx_generator_id ON generator_data(generator_id)',
            'CREATE INDEX idx_generator_health ON generator_data(health_status)',
            
            'CREATE INDEX idx_alerts_timestamp ON alerts(timestamp)',
            'CREATE INDEX idx_alerts_severity ON alerts(severity)',
            'CREATE INDEX idx_alerts_component ON alerts(component_type, component_id)',
            
            'CREATE INDEX idx_maintenance_date ON maintenance_schedule(scheduled_date)',
            'CREATE INDEX idx_maintenance_status ON maintenance_schedule(status)'
        ]
        
        for index_sql in indexes:
            try:
                self.cursor.execute(index_sql)
            except Exception as e:
                print(f"Warning: Could not create index: {e}")
        
        self.conn.commit()
        print("✓ Database indexes created")
    
    def print_database_summary(self):
        """Print summary of database contents"""
        print("\n" + "="*70)
        print("DATABASE SUMMARY")
        print("="*70)
        
        # Count records in each table
        tables = [
            'solar_panel_data', 'battery_data', 'generator_data',
            'alerts', 'maintenance_schedule', 'health_history'
        ]
        
        for table in tables:
            self.cursor.execute(f'SELECT COUNT(*) FROM {table}')
            count = self.cursor.fetchone()[0]
            print(f"{table:30s}: {count:6d} records")
        
        print("\n" + "-"*70)
        
        # Date ranges
        self.cursor.execute('SELECT MIN(timestamp), MAX(timestamp) FROM solar_panel_data')
        min_date, max_date = self.cursor.fetchone()
        print(f"Data Range: {min_date} to {max_date}")
        
        # Component health status distribution
        print("\n" + "-"*70)
        print("COMPONENT HEALTH STATUS:")
        
        for table, component in [
            ('solar_panel_data', 'Solar Panels'),
            ('battery_data', 'Batteries'),
            ('generator_data', 'Generators')
        ]:
            self.cursor.execute(f'''
                SELECT health_status, COUNT(*) 
                FROM {table} 
                GROUP BY health_status
            ''')
            print(f"\n{component}:")
            for status, count in self.cursor.fetchall():
                print(f"  {status:12s}: {count:5d}")
        
        # Alert summary
        print("\n" + "-"*70)
        print("ALERT SUMMARY:")
        self.cursor.execute('''
            SELECT severity, COUNT(*) 
            FROM alerts 
            GROUP BY severity
        ''')
        for severity, count in self.cursor.fetchall():
            print(f"  {severity:12s}: {count:5d}")
        
        print("\n" + "="*70)
    
    def export_sample_queries(self):
        """Export sample SQL queries to a file"""
        queries = """
-- SAMPLE SQL QUERIES FOR PREDICTIVE MAINTENANCE DATABASE (PostgreSQL)
-- =====================================================================

-- 1. Get latest readings for all solar panels
SELECT timestamp, voltage, current, power_output, temperature, health_status
FROM solar_panel_data
WHERE panel_id = 'SP001'
ORDER BY timestamp DESC
LIMIT 10;

-- 2. Get battery health over time
SELECT DATE(timestamp) as date, 
       AVG(soc) as avg_soc, 
       AVG(temperature) as avg_temp,
       AVG(internal_resistance) as avg_resistance
FROM battery_data
WHERE battery_id = 'BAT001'
GROUP BY DATE(timestamp)
ORDER BY date;

-- 3. Find all critical alerts (unresolved)
SELECT timestamp, component_type, component_id, message
FROM alerts
WHERE severity = 'critical' AND resolved = FALSE
ORDER BY timestamp DESC;

-- 4. Get generator operating hours and efficiency
SELECT timestamp, oil_pressure, coolant_temp, vibration, 
       load_percentage, health_status
FROM generator_data
WHERE generator_id = 'GEN001'
ORDER BY timestamp DESC;

-- 5. Upcoming maintenance schedule
SELECT component_type, component_id, scheduled_date, 
       priority, task_description, status
FROM maintenance_schedule
WHERE status = 'scheduled'
ORDER BY scheduled_date;

-- 6. Component health trends
SELECT component_type, timestamp, health_score, rul_days
FROM health_history
WHERE component_id IN ('SP001', 'BAT001', 'GEN001')
ORDER BY component_type, timestamp;

-- 7. Average daily solar production
SELECT DATE(timestamp) as date,
       AVG(power_output) as avg_power,
       MAX(power_output) as peak_power,
       AVG(irradiance) as avg_irradiance
FROM solar_panel_data
GROUP BY DATE(timestamp)
ORDER BY date;

-- 8. Battery charge/discharge cycles
SELECT DATE(timestamp) as date,
       charge_status,
       COUNT(*) as readings,
       AVG(soc) as avg_soc
FROM battery_data
GROUP BY DATE(timestamp), charge_status
ORDER BY date, charge_status;

-- 9. Generator maintenance needs
SELECT g.generator_id,
       AVG(g.oil_pressure) as avg_oil_pressure,
       AVG(g.coolant_temp) as avg_coolant_temp,
       AVG(g.vibration) as avg_vibration,
       MAX(g.operating_hours) as total_hours
FROM generator_data g
GROUP BY g.generator_id;

-- 10. Alert frequency by component
SELECT component_type, 
       COUNT(*) as total_alerts,
       SUM(CASE WHEN severity = 'critical' THEN 1 ELSE 0 END) as critical,
       SUM(CASE WHEN severity = 'high' THEN 1 ELSE 0 END) as high,
       SUM(CASE WHEN severity = 'medium' THEN 1 ELSE 0 END) as medium
FROM alerts
GROUP BY component_type;

-- 11. Recent anomalies (last 7 days) - PostgreSQL specific
SELECT 
    component_type,
    component_id,
    COUNT(*) as anomaly_count,
    MAX(timestamp) as last_occurrence
FROM alerts
WHERE timestamp > NOW() - INTERVAL '7 days'
GROUP BY component_type, component_id
ORDER BY anomaly_count DESC;

-- 12. Battery state transitions (using window functions)
SELECT 
    timestamp,
    battery_id,
    soc,
    charge_status,
    LAG(charge_status) OVER (PARTITION BY battery_id ORDER BY timestamp) as prev_status
FROM battery_data
WHERE battery_id = 'BAT001'
ORDER BY timestamp;

-- 13. Solar panel efficiency trend (7-day moving average)
SELECT 
    timestamp,
    efficiency,
    AVG(efficiency) OVER (
        ORDER BY timestamp 
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) as moving_avg_efficiency
FROM solar_panel_data
WHERE panel_id = 'SP001'
ORDER BY timestamp;

-- 14. Maintenance schedule with health correlation
SELECT 
    ms.component_type,
    ms.scheduled_date,
    ms.priority,
    hh.health_score,
    hh.rul_days
FROM maintenance_schedule ms
LEFT JOIN LATERAL (
    SELECT health_score, rul_days
    FROM health_history
    WHERE health_history.component_type = ms.component_type
      AND health_history.component_id = ms.component_id
    ORDER BY timestamp DESC
    LIMIT 1
) hh ON TRUE
WHERE ms.status = 'scheduled'
ORDER BY ms.scheduled_date;

-- 15. Time-series analysis: hourly aggregates
SELECT 
    DATE_TRUNC('hour', timestamp) as hour,
    AVG(power_output) as avg_power,
    MAX(power_output) as max_power,
    AVG(temperature) as avg_temp
FROM solar_panel_data
WHERE timestamp > NOW() - INTERVAL '7 days'
GROUP BY DATE_TRUNC('hour', timestamp)
ORDER BY hour;

-- 16. JSONB query for required maintenance parts
SELECT 
    component_type,
    scheduled_date,
    task_description,
    jsonb_array_elements_text(required_parts) as part
FROM maintenance_schedule
WHERE status = 'scheduled'
ORDER BY scheduled_date;

-- 17. Create a view for current system health
CREATE OR REPLACE VIEW current_system_health AS
SELECT 
    component_type,
    component_id,
    health_score,
    rul_days,
    prediction_confidence,
    timestamp,
    ROW_NUMBER() OVER (PARTITION BY component_type, component_id ORDER BY timestamp DESC) as rn
FROM health_history;

-- Query the view for latest health status
SELECT 
    component_type,
    component_id,
    health_score,
    rul_days,
    prediction_confidence
FROM current_system_health
WHERE rn = 1;

-- 18. Materialized view for performance optimization
CREATE MATERIALIZED VIEW daily_component_summary AS
SELECT 
    DATE(timestamp) as date,
    'solar_panels' as component_type,
    COUNT(*) as reading_count,
    AVG(power_output) as avg_power_output,
    MAX(temperature) as max_temperature
FROM solar_panel_data
GROUP BY DATE(timestamp)
UNION ALL
SELECT 
    DATE(timestamp) as date,
    'batteries' as component_type,
    COUNT(*) as reading_count,
    AVG(soc) as avg_soc,
    MAX(temperature) as max_temperature
FROM battery_data
GROUP BY DATE(timestamp)
UNION ALL
SELECT 
    DATE(timestamp) as date,
    'generators' as component_type,
    COUNT(*) as reading_count,
    AVG(load_percentage) as avg_load,
    MAX(coolant_temp) as max_temperature
FROM generator_data
GROUP BY DATE(timestamp);

-- Refresh materialized view
-- REFRESH MATERIALIZED VIEW daily_component_summary;

-- 19. Predictive query: Components requiring attention soon
SELECT 
    hh.component_type,
    hh.component_id,
    hh.health_score,
    hh.rul_days,
    COUNT(a.id) as recent_alerts
FROM health_history hh
LEFT JOIN alerts a ON 
    a.component_type = hh.component_type 
    AND a.component_id = hh.component_id
    AND a.timestamp > NOW() - INTERVAL '7 days'
WHERE hh.timestamp = (
    SELECT MAX(timestamp)
    FROM health_history
    WHERE component_type = hh.component_type
      AND component_id = hh.component_id
)
AND (hh.health_score < 80 OR hh.rul_days < 30)
GROUP BY hh.component_type, hh.component_id, hh.health_score, hh.rul_days
ORDER BY hh.health_score ASC, hh.rul_days ASC;

-- 20. Export data for machine learning (CSV format)
COPY (
    SELECT 
        timestamp,
        voltage,
        current,
        temperature,
        soc,
        internal_resistance,
        cycle_count,
        health_status
    FROM battery_data
    ORDER BY timestamp
) TO '/tmp/battery_ml_data.csv' WITH CSV HEADER;
"""
        
        with open('postgresql_sample_queries.sql', 'w') as f:
            f.write(queries)
        
        print("\n✓ Sample queries exported to 'postgresql_sample_queries.sql'")
    
    def create_useful_views(self):
        """Create useful database views"""
        print("\nCreating database views...")
        
        # View for latest component readings
        self.cursor.execute('''
            CREATE OR REPLACE VIEW latest_component_readings AS
            SELECT 
                'solar_panels' as component_type,
                panel_id as component_id,
                timestamp,
                power_output as primary_metric,
                temperature,
                health_status
            FROM solar_panel_data
            WHERE timestamp = (SELECT MAX(timestamp) FROM solar_panel_data WHERE panel_id = solar_panel_data.panel_id)
            
            UNION ALL
            
            SELECT 
                'batteries' as component_type,
                battery_id as component_id,
                timestamp,
                soc as primary_metric,
                temperature,
                health_status
            FROM battery_data
            WHERE timestamp = (SELECT MAX(timestamp) FROM battery_data WHERE battery_id = battery_data.battery_id)
            
            UNION ALL
            
            SELECT 
                'generators' as component_type,
                generator_id as component_id,
                timestamp,
                load_percentage as primary_metric,
                coolant_temp as temperature,
                health_status
            FROM generator_data
            WHERE timestamp = (SELECT MAX(timestamp) FROM generator_data WHERE generator_id = generator_data.generator_id)
        ''')
        
        # View for active alerts
        self.cursor.execute('''
            CREATE OR REPLACE VIEW active_alerts AS
            SELECT 
                id,
                timestamp,
                component_type,
                component_id,
                severity,
                alert_type,
                message,
                EXTRACT(EPOCH FROM (NOW() - timestamp))/3600 as hours_since_alert
            FROM alerts
            WHERE resolved = FALSE
            ORDER BY 
                CASE severity
                    WHEN 'critical' THEN 1
                    WHEN 'high' THEN 2
                    WHEN 'medium' THEN 3
                    ELSE 4
                END,
                timestamp DESC
        ''')
        
        self.conn.commit()
        print("✓ Database views created")
    
    def close(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        print(f"\n✓ Database connection closed")


def main():
    """Main execution function"""
    print("="*70)
    print("PREDICTIVE MAINTENANCE PostgreSQL DATABASE GENERATOR")
    print("="*70)
    print("\nIMPORTANT: Update database configuration before running!")
    print("Edit the db_config in the script with your PostgreSQL credentials.\n")
    
    # Database configuration
    db_config = {
        'host': 'localhost',
        'port': 5432,
        'database': 'predictive_maintenance',
        'user': 'postgres',
        'password': 'postgres'  # CHANGE THIS!
    }
    
    try:
        # Create database connection
        db = PostgreSQLMaintenanceDB(db_config)
        
        # Create all tables
        db.create_tables()
        
        # Generate sensor data for 2 months
        db.generate_solar_panel_data()
        db.generate_battery_data()
        db.generate_generator_data()
        
        # Generate supporting data
        db.generate_alerts()
        db.generate_maintenance_schedule()
        db.generate_health_history()
        
        # Create indexes for performance
        db.create_indexes()
        
        # Create useful views
        db.create_useful_views()
        
        # Print summary
        db.print_database_summary()
        
        # Export sample queries
        db.export_sample_queries()
        
        # Close connection
        db.close()
        
        print("\n" + "="*70)
        print("DATABASE CREATION COMPLETE!")
        print("="*70)
        print("\nYou can now connect to the database with:")
        print("  import psycopg2")
        print("  conn = psycopg2.connect(")
        print("      host='localhost',")
        print("      database='predictive_maintenance',")
        print("      user='postgres',")
        print("      password='your_password'")
        print("  )")
        print("\nOr use pgAdmin, DBeaver, or any PostgreSQL client.")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease ensure:")
        print("  1. PostgreSQL is installed and running")
        print("  2. Database credentials are correct")
        print("  3. User has permission to create databases")
        print("  4. psycopg2 package is installed: pip install psycopg2-binary")
        raise


if __name__ == "__main__":
    main()