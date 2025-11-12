
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
