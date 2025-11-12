// static/JS/script.js - Updated for Real Data
class PredictiveMaintenanceApp {
    constructor() {
        this.chartManager = new ChartManager();
        this.socket = io();
        this.currentData = null;
        this.healthTrendChart = null;
        
        this.init();
    }
    
    init() {
        this.initializeRealTimeUpdates();
        this.updateCurrentTime();
        this.loadInitialData();
        
        // Update time every minute
        setInterval(() => this.updateCurrentTime(), 60000);
        
        // Refresh data every 2 minutes
        setInterval(() => this.loadInitialData(), 120000);
    }
    
    initializeRealTimeUpdates() {
        this.socket.on('connect', () => {
            console.log('Connected to server');
            this.updateSystemStatus('operational');
        });
        
        this.socket.on('disconnect', () => {
            console.log('Disconnected from server');
            this.updateSystemStatus('offline');
        });
        
        this.socket.on('health_update', (data) => {
            this.handleHealthUpdate(data);
        });
        
        this.socket.on('alert_triggered', (alert) => {
            this.showAlert(alert);
        });
    }
    
    updateCurrentTime() {
        const now = new Date();
        const timeString = now.toLocaleTimeString('en-US', {
            hour12: true,
            hour: 'numeric',
            minute: '2-digit'
        });
        const dateString = now.toLocaleDateString('en-US', {
            weekday: 'short',
            year: 'numeric',
            month: 'short',
            day: 'numeric'
        });
        
        const timeElement = document.getElementById('currentTime');
        if (timeElement) {
            timeElement.textContent = `${dateString} • ${timeString}`;
        }
    }
    
    updateSystemStatus(status) {
        const statusElement = document.getElementById('systemStatus');
        if (statusElement) {
            statusElement.className = `status-indicator ${status}`;
            statusElement.innerHTML = `<i class="fas fa-circle"></i> ${status.charAt(0).toUpperCase() + status.slice(1)}`;
        }
    }
    
    async loadInitialData() {
        try {
            console.log('Loading real data from API...');
            const response = await fetch('/api/health_status');
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }
            
            this.currentData = data;
            this.updateDashboard(data);
            console.log('Real data loaded successfully:', data);
        } catch (error) {
            console.error('Failed to load real data:', error);
            this.showError('Failed to load data: ' + error.message);
        }
    }
    
    updateDashboard(data) {
        this.updateHealthCards(data.predictions);
        this.updateSystemHealth(data.predictions);
        this.updateAlerts(data.alerts);
        this.updateMaintenanceSchedule(data.schedule);
    }
    
    updateHealthCards(predictions) {
        for (const [component, data] of Object.entries(predictions)) {
            this.updateHealthCard(component, data);
        }
    }
    
    updateHealthCard(component, data) {
        const healthElement = document.getElementById(`${component.replace('s', 's_').replace('b', 'b_').replace('g', 'g_')}_health`);
        const progressElement = document.getElementById(`${component.replace('s', 's_').replace('b', 'b_').replace('g', 'g_')}_progress`);
        const rulElement = document.getElementById(`${component.replace('s', 's_').replace('b', 'b_').replace('g', 'g_')}_rul`);
        const confidenceElement = document.getElementById(`${component.replace('s', 's_').replace('b', 'b_').replace('g', 'g_')}_confidence`);
        const timestampElement = document.getElementById(`${component.replace('s', 's_').replace('b', 'b_').replace('g', 'g_')}_timestamp`);
        
        if (healthElement && data.health_score !== undefined) {
            healthElement.textContent = `${data.health_score.toFixed(1)}%`;
            healthElement.className = `health-score ${
                data.health_score >= 80 ? 'excellent' : 
                data.health_score >= 60 ? 'good' : 'poor'
            }`;
        }
        
        if (progressElement && data.health_score !== undefined) {
            progressElement.style.width = `${data.health_score}%`;
            progressElement.className = `progress-fill ${
                data.health_score >= 80 ? 'excellent' : 
                data.health_score >= 60 ? 'good' : 'poor'
            }`;
        }
        
        if (rulElement && data.rul !== undefined) {
            rulElement.textContent = `${data.rul.toFixed(1)} days`;
        }
        
        if (confidenceElement && data.confidence !== undefined) {
            confidenceElement.textContent = `${(data.confidence * 100).toFixed(1)}%`;
        }
        
        if (timestampElement && data.timestamp) {
            const date = new Date(data.timestamp);
            timestampElement.textContent = `Last updated: ${date.toLocaleTimeString()}`;
        }
    }
    
    updateSystemHealth(predictions) {
        // Calculate overall system health
        const healthScores = Object.values(predictions).map(p => p.health_score);
        const averageHealth = healthScores.reduce((a, b) => a + b, 0) / healthScores.length;
        
        const systemHealthElement = document.getElementById('system_health');
        const systemProgressElement = document.getElementById('system_progress');
        const mlAccuracyElement = document.getElementById('ml_accuracy');
        
        if (systemHealthElement) {
            systemHealthElement.textContent = `${averageHealth.toFixed(1)}%`;
            systemHealthElement.className = `health-score ${
                averageHealth >= 80 ? 'excellent' : 
                averageHealth >= 60 ? 'good' : 'poor'
            }`;
        }
        
        if (systemProgressElement) {
            systemProgressElement.style.width = `${averageHealth}%`;
            systemProgressElement.className = `progress-fill ${
                averageHealth >= 80 ? 'excellent' : 
                averageHealth >= 60 ? 'good' : 'poor'
            }`;
        }
        
        // Calculate average ML confidence
        const confidences = Object.values(predictions).map(p => p.confidence || 0.85);
        const averageConfidence = confidences.reduce((a, b) => a + b, 0) / confidences.length;
        
        if (mlAccuracyElement) {
            mlAccuracyElement.textContent = `${(averageConfidence * 100).toFixed(1)}%`;
        }
    }
    
    updateAlerts(alerts) {
        const alertsContainer = document.getElementById('alertsContainer');
        const alertsCountElement = document.getElementById('alertsCount');
        const activeAlertsCountElement = document.getElementById('activeAlertsCount');
        
        if (alertsCountElement) {
            alertsCountElement.textContent = alerts.length;
        }
        
        if (activeAlertsCountElement) {
            const criticalAlerts = alerts.filter(alert => alert.severity === 'critical').length;
            activeAlertsCountElement.textContent = criticalAlerts;
        }
        
        if (!alertsContainer) return;
        
        if (alerts.length === 0) {
            alertsContainer.innerHTML = `
                <div class="alert alert-medium">
                    <i class="fas fa-check-circle alert-icon"></i>
                    <div>
                        <strong>No Active Alerts</strong>
                        <p>All systems are operating normally</p>
                    </div>
                </div>
            `;
            return;
        }
        
        // Sort alerts by severity (critical first)
        const sortedAlerts = alerts.sort((a, b) => {
            const severityOrder = { 'critical': 0, 'high': 1, 'medium': 2 };
            return severityOrder[a.severity] - severityOrder[b.severity];
        });
        
        alertsContainer.innerHTML = sortedAlerts.map(alert => `
            <div class="alert alert-${alert.severity}">
                <i class="fas fa-${this.getAlertIcon(alert.severity)} alert-icon"></i>
                <div>
                    <strong>${alert.component.replace(/_/g, ' ').toUpperCase()}</strong>
                    <p>${alert.message}</p>
                    <small><strong>Action:</strong> ${alert.action_required}</small>
                    <div class="alert-time">${new Date(alert.timestamp).toLocaleString()}</div>
                </div>
            </div>
        `).join('');
    }
    
    updateMaintenanceSchedule(schedule) {
        const scheduleContainer = document.getElementById('scheduleContainer');
        if (!scheduleContainer) return;
        
        if (!schedule || schedule.length === 0) {
            scheduleContainer.innerHTML = `
                <div class="card">
                    <div class="card-header">
                        <h4>No Maintenance Scheduled</h4>
                    </div>
                    <p>No maintenance tasks are currently scheduled</p>
                </div>
            `;
            return;
        }
        
        scheduleContainer.innerHTML = schedule.map(item => `
            <div class="card fade-in">
                <div class="card-header">
                    <h4>${item.component.replace(/_/g, ' ').toUpperCase()}</h4>
                    <span class="priority-badge priority-${item.priority.toLowerCase()}">
                        ${item.priority}
                    </span>
                </div>
                <p><strong>Scheduled:</strong> ${item.scheduled_date}</p>
                <p><strong>Health Score:</strong> ${item.health_score}%</p>
                <p><strong>RUL:</strong> ${item.rul} days</p>
                <p><strong>Confidence:</strong> ${(item.confidence * 100).toFixed(1)}%</p>
                <p><strong>Type:</strong> ${item.type.replace(/_/g, ' ')}</p>
                <div class="schedule-actions">
                    <button class="btn btn-success btn-sm" onclick="completeMaintenance('${item.component}')">
                        <i class="fas fa-check"></i> Complete
                    </button>
                </div>
            </div>
        `).join('');
    }
    
    async loadPredictionHistory(componentType) {
        try {
            const response = await fetch(`/api/prediction_history/${componentType}`);
            const data = await response.json();
            
            if (data.history && data.history.length > 0) {
                this.createHealthTrendChart(componentType, data.history);
            } else {
                // Create sample data if no history available
                this.createSampleTrendChart(componentType);
            }
        } catch (error) {
            console.error('Failed to load prediction history:', error);
            this.createSampleTrendChart(componentType);
        }
    }
    
    createHealthTrendChart(componentType, history) {
        const ctx = document.getElementById('healthTrendChart').getContext('2d');
        
        // Process history data
        const labels = history.slice(-30).map(item => 
            new Date(item.timestamp).toLocaleDateString()
        );
        const healthScores = history.slice(-30).map(item => item.predicted_health);
        
        if (this.healthTrendChart) {
            this.healthTrendChart.destroy();
        }
        
        this.healthTrendChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: `${componentType.replace(/_/g, ' ').toUpperCase()} Health Score`,
                    data: healthScores,
                    borderColor: '#2563eb',
                    backgroundColor: 'rgba(37, 99, 235, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: `Health Trend - ${componentType.replace(/_/g, ' ').toUpperCase()}`
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        min: 0,
                        max: 100,
                        grid: {
                            color: 'rgba(0,0,0,0.1)'
                        },
                        title: {
                            display: true,
                            text: 'Health Score (%)'
                        }
                    },
                    x: {
                        grid: {
                            display: false
                        },
                        title: {
                            display: true,
                            text: 'Date'
                        }
                    }
                }
            }
        });
    }
    
    createSampleTrendChart(componentType) {
        const ctx = document.getElementById('healthTrendChart').getContext('2d');
        
        // Generate sample data
        const labels = [];
        const data = [];
        const baseHealth = 85;
        
        for (let i = 29; i >= 0; i--) {
            const date = new Date();
            date.setDate(date.getDate() - i);
            labels.push(date.toLocaleDateString());
            
            // Simulate slight degradation with random variation
            const health = Math.max(50, baseHealth - (i * 0.1) + (Math.random() * 4 - 2));
            data.push(parseFloat(health.toFixed(1)));
        }
        
        if (this.healthTrendChart) {
            this.healthTrendChart.destroy();
        }
        
        this.healthTrendChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: `${componentType.replace(/_/g, ' ').toUpperCase()} Health Score`,
                    data: data,
                    borderColor: '#2563eb',
                    backgroundColor: 'rgba(37, 99, 235, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: `Health Trend - ${componentType.replace(/_/g, ' ').toUpperCase()} (Sample Data)`
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        min: 0,
                        max: 100,
                        grid: {
                            color: 'rgba(0,0,0,0.1)'
                        },
                        title: {
                            display: true,
                            text: 'Health Score (%)'
                        }
                    },
                    x: {
                        grid: {
                            display: false
                        },
                        title: {
                            display: true,
                            text: 'Date'
                        }
                    }
                }
            }
        });
    }
    
    handleHealthUpdate(data) {
        this.updateHealthCard(data.component, data);
        
        // Update system health if we have current data
        if (this.currentData && this.currentData.predictions) {
            this.currentData.predictions[data.component] = data;
            this.updateSystemHealth(this.currentData.predictions);
        }
    }
    
    showAlert(alert) {
        // Create toast notification
        const toast = document.createElement('div');
        toast.className = `alert alert-${alert.severity} fade-in toast-alert`;
        toast.innerHTML = `
            <i class="fas fa-${this.getAlertIcon(alert.severity)} alert-icon"></i>
            <div>
                <strong>${alert.component.replace(/_/g, ' ').toUpperCase()}</strong>
                <p>${alert.message}</p>
                <small>${new Date(alert.timestamp).toLocaleTimeString()}</small>
            </div>
            <button class="toast-close" onclick="this.parentElement.remove()">
                <i class="fas fa-times"></i>
            </button>
        `;
        
        // Add styles for toast
        toast.style.position = 'fixed';
        toast.style.top = '20px';
        toast.style.right = '20px';
        toast.style.zIndex = '1000';
        toast.style.minWidth = '300px';
        toast.style.maxWidth = '400px';
        
        document.body.appendChild(toast);
        
        // Remove toast after 8 seconds
        setTimeout(() => {
            if (toast.parentElement) {
                toast.remove();
            }
        }, 8000);
    }
    
    showError(message) {
        const toast = document.createElement('div');
        toast.className = 'alert alert-critical fade-in toast-alert';
        toast.innerHTML = `
            <i class="fas fa-exclamation-triangle alert-icon"></i>
            <div>
                <strong>Error</strong>
                <p>${message}</p>
            </div>
            <button class="toast-close" onclick="this.parentElement.remove()">
                <i class="fas fa-times"></i>
            </button>
        `;
        
        toast.style.position = 'fixed';
        toast.style.top = '20px';
        toast.style.right = '20px';
        toast.style.zIndex = '1000';
        toast.style.minWidth = '300px';
        
        document.body.appendChild(toast);
        
        setTimeout(() => {
            if (toast.parentElement) {
                toast.remove();
            }
        }, 8000);
    }
    
    getAlertIcon(severity) {
        const icons = {
            'critical': 'exclamation-triangle',
            'high': 'exclamation-circle',
            'medium': 'info-circle'
        };
        return icons[severity] || 'info-circle';
    }
}

// Utility functions
const Utils = {
    formatDate: (dateString) => {
        return new Date(dateString).toLocaleDateString();
    },
    
    formatTime: (dateString) => {
        return new Date(dateString).toLocaleTimeString();
    },
    
    debounce: (func, wait) => {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }
};

// Global functions
function completeMaintenance(component) {
    if (confirm(`Mark maintenance for ${component} as completed?`)) {
        alert('Maintenance marked as completed. In a real application, this would update the database.');
        // Here you would typically make an API call to update the maintenance status
        window.app.loadInitialData(); // Refresh data
    }
}

function refreshAllData() {
    window.app.loadInitialData();
    const trendComponent = document.getElementById('trendComponent');
    if (trendComponent) {
        window.app.loadPredictionHistory(trendComponent.value);
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new PredictiveMaintenanceApp();
});