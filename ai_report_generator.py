# ai_report_generator.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from typing import Dict, List, Any
import logging

class AIReportGenerator:
    """
    AI-powered analysis and report generation for predictive maintenance system
    """
    
    def __init__(self, db_integration):
        self.db = db_integration
        self.logger = logging.getLogger(__name__)
    
    def analyze_component_health(self, component_type: str, component_id: str) -> Dict[str, Any]:
        """AI analysis of component health and trends"""
        
        try:
            # Get recent predictions
            df_history = self.db.get_prediction_history(component_type, component_id, days=30)
            
            if df_history.empty:
                return self._generate_fallback_analysis(component_type)
            
            # Calculate trends
            recent_health = df_history['predicted_health'].tail(7).mean()
            previous_health = df_history['predicted_health'].head(7).mean()
            health_trend = recent_health - previous_health
            
            # Calculate degradation rate
            if len(df_history) > 1:
                days_data = (df_history['timestamp'].max() - df_history['timestamp'].min()).days
                if days_data > 0:
                    health_change = df_history['predicted_health'].iloc[-1] - df_history['predicted_health'].iloc[0]
                    degradation_rate = health_change / days_data
                else:
                    degradation_rate = 0
            else:
                degradation_rate = 0
            
            # Get current prediction
            current_health = df_history['predicted_health'].iloc[-1] if not df_history.empty else 85.0
            current_rul = df_history['predicted_rul'].iloc[-1] if not df_history.empty else 180.0
            
            # Generate insights
            insights = self._generate_health_insights(
                component_type, current_health, health_trend, degradation_rate, current_rul
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                component_type, current_health, health_trend, degradation_rate, current_rul
            )
            
            # Calculate risk score
            risk_score = self._calculate_risk_score(current_health, degradation_rate, current_rul)
            
            return {
                'component': component_type,
                'component_id': component_id,
                'current_health': round(current_health, 1),
                'current_rul': round(current_rul, 1),
                'health_trend': round(health_trend, 2),
                'degradation_rate': round(degradation_rate, 3),
                'risk_score': round(risk_score, 1),
                'risk_level': self._get_risk_level(risk_score),
                'insights': insights,
                'recommendations': recommendations,
                'analysis_timestamp': datetime.now().isoformat(),
                'data_points_analyzed': len(df_history)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing {component_type}: {e}")
            return self._generate_fallback_analysis(component_type)
    
    def analyze_system_health(self) -> Dict[str, Any]:
        """AI analysis of overall system health"""
        
        components = ['solar_panels', 'batteries', 'generators']
        component_analyses = {}
        system_metrics = {
            'total_components': len(components),
            'components_at_risk': 0,
            'avg_system_health': 0,
            'total_maintenance_cost_savings': 0,
            'system_risk_score': 0
        }
        
        # Analyze each component
        component_ids = {
            'solar_panels': 'SP001',
            'batteries': 'BAT001', 
            'generators': 'GEN001'
        }
        
        for component in components:
            analysis = self.analyze_component_health(component, component_ids.get(component))
            component_analyses[component] = analysis
            
            # Update system metrics
            system_metrics['avg_system_health'] += analysis['current_health']
            system_metrics['system_risk_score'] += analysis['risk_score']
            
            if analysis['risk_level'] in ['high', 'critical']:
                system_metrics['components_at_risk'] += 1
        
        # Calculate averages
        system_metrics['avg_system_health'] = round(system_metrics['avg_system_health'] / len(components), 1)
        system_metrics['system_risk_score'] = round(system_metrics['system_risk_score'] / len(components), 1)
        
        # Generate system-wide insights
        system_insights = self._generate_system_insights(component_analyses, system_metrics)
        system_recommendations = self._generate_system_recommendations(component_analyses, system_metrics)
        
        # Calculate cost savings (estimated)
        system_metrics['total_maintenance_cost_savings'] = self._calculate_cost_savings(component_analyses)
        
        return {
            'system_metrics': system_metrics,
            'component_analyses': component_analyses,
            'system_insights': system_insights,
            'system_recommendations': system_recommendations,
            'analysis_timestamp': datetime.now().isoformat(),
            'report_id': f"SR-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        }
    
    def analyze_maintenance_patterns(self) -> Dict[str, Any]:
        """AI analysis of maintenance patterns and optimization opportunities"""
        
        try:
            # Get maintenance history from database
            query = """
                SELECT component_type, scheduled_date, priority, status, health_score, rul
                FROM maintenance_schedule 
                WHERE scheduled_date >= NOW() - INTERVAL '90 days'
                ORDER BY scheduled_date
            """
            
            self.db.cursor.execute(query)
            maintenance_data = self.db.cursor.fetchall()
            
            # Analyze patterns
            maintenance_analysis = self._analyze_maintenance_patterns(maintenance_data)
            optimization_opportunities = self._find_optimization_opportunities(maintenance_data)
            
            return {
                'maintenance_analysis': maintenance_analysis,
                'optimization_opportunities': optimization_opportunities,
                'total_maintenance_events': len(maintenance_data),
                'analysis_period': '90 days'
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing maintenance patterns: {e}")
            return {'error': 'Failed to analyze maintenance patterns'}
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive AI-powered report"""
        
        system_analysis = self.analyze_system_health()
        maintenance_analysis = self.analyze_maintenance_patterns()
        
        # Generate executive summary
        executive_summary = self._generate_executive_summary(system_analysis, maintenance_analysis)
        
        # Calculate ROI metrics
        roi_metrics = self._calculate_roi_metrics(system_analysis, maintenance_analysis)
        
        return {
            'executive_summary': executive_summary,
            'system_analysis': system_analysis,
            'maintenance_analysis': maintenance_analysis,
            'roi_metrics': roi_metrics,
            'report_generated': datetime.now().isoformat(),
            'report_period': 'Last 30 days',
            'ai_model_version': 'v1.0'
        }
    
    def _generate_health_insights(self, component_type: str, health: float, trend: float, 
                                degradation_rate: float, rul: float) -> List[str]:
        """Generate AI insights for component health"""
        
        insights = []
        
        # Health-based insights
        if health >= 90:
            insights.append(f"✅ {component_type.replace('_', ' ').title()} are in excellent condition")
        elif health >= 75:
            insights.append(f"⚠️ {component_type.replace('_', ' ').title()} show normal wear patterns")
        elif health >= 60:
            insights.append(f"🔶 {component_type.replace('_', ' ').title()} require monitoring")
        else:
            insights.append(f"🚨 {component_type.replace('_', ' ').title()} need immediate attention")
        
        # Trend-based insights
        if trend < -2:
            insights.append("📉 Health is declining rapidly - investigate root causes")
        elif trend < -0.5:
            insights.append("📉 Moderate degradation detected")
        elif trend > 0.5:
            insights.append("📈 Health is improving - maintenance effective")
        else:
            insights.append("➡️ Health trend is stable")
        
        # RUL-based insights
        if rul < 30:
            insights.append(f"⏰ Remaining useful life is critical: {rul:.0f} days")
        elif rul < 90:
            insights.append(f"📅 Plan maintenance within next {rul:.0f} days")
        else:
            insights.append(f"✅ Long remaining useful life: {rul:.0f} days")
        
        # Degradation rate insights
        if degradation_rate < -1:
            insights.append("🔻 High degradation rate - consider proactive measures")
        elif degradation_rate < -0.3:
            insights.append("🔻 Normal degradation rate for age and usage")
        
        return insights
    
    def _generate_recommendations(self, component_type: str, health: float, trend: float,
                                degradation_rate: float, rul: float) -> List[str]:
        """Generate AI recommendations"""
        
        recommendations = []
        
        if health < 60 or rul < 30:
            recommendations.append(f"🚨 IMMEDIATE: Schedule emergency maintenance for {component_type}")
            recommendations.append("🔧 Perform comprehensive diagnostic testing")
            recommendations.append("📋 Review operating parameters and environmental conditions")
        
        elif health < 75 or rul < 90:
            recommendations.append(f"📅 Schedule maintenance within next 30 days for {component_type}")
            recommendations.append("🔍 Conduct detailed performance analysis")
            recommendations.append("📊 Monitor key performance indicators closely")
        
        else:
            recommendations.append(f"✅ Continue routine monitoring of {component_type}")
            recommendations.append("📈 Focus on preventive maintenance schedule")
        
        # Specific recommendations based on component type
        if component_type == 'batteries':
            if degradation_rate < -0.5:
                recommendations.append("🔋 Optimize charging cycles to extend battery life")
                recommendations.append("🌡️ Monitor temperature controls more frequently")
        
        elif component_type == 'solar_panels':
            if health < 80:
                recommendations.append("🧹 Schedule panel cleaning to improve efficiency")
                recommendations.append("🔍 Inspect for physical damage or shading issues")
        
        elif component_type == 'generators':
            if trend < -1:
                recommendations.append("⛽ Check fuel quality and filtration systems")
                recommendations.append("🔧 Inspect lubrication and cooling systems")
        
        # General recommendations
        recommendations.append("📱 Ensure remote monitoring systems are operational")
        recommendations.append("📚 Review maintenance logs for patterns")
        
        return recommendations
    
    def _calculate_risk_score(self, health: float, degradation_rate: float, rul: float) -> float:
        """Calculate comprehensive risk score (0-100, higher = more risk)"""
        
        health_risk = max(0, 100 - health) * 0.4  # 40% weight
        degradation_risk = min(100, abs(degradation_rate) * 50) * 0.3  # 30% weight
        rul_risk = max(0, min(100, (365 - rul) / 3.65)) * 0.3  # 30% weight
        
        return health_risk + degradation_risk + rul_risk
    
    def _get_risk_level(self, risk_score: float) -> str:
        """Convert risk score to risk level"""
        
        if risk_score >= 70:
            return 'critical'
        elif risk_score >= 50:
            return 'high'
        elif risk_score >= 30:
            return 'medium'
        elif risk_score >= 15:
            return 'low'
        else:
            return 'very_low'
    
    def _generate_system_insights(self, component_analyses: Dict, system_metrics: Dict) -> List[str]:
        """Generate system-wide insights"""
        
        insights = []
        avg_health = system_metrics['avg_system_health']
        components_at_risk = system_metrics['components_at_risk']
        
        insights.append(f"🏭 Overall system health: {avg_health}%")
        
        if components_at_risk == 0:
            insights.append("✅ All components operating within safe parameters")
        else:
            insights.append(f"⚠️ {components_at_risk} component(s) require attention")
        
        # Find worst performing component
        worst_component = max(component_analyses.values(), key=lambda x: x['risk_score'])
        if worst_component['risk_score'] > 50:
            insights.append(f"🔴 Highest risk: {worst_component['component']} ({worst_component['risk_level']} risk)")
        
        # System efficiency insight
        if avg_health > 85:
            insights.append("📈 System operating at peak efficiency")
        elif avg_health > 70:
            insights.append("📊 System performance is satisfactory")
        else:
            insights.append("📉 System performance needs improvement")
        
        return insights
    
    def _generate_system_recommendations(self, component_analyses: Dict, system_metrics: Dict) -> List[str]:
        """Generate system-wide recommendations"""
        
        recommendations = []
        components_at_risk = system_metrics['components_at_risk']
        
        if components_at_risk > 0:
            recommendations.append("🎯 Prioritize maintenance for high-risk components")
            recommendations.append("📋 Develop contingency plans for critical assets")
        
        # Check if any components are trending poorly
        declining_components = [
            comp for comp, analysis in component_analyses.items() 
            if analysis['health_trend'] < -1
        ]
        
        if declining_components:
            recommendations.append(f"🔍 Investigate root causes for {', '.join(declining_components)}")
        
        # General system recommendations
        recommendations.append("📊 Continue predictive maintenance program")
        recommendations.append("🔧 Regular calibration of monitoring equipment")
        recommendations.append("📚 Document all maintenance activities")
        
        return recommendations
    
    def _analyze_maintenance_patterns(self, maintenance_data: List) -> Dict[str, Any]:
        """Analyze maintenance patterns"""
        
        if not maintenance_data:
            return {'message': 'Insufficient maintenance data for analysis'}
        
        df = pd.DataFrame(maintenance_data)
        
        analysis = {
            'total_maintenance_events': len(df),
            'preventive_maintenance': len(df[df['priority'].isin(['LOW', 'MEDIUM'])]),
            'corrective_maintenance': len(df[df['priority'].isin(['HIGH', 'CRITICAL'])]),
            'avg_time_between_maintenance': 0,
            'most_frequent_issues': [],
            'maintenance_cost_trend': 'stable'
        }
        
        # Calculate average time between maintenance for each component
        component_intervals = {}
        for component in df['component_type'].unique():
            comp_data = df[df['component_type'] == component].sort_values('scheduled_date')
            if len(comp_data) > 1:
                intervals = (comp_data['scheduled_date'].diff().dt.days).mean()
                component_intervals[component] = round(intervals, 1)
        
        analysis['component_maintenance_intervals'] = component_intervals
        analysis['avg_time_between_maintenance'] = round(np.mean(list(component_intervals.values())), 1) if component_intervals else 0
        
        return analysis
    
    def _find_optimization_opportunities(self, maintenance_data: List) -> List[str]:
        """Find maintenance optimization opportunities"""
        
        opportunities = []
        
        if not maintenance_data:
            return ["Collect more maintenance data for optimization analysis"]
        
        df = pd.DataFrame(maintenance_data)
        
        # Check for frequent high-priority maintenance
        critical_maintenance = df[df['priority'] == 'CRITICAL']
        if len(critical_maintenance) > 2:
            opportunities.append("Reduce emergency maintenance through better prediction")
        
        # Check maintenance intervals
        for component in df['component_type'].unique():
            comp_data = df[df['component_type'] == component]
            if len(comp_data) > 3:  # Enough data for pattern analysis
                avg_health_before_maintenance = comp_data['health_score'].mean()
                if avg_health_before_maintenance < 50:
                    opportunities.append(f"Earlier intervention needed for {component}")
        
        # General optimization opportunities
        opportunities.append("Implement condition-based maintenance scheduling")
        opportunities.append("Optimize spare parts inventory based on failure patterns")
        opportunities.append("Train maintenance staff on predictive techniques")
        
        return opportunities
    
    def _generate_executive_summary(self, system_analysis: Dict, maintenance_analysis: Dict) -> Dict[str, Any]:
        """Generate executive summary"""
        
        system_metrics = system_analysis['system_metrics']
        
        summary = {
            'overall_health': system_metrics['avg_system_health'],
            'system_status': 'Excellent' if system_metrics['avg_system_health'] > 85 else 
                           'Good' if system_metrics['avg_system_health'] > 70 else 
                           'Needs Attention',
            'components_at_risk': system_metrics['components_at_risk'],
            'estimated_savings': f"${system_metrics['total_maintenance_cost_savings']:,.0f}",
            'key_achievements': [],
            'areas_for_improvement': []
        }
        
        # Key achievements
        if system_metrics['components_at_risk'] == 0:
            summary['key_achievements'].append("All components operating safely")
        
        if system_metrics['avg_system_health'] > 80:
            summary['key_achievements'].append("High overall system reliability")
        
        # Areas for improvement
        critical_components = [
            comp for comp, analysis in system_analysis['component_analyses'].items()
            if analysis['risk_level'] in ['high', 'critical']
        ]
        
        if critical_components:
            summary['areas_for_improvement'].append(f"Address {len(critical_components)} high-risk components")
        
        return summary
    
    def _calculate_cost_savings(self, component_analyses: Dict) -> float:
        """Calculate estimated cost savings from predictive maintenance"""
        
        # Base costs (simplified estimation)
        emergency_repair_cost = 5000
        planned_maintenance_cost = 1000
        component_value = 20000
        
        total_savings = 0
        
        for component, analysis in component_analyses.items():
            risk_level = analysis['risk_level']
            
            if risk_level == 'critical':
                # Savings from avoiding emergency repair
                savings = emergency_repair_cost - planned_maintenance_cost
            elif risk_level == 'high':
                # Savings from early detection
                savings = 0.7 * emergency_repair_cost - planned_maintenance_cost
            else:
                # Savings from optimized scheduling
                savings = 0.3 * planned_maintenance_cost
            
            total_savings += max(0, savings)
        
        return round(total_savings)
    
    def _calculate_roi_metrics(self, system_analysis: Dict, maintenance_analysis: Dict) -> Dict[str, Any]:
        """Calculate ROI metrics"""
        
        cost_savings = system_analysis['system_metrics']['total_maintenance_cost_savings']
        estimated_system_value = 150000  # Simplified system value
        
        roi_metrics = {
            'estimated_annual_savings': cost_savings * 12,
            'system_uptime_improvement': '5-15%',
            'maintenance_cost_reduction': '20-40%',
            'equipment_life_extension': '15-25%',
            'roi_period': '6-18 months',
            'total_system_value_protected': f"${estimated_system_value:,.0f}"
        }
        
        return roi_metrics
    
    def _generate_fallback_analysis(self, component_type: str) -> Dict[str, Any]:
        """Generate fallback analysis when data is insufficient"""
        
        return {
            'component': component_type,
            'current_health': 85.0,
            'current_rul': 180.0,
            'health_trend': 0.0,
            'degradation_rate': -0.1,
            'risk_score': 25.0,
            'risk_level': 'low',
            'insights': [f"Limited data available for {component_type}. Collect more sensor readings for accurate analysis."],
            'recommendations': ["Install additional sensors", "Increase data collection frequency", "Validate ML model performance"],
            'analysis_timestamp': datetime.now().isoformat(),
            'data_points_analyzed': 0,
            'data_quality': 'low'
        }