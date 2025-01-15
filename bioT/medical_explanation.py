import numpy as np
from collections import defaultdict
from enum import Enum

class MedicalExplanationGenerator:
    """
    Medical explanation generator focused on specific CKD features
    """
    def __init__(self):
        self.ckd_knowledge_base = {
            'albumin': {
                'units': 'nominal (0-5)',
                'normal_range': '0',
                'interpretation': {
                    '0': 'Normal albumin levels - no significant protein in urine',
                    '1': 'Trace amounts of protein in urine',
                    '2': 'Mild proteinuria (+)',
                    '3': 'Moderate proteinuria (++)',
                    '4': 'Severe proteinuria (+++)',
                    '5': 'Very severe proteinuria (++++)'
                },
                'precautions': {
                    'general': [
                        "Regular protein monitoring in urine",
                        "Follow prescribed dietary protein restrictions",
                        "Control blood pressure and blood sugar",
                        "Regular kidney function monitoring"
                    ],
                    'high_risk': [  # For values 3-5
                        "Strict dietary protein control",
                        "More frequent medical monitoring",
                        "Salt restriction",
                        "Close blood pressure monitoring"
                    ]
                },
                'treatment': {
                    'lifestyle': [
                        "Balanced protein intake as per doctor's advice",
                        "Regular exercise within limits",
                        "Maintain healthy weight",
                        "Adequate hydration"
                    ],
                    'monitoring': [
                        "Regular urine protein testing",
                        "Blood pressure monitoring",
                        "Regular kidney function assessment"
                    ]
                }
            },
            'serum_creatinine': {
                'units': 'mgs/dl',
                'normal_range': {
                    'male': '0.7-1.3',
                    'female': '0.6-1.1'
                },
                'precautions': {
                    'general': [
                        "Regular creatinine monitoring",
                        "Maintain proper hydration",
                        "Avoid nephrotoxic medications",
                        "Regular exercise within limits"
                    ],
                    'high_risk': [
                        "Strict monitoring of kidney function",
                        "Dietary modifications",
                        "Medication review with healthcare provider",
                        "Regular consultation with nephrologist"
                    ]
                },
                'treatment': {
                    'lifestyle': [
                        "Balanced diet with appropriate protein intake",
                        "Regular physical activity as tolerated",
                        "Proper hydration",
                        "Stress management"
                    ],
                    'monitoring': [
                        "Regular blood tests",
                        "GFR estimation",
                        "Blood pressure monitoring"
                    ]
                }
            },
            'hemoglobin': {
                'units': 'gms',
                'normal_range': {
                    'male': '13.5-17.5',
                    'female': '12.0-15.5'
                },
                'precautions': {
                    'general': [
                        "Regular hemoglobin monitoring",
                        "Iron-rich diet when appropriate",
                        "Regular exercise as tolerated",
                        "Proper nutrition"
                    ],
                    'low': [
                        "Iron supplementation if prescribed",
                        "More frequent monitoring",
                        "Fatigue management",
                        "Dietary modifications"
                    ]
                },
                'treatment': {
                    'lifestyle': [
                        "Iron-rich diet",
                        "Vitamin C intake with iron-rich foods",
                        "Regular mild exercise",
                        "Adequate rest"
                    ],
                    'monitoring': [
                        "Regular blood count tests",
                        "Iron studies",
                        "Vitamin B12 and folate levels"
                    ]
                }
            },
            'packed_cell_volume': {
                'units': 'percentage',
                'normal_range': {
                    'male': '40-54',
                    'female': '36-48'
                },
                'precautions': [
                    "Regular monitoring of blood counts",
                    "Maintain proper hydration",
                    "Report unusual fatigue or weakness",
                    "Follow prescribed treatment plan"
                ]
            },
            'red_blood_cell_count': {
                'units': 'millions/cmm',
                'normal_range': {
                    'male': '4.5-5.9',
                    'female': '4.1-5.1'
                },
                'precautions': [
                    "Regular blood count monitoring",
                    "Proper nutrition",
                    "Report unusual symptoms",
                    "Follow prescribed medications"
                ]
            },
            'diabetes_mellitus': {
                'units': 'nominal (yes-1, no-0)',
                'precautions': {
                    'general': [
                        "Regular blood sugar monitoring",
                        "Medication compliance",
                        "Regular exercise",
                        "Proper foot care"
                    ],
                    'with_ckd': [
                        "Strict blood sugar control",
                        "Regular kidney function monitoring",
                        "Blood pressure control",
                        "Careful medication management"
                    ]
                },
                'treatment': {
                    'lifestyle': [
                        "Balanced diabetic diet",
                        "Regular physical activity",
                        "Weight management",
                        "Stress reduction"
                    ],
                    'monitoring': [
                        "Regular HbA1c testing",
                        "Blood sugar monitoring",
                        "Kidney function tests",
                        "Regular medical check-ups"
                    ]
                }
            },
            'sugar': {
                'units': 'nominal (0-5)',
                'interpretation': {
                    '0': 'Normal',
                    '1': 'Trace',
                    '2': 'Mild elevation (+)',
                    '3': 'Moderate elevation (++)',
                    '4': 'High elevation (+++)',
                    '5': 'Very high elevation (++++)'
                }
            },
            'blood_glucose_random': {
                'units': 'mgs/dl',
                'normal_range': '<140',
                'precautions': {
                    'general': [
                        "Regular blood sugar monitoring",
                        "Balanced diet",
                        "Regular exercise",
                        "Medication compliance"
                    ],
                    'high': [
                        "More frequent blood sugar checks",
                        "Dietary adjustments",
                        "Medication review",
                        "Lifestyle modifications"
                    ]
                }
            },
            'hypertension': {
                'units': 'nominal (yes-1, no-0)',
                'precautions': {
                    'general': [
                        "Regular blood pressure monitoring",
                        "Low sodium diet",
                        "Regular exercise",
                        "Stress management"
                    ],
                    'with_ckd': [
                        "Strict blood pressure control",
                        "Regular medication review",
                        "Dietary sodium restriction",
                        "Regular kidney function monitoring"
                    ]
                },
                'treatment': {
                    'lifestyle': [
                        "DASH diet",
                        "Regular physical activity",
                        "Weight management",
                        "Stress reduction"
                    ],
                    'monitoring': [
                        "Regular blood pressure checks",
                        "Kidney function monitoring",
                        "Heart health assessment",
                        "Medication effectiveness review"
                    ]
                }
            },
            'appetite': {
                'units': 'nominal (good-0, poor-1)',
                'precautions': {
                    'poor': [
                        "Regular nutritional assessment",
                        "Small, frequent meals",
                        "Monitor weight changes",
                        "Consider nutritional supplements"
                    ]
                },
                'recommendations': {
                    'poor': [
                        "Eat small, frequent meals",
                        "Choose nutrient-dense foods",
                        "Consider nutrition counseling",
                        "Monitor weight regularly"
                    ]
                }
            }
        }

    def get_feature_interpretation(self, feature_name, value):
        """Generate interpretation for a single feature based on its value."""
        feature_info = self.ckd_knowledge_base.get(feature_name.lower().replace(" ", "_"))
        if not feature_info:
            return f"Impact of {feature_name} requires medical interpretation."

        if 'interpretation' in feature_info:
            return feature_info['interpretation'].get(str(value), 
                   "Value requires medical interpretation")
        
        if 'normal_range' in feature_info:
            if isinstance(feature_info['normal_range'], dict):
                return (f"Normal ranges: Male: {feature_info['normal_range']['male']}, "
                       f"Female: {feature_info['normal_range']['female']}")
            return f"Normal range: {feature_info['normal_range']}"

        return "Interpretation requires medical evaluation"

    def get_feature_precautions(self, feature_name, value):
        """Get precautions for a specific feature value."""
        feature = self.ckd_knowledge_base.get(feature_name.lower().replace(" ", "_"), {})
        precautions = feature.get('precautions', {})
        
        if isinstance(precautions, dict):
            if 'high_risk' in precautions and self.is_high_risk(feature_name, value):
                return precautions['high_risk']
            return precautions.get('general', [])
        
        return precautions if isinstance(precautions, list) else []

    def is_high_risk(self, feature_name, value):
        """Determine if a value indicates high risk."""
        feature = self.ckd_knowledge_base.get(feature_name.lower().replace(" ", "_"), {})
        
        if feature_name == 'albumin':
            return int(value) >= 3
        elif feature_name == 'serum_creatinine':
            return float(value) > 1.3  # Using male upper limit as reference
        elif feature_name == 'blood_glucose_random':
            return float(value) > 200
        
        return False

    def explain_prediction(self, feature_labels, shap_values, feature_values=None):
        
        """
        Generate a detailed explanation with feature-specific interpretations
        
        Args:
            feature_labels (list): Names of the features
            shap_values (list): SHAP values corresponding to the features
            feature_values (dict): Actual values for each feature (optional)
        """
        try:
            explanation = []
            medical_insights = []
            precautions = []
            print("starting")
            # Generate SHAP-based explanation and collect insights
            feature_importance = list(zip(feature_labels, shap_values))
            feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
            print("Generate SHAP-based explanation and collect insights")
            for feature, shap_value in feature_importance:
                # Add SHAP explanation
                if shap_value < 0:
                    explanation.append( 
                        f"{feature}: Reduces CKD risk (impact: {shap_value:.3f})"
                    )
                elif shap_value > 0:
                    explanation.append(
                        f"{feature}: Increases CKD risk (impact: {shap_value:.3f})"
                    )
                
                # Add feature-specific insights if values are provided
                if feature_values and feature in feature_values:
                    value = feature_values[feature]
                    interpretation = self.get_feature_interpretation(feature, value)
                    medical_insights.append(f"{feature}: {interpretation}")
                    
                    # Add precautions for high-impact features
                    if abs(shap_value) > 0.1:  # Threshold for significant impact
                        feature_precautions = self.get_feature_precautions(feature, value)
                        precautions.extend(feature_precautions)

            return {
                "shap_explanation": "\n".join(explanation),
                "medical_insights": "\n".join(medical_insights),
                "precautions": list(set(precautions)),  # Remove duplicates
                "feature_units": {
                    feature: self.ckd_knowledge_base.get(
                        feature.lower().replace(" ", "_"), {}
                    ).get('units', 'Not specified')
                    for feature in feature_labels
                }
            }
            
        except Exception as e:
            print("An error occurred:", e)
            raise
