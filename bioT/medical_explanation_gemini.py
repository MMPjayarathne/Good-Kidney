import google.generativeai as genai




def explain_prediction_with_gemini(feature_labels, shap_values, feature_values, type):
    """
    Generate a detailed explanation using SHAP values and GPT for medical insights.

    Args:
        feature_labels (list): Names of the features.
        shap_values (list): SHAP values corresponding to the features.

    Returns:
        dict: A dictionary containing SHAP-based explanations and GPT-generated medical insights.
    """
    try:
        # Ensure feature_labels and shap_values are aligned
        if len(feature_labels) != len(shap_values):
            raise ValueError("The lengths of feature_labels and shap_values do not match.")

        # Basic SHAP-based explanation
        explanation = []
        for feature, shap_value in zip(feature_labels, shap_values):
            if shap_value < 0:
                explanation.append(
                    f"{feature}: A negative contribution ({shap_value:.3f}) indicates a reduced likelihood of CKD.\n"
                )
            elif shap_value > 0:
                explanation.append(
                    f"{feature}: A positive contribution ({shap_value:.3f}) suggests an increased likelihood of CKD.\n"
                )
            else:
                explanation.append(
                    f"{feature}: No significant contribution to the prediction.\n"
                )

        # Combine the SHAP explanations
        explanation_text = "\n".join(explanation)
        print("Generated SHAP explanation:", explanation_text)

        # Prepare the input prompt for GPT
        if(type == 1):
            context = (
                f"These are the feature values was given for Chronic Kidney Disease (CKD) prediction using deep learning : "
                f"{feature_values}\n\n"
                f"then the model indicate the patient has a posibility of CKD"
                f"and the following feature contributions were observed by the SHAP:\n\n"
                f"{explanation_text}\n\n"
                "Based on these observations, provide potential causes of CKD and suggested steps the patient should take to improve their health outcomes.  Need only the medical insights."
            )
        else:
            context = (
                f"These are the feature values was given for Chronic Kidney Disease (CKD) prediction using deep learning : "
                f"{feature_values}\n\n"
                f"then the model indicate the patient is Healthy"
                f"and the following feature contributions were observed by the SHAP:\n\n"
                f"{explanation_text}\n\n"
                "Based on these observations, eventhough the patient do not have CKD, suggest steps that patient should take to improve their health outcomes if need. Need only the medical insights."
            )
    

        # Use GPT for medical guidance
        try:
            genai.configure(api_key="AIzaSyD9nsmYXpHiEto6OSnnwU7h76o2R89vbcI")
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(context)
            # medical_guidance = response["choices"][0]["text"].strip()
            # print("GPT-generated medical guidance:", medical_guidance)

            # Return the combined explanation
            return {
                "shap_explanation": explanation_text,
                "medical_guidance": response.text
            }

        except Exception as e:
            print("Error during GPT request:", e)
            return {
                "shap_explanation": explanation_text,
                "medical_guidance": "Failed to generate guidance due to an error."
            }

    except Exception as e:
        print("An error occurred:", e)
        raise
