import openai
from dotenv import load_dotenv
import os
from openai import OpenAI

# Load the .env file
load_dotenv()

# Get the API key from the environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")
projectId = os.getenv("PROJECT_ID")

client = OpenAI(
    organization='org-wfULhgE5Tk9jJEo47dmdKwxd',
    project=projectId,
)

def explain_prediction_with_gpt(feature_labels, shap_values):
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
                    f"{feature}: A negative contribution ({shap_value:.3f}) indicates a reduced likelihood of CKD."
                )
            elif shap_value > 0:
                explanation.append(
                    f"{feature}: A positive contribution ({shap_value:.3f}) suggests an increased likelihood of CKD."
                )
            else:
                explanation.append(
                    f"{feature}: No significant contribution to the prediction."
                )

        # Combine the SHAP explanations
        explanation_text = "\n".join(explanation)
        print("Generated SHAP explanation:", explanation_text)

        # Prepare the input prompt for GPT
        context = (
            f"The following feature contributions were observed in the prediction of chronic kidney disease (CKD):\n\n"
            f"{explanation_text}\n\n"
            "Based on these observations, provide potential causes of CKD and suggested steps the patient should take to improve their health outcomes."
        )

        # Use GPT for medical guidance
        try:
            response = client.chat.completions.with_raw_response.create(
                messages=[{
                    "role": "user",
                    "content": context,
                }],
                model="gpt-4o-mini",
            )

            medical_guidance = response["choices"][0]["text"].strip()
            print("GPT-generated medical guidance:", medical_guidance)

            # Return the combined explanation
            return {
                "shap_explanation": explanation_text,
                "medical_guidance": medical_guidance
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
