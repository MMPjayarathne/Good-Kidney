# GOOD KIDNEY
![good kidney](https://github.com/user-attachments/assets/3e18d280-d82d-42fa-a264-6248364067cf)


## Overview
GOOD KIDNEY is a user-friendly application developed as part of my final year research project. The project explores the use of **Explainable Deep Learning** for **Chronic Kidney Disease (CKD) Prediction**, focusing on enhancing transparency and usability in medical decision support systems.

### Research Title:
**EXPLAINABLE DEEP LEARNING FOR CHRONIC KIDNEY DISEASE PREDICTION: ENHANCING TRANSPARENCY AND USER-FRIENDLINESS IN MEDICAL DECISION SUPPORT**

### Purpose:
The application predicts the likelihood of a patient having Chronic Kidney Disease (CKD) based on input features and provides detailed explanations for the prediction. This includes highlighting the contribution of individual features, improving transparency, and helping medical professionals make informed decisions.

---

## Features
- Predicts the probability of a patient having CKD using deep learning models.
- Provides **explainable AI** insights, showing feature contributions to the prediction.
- Developed with **Python** and **Streamlit** for simplicity and interactivity.

---

## Prerequisites
- **Python 3.9** (strictly required for compatibility).
- A system with sufficient resources to run Streamlit and the machine learning models.

---

## Installation and Setup

Follow these steps to set up and run the application on your local machine:

1. **Clone the Repository**  
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Install Dependencies**  
   Install the required Python packages specified in the `requirements.txt` file:  
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**  
   Launch the application using Streamlit:  
   ```bash
   streamlit run app.py
   ```

---

## How It Works
1. **Input Features**: Provide patient data through the application interface.
2. **Prediction**: The model predicts the likelihood of CKD.
3. **Explanation**: The application explains the model's prediction by visualizing the contribution of each input feature.

---

## Technologies Used
- **Python 3.9**
- **Streamlit**: For building the user interface.
- **Scikit-learn**: For preprocessing and model evaluation.
- **SHAP**: For explainable AI visualizations.
- **TensorFlow/Keras**: For implementing the deep learning model.

---
<!--
## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests.  
To contribute:
1. Fork the repository.
2. Create a new branch.
3. Commit your changes and submit a pull request.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---
-->

## Troubleshooting
- **Environment Issues**: Ensure you're using Python 3.9, as other versions may cause compatibility issues.
- **Dependencies**: If issues arise with dependency installations, consider using a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # On Windows, use .venv\Scripts\activate
   pip install -r requirements.txt
   ```

---

## Acknowledgments
This project is part of my final year research at **University of Kelaniya, Sri Lanka**, under the supervision of Dr Isuru Hewapathirana, Sr. Lecturer. Thanks to all contributors and supporters for their guidance and feedback.

---
