import streamlit as st
import matplotlib.pyplot as plt

def plotPieChart(prediction,probability_ckd, probability_healthy):
        # Pie chart display with transparent background
        labels = ['CKD', 'Healthy']
        sizes = [probability_ckd, probability_healthy]
        colors = [(255/255, 0/255, 0/255, 0.5), (61/255, 213/255, 109/255, 0.5)] 

        fig, ax = plt.subplots(figsize=(3, 3), dpi=80)

        # Create the pie chart
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, pctdistance=0.85, textprops={'color': 'white'})


        # Set percentage text color to white and inside the pie
        for autotext in autotexts:
            autotext.set_color('white')

        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # Make the background transparent
        fig.patch.set_alpha(0.0)

        # Display the pie chart in Streamlit
        st.sidebar.pyplot(fig)
        
        


def visualize_lime_explanation_from_text(explanation_text):
    """
    Visualize LIME explanation as a bar chart with vertical feature labels.

    Parameters:
    explanation_text (str): A string of feature contributions, formatted as "Feature: Weight" on each line.
                            Example:
                            Hypertension: 0.126
                            Diabetes Mellitus: 0.116
                            ...
    """
    print(f"Get the text",explanation_text)
    # Parse explanation_text into a dictionary
    explanation_dict = {}
    for line in explanation_text.split("\n"):
        if line.strip():  # Skip empty lines
            feature, weight = line.split(":")
            explanation_dict[feature.strip()] = float(weight.strip())
    
    # Sort features by their absolute contribution
    sorted_explanation = dict(sorted(explanation_dict.items(), key=lambda x: abs(x[1]), reverse=True))
    
    # Split keys and values for visualization
    features = list(sorted_explanation.keys())
    contributions = list(sorted_explanation.values())
    
    # Define colors for positive and negative contributions
    colors = ['green' if value > 0 else 'red' for value in contributions]
    
    # Plot the bar chart
    plt.figure(figsize=(12, 6))
    bars = plt.bar(features, contributions, color=colors)
    
    # Rotate feature labels to vertical
    plt.xticks(rotation=90, fontsize=10)
    
    # Add contribution values above bars
    for bar, contribution in zip(bars, contributions):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), 
                 f"{contribution:.3f}", 
                 ha='center', va='bottom' if contribution > 0 else 'top', 
                 fontsize=9)
    
    # Chart settings
    plt.title('LIME Explanation - Feature Contributions', fontsize=16)
    plt.ylabel('Contribution to Prediction', fontsize=12)
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')  # Add horizontal line at 0
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


