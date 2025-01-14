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