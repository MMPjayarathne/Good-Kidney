import streamlit as st
import time

# Initialize session state if not already initialized
if 'page_loaded' not in st.session_state:
    st.session_state.page_loaded = False  # Control the loading state

def start_loading():
    """
    Starts the full-page loading process with blur effect.
    """
    st.session_state.page_loaded = False  # Set the loading state to False (loading)
    
    # Add the full-page loading indicator with blur effect
    st.markdown(
        """
        <style>
            .loading-overlay {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(255, 255, 255, 0.8); /* Light overlay for visibility */
                display: flex;
                justify-content: center;
                align-items: center;
                z-index: 9999;
                backdrop-filter: blur(8px); /* Apply blur effect */
                -webkit-backdrop-filter: blur(8px); /* For Safari */
            }
            .spinner {
                border: 8px solid #f3f3f3;
                border-top: 8px solid #3498db;
                border-radius: 50%;
                width: 50px;
                height: 50px;
                animation: spin 2s linear infinite;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
        <div class="loading-overlay">
            <div class="spinner"></div>
        </div>
        """, unsafe_allow_html=True
    )
    
    # Simulate loading process (replace with actual loading logic)
    time.sleep(3)  # Simulate a delay for loading, replace with actual process

    # Once loading is finished, set the page as loaded
    st.session_state.page_loaded = True

def stop_loading():
    """
    Stops the full-page loading process and shows the page content.
    """
    st.session_state.page_loaded = True
    
    # Hide the loading overlay
    st.markdown("<style>.loading-overlay { display: none; }</style>", unsafe_allow_html=True)
