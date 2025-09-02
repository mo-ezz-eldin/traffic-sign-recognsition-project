"""
Main Streamlit application for traffic sign recognition
"""
import streamlit as st
from PIL import Image
import time

# Import custom modules
from model.model_utils import (
    load_model, preprocess_image, predict, 
    get_top_predictions, CLASS_NAMES
)
from ui.components import (
    apply_custom_css, render_header, render_upload_section,
    render_success_message, render_prediction_result,
    render_image_details, render_top_predictions,
    render_stats_section, render_instructions,
    get_confidence_message
)
from config.settings import PAGE_CONFIG, UPLOAD_CONFIG


def main():
    """Main application function"""
    
    # Configure page
    st.set_page_config(**PAGE_CONFIG)
    
    # Apply custom styling
    apply_custom_css()
    
    # Render header
    render_header()
    
    # Load model with status indicator
    with st.spinner("🤖 Loading AI model..."):
        model = load_model()
    
    if model is None:
        st.error("Failed to load the AI model. Please check if 'traffic_sign_model.pth' exists.")
        st.stop()
    
    # Show success message
    render_success_message()
    
    # Render upload section
    render_upload_section()
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=UPLOAD_CONFIG["allowed_types"],
        help=UPLOAD_CONFIG["help_text"]
    )
    
    if uploaded_file is not None:
        # Process uploaded image
        process_uploaded_image(uploaded_file, model)
    
    # Render footer sections
    st.markdown("---")
    render_stats_section()
    render_instructions()


def process_uploaded_image(uploaded_file, model):
    """
    Process the uploaded image and display results
    
    Args:
        uploaded_file: Streamlit uploaded file object
        model: Loaded PyTorch model
    """
    # Create layout columns
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.markdown("### 🖼️ Your Image")
        st.image(image, use_column_width=True, caption="Uploaded Traffic Sign")
        
        # Show image details
        render_image_details(image)
    
    with col2:
        st.markdown("### 🧠 AI Analysis")
        
        # Make prediction with loading animation
        with st.spinner("🔍 Analyzing your traffic sign..."):
            time.sleep(0.5)  # Dramatic effect
            
            try:
                # Preprocess and predict
                image_tensor = preprocess_image(image)
                predicted_class, confidence, probabilities = predict(model, image_tensor)
                
                # Display main prediction
                predicted_name = CLASS_NAMES[predicted_class]
                render_prediction_result(predicted_name, confidence)
                
                # Show confidence progress bar
                st.markdown("#### 📊 Confidence Level")
                st.progress(confidence)
                
                # Display confidence message
                msg_type, msg_text = get_confidence_message(confidence)
                if msg_type == "success":
                    st.success(msg_text)
                elif msg_type == "info":
                    st.info(msg_text)
                else:
                    st.warning(msg_text)
                
                # Show top 3 predictions
                with st.expander("🏆 See Top 3 Predictions", expanded=True):
                    top_predictions = get_top_predictions(probabilities, top_k=3)
                    render_top_predictions(top_predictions)
                    
            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")


if __name__ == "__main__":
    main()
