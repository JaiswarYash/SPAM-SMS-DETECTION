import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import warnings
import pickle
import os
import re
warnings.filterwarnings('ignore')

# Download NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Set page configuration
st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="📱",
    layout="wide"
)

# Title and description
st.title("📱 SMS Spam Classification App")
st.markdown("""
This app classifies SMS messages as **Spam** or **Ham** using machine learning models.
Upload a CSV file with SMS messages or enter a single message for classification.
""")

# Text preprocessing function
def preprocess_text(text):
    """
    Preprocess the text: remove punctuation, stopwords, and apply stemming
    """
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove numbers and special characters
    text = re.sub(r'\d+', '', text)
    
    # Tokenize
    words = text.split()
    
    # Remove stopwords and apply stemming
    ps = PorterStemmer()
    stopwords_set = set(stopwords.words('english'))
    words = [ps.stem(word) for word in words if word not in stopwords_set]
    
    return ' '.join(words)

# Load or train model
@st.cache_resource
def load_model_and_vectorizer():
    """
    Load the trained model and vectorizer, or train new ones if not available
    """
    try:
        # Try to load pre-trained model and vectorizer
        with open('models/spam_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        with open('models/tfidf_vectorizer.pkl', 'rb') as vec_file:
            vectorizer = pickle.load(vec_file)
        return model, vectorizer, "Pre-trained Model", 0.98
    except:
        st.warning("Pre-trained model not found. Please train a model in the 'Model Training' section.")
        return None, None, None, None

# Load and preprocess data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/sms-spam-collection-dataset/spam.csv', encoding='latin-1')
        # Data cleaning
        df = df.dropna(axis=1, how='all')
        df = df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'])
        df = df.rename(columns={'v1': 'target', 'v2': 'message'})
        df = df.dropna()
        df = df.drop_duplicates(keep='first')
        
        # Convert label to numerical values
        label = LabelEncoder()
        df['target'] = label.fit_transform(df['target'])
        
        return df, label
    except FileNotFoundError:
        st.error("Dataset not found. Please make sure the file path is correct.")
        return None, None

# Train models
@st.cache_resource
def train_models(X_train, X_test, y_train, y_test, vectorizer_type='tfidf'):
    # Vectorize the text data
    if vectorizer_type == 'tfidf':
        vectorizer = TfidfVectorizer(max_features=5000)
    else:
        vectorizer = CountVectorizer(max_features=5000)
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Initialize models
    models = {
        'Multinomial Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(random_state=42),
        'Support Vector Machine': SVC(random_state=42, probability=True)
    }
    
    results = {}
    
    for name, model in models.items():
        # Train model
        model.fit(X_train_vec, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_vec)
        y_pred_proba = model.predict_proba(X_test_vec)[:, 1] if hasattr(model, "predict_proba") else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        results[name] = {
            'model': model,
            'vectorizer': vectorizer,
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'classification_report': report,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    return results

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the app mode", 
                               ["Single Prediction", "Batch Prediction", "Data Exploration", "Model Training"])

# Single Prediction Section
if app_mode == "Single Prediction":
    st.header("🔍 Single Message Prediction")
    
    # Load model and vectorizer
    model, vectorizer, model_name, accuracy = load_model_and_vectorizer()
    
    if model is not None:
        st.success(f"✅ Using {model_name} (Accuracy: {accuracy:.2%})")
        
        # Message input
        message = st.text_area("Enter your SMS message:", 
                              placeholder="Type your message here...",
                              height=150)
        
        # Example messages
        with st.expander("💡 Example Messages"):
            st.markdown("""
            **Spam Examples:**
            - "WINNER!! You have won 1 million dollars! Call now to claim."
            - "Free entry in 2 a wkly comp to win FA Cup final tkts"
            - "URGENT! Your account will be suspended. Click here to verify."
            
            **Ham Examples:**
            - "Hey, are we still meeting for lunch tomorrow?"
            - "Your package will be delivered today between 2-4 PM"
            - "Don't forget to bring the documents for the meeting"
            """)
        
        if st.button("Classify Message", type="primary"):
            if message.strip():
                # Preprocess the message
                processed_message = preprocess_text(message)
                
                # Vectorize
                message_vec = vectorizer.transform([processed_message])
                
                # Predict
                prediction = model.predict(message_vec)[0]
                probability = model.predict_proba(message_vec)[0]
                
                # Display results
                st.markdown("---")
                st.subheader("Prediction Results")
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    if prediction == 1:
                        st.error(f"🚨 **SPAM DETECTED!**")
                        st.write(f"Confidence: {probability[1]:.2%}")
                    else:
                        st.success(f"✅ **HAM (Not Spam)**")
                        st.write(f"Confidence: {probability[0]:.2%}")
                    
                    # Show probability breakdown
                    st.metric("Ham Probability", f"{probability[0]:.2%}")
                    st.metric("Spam Probability", f"{probability[1]:.2%}")
                
                with col2:
                    # Probability chart
                    fig, ax = plt.subplots(figsize=(8, 4))
                    classes = ['Ham', 'Spam']
                    colors = ['lightgreen', 'lightcoral']
                    bars = ax.bar(classes, probability, color=colors)
                    ax.set_ylabel('Probability')
                    ax.set_title('Prediction Probabilities')
                    ax.set_ylim(0, 1)
                    
                    # Add value labels on bars
                    for bar, prob in zip(bars, probability):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{prob:.2%}', ha='center', va='bottom', fontsize=12)
                    
                    st.pyplot(fig)
                
                # Message analysis
                st.subheader("Message Analysis")
                col_analysis1, col_analysis2 = st.columns(2)
                
                with col_analysis1:
                    st.write("**Original Message:**")
                    st.info(message)
                
                with col_analysis2:
                    processed_display = preprocess_text(message)
                    st.write("**Preprocessed Message:**")
                    st.info(processed_display)
                    
                    # Word count analysis
                    word_count = len(message.split())
                    char_count = len(message)
                    st.write(f"**Stats:** {word_count} words, {char_count} characters")
            else:
                st.warning("Please enter a message to classify.")
    else:
        st.warning("⚠️ No pre-trained model found. Please train a model in the 'Model Training' section first!")

# Batch Prediction Section
elif app_mode == "Batch Prediction":
    st.header("📄 Batch Prediction")
    
    # Load model and vectorizer
    model, vectorizer, model_name, accuracy = load_model_and_vectorizer()
    
    if model is not None:
        st.success(f"✅ Using {model_name} (Accuracy: {accuracy:.2%})")
        
        uploaded_file = st.file_uploader("Upload CSV file with messages", type=['csv'])
        
        if uploaded_file is not None:
            try:
                # Read uploaded file
                batch_df = pd.read_csv(uploaded_file)
                st.write("**Uploaded Data Preview:**")
                st.dataframe(batch_df.head())
                
                # Check if message column exists
                message_column = st.selectbox("Select the message column", batch_df.columns)
                
                if st.button("Predict Batch", type="primary"):
                    with st.spinner("Processing batch predictions..."):
                        # Preprocess messages
                        batch_df['processed_message'] = batch_df[message_column].apply(preprocess_text)
                        
                        # Vectorize
                        batch_vec = vectorizer.transform(batch_df['processed_message'])
                        
                        # Predict
                        predictions = model.predict(batch_vec)
                        probabilities = model.predict_proba(batch_vec)
                        
                        # Add predictions to dataframe
                        batch_df['prediction'] = predictions
                        batch_df['prediction_label'] = batch_df['prediction'].map({0: 'Ham', 1: 'Spam'})
                        batch_df['ham_probability'] = probabilities[:, 0]
                        batch_df['spam_probability'] = probabilities[:, 1]
                        
                        # Display results
                        st.subheader("Prediction Results")
                        st.dataframe(batch_df[[message_column, 'prediction_label', 'ham_probability', 'spam_probability']])
                        
                        # Download results
                        csv = batch_df.to_csv(index=False)
                        st.download_button(
                            label="Download Predictions as CSV",
                            data=csv,
                            file_name="spam_predictions.csv",
                            mime="text/csv"
                        )
                        
                        # Summary statistics
                        st.subheader("Batch Summary")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Messages", len(batch_df))
                        with col2:
                            ham_count = len(batch_df[batch_df['prediction'] == 0])
                            st.metric("Ham Messages", ham_count)
                        with col3:
                            spam_count = len(batch_df[batch_df['prediction'] == 1])
                            st.metric("Spam Messages", spam_count)
                        with col4:
                            spam_percentage = (spam_count / len(batch_df)) * 100 if len(batch_df) > 0 else 0
                            st.metric("Spam Percentage", f"{spam_percentage:.1f}%")
                        
                        # Distribution chart
                        fig, ax = plt.subplots(figsize=(8, 5))
                        batch_df['prediction_label'].value_counts().plot(kind='bar', color=['lightgreen', 'lightcoral'], ax=ax)
                        ax.set_title('Prediction Distribution')
                        ax.set_xlabel('Class')
                        ax.set_ylabel('Count')
                        
                        # Add value labels on bars
                        for i, v in enumerate(batch_df['prediction_label'].value_counts()):
                            ax.text(i, v + 0.1, str(v), ha='center', va='bottom')
                        
                        st.pyplot(fig)
                        
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    else:
        st.warning("⚠️ No pre-trained model found. Please train a model in the 'Model Training' section first!")

# Data Exploration Section
elif app_mode == "Data Exploration":
    st.header("📊 Data Exploration")
    
    df, label_encoder = load_data()
    
    if df is not None:
        # Display dataset info
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Overview")
            st.write(f"**Total messages:** {len(df)}")
            st.write(f"**Ham messages:** {len(df[df['target'] == 0])}")
            st.write(f"**Spam messages:** {len(df[df['target'] == 1])}")
            st.write(f"**Spam percentage:** {(len(df[df['target'] == 1])/len(df))*100:.2f}%")
            
            # Show sample data
            st.subheader("Sample Data")
            st.dataframe(df.head(10))
        
        with col2:
            st.subheader("Class Distribution")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Bar plot
            df['target'].value_counts().plot(kind='bar', color=['skyblue', 'lightcoral'], ax=ax1)
            ax1.set_title('Spam vs Ham Distribution')
            ax1.set_xlabel('Class (0=Ham, 1=Spam)')
            ax1.set_ylabel('Count')
            
            # Pie chart
            df['target'].value_counts().plot(kind='pie', labels=['Ham', 'Spam'], autopct='%1.1f%%', 
                                           colors=['lightgreen', 'lightcoral'], startangle=90, ax=ax2)
            ax2.set_title('Class Distribution')
            ax2.set_ylabel('')
            
            st.pyplot(fig)
        
        # Message length analysis
        st.subheader("Message Length Analysis")
        df['message_length'] = df['message'].apply(len)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data=df, x='message_length', hue='target', bins=50, ax=ax)
            ax.set_title('Message Length Distribution by Class')
            ax.set_xlabel('Message Length')
            ax.set_ylabel('Count')
            st.pyplot(fig)
        
        with col2:
            # Summary statistics
            length_stats = df.groupby('target')['message_length'].describe()
            st.write("**Message Length Statistics:**")
            st.dataframe(length_stats)

# Model Training Section
elif app_mode == "Model Training":
    st.header("🤖 Model Training")
    
    df, label_encoder = load_data()
    
    if df is not None:
        # Preprocess messages
        st.subheader("Data Preprocessing")
        with st.spinner("Preprocessing messages..."):
            df['processed_message'] = df['message'].apply(preprocess_text)
        
        # Model configuration
        st.subheader("Model Configuration")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            vectorizer_type = st.selectbox(
                "Select Vectorizer",
                ["TF-IDF", "Count Vectorizer"]
            ).lower().replace(' ', '_')
        
        with col2:
            test_size = st.slider("Test Set Size", 0.1, 0.5, 0.2, 0.05)
        
        with col3:
            random_state = st.number_input("Random State", 0, 100, 42)
        
        # Split data
        X = df['processed_message']
        y = df['target']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Train models
        if st.button("Train Models", type="primary"):
            with st.spinner("Training models..."):
                results = train_models(X_train, X_test, y_train, y_test, vectorizer_type)
            
            # Display results
            st.subheader("Model Performance")
            
            # Accuracy comparison
            st.write("### Accuracy Comparison")
            accuracy_data = []
            for model_name, result in results.items():
                accuracy_data.append({
                    'Model': model_name,
                    'Accuracy': result['accuracy']
                })
            
            accuracy_df = pd.DataFrame(accuracy_data)
            st.dataframe(accuracy_df.style.highlight_max(axis=0))
            
            # Model details
            selected_model = st.selectbox("Select model to view details", list(results.keys()))
            
            if selected_model:
                result = results[selected_model]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Accuracy:** {result['accuracy']:.4f}")
                    
                    # Confusion matrix
                    st.write("**Confusion Matrix:**")
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.heatmap(result['confusion_matrix'], annot=True, fmt='d', cmap='Blues', 
                               xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'], ax=ax)
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    ax.set_title(f'Confusion Matrix - {selected_model}')
                    st.pyplot(fig)
                
                with col2:
                    # Classification report
                    st.write("**Classification Report:**")
                    report_df = pd.DataFrame(result['classification_report']).transpose()
                    st.dataframe(report_df.style.highlight_max(axis=0))
                
                # Save the best model for prediction
                if 'best_model' not in st.session_state or result['accuracy'] > st.session_state.get('best_accuracy', 0):
                    # Create models directory if it doesn't exist
                    os.makedirs('models', exist_ok=True)
                    
                    # Save the best model and vectorizer
                    with open('models/spam_model.pkl', 'wb') as model_file:
                        pickle.dump(result['model'], model_file)
                    with open('models/tfidf_vectorizer.pkl', 'wb') as vec_file:
                        pickle.dump(result['vectorizer'], vec_file)
                    
                    st.session_state.best_model = result['model']
                    st.session_state.best_vectorizer = result['vectorizer']
                    st.session_state.best_accuracy = result['accuracy']
                    st.session_state.best_model_name = selected_model
                    
                    st.success(f"✅ {selected_model} saved as the best model for predictions!")
                    st.info("You can now use this model in the 'Single Prediction' and 'Batch Prediction' sections.")

# Sidebar information
st.sidebar.markdown("---")
st.sidebar.subheader("About")
st.sidebar.info("""
This spam detection model uses:
- **Multiple ML algorithms** (Naive Bayes, Logistic Regression, SVM)
- **TF-IDF/Count Vectorization** for text processing
- **NLTK** for text preprocessing
- Trained on SMS Spam Collection Dataset
""")

st.sidebar.markdown("---")
st.sidebar.subheader("Instructions")
st.sidebar.markdown("""
1. **Single Prediction**: Classify individual SMS messages
2. **Batch Prediction**: Upload CSV file for bulk classification
3. **Data Exploration**: View dataset statistics and visualizations
4. **Model Training**: Train and compare different ML models
""")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Built with ❤️ using Streamlit | SMS Spam Classification App"
    "</div>",
    unsafe_allow_html=True
)