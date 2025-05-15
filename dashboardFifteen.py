import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time
import re
import string

import nltk
import emoji
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, TweetTokenizer

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading required NLTK resources...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
# Download necessary NLTK packages
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Function definitions from your model
def clean_text(text):
    """Clean and preprocess tweet text"""
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Remove mentions and hashtags
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove special characters and punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Rejoin into a single string
    clean_text = ' '.join(tokens)
    
    return clean_text

def extract_rich_text_features(text):
    """Extract rich text features including punctuation, capitalization, and emojis"""
    if pd.isna(text):
        return {
            'exclamation_count': 0,
            'question_count': 0,
            'uppercase_ratio': 0,
            'uppercase_word_count': 0,
            'emoji_count': 0,
            'has_disaster_emoji': 0,
            'punctuation_ratio': 0,
            'emphasis_punctuation': 0,
            'avg_word_length': 0,
            'hashtag_count': 0,
            'mention_count': 0,
            'disaster_emoji_list': '',
            'multiple_punctuation': 0,
            'all_caps_words': 0
        }
    
    text = str(text)
    original_length = len(text)
    
    features = {}
    
    # Punctuation features
    features['exclamation_count'] = text.count('!')
    features['question_count'] = text.count('?')
    
    # Total punctuation count
    punctuation_count = sum(1 for char in text if char in string.punctuation)
    features['punctuation_ratio'] = punctuation_count / original_length if original_length > 0 else 0
    
    # Emphasis patterns (!!!, ???, etc.)
    emphasis_pattern = re.compile(r'(!{2,}|\?{2,}|\.{3,})')
    emphasis_matches = emphasis_pattern.findall(text)
    features['emphasis_punctuation'] = len(emphasis_matches)
    
    # Multiple punctuation (e.g., '?!', '!!?')
    multiple_punct_pattern = re.compile(r'[!?.]{2,}')
    features['multiple_punctuation'] = len(multiple_punct_pattern.findall(text))
    
    # Tokenize with TweetTokenizer to better handle social media text
    tokenizer = TweetTokenizer(preserve_case=True)
    words = tokenizer.tokenize(text)
    
    # Capitalization features
    uppercase_words = [w for w in words if w.isupper() and len(w) > 1]
    features['uppercase_word_count'] = len(uppercase_words)
    
    word_count = len([w for w in words if w.isalpha()])
    features['uppercase_ratio'] = len(uppercase_words) / word_count if word_count > 0 else 0
    
    # All caps words feature
    features['all_caps_words'] = len([w for w in words if w.isupper() and len(w) >= 3])
    
    # Word features
    if word_count > 0:
        features['avg_word_length'] = sum(len(w) for w in words if w.isalpha()) / word_count
    else:
        features['avg_word_length'] = 0
    
    # Twitter-specific features
    features['hashtag_count'] = len(re.findall(r'#\w+', text))
    features['mention_count'] = len(re.findall(r'@\w+', text))
    
    # Emoji features
    emoji_list = [c for c in text if c in emoji.EMOJI_DATA]
    features['emoji_count'] = len(emoji_list)
    
    # Disaster-related emojis (expanded list)
    disaster_emojis = {
        '🔥': 'fire', '💧': 'water', '🌊': 'wave', '💥': 'explosion',
        '🌪️': 'tornado', '🌫️': 'fog', '⚡': 'lightning', '🌩️': 'thunderstorm',
        '🌧️': 'rain', '🌨️': 'snow', '🏚️': 'damaged_house', '🚨': 'alarm',
        '🚑': 'ambulance', '🚒': 'fire_truck', '⚠️': 'warning', '☣️': 'biohazard',
        '☢️': 'radioactive', '🆘': 'sos', '📢': 'loudspeaker', '🚁': 'helicopter',
        '🚓': 'police_car', '🌋': 'volcano', '💨': 'wind', '🌀': 'cyclone'
    }
    
    disaster_emojis_found = [emoji for emoji in emoji_list if emoji in disaster_emojis]
    features['has_disaster_emoji'] = 1 if disaster_emojis_found else 0
    features['disaster_emoji_list'] = ','.join([disaster_emojis.get(e, '') for e in disaster_emojis_found])
    
    return features

def enrich_dataframe_with_text_features(df, text_column='text'):
    """Extract rich text features and add them to the DataFrame"""
    # Apply the feature extraction function to each tweet
    rich_features = df[text_column].apply(extract_rich_text_features)
    
    # Convert list of dictionaries to DataFrame
    rich_features_df = pd.DataFrame(rich_features.tolist())
    
    # Combine with original DataFrame
    enriched_df = pd.concat([df.reset_index(drop=True), rich_features_df.reset_index(drop=True)], axis=1)
    
    return enriched_df

def predict_single_tweet(model_data, tweet_text, keyword="", location=""):
    """Predict class for a single tweet"""
    
    model = model_data['model']
    tfidf = model_data['tfidf']
    encoder = model_data['encoder']
    sentiment_model = model_data['sentiment_model']
    sentiment_vectorizer = model_data['sentiment_vectorizer']
    sentiment_mapping = model_data['sentiment_mapping']
    sentiment_feature_names = model_data['sentiment_feature_names']
    feature_names = model_data['feature_names']
    use_rich_features = model_data.get('use_rich_features', True)
    
    # Create a DataFrame with the tweet
    tweet_df = pd.DataFrame({
        'text': [tweet_text],
        'keyword': [keyword],
        'location': [location]
    })
    
    # Preprocess the tweet
    tweet_df['clean_text'] = tweet_df['text'].apply(clean_text)
    
    # Extract rich text features if enabled
    if use_rich_features:
        tweet_df = enrich_dataframe_with_text_features(tweet_df)
    
    # Extract sentiment features
    X_sentiment = sentiment_vectorizer.transform(tweet_df['clean_text'])
    X_sentiment_df = pd.DataFrame(X_sentiment.toarray(), columns=sentiment_feature_names)
    sentiment_pred = sentiment_model.predict(X_sentiment_df)
    sentiment_proba = sentiment_model.predict_proba(X_sentiment_df)
    
    # Create a DataFrame with sentiment probabilities
    sentiment_columns = [f'sentiment_{key}' for key in sentiment_mapping.keys()]
    sentiment_df = pd.DataFrame(sentiment_proba, columns=sentiment_columns)
    
    # Add predicted sentiment class
    tweet_df['predicted_sentiment'] = sentiment_pred
    
    # Combine with sentiment probabilities
    tweet_df = pd.concat([tweet_df.reset_index(drop=True), sentiment_df.reset_index(drop=True)], axis=1)
    
    # Extract TF-IDF features
    tweet_tfidf = tfidf.transform(tweet_df['clean_text'])
    tfidf_feature_names = []
    for name in tfidf.get_feature_names_out():
        clean_name = re.sub(r'[\[\]<>]', '_', name)
        tfidf_feature_names.append(f'tfidf_{clean_name}')
    
    tweet_tfidf_df = pd.DataFrame(tweet_tfidf.toarray(), columns=tfidf_feature_names)
    
    # Extract categorical features
    categorical_features = ['keyword', 'location']
    tweet_cat = encoder.transform(tweet_df[categorical_features])
    
    cat_feature_names = []
    for i, category in enumerate(categorical_features):
        for value in encoder.categories_[i]:
            clean_value = re.sub(r'[\[\]<>]', '_', str(value))
            cat_feature_names.append(f'{category}_{clean_value}')
    
    tweet_cat_df = pd.DataFrame(tweet_cat, columns=cat_feature_names)
    
    # Combine features
    tweet_features = pd.concat([tweet_tfidf_df, tweet_cat_df], axis=1)
    
    # Add rich text features if enabled
    if use_rich_features:
        rich_feature_cols = [
            'exclamation_count', 'question_count', 'uppercase_ratio', 
            'uppercase_word_count', 'emoji_count', 'has_disaster_emoji',
            'punctuation_ratio', 'emphasis_punctuation', 'avg_word_length',
            'hashtag_count', 'mention_count', 'multiple_punctuation', 'all_caps_words'
        ]
        if all(col in tweet_df.columns for col in rich_feature_cols):
            rich_features_df = tweet_df[rich_feature_cols]
            tweet_features = pd.concat([tweet_features, rich_features_df], axis=1)
    
    # Ensure all features are present
    for col in feature_names:
        if col not in tweet_features.columns:
            tweet_features[col] = 0
    
    # Reorder to match training features
    tweet_features = tweet_features[feature_names]
    
    # Make prediction
    prediction = model.predict(tweet_features)[0]
    prediction_proba = model.predict_proba(tweet_features)[0]
    
    class_labels = {
        0: "Disaster word used but no disaster",
        1: "Disaster word used and disaster occurred",
        2: "Not disaster related"
    }
    
    binary_pred = 1 if prediction in [0, 1] else 0
    binary_label = "Disaster Related" if binary_pred == 1 else "Not Disaster Related"
    
    sentiment_label = list(sentiment_mapping.keys())[list(sentiment_mapping.values()).index(tweet_df['predicted_sentiment'].iloc[0])]
    
    result = {
        'prediction_class': int(prediction),
        'prediction_label': class_labels[prediction],
        'binary_class': binary_pred,
        'binary_label': binary_label,
        'probabilities': {
            'class_0': float(prediction_proba[0]),
            'class_1': float(prediction_proba[1]),
            'class_2': float(prediction_proba[2])
        },
        'sentiment': sentiment_label,
        'rich_features': {k: tweet_df[k].iloc[0] for k in rich_feature_cols} if use_rich_features else {}
    }
    
    return result

# Load sample tweets for demo
def load_sample_tweets():
    return [
        "BREAKING: Massive earthquake hits Japan, tsunami warning issued. Thousands evacuating coastal areas. #emergency",
        "The hurricane season this year is going to be worse than last year according to experts. Stay prepared!",
        "My heart is on fire and crashing like a wave against the shore after watching that movie! #emotional",
        "Just saw a fire truck race by my office window. Hope everyone is safe.",
        "This new game is literally blowing up on social media! Everyone is playing it. #gaming #explosion",
        "URGENT: Dam failure reported in Colorado. Flash flooding expected. Please evacuate immediately if in affected areas.",
        "My new mixtape is straight fire! It's going to cause an earthquake in the music industry! 🔥",
        "The stock market crashed today, wiping out billions in value. #financialdisaster",
        "Firefighters battling wildfire near California town, residents evacuated. Wind making conditions worse."
    ]

# Function to create visualizations for predictions
def create_prediction_visualizations(prediction_results):
    # Class probabilities
    probabilities = prediction_results['probabilities']
    class_names = ["Disaster word, no disaster", "Disaster word and disaster", "Not disaster related"]
    
    # Create probability bar chart
    fig_prob = go.Figure(data=[
        go.Bar(
            x=class_names,
            y=[probabilities['class_0'], probabilities['class_1'], probabilities['class_2']],
            marker_color=['#FF9999', '#FF3333', '#99CC99']
        )
    ])
    fig_prob.update_layout(
        title="Prediction Probabilities",
        xaxis_title="Class",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1]),
        height=400
    )
    
    # Create gauge chart for binary classification
    binary_prob = probabilities['class_0'] + probabilities['class_1']  # Sum of disaster-related classes
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=binary_prob,
        title={'text': "Disaster Related Probability"},
        gauge={
            'axis': {'range': [0, 1]},
            'bar': {'color': "#FF3333" if binary_prob > 0.5 else "#99CC99"},
            'steps': [
                {'range': [0, 0.5], 'color': "#99CC99"},
                {'range': [0.5, 1], 'color': "#FF9999"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.5
            }
        }
    ))
    fig_gauge.update_layout(height=300)
    
    return fig_prob, fig_gauge

# Function to create rich features visualization
def create_rich_features_viz(rich_features):
    # Only include numeric features
    numeric_features = {k: v for k, v in rich_features.items() 
                        if isinstance(v, (int, float)) and k != 'disaster_emoji_list'}
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(numeric_features.keys()),
            y=list(numeric_features.values()),
            marker_color='#5588BB'
        )
    ])
    fig.update_layout(
        title="Rich Text Features",
        xaxis_title="Feature",
        yaxis_title="Value",
        height=400,
        xaxis={'categoryorder': 'total descending'}
    )
    
    return fig

# Define the Streamlit app
def main():
    st.set_page_config(
        page_title="Disaster Tweet Classifier",
        page_icon="🚨",
        layout="wide"
    )
    
    # Load model data
    model_path = st.sidebar.file_uploader("Upload your model file", type=["pkl"])
    
    if model_path is None:
        st.sidebar.warning("Please upload your trained model file (.pkl)")
        st.title("Disaster Tweet Classification Dashboard")
        st.write("Upload your model file to begin.")
        return
    
    try:
        model_data = pickle.load(model_path)
        st.sidebar.success("Model loaded successfully!")
        model_name = model_data.get('model_type', 'Unknown')
        st.sidebar.write(f"Model type: {model_name}")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {str(e)}")
        return
    
    # Main layout
    st.title("Disaster Tweet Classification Dashboard")
    st.write("This dashboard demonstrates a multiclass classification model for disaster tweets.")
    
    # Create tabs
    tab1, tab2 = st.tabs(["Predict Single Tweet", "Model Information"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Enter a Tweet")
            
            # Radio option for input method
            input_method = st.radio(
                "Select input method:",
                ["Enter manually", "Choose from samples"]
            )
            
            if input_method == "Enter manually":
                tweet_text = st.text_area("Tweet text", height=100, 
                                        placeholder="Enter a tweet to classify...")
                keyword = st.text_input("Keyword (optional)")
                location = st.text_input("Location (optional)")
            else:
                sample_tweets = load_sample_tweets()
                selected_index = st.selectbox("Sample tweets:", 
                                            range(len(sample_tweets)),
                                            format_func=lambda i: sample_tweets[i])
                tweet_text = sample_tweets[selected_index]
                keyword = ""
                location = ""
                
                st.text_area("Selected tweet", tweet_text, height=100, disabled=True)
            
            if st.button("Predict", type="primary"):
                if tweet_text.strip():
                    with st.spinner("Analyzing tweet..."):
                        # Add slight delay for effect
                        time.sleep(1)
                        result = predict_single_tweet(model_data, tweet_text, keyword, location)
                        
                        # Show prediction
                        st.success("Prediction Complete!")
                        
                        # Display results
                        st.subheader("Prediction Results")
                        
                        # Create columns for multiclass and binary results
                        res_col1, res_col2 = st.columns(2)
                        
                        with res_col1:
                            st.write("**Multiclass Prediction:**")
                            st.markdown(f"""
                            <div style="background-color:#f0f2f6;padding:15px;border-radius:10px;">
                                <h3 style="margin:0;color:{'#ff6347' if result['prediction_class'] == 1 else '#4682b4'};">
                                    {result['prediction_label']}
                                </h3>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        with res_col2:
                            st.write("**Binary Prediction:**")
                            st.markdown(f"""
                            <div style="background-color:{'#ffcccc' if result['binary_class'] == 1 else '#ccffcc'};
                                padding:15px;border-radius:10px;">
                                <h3 style="margin:0;color:{'#cc0000' if result['binary_class'] == 1 else '#006600'};">
                                    {result['binary_label']}
                                </h3>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Show sentiment
                        st.write("**Detected Sentiment:**", result['sentiment'])
                        
                        # Visualizations
                        fig_prob, fig_gauge = create_prediction_visualizations(result)
                        st.plotly_chart(fig_prob, use_container_width=True)
                        st.plotly_chart(fig_gauge, use_container_width=True)
                        
                        # Rich features
                        if result['rich_features']:
                            st.subheader("Rich Text Features")
                            rich_fig = create_rich_features_viz(result['rich_features'])
                            st.plotly_chart(rich_fig, use_container_width=True)
                else:
                    st.error("Please enter or select a tweet first.")
        
        with col2:
            st.subheader("How It Works")
            st.write("""
            This model classifies tweets into three categories:
            
            1. **Disaster word used but no disaster** - Uses disaster terminology but doesn't describe an actual disaster
            
            2. **Disaster word used and disaster occurred** - Describes a real disaster event
            
            3. **Not disaster related** - Completely unrelated to disasters
            
            The model uses:
            - TF-IDF vectorization
            - Rich text features (emojis, capitalization, etc.)
            - Sentiment analysis
            - Categorical features (keywords, locations)
            """)
            
            st.info("""
            **Tip**: Try tweets with:
            - Disaster terminology used figuratively
            - Real disaster reports
            - Ambiguous disaster references
            """)
    
    with tab2:
        st.subheader("Model Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Model Details**")
            st.write(f"Model Type: {model_name}")
            st.write(f"Features Used: {len(model_data['feature_names'])}")
            st.write(f"Rich Features Enabled: {model_data.get('use_rich_features', True)}")
            
            # Feature importance if available
            if hasattr(model_data['model'], 'feature_importances_'):
                st.subheader("Feature Importance")
                
                # Get top 15 features
                feature_imp = pd.DataFrame({
                    'feature': model_data['feature_names'],
                    'importance': model_data['model'].feature_importances_
                }).sort_values('importance', ascending=False).head(15)
                
                fig = px.bar(feature_imp, x='importance', y='feature', orientation='h',
                            title="Top 15 Important Features")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**Classification Overview**")
            st.markdown("""
            This model distinguishes between:
            
            - **False Positives**: Tweets that use disaster terminology but don't refer to actual disasters
            - **True Positives**: Tweets about real disaster events
            - **True Negatives**: Tweets unrelated to disasters
            
            The multiclass approach allows emergency responders to focus on Class 1 tweets (real disasters) while reducing noise from Class 0 tweets (figurative language using disaster terms).
            """)
            
            # Class distribution chart
            st.subheader("Class Distribution")
            fig = go.Figure(data=[
                go.Pie(
                    labels=["False alarm tweets", "Real disaster tweets", "Non-disaster tweets"],
                    values=[25, 40, 35],  # Example values - would need actual data 
                    hole=.3,
                    marker_colors=['#FF9999', '#FF3333', '#99CC99']
                )
            ])
            fig.update_layout(title_text="Example Class Distribution")
            st.plotly_chart(fig, use_container_width=True)

    # Add footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center">
        <p>Disaster Tweet Classification Dashboard | Created with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()