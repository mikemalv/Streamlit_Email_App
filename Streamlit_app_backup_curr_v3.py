##########################################
#             Imports
##########################################
import pandas as pd
import streamlit as st
import json
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components
from snowflake.snowpark.context import get_active_session
from langdetect import detect
from datetime import datetime

##########################################
#             Page | Session
##########################################
st.set_page_config(page_title="Callaway Support Dashboard", layout="wide")

##########################################
#             CSS Color Picker: https://www.color-hex.com/color/000000
##########################################
def load_css():
    st.markdown("""
        <style>
            /* Reset and base styles */
            section[data-testid="stSidebar"] {
                background-color: black !important;
                color: white !important;
            }
            
            /* Main content area */
            .main .block-container {
                background-color: black !important;
                color: white !important;
                padding: 2rem;
            }
            
            /* Override default white background */
            .stApp {
                background-color: black !important;
            }
            
            /* Header and Title Styling */
            h1, h2, h3 {
                color: white !important;
                font-family: 'Verdana', serif !important;
            }
            
            /* Button and Selectbox Styling */
            .stButton > button {
                background-color: #5c9e57 !important;
                color: white !important;
                border-radius: 8px !important;
                padding: 0.5rem 1rem !important;
                border: 2px solid #a8d5a2 !important;
            }
            
            .stButton > button:hover {
                background-color: #3b5f3a !important;
                border-color: white !important;
            }
            
            /* Selectbox styling */
            .stSelectbox > div > div {
                background-color: #5c9e57 !important;
                color: white !important;
            }
            
            /* DataFrames and tables */
            .dataframe {
                background-color: #1e1e1e !important;
                color: white !important;
            }
            
            /* Cards and containers */
            .element-container {
                background-color: black !important;
            }
            
            /* Alert/Info boxes */
            .stAlert {
                background-color: #1e1e1e !important;
                color: white !important;
                border: 1px solid #5c9e57 !important;
            }
            
            /* Text elements */
            .stMarkdown {
                color: white !important;
            }
            
            /* Code blocks */
            .stCodeBlock {
                background-color: #1e1e1e !important;
            }
            
            /* Expander */
            .streamlit-expanderHeader {
                background-color: #1e1e1e !important;
                color: white !important;
            }
            
            /* Success/Error messages */
            .success {
                background-color: #1e8320 !important;
                color: white !important;
            }
            
            .error {
                background-color: #8b0000 !important;
                color: white !important;
            }
            
            /* Widgets */
            .stCheckbox, .stRadio {
                color: white !important;
            }
            
            /* Text input */
            .stTextInput > div > div {
                background-color: #1e1e1e !important;
                color: white !important;
            }
            
            /* Spinner */
            .stSpinner > div {
                border-color: white !important;
            }
            
            /* Progress bar background */
            .stProgress > div > div {
                background-color: #1e1e1e !important;
            }
            
            /* Progress bar fill */
            .stProgress > div > div > div {
                background-color: #5c9e57 !important;
            }
            
            /* Email styling */
            .email-container {
                background-color: #1e1e1e;
                border-radius: 10px;
                padding: 20px;
                margin: 20px 0;
                border: 1px solid #5c9e57;
                font-family: Arial, sans-serif;
            }
            .email-header {
                border-bottom: 1px solid #5c9e57;
                padding-bottom: 15px;
                margin-bottom: 15px;
            }
            .email-field {
                margin: 5px 0;
                color: #cccccc;
            }
            .email-label {
                color: #888888;
                display: inline-block;
                width: 100px;
            }
            .email-value {
                color: #ffffff;
            }
            .email-body {
                padding: 15px 0;
                line-height: 1.6;
                white-space: pre-wrap;
                color: white;
            }
            .email-metadata {
                color: #888888;
                font-size: 0.9em;
                margin-top: 15px;
                padding-top: 15px;
                border-top: 1px solid #5c9e57;
            }

            /* Priority classes */
            .priority-high {
                color: #ff6b6b;
                font-weight: bold;
            }
            .priority-medium {
                color: #ffd93d;
            }
            .priority-low {
                color: #95d5b2;
            }

            /* Status badges */
            .status-badge {
                display: inline-block;
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 0.9em;
                margin-left: 10px;
            }
            .status-open {
                background-color: #4ecdc4;
                color: black;
            }
            .status-in-progress {
                background-color: #ffd93d;
                color: black;
            }
            .status-closed {
                background-color: #95d5b2;
                color: black;
            }

            /* Tag styling */
            .tag-container {
                margin-top: 10px;
            }
            .tag {
                display: inline-block;
                background-color: #2d2d2d;
                padding: 2px 8px;
                border-radius: 12px;
                margin: 2px;
                font-size: 0.8em;
            }

            /* Reply card styling */
            .reply-card {
                background-color: #1e1e1e;
                border-radius: 10px;
                padding: 20px;
                margin: 20px 0;
                border: 1px solid #5c9e57;
            }
            .reply-content {
                margin: 15px 0;
                white-space: pre-wrap;
                color: white;
            }

            /* Selectbox styling - Modified for dark green */
            .stSelectbox > div > div {
                background-color: #1e4620 !important;  /* Dark green color */
                color: white !important;
                border: 1px solid #2c662d !important;  /* Slightly lighter green border */
            }
            
            /* Selectbox hover state */
            .stSelectbox > div > div:hover {
                border-color: #3d8b40 !important;  /* Lighter green on hover */
            }
            
            /* Selectbox options menu */
            .stSelectbox > div > div[data-baseweb="select"] > div {
                background-color: #1e4620 !important;
                border-color: #2c662d !important;
            }
            
            /* Selected option */
            .stSelectbox [data-baseweb="select"] [aria-selected="true"] {
                background-color: #2c662d !important;
            }   
        </style>
    """, unsafe_allow_html=True)

##########################################
#             Database Functions
##########################################
def get_snowflake_session():
    """Initialize and return Snowflake session with error handling"""
    try:
        return get_active_session()
    except Exception as e:
        st.error("Error connecting to Snowflake. Please check your connection settings.")
        st.stop()

def safe_sql_string(text):
    """Safely escape SQL string parameters"""
    if text is None:
        return None
    return text.replace("'", "''")

def execute_sql_safely(session, query, params=None):
    """Execute SQL with error handling"""
    try:
        if params:
            query = query.format(*[safe_sql_string(p) for p in params])
            
        # Execute query
        result = session.sql(query)
        
        # If it's a SELECT query, convert to pandas DataFrame
        if query.strip().upper().startswith('SELECT'):
            return pd.DataFrame(result.collect())
        else:
            # For non-SELECT queries (UPDATE, INSERT, DELETE)
            result.collect()
            return None
            
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        return None

##########################################
#             Email Functions
##########################################
def detect_language(text):
    """Detect language of text with error handling"""
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        language = detect(text)
        return language if language != 'en' else None
    except Exception:
        return None

def format_email_html(selected_email):
    """Generate HTML for email display"""
    email_body_formatted = selected_email['EMAIL_BODY'].replace('\n', '<br>')
    priority_class = f"priority-{selected_email.get('PRIORITY', 'medium').lower()}"
    status_class = f"status-{selected_email.get('EMAIL_STATUS', 'open').lower().replace(' ', '-')}"
    
    # Build email HTML parts
    email_html = [
        '<div class="email-container">',
        '<div class="email-header">',
        
        # From field
        '<div class="email-field">',
        '<span class="email-label">From:</span>',
        f'<span class="email-value">{selected_email.get("EMAIL_FROM", "Customer")}</span>',
        '</div>',
        
        # To field
        '<div class="email-field">',
        '<span class="email-label">To:</span>',
        f'<span class="email-value">{selected_email.get("EMAIL_TO", "Support")}</span>',
        '</div>'
    ]
    
    # Optional CC field
    if pd.notna(selected_email.get('EMAIL_CC')):
        email_html.extend([
            '<div class="email-field">',
            '<span class="email-label">CC:</span>',
            f'<span class="email-value">{selected_email["EMAIL_CC"]}</span>',
            '</div>'
        ])
    
    # Subject field
    email_html.extend([
        '<div class="email-field">',
        '<span class="email-label">Subject:</span>',
        f'<span class="email-value">{selected_email.get("SUBJECT", "No Subject")}</span>',
        '</div>'
    ])
    
    # Received Time
    if pd.notna(selected_email.get('RECEIVED_TIME')):
        email_html.extend([
            '<div class="email-field">',
            '<span class="email-label">Received:</span>',
            f'<span class="email-value">{pd.to_datetime(selected_email["RECEIVED_TIME"]).strftime("%B %d, %Y %I:%M %p")}</span>',
            '</div>'
        ])
    
    # Priority and Status
    email_html.extend([
        '<div class="email-field">',
        '<span class="email-label">Priority:</span>',
        f'<span class="{priority_class}">{selected_email.get("PRIORITY", "Medium")}</span>',
        f'<span class="status-badge {status_class}">{selected_email.get("EMAIL_STATUS", "Open")}</span>',
        '</div>'
    ])
    
    # Optional Due Time
    #if pd.notna(selected_email.get('RESPONSE_DUE_BY')):
    #    email_html.extend([
    #        '<div class="email-field">',
    #        '<span class="email-label">Due By:</span>',
    #        f'<span class="email-value">{pd.to_datetime(selected_email["RESPONSE_DUE_BY"]).strftime("%B %d, %Y %I:%M %p")}</span>',
    #        '</div>'
    #    ])
    
    # Optional Assigned To
    if pd.notna(selected_email.get('ASSIGNED_TO')):
        email_html.extend([
            '<div class="email-field">',
            '<span class="email-label">Assigned To:</span>',
            f'<span class="email-value">{selected_email["ASSIGNED_TO"]}</span>',
            '</div>'
        ])
    
    # Case Number
    if pd.notna(selected_email.get('CASE_NUMBER')) and len(selected_email['CASE_NUMBER']) > 0:
        tags_html = ' '.join([f'<span class="tag">{tag}</span>' for tag in selected_email['CASE_NUMBER']])
        email_html.extend([
            '<div class="email-field">',
            '<span class="email-label">Case:</span>',
            f'<span class="email-value">{selected_email["CASE_NUMBER"]}</span>',
            '</div>'
        ])
    
    # Close header and add body
    email_html.extend([
        '</div>',
        '<div class="email-body">',
        email_body_formatted,
        '</div>',
        '<div class="email-metadata">',
        f'<div>Email ID: {selected_email["EMAIL_ID"]}</div>',
        '</div>',
        '</div>'
    ])
    
    return ''.join(email_html)

##########################################
#             Email Display|Select
##########################################
def display_email(selected_email):
    """Display formatted email in Streamlit"""
    email_html = format_email_html(selected_email)
    st.markdown(email_html, unsafe_allow_html=True)

def select_email(data_df, key_prefix, preview_length=100):
    """Common email selection interface"""
    email_options = data_df['EMAIL_BODY'].apply(
        lambda x: x[:preview_length] + "..." if len(x) > preview_length else x
    ).tolist()
    
    selected_index = st.selectbox(
        "Choose an email to analyze:",
        range(len(email_options)),
        format_func=lambda x: email_options[x],
        key=f"{key_prefix}_email_selector"
    )
    
    return data_df.iloc[selected_index]

##########################################
#             Sentiment Function
##########################################
def analyze_sentiment(session, email_body):
    """Perform sentiment analysis"""
    sentiment_query = f"""
        SELECT snowflake.cortex.sentiment(
            '{safe_sql_string(email_body)}'
        ) AS sentiment
    """
    result = execute_sql_safely(session, sentiment_query)
    if result is not None and not result.empty:
        sentiment_score = float(result['SENTIMENT'][0])
        scaled_score = round(((sentiment_score + 1) / 2) * 9 + 1)
        return {
            'raw_score': sentiment_score,
            'scaled_score': scaled_score,
            'category': "Negative" if scaled_score <= 3 else "Neutral" if scaled_score <= 6 else "Positive",
            'emoji': "🔴" if scaled_score <= 3 else "🟡" if scaled_score <= 6 else "🟢"
        }
    return None

def create_sentiment_gauge(sentiment_score):
    """
    Create a gauge visualization for sentiment score directly using the provided 1-10 score
    """
    if sentiment_score is None:
        return None
        
    # Use the sentiment score directly since it's already on 1-10 scale
    score = float(sentiment_score)
    
    # Gauge configuration for sentiment from 0 (negative) to 10 (positive)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,  # Use score directly
        title={'text': "Sentiment Score", 'font': {'color': 'white'}},
        gauge={
            'axis': {'range': [0, 10], 'tickwidth': 1, 'tickcolor': "white", 'tickfont': {'color': 'white'}},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 3], 'color': "#ff6666"},  # Light Red
                {'range': [3, 7], 'color': "#ffff99"},  # Light Yellow
                {'range': [7, 10], 'color': "#99ff99"}  # Light Green
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        },
        number={'font': {'color': 'white'}}
    ))
    
    # Update layout for better visualization with black background
    fig.update_layout(
        height=160,  
        margin=dict(l=14, r=14, t=35, b=14),  
        paper_bgcolor="black",
        plot_bgcolor="black",
        font={'size': 12, 'color': 'white'}  
    )    
    return fig

def process_email_sentiment(session, email_body):
    # First get the sentiment analysis
    sentiment_result = analyze_sentiment(session, email_body)
    
    if sentiment_result:
        # Use the scaled score for the gauge
        gauge_fig = create_sentiment_gauge(sentiment_result['scaled_score'])
        return gauge_fig, sentiment_result
    return None, None
##########################################
#             Translate Function
##########################################
def translate_text(session, email_body, source_lang, target_lang='en'):
    """Translate text"""
    translate_query = f"""
        SELECT snowflake.cortex.translate(
            '{safe_sql_string(email_body)}',
            '{source_lang}',
            '{target_lang}'
        ) AS translation
    """
    result = execute_sql_safely(session, translate_query)
    if result is not None and not result.empty:
        return result['TRANSLATION'][0]
    return None

##########################################
#             Summarize Function
##########################################
def summarize_text(session, email_body):
    """Summarize text"""
    summarize_query = f"""
        SELECT snowflake.cortex.summarize(
            '{safe_sql_string(email_body)}'
        ) AS summary
    """
    result = execute_sql_safely(session, summarize_query)
    if result is not None and not result.empty:
        return result['SUMMARY'][0]
    return None

##########################################
#        Generate Reply Function
##########################################
def generate_reply(session, email_body, model):
    """Generate email reply"""
    reply_query = f"""
        SELECT SNOWFLAKE.CORTEX.COMPLETE(
            '{model}', 
            'Generate a reply for the following email with sympathy:\n\n{safe_sql_string(email_body)}'
        ) AS suggested_reply
    """
    result = execute_sql_safely(session, reply_query)
    if result is not None and not result.empty:
        return result['SUGGESTED_REPLY'][0]
    return None


##########################################
#         Classify Text Function
##########################################
def classify_text(session, email_body):
    """Classify email text"""
    categories = [
        "Drivers",
        "Irons",
        "Wedges",
        "Putters",
        "Fairway Woods",
        "Golf Balls",
        "Bags",
        "Accessories"
    ]
    
    patterns = {
        'Drivers': '(DRIVER|DRIVERS|DRIVING|TEE OFF|OFF THE TEE|BIG BERTHA)',
        'Hybrid': '(HYBRID)',
        'Irons': '(IRON|IRONS|[0-9]+ IRON|[0-9]+ IRONS|APEX)',
        'Wedges': '(WEDGE|WEDGES|SAND WEDGE|PITCHING WEDGE)',
        'Putters': '(PUTTER|PUTTERS|PUTTING|PUTT|ODYSSEY)',
        'Fairway Woods': '(FAIRWAY WOOD|WOOD|WOODS|3 WOOD|5 WOOD)',
        'Golf Balls': '(BALL|BALLS|GOLF BALL|GOLF BALLS)',
        'Bags': '(BAG|BAGS|GOLF BAG|CARRY|CART BAG)',
        'Accessories': '(GLOVE|TEE|MARKER|TOWEL|ACCESSORY|ACCESSORIES|GPS)'
    }
    
    classification_scores = {}
    for category in categories:
        count_query = f"""
            SELECT REGEXP_COUNT(
                UPPER('{safe_sql_string(email_body)}'), 
                '{patterns[category]}'
            ) as mention_count
        """
        result = execute_sql_safely(session, count_query)
        if result is not None and not result.empty:
            classification_scores[category] = result['MENTION_COUNT'][0]
    
    total_mentions = sum(classification_scores.values())
    if total_mentions > 0:
        for category in classification_scores:
            classification_scores[category] = classification_scores[category] / total_mentions
        primary_category = max(classification_scores.items(), key=lambda x: x[1])[0]
    else:
        primary_category = "Unclassified"
        
    return {
        'primary_category': primary_category,
        'scores': classification_scores,
        'total_mentions': total_mentions
    }

##########################################
#        Extract Answer Function
##########################################
def extract_answer(session, email_body, question):
    """Extract answer from email based on question"""
    extract_query = f"""
        SELECT snowflake.cortex.extract_answer(
            '{safe_sql_string(email_body)}',
            '{safe_sql_string(question)}'
        ) AS answer
    """
    result = execute_sql_safely(session, extract_query)
    if result is not None and not result.empty:
        return result['ANSWER'][0]
    return None

##########################################
#             Feature Handlers
##########################################
def config_options():
    """Sidebar configuration options"""

    st.sidebar.title("Select Option")
    # Main option selection with Dashboard as default
    option = st.sidebar.selectbox(
        'Select Analysis Option:',
        ("Dashboard", "Translate Text", "Detect Sentiment", "Summarize Text", 
         "Classify Text", "Extract Answer", "Suggest Email Reply", "Process All Data"
    ), key="option"
    )
    
    st.sidebar.title("Model Option")
    selected_model = st.sidebar.selectbox(
        "Select your model:",
        (
            'snowflake-arctic', 'mistral-large', 'mistral-large2', 'reka-flash', 
            'reka-core', 'jamba-instruct', 'jamba-1.5-mini', 'jamba-1.5-large', 
            'mixtral-8x7b', 'llama2-70b-chat', 'llama3-8b', 'llama3-70b', 
            'llama3.1-8b', 'llama3.1-70b', 'llama3.1-405b', 'llama3.2-1b', 
            'llama3.2-3b', 'mistral-7b', 'gemma-7b'
        ),
        key="model_name"
    )

    use_chat_history = st.sidebar.checkbox('Remember chat history?', value=True)
    debug = st.sidebar.checkbox('Debug: Show summary of previous conversation', value=False)
    
    if st.sidebar.button("Start Over"):
        try:
            for key in list(st.session_state.keys()):
                del st.session_state[key]
        except Exception as e:
            st.sidebar.error(f"Error clearing session state: {str(e)}")

    if debug:
        with st.sidebar.expander("Session State"):
            st.write(st.session_state)
            
    #return selected_model, use_chat_history, debug
    return selected_model, use_chat_history, debug, option

##########################################
#               Dashboard
##########################################
def handle_dashboard(session):
    """Handle Dashboard View"""
    st.markdown("## 📊 Support Email Analytics Dashboard")
    
    try:
        # Get comprehensive statistics with better error handling
        stats_query = """
            WITH EmailStats AS (
                SELECT 
                    COUNT(*) as total_emails,
                    COUNT(CASE WHEN CORTEX_DETECTED_LANGUAGE != 'en' 
                              AND CORTEX_DETECTED_LANGUAGE IS NOT NULL 
                              AND CORTEX_TRANSLATE IS NOT NULL 
                              AND CORTEX_TRANSLATE != '' THEN 1 END) as translated_count,
                    COUNT(CASE WHEN CORTEX_SENTIMENT IS NOT NULL 
                              AND CORTEX_SENTIMENT != '' THEN 1 END) as sentiment_count,
                    COUNT(CASE WHEN CORTEX_SUMMARIZE IS NOT NULL 
                              AND CORTEX_SUMMARIZE != '' 
                              AND LENGTH(EMAIL_BODY) > 50 THEN 1 END) as summarized_count,
                    COUNT(CASE WHEN LENGTH(EMAIL_BODY) > 50 THEN 1 END) as long_emails_count,
                    COUNT(CASE WHEN CORTEX_DETECTED_LANGUAGE != 'en' 
                              AND CORTEX_DETECTED_LANGUAGE IS NOT NULL THEN 1 END) as non_english_count
                FROM CUSTOMER_SUPPORT_EMAILS
            )
            SELECT 
                total_emails,
                translated_count,
                sentiment_count,
                summarized_count,
                long_emails_count,
                non_english_count,
                CASE 
                    WHEN non_english_count = 0 THEN 0 
                    ELSE ROUND(translated_count * 100.0 / non_english_count, 1) 
                END as translation_percentage,
                CASE 
                    WHEN total_emails = 0 THEN 0 
                    ELSE ROUND(sentiment_count * 100.0 / total_emails, 1) 
                END as sentiment_percentage,
                CASE 
                    WHEN long_emails_count = 0 THEN 0 
                    ELSE ROUND(summarized_count * 100.0 / long_emails_count, 1) 
                END as summary_percentage
            FROM EmailStats
        """
        
        # Execute query with modified execute_sql_safely function
        stats_df = pd.DataFrame(session.sql(stats_query).collect())
        
        if stats_df.empty:
            st.error("No data available in the CUSTOMER_SUPPORT_EMAILS table.")
            return
            
        # Display overall statistics
        st.markdown("### 📈 Overall Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Total Emails", 
                f"{int(stats_df.loc[0, 'TOTAL_EMAILS']):,}",
                help="Total number of emails in the database"
            )
            
        with col2:
            st.metric(
                "Non-English Emails", 
                f"{int(stats_df.loc[0, 'NON_ENGLISH_COUNT']):,}",
                f"{stats_df.loc[0, 'TRANSLATION_PERCENTAGE']}% Translated"
            )
            
        with col3:
            st.metric(
                "Long Emails (>50 chars)", 
                f"{int(stats_df.loc[0, 'LONG_EMAILS_COUNT']):,}",
                f"{stats_df.loc[0, 'SUMMARY_PERCENTAGE']}% Summarized"
            )

        # Processing Status
        st.markdown("### 🔄 Processing Status")
        col4, col5, col6 = st.columns(3)
        
        with col4:
            translated = int(stats_df.loc[0, 'TRANSLATED_COUNT'])
            needs_translation = int(stats_df.loc[0, 'NON_ENGLISH_COUNT']) - translated
            st.metric(
                "Translations Complete", 
                f"{translated:,}",
                f"{needs_translation:,} pending" if needs_translation > 0 else "All complete!"
            )
            
        with col5:
            sentiment_complete = int(stats_df.loc[0, 'SENTIMENT_COUNT'])
            needs_sentiment = int(stats_df.loc[0, 'TOTAL_EMAILS']) - sentiment_complete
            st.metric(
                "Sentiment Analysis Complete", 
                f"{sentiment_complete:,}",
                f"{needs_sentiment:,} pending" if needs_sentiment > 0 else "All complete!"
            )
            
        with col6:
            summarized = int(stats_df.loc[0, 'SUMMARIZED_COUNT'])
            needs_summary = int(stats_df.loc[0, 'LONG_EMAILS_COUNT']) - summarized
            st.metric(
                "Summarizations Complete", 
                f"{summarized:,}",
                f"{needs_summary:,} pending" if needs_summary > 0 else "All complete!"
            )

        # Create two columns for the charts
        chart_col1, chart_col2 = st.columns(2)

        # Sentiment Distribution
        sentiment_query = """
            SELECT 
                CASE 
                    WHEN CORTEX_SENTIMENT_NORMALIZED <= 3 THEN 'Negative (1-3)'
                    WHEN CORTEX_SENTIMENT_NORMALIZED <= 6 THEN 'Neutral (4-6)'
                    ELSE 'Positive (7-10)'
                END as SENTIMENT_CATEGORY,
                COUNT(*) as COUNT
            FROM CUSTOMER_SUPPORT_EMAILS
            WHERE CORTEX_SENTIMENT_NORMALIZED IS NOT NULL
            GROUP BY SENTIMENT_CATEGORY
            ORDER BY SENTIMENT_CATEGORY
        """
        
        sentiment_df = pd.DataFrame(session.sql(sentiment_query).collect())
        
        if not sentiment_df.empty:
            with chart_col1:
                st.markdown("### 📊 Sentiment Distribution")
                fig = px.pie(
                    sentiment_df, 
                    values='COUNT', 
                    names='SENTIMENT_CATEGORY',
                    color='SENTIMENT_CATEGORY',
                    color_discrete_map={
                        'Negative (1-3)': '#ff6666',  # Lighter Red
                        'Neutral (4-6)': '#ffff99',   # Lighter Yellow
                        'Positive (7-10)': '#99ff99'  # Lighter Green
                    }
                )
                fig.update_layout(
                    plot_bgcolor='black',
                    paper_bgcolor='black',
                    font_color='white'
                )
                st.plotly_chart(fig, use_container_width=True)

        # Classification Distribution
        classification_query = """
            SELECT 
                CORTEX_CLASSIFY_TEXT as CATEGORY,
                COUNT(*) as COUNT
            FROM CUSTOMER_SUPPORT_EMAILS
            WHERE 
                CORTEX_CLASSIFY_TEXT IS NOT NULL 
                AND CORTEX_CLASSIFY_TEXT != ''
                AND CORTEX_CLASSIFY_TEXT NOT LIKE '{%}'
                AND CORTEX_CLASSIFY_TEXT NOT LIKE '[]'
            GROUP BY CORTEX_CLASSIFY_TEXT
            ORDER BY COUNT DESC
        """
        
        classification_df = pd.DataFrame(session.sql(classification_query).collect())
        
        if not classification_df.empty:
            with chart_col2:
                st.markdown("### 🏷️ Product Category Distribution")
                fig2 = px.pie(
                    classification_df, 
                    values='COUNT', 
                    names='CATEGORY',
                    color='CATEGORY',
                    color_discrete_map = {
                        'Drivers': '#a3e4d7',        # Light Greenish
                        'Hybrid': '#a3e4d7',         # Light Greenish
                        'Irons': '#e67e22',          # Orange
                        'Wedges': '#f1c40f',         # Yellow
                        'Putters': '#f1948a',        # Light Red
                        'Fairway Woods': '#1abc9c',  # Light Turquoise
                        'Golf Balls': '#f8f4e3',     # Off White
                        'Bags': '#a0522d',           # Brown
                        'Accessories': '#3498db',    # Blue
                        'Unclassified': '#7f8c8d'    # Grey
                    }
                )
                fig2.update_layout(
                    plot_bgcolor='black',
                    paper_bgcolor='black',
                    font_color='white'
                )
                st.plotly_chart(fig2, use_container_width=True)

        # Download capability
        st.markdown("### 📥 Download Analytics Report")
        
        # Create base metrics
        base_metrics = {
            'Metric': [
                'Total Emails',
                'Non-English Emails',
                'Translated Emails',
                'Translation Coverage',
                'Emails with Sentiment Analysis',
                'Sentiment Analysis Coverage',
                'Long Emails (>50 chars)',
                'Summarized Emails',
                'Summarization Coverage'
            ],
            'Value': [
                int(stats_df.loc[0, 'TOTAL_EMAILS']),
                int(stats_df.loc[0, 'NON_ENGLISH_COUNT']),
                int(stats_df.loc[0, 'TRANSLATED_COUNT']),
                f"{stats_df.loc[0, 'TRANSLATION_PERCENTAGE']}%",
                int(stats_df.loc[0, 'SENTIMENT_COUNT']),
                f"{stats_df.loc[0, 'SENTIMENT_PERCENTAGE']}%",
                int(stats_df.loc[0, 'LONG_EMAILS_COUNT']),
                int(stats_df.loc[0, 'SUMMARIZED_COUNT']),
                f"{stats_df.loc[0, 'SUMMARY_PERCENTAGE']}%"
            ]
        }

        # Create the download DataFrame
        download_df = pd.DataFrame(base_metrics)

        # Add classification metrics if available
        if not classification_df.empty:
            base_metrics['Metric'].append('--- Product Categories ---')
            base_metrics['Value'].append('')
            
            total_classified = classification_df['COUNT'].sum()
            for _, row in classification_df.iterrows():
                percentage = (row['COUNT'] / total_classified) * 100
                base_metrics['Metric'].append(f"Category: {row['CATEGORY']}")
                base_metrics['Value'].append(f"{row['COUNT']:,} ({percentage:.1f}%)")
        
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        csv = download_df.to_csv(index=False)
        
        st.download_button(
            label="Download Analytics Report (CSV)",
            data=csv,
            file_name=f"support_analytics_{current_time}.csv",
            mime="text/csv"
        )
                
    except Exception as e:
        st.error("Error loading dashboard. Please check the following:")
        st.error("1. The database connection is active")
        st.error("2. The CUSTOMER_SUPPORT_EMAILS table exists")
        st.error("3. All required columns are present")
        st.error(f"Technical details: {str(e)}")
        
        if st.checkbox("Show detailed error information"):
            st.exception(e)
            
##########################################
#             Translation
##########################################
def handle_translation(session):
    """Handle Translation Feature"""
    try:
        # Fetch untranslated emails
        table_query = """
            SELECT * 
            FROM CUSTOMER_SUPPORT_EMAILS 
            WHERE NULLIF(CORTEX_TRANSLATE, '') IS NULL;
        """
        data_df = execute_sql_safely(session, table_query)

        if data_df is None or data_df.empty:
            st.warning("No entries available for translation.")
            return

        # Process and filter non-English rows
        data_df["CORTEX_DETECTED_LANGUAGE"] = data_df["EMAIL_BODY"].apply(detect_language)
        non_english_df = data_df.dropna(subset=["CORTEX_DETECTED_LANGUAGE"])

        if non_english_df.empty:
            st.warning("No non-English entries available for translation.")
            return

        # Select and display email
        selected_email = select_email(non_english_df, "translate")
        display_email(selected_email)

        email_body = selected_email["EMAIL_BODY"]
        email_id = selected_email["EMAIL_ID"]
        detected_language = selected_email["CORTEX_DETECTED_LANGUAGE"]

        st.info(f"Detected Language: {detected_language}")

        # Translate
        translation = translate_text(session, email_body, detected_language)
        if translation:
            # Update database
            update_query = f"""
                UPDATE CUSTOMER_SUPPORT_EMAILS
                SET 
                    CORTEX_TRANSLATE = '{safe_sql_string(translation)}',
                    CORTEX_DETECTED_LANGUAGE = '{detected_language}'
                WHERE EMAIL_ID = '{email_id}'
            """
            execute_sql_safely(session, update_query)
            
            st.success(f"Successfully updated translation for Email ID: {email_id}")
            st.markdown("### Translation")
            st.success(translation)
        else:
            st.error("Translation failed. Please try again.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

##########################################
#             Sentiment
##########################################
def handle_sentiment(session, model):
    """Handle Sentiment Analysis Feature"""
    try:
        if 'selected_email' not in st.session_state:
            st.session_state.selected_email = None
            
        # Fetch unprocessed emails
        table_query = """
            SELECT * 
            FROM CUSTOMER_SUPPORT_EMAILS 
            WHERE NULLIF(CORTEX_SENTIMENT, '') IS NULL
            LIMIT 10;
        """
        data_df = execute_sql_safely(session, table_query)

        if data_df is None or data_df.empty:
            st.warning("No entries available for sentiment analysis.")
            return

        # Select and display email
        selected_email = select_email(data_df, "sentiment")
        st.session_state.selected_email = selected_email
        display_email(selected_email)

        # Add analysis button
        if st.button("Analyze Sentiment", key="analyze_sentiment_button"):
            with st.spinner(f'Analyzing sentiment using {model}...'):
                email_body = selected_email["EMAIL_BODY"]
                sentiment_results = analyze_sentiment(session, email_body)
                
                if sentiment_results:
                    # Update database
                    update_query = f"""
                        UPDATE CUSTOMER_SUPPORT_EMAILS
                        SET 
                            CORTEX_SENTIMENT = '{sentiment_results['raw_score']}',
                            CORTEX_SENTIMENT_NORMALIZED = {sentiment_results['scaled_score']}
                        WHERE EMAIL_ID = '{selected_email["EMAIL_ID"]}'
                    """
                    execute_sql_safely(session, update_query)
                    
                    # Display results
                    bg_color = (
                        "#ff6b6b" if sentiment_results['scaled_score'] <= 3 
                        else "#4ecdc4" if sentiment_results['scaled_score'] <= 6 
                        else "#95d5b2"
                    )
                    
                    st.markdown(f"""
                        <div class="sentiment-card">
                            <div class="sentiment-score">
                                {sentiment_results['scaled_score']}/10 {sentiment_results['emoji']}
                            </div>
                            <div class="sentiment-category" style="background-color: {bg_color};">
                                {sentiment_results['category']}
                            </div>
                            <div class="sentiment-details">
                                <!-- 
                                <div class="sentiment-detail-item">
                                    <div style="font-size: 14px; color: #888;">Raw Score</div>
                                    <div style="font-size: 20px; margin-top: 5px;">{sentiment_results['raw_score']:.3f}</div>
                                </div>
                                <div class="sentiment-detail-item">
                                    <div style="font-size: 14px; color: #888;">Confidence</div>
                                    <div style="font-size: 20px; margin-top: 5px;">{abs(sentiment_results['raw_score'] * 100):.1f}%</div>
                                </div>
                                -->
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                    # Gauge Chart
                    st.subheader("Sentiment Gauge Chart")
                    
                    # Create and display the gauge chart using actual sentiment score
                    fig = create_sentiment_gauge(sentiment_results['scaled_score'])
                    st.plotly_chart(fig)
                    
                    # Add interpretation
                    interpretation = (
                        "😢 Very Negative" if sentiment_results['scaled_score'] <= 2
                        else "☹️ Negative" if sentiment_results['scaled_score'] <= 4
                        else "😐 Neutral" if sentiment_results['scaled_score'] <= 6
                        else "🙂 Positive" if sentiment_results['scaled_score'] <= 8
                        else "😊 Very Positive"
                    )
                    
                    st.markdown(f"""
                        <div style="text-align: center; margin-top: 10px; font-size: 20px;">
                            {interpretation}
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error("Sentiment analysis failed. Please try again.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

##########################################
#             Summarization
##########################################
def handle_summarization(session):
    """Handle Text Summarization Feature"""
    try:
        # Fetch long emails for summarization
        table_query = """
            SELECT * 
            FROM CUSTOMER_SUPPORT_EMAILS 
            WHERE NULLIF(CORTEX_SUMMARIZE, '') IS NULL
            AND LENGTH(EMAIL_BODY) > 100
            LIMIT 10;
        """
        data_df = execute_sql_safely(session, table_query)

        if data_df is None or data_df.empty:
            st.warning("No entries available for summarization.")
            return

        # Select and display email
        selected_email = select_email(data_df, "summarize")
        display_email(selected_email)

        if st.button("Generate Summary", key="generate_summary_button"):
            with st.spinner('Generating summary...'):
                summary = summarize_text(session, selected_email["EMAIL_BODY"])
                
                if summary:
                    # Update database
                    update_query = f"""
                        UPDATE CUSTOMER_SUPPORT_EMAILS
                        SET CORTEX_SUMMARIZE = '{safe_sql_string(summary)}'
                        WHERE EMAIL_ID = '{selected_email["EMAIL_ID"]}'
                    """
                    execute_sql_safely(session, update_query)
                    
                    st.success(f"Successfully updated summary for Email ID: {selected_email['EMAIL_ID']}")
                    
                    # Display summary in a card
                    st.markdown("""
                        <div class="email-container">
                            <div style="font-size: 1.2em; margin-bottom: 10px;">📝 Summary</div>
                            <div style="line-height: 1.6;">
                    """, unsafe_allow_html=True)
                    st.write(summary)
                    st.markdown("</div></div>", unsafe_allow_html=True)
                else:
                    st.error("Summarization failed. Please try again.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

##########################################
#             Classification
##########################################
def handle_classification(session):
    """Handle Text Classification Feature"""
    try:
        table_query = """
            SELECT * 
            FROM CUSTOMER_SUPPORT_EMAILS 
            WHERE NULLIF(CORTEX_CLASSIFY_TEXT, '') IS NULL
            AND LENGTH(EMAIL_BODY) > 5 
            LIMIT 10;
        """
        data_df = execute_sql_safely(session, table_query)

        if data_df is None or data_df.empty:
            st.warning("No entries available for classification.")
            return

        # Select and display email
        selected_email = select_email(data_df, "classify")
        display_email(selected_email)

        if st.button("Classify Email", key="classify_email_button"):
            with st.spinner('Classifying email...'):
                classification_results = classify_text(session, selected_email["EMAIL_BODY"])
                
                if classification_results:
                    # Update database
                    update_query = f"""
                        UPDATE CUSTOMER_SUPPORT_EMAILS
                        SET CORTEX_CLASSIFY_TEXT = '{classification_results['primary_category']}'
                        WHERE EMAIL_ID = '{selected_email["EMAIL_ID"]}'
                    """
                    execute_sql_safely(session, update_query)
                    
                    # Display results
                    st.markdown("### Classification Results")
                    if classification_results['primary_category'] == "Unclassified":
                        st.warning("No specific product category detected in the email.")
                    else:
                        st.success(f"Primary Product Category: {classification_results['primary_category']}")
                    
                    # Show confidence scores
                    if classification_results['total_mentions'] > 0:
                        st.write("#### Category Relevance Scores:")
                        sorted_scores = sorted(
                            classification_results['scores'].items(), 
                            key=lambda x: x[1], 
                            reverse=True
                        )
                        
                        for category, score in sorted_scores:
                            if score > 0:
                                score_percentage = score * 100
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.progress(score)
                                with col2:
                                    st.write(f"{score_percentage:.1f}%")
                                st.markdown(f"**{category}**")
                                st.write("---")
                    
                    st.success(f"Classification completed for Email ID: {selected_email['EMAIL_ID']}")
                else:
                    st.error("Classification failed. Please try again.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

##########################################
#             Answer Extraction
##########################################
def handle_answer_extraction(session):
    """Handle Answer Extraction Feature"""
    try:
        # Fetch emails
        table_query = """
            SELECT * 
            FROM CUSTOMER_SUPPORT_EMAILS 
            WHERE NULLIF(cortex_extract_answer, '') IS NULL
            LIMIT 10;
        """
        data_df = execute_sql_safely(session, table_query)

        if data_df is None or data_df.empty:
            st.warning("No entries available for answer extraction.")
            return

        # Select and display email
        selected_email = select_email(data_df, "extract")
        display_email(selected_email)

        # Question input
        question = st.text_input(
            "Enter your question about this email:",
            placeholder="e.g., What product is the customer inquiring about?"
        )

        if question:
            if st.button("Extract Answer", key="extract_answer_button"):
                with st.spinner('Extracting answer...'):
                    answer = extract_answer(session, selected_email["EMAIL_BODY"], question)
                    
                    if answer:
                        # Display Q&A
                        st.markdown("""
                            <div class="email-container">
                                <div style="margin-bottom: 15px;">
                                    <div style="color: #888; margin-bottom: 5px;">Question:</div>
                                    <div style="color: white; font-size: 1.1em;">%s</div>
                                </div>
                                <div>
                                    <div style="color: #888; margin-bottom: 5px;">Answer:</div>
                                    <div style="color: white; font-size: 1.1em;">%s</div>
                                </div>
                            </div>
                        """ % (question, answer), unsafe_allow_html=True)
                        
                        # Store in session state
                        if 'qa_history' not in st.session_state:
                            st.session_state.qa_history = []
                        
                        st.session_state.qa_history.append({
                            'email_id': selected_email["EMAIL_ID"],
                            'question': question,
                            'answer': answer,
                            'timestamp': pd.Timestamp.now()
                        })
                    else:
                        st.error("Answer extraction failed. Please try again.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

##########################################
#             Email Copy 
##########################################

# Function to display code with a copy button
def add_copy_button(code_string, unique_id):
    components.html(f"""
    <div style="position: relative; margin-bottom: 20px;">
        <button onclick="copyToClipboard('code-{unique_id}')" style="margin-bottom: 10px;">Copy Code</button>
        <pre id="code-{unique_id}" style="display: none;">{code_string}</pre>
    </div>
    <script>
    function copyToClipboard(elementId) {{
        var text = document.getElementById(elementId).textContent;
        navigator.clipboard.writeText(text).then(function() {{
            alert('Code copied to clipboard!');
        }}).catch(function(error) {{
            console.error('Could not copy text: ', error);
        }});
    }}
    </script>
    """, height=100)

##########################################
#             Email Reply 
##########################################
def handle_reply_generation(session, model):
    """Handle Reply Generation Feature"""
    try:
        # Initialize session state
        if 'selected_email_reply' not in st.session_state:
            st.session_state.selected_email_reply = None
        
        # Fetch emails
        table_query = """
            SELECT *,
                   TO_CHAR(received_time, 'Month DD, YYYY HH:MI:SS AM') as FORMATTED_RECEIVED_TIME,
                   TO_CHAR(response_due_by, 'Month DD, YYYY HH:MI:SS AM') as FORMATTED_DUE_TIME
            FROM CUSTOMER_SUPPORT_EMAILS
            TABLESAMPLE (10 ROWS)
        """
        data_df = execute_sql_safely(session, table_query)

        if data_df is None or data_df.empty:
            st.warning("No entries available for reply generation.")
            return

        # Select and display email
        selected_email = select_email(data_df, "reply")
        st.session_state.selected_email_reply = selected_email
        display_email(selected_email)

        if st.button("Generate Reply", key="generate_reply_button"):
            with st.spinner(f'Generating a sympathetic reply using {model} model...'):
                suggested_reply = generate_reply(session, selected_email["EMAIL_BODY"], model)
                
                if suggested_reply:
                    # Store the reply in session state
                    st.session_state.current_reply = suggested_reply
                    
                    # Display the reply
                    st.markdown("### 📧 Suggested Reply")
                    st.markdown("""
                        <div class="reply-card">
                            <div class="reply-content">%s</div>
                        </div>
                    """ % suggested_reply.replace('\n', '<br>'), unsafe_allow_html=True)
                    
                    # Add text area for editing
                    st.markdown("### ✏️ Edit Reply")
                    edited_reply = st.text_area(
                        "Edit the suggested reply:",
                        value=suggested_reply,
                        height=200,
                        key="reply_editor"
                    )
                    
                    # Add copy buttons for original and edited replies
                    st.markdown("### 📋 Copy Original Reply")
                    add_copy_button(suggested_reply, unique_id='original_reply')

                    st.markdown("### 📋 Copy Edited Reply")
                    add_copy_button(edited_reply, unique_id='edited_reply')
                else:
                    st.error("Reply generation failed. Please try again.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")


##########################################
#                Batch
##########################################
def handle_batch_processing(session):
    """Handle Batch Processing Feature"""
    st.write("### Process All Data")
    st.write("This will process unprocessed emails (those missing translation, sentiment, or summarization).")
    
    try:
        # Get counts of unprocessed emails
        stats_query = """
            SELECT 
                COUNT(*) as total,
                COUNT(CASE WHEN (CORTEX_TRANSLATE IS NULL OR CORTEX_TRANSLATE = '') 
                          AND (CORTEX_DETECTED_LANGUAGE IS NULL OR CORTEX_DETECTED_LANGUAGE = '') THEN 1 END) as needs_lang_check,
                COUNT(CASE WHEN CORTEX_SENTIMENT IS NULL OR CORTEX_SENTIMENT = '' THEN 1 END) as needs_sentiment,
                COUNT(CASE WHEN (CORTEX_SUMMARIZE IS NULL OR CORTEX_SUMMARIZE = '') 
                          AND LENGTH(EMAIL_BODY) > 5 THEN 1 END) as needs_summary
            FROM CUSTOMER_SUPPORT_EMAILS
            WHERE EMAIL_BODY IS NOT NULL
        """
        stats = execute_sql_safely(session, stats_query)
        
        if stats is None:
            st.error("Unable to fetch processing statistics.")
            return
            
        # Display current status
        st.write("### Current Status")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Emails Needing Language Check", stats['NEEDS_LANG_CHECK'][0])
        with col2:
            st.metric("Emails Needing Sentiment Analysis", stats['NEEDS_SENTIMENT'][0])
        with col3:
            st.metric("Emails Needing Summarization", stats['NEEDS_SUMMARY'][0])
        
        # Add confirmation button
        st.warning("⚠️ This process may take a while depending on the number of emails.")
        confirm = st.button("Start Processing All Unprocessed Data", type="primary")
        
        if confirm:
            process_all_emails(session)

    except Exception as e:
        st.error(f"Error during batch processing: {str(e)}")

##########################################
#             Main Application
##########################################
def main():
    """Main application entry point"""
    
    # Load CSS
    load_css()
    
    # Initialize Snowflake session
    session = get_snowflake_session()
    
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Title and Logo
    st.image("https://moongolf.com/wp-content/uploads/2017/03/Callaway-logo-WHITE-1024x591-small-300x173.png", width=150)
    st.divider()
    st.subheader("⛳ Support Caddy")
    st.divider()
    
    # Get configuration options
    #selected_model, use_chat_history, debug = config_options()
    selected_model, use_chat_history, debug, option = config_options()
    
    # Main option selection with Dashboard as default
   # option = st.selectbox(
   #     'Select Analysis Option:',
   #     ("Dashboard", "Translate Text", "Detect Sentiment", "Summarize Text", 
   #      "Classify Text", "Extract Answer", "Suggest Email Reply", "Process All Data")
   # )
    
    # Handle each option
    try:
        if option == "Dashboard":
            handle_dashboard(session)
            
        elif option == "Translate Text":
            st.markdown("## 🌍 Email Translation")
            handle_translation(session)
            
        elif option == "Detect Sentiment":
            st.markdown("## 😊 Sentiment Analysis")
            handle_sentiment(session, selected_model)
            
        elif option == "Summarize Text":
            st.markdown("## 📝 Email Summarization")
            handle_summarization(session)
            
        elif option == "Classify Text":
            st.markdown("## 🏷️ Email Classification")
            handle_classification(session)
            
        elif option == "Extract Answer":
            st.markdown("## ❓ Answer Extraction")
            handle_answer_extraction(session)
            
        elif option == "Suggest Email Reply":
            st.markdown("## 📧 Email Reply Generation")
            handle_reply_generation(session, selected_model)
            
        elif option == "Process All Data":
            st.markdown("## 🔄 Batch Processing")
            handle_batch_processing(session)
            
        # Show debug information if enabled
        if debug:
            with st.expander("Debug Information"):
                st.write("### Session State")
                st.json(st.session_state)
                
                st.write("### Current Configuration")
                st.write({
                    "Selected Model": selected_model,
                    "Chat History Enabled": use_chat_history,
                    "Current Option": option
                })
                
                st.write("### System Status")
                try:
                    status_query = """
                        SELECT 
                            COUNT(*) as total_emails,
                            COUNT(DISTINCT email_status) as status_count,
                            COUNT(DISTINCT priority) as priority_count
                        FROM customer_support_emails
                    """
                    status_df = execute_sql_safely(session, status_query)
                    if status_df is not None:
                        st.write(status_df)
                except Exception as e:
                    st.write(f"Error fetching system status: {str(e)}")
        
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        if debug:
            st.exception(e)
    
    # Add footer
    st.markdown("""
        ---
        <div style="text-align: center; color: #666666; padding: 10px;">
            Support Caddy • Powered by Snowflake Cortex • Built By Mike Malveira
        </div>
    """, unsafe_allow_html=True)

def process_all_emails(session):
    """Process all unprocessed emails in batch"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Get unprocessed emails
    emails_query = """
        SELECT *
        FROM CUSTOMER_SUPPORT_EMAILS
        WHERE EMAIL_BODY IS NOT NULL
        AND (
            (CORTEX_TRANSLATE IS NULL OR CORTEX_TRANSLATE = '') 
            AND (CORTEX_DETECTED_LANGUAGE IS NULL OR CORTEX_DETECTED_LANGUAGE = '')
            OR CORTEX_SENTIMENT IS NULL OR CORTEX_SENTIMENT = ''
            OR (CORTEX_SUMMARIZE IS NULL OR CORTEX_SUMMARIZE = '') 
            AND LENGTH(EMAIL_BODY) > 5
        )
    """
    emails_df = execute_sql_safely(session, emails_query)
    
    if emails_df is None or emails_df.empty:
        st.success("No unprocessed emails found!")
        return
    
    total_to_process = len(emails_df)
    status_text.write(f"Found {total_to_process} emails to process...")
    processed_count = 0
    
    for _, row in emails_df.iterrows():
        email_body = row['EMAIL_BODY']
        email_id = row['EMAIL_ID']
        
        # Update progress
        progress = (processed_count + 1) / total_to_process
        progress_bar.progress(progress)
        status_text.write(f"Processing Email ID: {email_id} ({processed_count + 1}/{total_to_process})")
        
        try:
            updates = []
            
            # 1. Language detection and translation
            if pd.isna(row['CORTEX_DETECTED_LANGUAGE']):
                language = detect_language(email_body)
                if language:
                    translation = translate_text(session, email_body, language)
                    if translation:
                        updates.extend([
                            f"CORTEX_TRANSLATE = '{safe_sql_string(translation)}'",
                            f"CORTEX_DETECTED_LANGUAGE = '{language}'"
                        ])
                else:
                    updates.append("CORTEX_DETECTED_LANGUAGE = 'en'")
            
            # Use translated text if available
            processing_text = translation if 'translation' in locals() else email_body
            
            # 2. Sentiment analysis
            if pd.isna(row['CORTEX_SENTIMENT']):
                sentiment_results = analyze_sentiment(session, processing_text)
                if sentiment_results:
                    updates.extend([
                        f"CORTEX_SENTIMENT = '{sentiment_results['raw_score']}'",
                        f"CORTEX_SENTIMENT_NORMALIZED = {sentiment_results['scaled_score']}"
                    ])
            
            # 3. Summarization
            if pd.isna(row['CORTEX_SUMMARIZE']) and len(processing_text) > 5:
                summary = summarize_text(session, processing_text)
                if summary:
                    updates.append(f"CORTEX_SUMMARIZE = '{safe_sql_string(summary)}'")
            
            # Apply updates
            if updates:
                update_query = f"""
                    UPDATE CUSTOMER_SUPPORT_EMAILS
                    SET {', '.join(updates)}
                    WHERE EMAIL_ID = '{email_id}'
                """
                execute_sql_safely(session, update_query)
                
                # Show results in expander
                with st.expander(f"Results for Email ID: {email_id}", expanded=False):
                    st.write("Original Text:", email_body[:200] + "..." if len(email_body) > 200 else email_body)
                    if 'language' in locals() and language:
                        st.write("Detected Language:", language)
                        st.write("Translated Text:", translation[:200] + "..." if len(translation) > 200 else translation)
                    if 'sentiment_results' in locals():
                        st.write(f"Sentiment Score: {sentiment_results['scaled_score']}/10")
                    if 'summary' in locals():
                        st.write("Summary:", summary)
                    st.success(f"Successfully processed Email ID: {email_id}")
            
            processed_count += 1
            
        except Exception as e:
            st.error(f"Error processing Email ID {email_id}: {str(e)}")
            continue
    
    # Final status
    progress_bar.progress(1.0)
    st.success(f"Successfully processed {processed_count} out of {total_to_process} emails!")
    
    # Show updated statistics
    show_final_processing_stats(session)

def show_final_processing_stats(session):
    """Show final processing statistics"""
    st.write("### Final Processing Summary")
    final_stats_query = """
        WITH Stats AS (
            SELECT 
                COUNT(CASE WHEN CORTEX_TRANSLATE IS NOT NULL 
                          AND CORTEX_TRANSLATE != ''
                          AND CORTEX_DETECTED_LANGUAGE != 'en' 
                          AND CORTEX_DETECTED_LANGUAGE != ''
                          THEN 1 END) as translations,
                COUNT(CASE WHEN CORTEX_SENTIMENT IS NOT NULL 
                          AND CORTEX_SENTIMENT != '' 
                          THEN 1 END) as sentiments,
                COUNT(CASE WHEN CORTEX_SUMMARIZE IS NOT NULL 
                          AND CORTEX_SUMMARIZE != '' 
                          THEN 1 END) as summaries,
                COUNT(*) as total_emails
            FROM CUSTOMER_SUPPORT_EMAILS
        )
        SELECT 
            translations,
            sentiments,
            summaries,
            ROUND(translations / NULLIF(total_emails, 0) * 100, 1) as translation_pct,
            ROUND(sentiments / NULLIF(total_emails, 0) * 100, 1) as sentiment_pct,
            ROUND(summaries / NULLIF(total_emails, 0) * 100, 1) as summary_pct
        FROM Stats
    """
    final_stats = execute_sql_safely(session, final_stats_query)
    
    if final_stats is not None:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Non-English Translations", 
                final_stats['TRANSLATIONS'][0],
                f"{final_stats['TRANSLATION_PCT'][0]}%"
            )
        with col2:
            st.metric(
                "Sentiment Analyses", 
                final_stats['SENTIMENTS'][0],
                f"{final_stats['SENTIMENT_PCT'][0]}%"
            )
        with col3:
            st.metric(
                "Summarizations", 
                final_stats['SUMMARIES'][0],
                f"{final_stats['SUMMARY_PCT'][0]}%"
            )

if __name__ == "__main__":
    main()