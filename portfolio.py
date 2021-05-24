#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 2021
@author: Jaryd Thornton
"""
import streamlit as st
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from category_encoders import OrdinalEncoder
from sklearn.pipeline import make_pipeline
from pdpbox import pdp
from joblib import load
from sklearn.metrics import f1_score
from sklearn.utils import compute_class_weight
from datetime import time, timedelta

# format tables to not show index
st.markdown(""" 
    <style>
    table td:nth-child(1) {
        display: none
        }
    table th:nth-child(1) {
        display: none
        }
    </style>
    """, unsafe_allow_html=True)

# create pages
page = st.sidebar.radio('Navigation',
        ['About', 
        'NLP Sentiment analysis',
        'Spotify algorithm',
        'Kickstarter planner',
        'Fuel efficiency'])

# png file for about page
picture = "https://www.newzealand.com/assets/Tourism-NZ/Auckland/1975559ab3/img-1536201939-3159-8823-717CA83C-0811-08A9-5BCA19BBB934D606__aWxvdmVrZWxseQo_FocalPointCropWzY2MCwxOTIwLDQwLDY2LDc1LCJqcGciLDY1LDIuNV0.jpg"

if page == 'About':
    st.title('About')
    st.image(picture, width=400, output_format='PNG')

    st.markdown("""
                My name is Jaryd Thornton, I'm a Data Engineer from Auckland, New Zealand. 
                I created this platform not only as a place to submit my work in the public domain but also 
                to allow anyone to experience machine learning in a fun and interactive way.
                """)
                
    st.subheader('**NLP Sentiment analysis**')
    st.markdown("""
                Natural language processors (NLP) uses neural networks to process and understand human language. 
                Neural networks are designed to simulate the way the human brain analyzes and processes 
                information. I have built a NLP that processes a 
                message of your choosing to determine the sentiment of the language. 
                This can be used for checking an email before clicking send to determine if 
                your message is coming across the right way.
                """)
                
    st.subheader('**Spotify algorithm**')
    st.markdown("""
                Recently gradient boosting has taken over as the prefered and most accurate machine learning method for regression and classification on structured datasets. 
                By tapping in and analysing Spotify's algorithm, this regression model takes in characteristics and parameters of a song and returns 
                the probability of it being popular. As a musician myself I've always been curious as to what makes a song a number one hit.
                """)
    
    st.subheader('**Kickstarter planner**')
    st.markdown("""
                Using advanced regularized gradient boosting for classification, this model 
                enables us to identify the parameters of a successful or unsuccessful Kickstarter campaign. 
                With this tool, we can check each parameter of our kickstarter campaign and make adjustments before launch to increase the chance 
                of success. 
                """)

    st.subheader('**Fuel efficiency**')
    st.markdown("""
                Another regression problem with a twist. All the models on this platform with the exception of this one, have been trained and tuned prior to being 
                loaded. This model has the benefit of using a much smaller dataset, so that we can make tuning adjustments, change what we are wanting to predict and 
                analyse feature correlations all in real-time. This enables you the user to not only interact with the results of a machine learning model but also to 
                adapt and tweak the model to improve its performance. The model works with vehicle engine specifications to predict the following: Annual fuel costs, 
                fuel economy (MPG/PMPL), and COS emissions.
                """)

    
if page == 'NLP Sentiment analysis':
    
    @st.cache()
    def build_model():
        mod = load_model('nlp_files')
        return mod
    
    st.title("NLP Sentiment analysis")
    st.markdown("Check your message to see how it conveys.")

    mod = build_model()
    
    text = st.text_input('Write your message here...')
    if text != "":
        # grab the predicted value
        pred = mod.predict([text])[0][0]
        predformat = "{0:.2f}%".format(pred * 100) + " positive"
        predchartformat = round(pred * 100)
        # check the value and return the results
        st.markdown("**Message:**")
        st.text(text)
        if pred > 0.75:
            sentiment = 'positive'
        elif pred > 0.5:
            sentiment = 'warm'
        elif pred > 0.25:
            sentiment = 'cold'
        else:
            sentiment = 'negative'
    
        st.header(f"Your message can be regarded as **{sentiment}**")
        st.subheader(f"{predformat}")
    
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = predchartformat,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Positivity meter"},
            gauge = {'axis': {'range': [None, 100]},
                     'bar': {'color': '#1394ad'},
                     'steps' : [
                     {'range': [0, 25], 'color': "#383737"},
                     {'range': [25, 50], 'color': "#515050"},
                     {'range': [50, 75], 'color': "#6b6b6b"},
                     {'range': [75, 100], 'color': "#878787"},],
                     }))
        st.write(fig)

    st.markdown(r"""
    **How this model works:** This NLP model has been trained using an artificial neural network. 
    I've trained this model on a tensorflow-hub dataset that has analysed 1.6 Millon tweets to determine if the tweet is positive or negative.
    Input layers in the neural network take vectorization of the twitter text 
    which then sends these inputs through a series of hidden weighted layers. Each layer has been set to use the 
    Logistic Sigmoid Function: 
    $$ 
    f(x) = \frac{1}{1 + e^{-x}} 
    $$
    The weighted sum of the inputs is passed to the Sigmoid function to produce an output.
    Model training is performed by reducing the loss function between each output and its predictions.
    The final accuracy score of this model is 87.3% which enables us to not only predict the sentiment of words used in text but also the combination of
    words.
    """)
        
if page == 'Kickstarter planner':
    
    #@st.cache()
    def load_k2():
        k_mod = load("ks_project.joblib.dat")
        return k_mod 
    k_mod = load_k2()
    
    #@st.cache()
    def load_k2_files():
        df1 = pd.read_csv('https://raw.githubusercontent.com/jcolethornton/datasets/main/ks1.csv')
        df2 = pd.read_csv('https://raw.githubusercontent.com/jcolethornton/datasets/main/ks2.csv')
        df3 = pd.read_csv('https://raw.githubusercontent.com/jcolethornton/datasets/main/ks3.csv')
        df4 = pd.read_csv('https://raw.githubusercontent.com/jcolethornton/datasets/main/ks4.csv')
        df5 = pd.read_csv('https://raw.githubusercontent.com/jcolethornton/datasets/main/ks5.csv')
        df = pd.concat([df1,df2,df3,df4,df5])
        df = df.loc[(df['time'] <= 60) & (df['time'] > 0)]
        df = df.drop('Unnamed: 0', axis=1)
        return df
    df = load_k2_files()
    
    #class weights
    X_train_all = df.drop('win', axis=1).copy()
    y_train_all = df['win'].copy()
    class_weight_all = compute_class_weight('balanced', y_train_all.unique(), y_train_all)
    weight_dict_all = {y_: weight for y_, weight in zip(y_train_all.unique(), class_weight_all)}
    y_weights_all = y_train_all.map(weight_dict_all)
        
    st.title("Kickstarter campaign predictor")
    st.markdown("""Enter the details your next kickstarter campaign to the left. Then when your ready give your campaign 
                a name to predict the likelihood that it will succeed.""")
    
    # create new kickstarter campaign
    st.sidebar.markdown('Input campaign details')
    cats = df[['main_category', 'category']].drop_duplicates()
    name = st.text_input("Name of your project")
    mcat = st.sidebar.selectbox('Main Category', cats.main_category.unique())
    cat = st.sidebar.selectbox('Sub Category', cats.loc[cats['main_category'] == mcat]['category'].unique())
    goal = st.sidebar.number_input("Set Funding goal",min_value=100, step=100, value=100)
    time = st.sidebar.number_input("Campaign runtime 1-60 days", min_value=1, max_value=60, value=30)

    if name != '':
        name_char_count = len(name)
        name_word_count = len(name.split(' '))
        
        goal_group = df.loc[df['goal_adj'] == goal]['goal_group'].unique()[0]
        test = pd.DataFrame({
        'category' : cat,
        'main_category' : mcat,
        'goal_adj' : goal,
        'time' : time,
        'goal_group' : goal_group,
        'name_charc' : name_char_count,
        'name_wc' : name_word_count,
        'name_the' : np.where('The' in  name , 1, 0),
        'cat_backers' : df.loc[df['category'] == cat].iloc[0][9],
        'cat_goal' : df.loc[df['category'] == cat].iloc[0][10],
        'cat_pledged' : df.loc[df['category'] == cat].iloc[0][11],
        'cat_win_rate' : df.loc[df['category'] == cat].iloc[0][12],
        'mcat_backers' : df.loc[df['main_category'] == mcat].iloc[0][13],
        'mcat_goal' : df.loc[df['main_category'] == mcat].iloc[0][14],
        'mcat_pledged' : df.loc[df['main_category'] == mcat].iloc[0][15],
        'mcat_win_rate' : df.loc[df['main_category'] == mcat].iloc[0][16],
        'goalgroup_backers' : df.loc[df['goal_group'] == goal_group].iloc[0][17],
        'goalgroup_goal' : df.loc[df['goal_group'] == goal_group].iloc[0][18],
        'goalgroup_pledged' : df.loc[df['goal_group'] == goal_group].iloc[0][19],
        'goalgroup_win_rate' : df.loc[df['goal_group'] == goal_group].iloc[0][20],
        }, index=[0])
        
        success = k_mod.predict_proba(test)[0][1]
        success_form = "{0:.2f}%".format(success * 100)
        st.header(f"This campaign has a {success_form} chance in succeeding")

        
        # reasons
        def roundup(x):
            return int(np.around(x))
        
        reason_ch_cats_goal = df.groupby(['main_category', 'category', 'win'])['goal_adj']\
                                    .mean().reset_index()
        reason_ch_cats_goal = reason_ch_cats_goal.loc[(reason_ch_cats_goal['win'] == 1) &\
                            (reason_ch_cats_goal['main_category'] == mcat) & (reason_ch_cats_goal['category'] == cat)]
        mean_goal = roundup(reason_ch_cats_goal['goal_adj'].values)    

        st.subheader("**Campaign name**")
        if ('The' not in name) | (name_word_count < 5):
            st.markdown("It is usually overlooked but the name of your campaign is important in it's success.")
            if name.split(' ')[0] != 'The':
                st.markdown("Try prefacing your campaign name with a **The**")
            if name_word_count < 5:
                words_add = (5 - name_word_count)     
                if name_char_count < 60:
                    if words_add == 1:
                        st.markdown(f"Longer names are more descriptive and usually do better than those with shorter names.\
                                Try adding juat one more word to your campaign name. Names with 5 or more words do the best.\
                                    However, too many characters in your name will cause the reader to disengage.\
                                        Your current character count is {name_char_count} which is below the recommended limit of 60")
                    else:
                        st.markdown(f"Longer names are more descriptive and usually do better than those with shorter names.\
                                Try adding {words_add} words to your campaign name. Names with 5 or more words do the best.\
                                    However, too many characters in your name will cause the reader to disengage.\
                                        Your current character count is {name_char_count} which is below the recommended limit of 60")
                else:
                    st.markdown("""Interesting name choice! Not only does this name exceed the recommended character limit
                                that can cause disengagement but it is also under 5 words which is considered too simplistic.
                                Consider a new name for this project that is both descriptive yet short.""")   
        else:
            st.markdown("""It is usually overlooked but the name of your campaign is important in its success.
                        Your name scores high in this model, nice choice.""")
            
        st.subheader("**Funding goal**")
        if goal > mean_goal:
            st.markdown(f"""Your funding goal of ${goal} could be too ambitious. 
                        Other successful campaigns for {cat} usually have a goal set to ${mean_goal}""")
        else:
            st.markdown(f"Your funding goal is within the expected range for {cat}")
        
        st.subheader("**Runtime**")
        if time > 30:
            st.markdown(f"Your campaign runtime of {time}-days is over the recommended.\
                        Campaigns with a runtime of 30-days or less have higher success rates,\
                            and create a helpful sense of urgency around your project.")
        elif time < 10:
            st.markdown(f"""{time} days may not be enough to achieve your goal. To increase your success
                        try keeping your campaign at a 30-day runtime. If you need a quicker campaign 22-days is a good start.
                            """)
        else:
            st.markdown("Your runtime is within the recommended range for a successful campaign.")
            
        # Model score
        st.subheader("**Model Score**")    
        score = np.round(f1_score(df['win'], k_mod.predict(df.drop('win',axis=1)),sample_weight=y_weights_all),2)
        st.markdown(f"F1 Score: {score}")

        # Model params
        st.markdown("""
        **How this model works:** This model uses XGBoost Classification, an advanced regularized gradient boosting tool. 
        Gradient boosting iterates through a series of decision trees each time reducing log loss between the predicted and actual outcomes.
        This model has been fine-tuned resulting in the following parameters:
        """)
        k_mod.named_steps['xgbclassifier']
                    
        
if page == 'Spotify algorithm':
    
    st.title('Compose a number one song')
    st.markdown("""By analysing Spotify's algorithm data we can determine the musical parameters that make a popular song.
                Input your musical paramters of your song to the left, then give your song a name to predict its popularity.""")
    
    #@st.cache()
    def load_spot():
        s_mod = load("spot_project.joblib.dat")
        return s_mod
    s_mod = load_spot()
    
    keys = {0: 'C',1:'C#',2:'D',3:'D#',4:'E',5:'F',6:'F#',7:'G',8:'G#',9:'A',10:'A#',11:'B'}
    inv_keys = {v: k for k, v in keys.items()}
    modes = {0: 'Minor', 1: 'Major'}
    inv_modes = {v: k for k, v in modes.items()}
    
    genres = [   'alternative metal',
                 'alternative rock',
                 'bebop',
                 'big band',
                 'blues',
                 'blues rock',
                 'classic bollywood',
                 'classic rock',
                 'classic soul',
                 'classical',
                 'classical piano',
                 'contemporary country',
                 'cool jazz',
                 'country',
                 'country rock',
                 'dance pop',
                 'dance rock',
                 'disco',
                 'easy listening'
                 'electro',
                 'electronica',
                 'folk',
                 'folk rock',
                 'funk',
                 'gangster rap',
                 'glam rock',
                 'hard bop',
                 'hard rock',
                 'heartland rock',
                 'hip hop',
                 'historic orchestral performance'
                 'house',
                 'jazz',
                 'jazz blues',
                 'jazz funk',
                 'jazz fusion',
                 'jazz piano',
                 'latin',
                 'latin pop',
                 'lounge',
                 'metal',
                 'modern rock',
                 'motown',
                 'new wave',
                 'new wave pop',
                 'nu metal',
                 'opera',
                 'orchestra',
                 'pop',
                 'pop dance',
                 'pop rap',
                 'pop rock',
                 'post-teen pop',
                 'progressive house',
                 'psychedelic rock',
                 'punk',
                 'r&b',
                 'rap',
                 'rock',
                 'rock-and-roll',
                 'soft rock',
                 'soul',
                 'soul jazz',
                 'swing',
                 'symphonic rock',
                 'tango',
                 'traditional folk',
                 'trance',
                 'trap',
                 'uplifting trance']
    
    
    name = st.text_input("Name of your song") # doesn't input into model
    st.sidebar.markdown('Input song parameters')
    genre = st.sidebar.selectbox('Genre', genres, index=46) 
    tempo = st.sidebar.slider('Tempo', min_value=0.0, max_value=250.0, value=120.0)
    key = st.sidebar.selectbox('Key', list(keys.values()), index=1)
    mode = st.sidebar.selectbox('Mode', ['Minor', 'Major'], index=0)
    explicit = st.sidebar.selectbox('Explicit concent', ['Yes', 'No'], index=0)
    time = st.sidebar.slider('Song duration', min_value=time(0,0,30), max_value=time(0,10,0),
                     step=timedelta(seconds=10), format='mm:ss', value=time(0,4,0))
    
    acoustic = st.sidebar.slider('Acousticness', min_value=0.0, max_value=1.0, value=0.3)
    speech = st.sidebar.slider('Speechiness', min_value=0.0, max_value=1.0, value=0.15)
    instruments = st.sidebar.slider('Instrumentalness', min_value=0.0, max_value=1.0, value=0.0)
    dance = st.sidebar.slider('Danceability', min_value=0.0, max_value=1.0, value=0.75)
    energy = st.sidebar.slider('Energy', min_value=0.0, max_value=1.0, value=0.6)
    live = st.sidebar.slider('Liveness', min_value=0.0, max_value=1.0, value=0.15)
    loudness = st.sidebar.slider('Loudness', min_value=-60.0, max_value=4.0, value=-7.0)
    valence = st.sidebar.slider('Valence', min_value=0.0, max_value=1.0, value=0.45)
    
    if name != '':
        # Prediction DataFrame
        df_pred = pd.DataFrame({
            'acousticness'    : acoustic, #float
            'danceability'    : dance, #float
            'duration_ms'     : ((time.minute * 60) + time.second) * 1000, # int
            'energy'          : energy, #float
            'explicit'        : explicit,# int
            'instrumentalness': instruments, #float
            'key'             : None, # int
            'liveness'        : live, # float
            'loudness'        : loudness, #float
            'mode'            : None, # int
            'speechiness'     : speech, #float
            'tempo'           : tempo, #float
            'valence'         : valence, #float
            'genres'          : genre, 
            'key_str'         : key,
            'mode_str'        : mode,
            'key_mode'        : None,
            'duration_min'    : time.minute,
            'duration_sec'    : time.second,
            'duration_mm:ss'  : str(time.minute) + ":" + str(time.second)},
            index=[0])
            
        df_pred['key'] = df_pred['key_str'].replace((inv_keys))
        df_pred['mode'] = df_pred['mode_str'].replace((inv_modes))
        df_pred['mode'] = df_pred['mode'].astype(int)
        df_pred['key_mode'] = df_pred['key_str'] + " " + df_pred['mode_str']
        df_pred['duration_min'] = df_pred['duration_min'].astype('str')
        df_pred['duration_sec'] = df_pred['duration_sec'].astype('str')
        df_pred['acousticness'] = df_pred['acousticness'].astype(float)
        df_pred['danceability'] = df_pred['danceability'].astype(float)
        df_pred['duration_ms'] = df_pred['duration_ms'].astype(int)
        df_pred['energy'] = df_pred['energy'].astype(float)
        df_pred['explicit'] = np.where(df_pred['explicit']== "No", 0,1)
        df_pred['instrumentalness'] = df_pred['instrumentalness'].astype(float)
        df_pred['speechiness'] = df_pred['speechiness'].astype(float)
        df_pred['tempo'] = df_pred['tempo'].astype(float)
        df_pred['valence'] = df_pred['valence'].astype(float)
        
        # Predictions
        popularrity = s_mod.predict(df_pred)[0]
        popularrity_form = "{0:.2f}".format(popularrity /10)  
        st.header(f"{name} has a popularity score of {popularrity_form} out of 10")

        df1 = pd.read_csv('https://raw.githubusercontent.com/jcolethornton/datasets/main/spotify_summary_duration.csv')
        df2 = pd.read_csv('https://raw.githubusercontent.com/jcolethornton/datasets/main/spotify_summary_genre.csv')
        df3 = pd.read_csv('https://raw.githubusercontent.com/jcolethornton/datasets/main/spotify_summary_keys.csv')
        df4 = pd.read_csv('https://raw.githubusercontent.com/jcolethornton/datasets/main/spotify_summary_key-genre.csv')

        # Genre
        df2 = df2.loc[df2['genres'].isin(genres)].sort_values(by='popularity')
        df2['selected'] = np.where(
            df2['genres'] == genre, 'y','n')
        genre_chart = px.bar(
            df2, x='popularity', y='genres', color='selected',
            title=f'{genre} popularity compared to other genres')

        genre_chart.layout.update(
            showlegend=False)
        genre_chart.update_layout(
            yaxis_categoryorder = 'total ascending')
        
        st.write(genre_chart)

        # Keys
        df3 = df3.sort_values(by='popularity')
        df3['selected'] = np.where(
            df3['key_mode'] == df_pred['key_mode'].iloc[0], 'y','n')
        key_chart = px.bar(
            df3, x='popularity', y='key_mode', color='selected',
            title=f"{df_pred['key_mode'].iloc[0]} popularity")

        key_chart.layout.update(
            showlegend=False)
        key_chart.update_layout(
            yaxis_categoryorder = 'category ascending')
        
        st.write(key_chart)

        # Genre keys
        df4 = df4.loc[df4['genres'].isin(genres)]
        fig = go.Figure(data=[go.Mesh3d(x=(df4.genres),
                    y=(df4.key_mode),
                    z=(df4.popularity),
                    opacity=0.5,
                    color='rgba(244,22,100,0.6)'
                    )])

        fig.update_layout(
        scene = dict(
            xaxis = dict(nticks=24,),
            yaxis = dict(nticks=24,),
            zaxis = dict(nticks=50, range=[0,100],),),
        width=700, height=700,
        margin=dict(r=0, l=0, b=0, t=0))

        st.write(fig)

        # Model params
        st.markdown("""
        **How this model works:** This model uses XGBoost Regression, an advanced regularized gradient boosting tool. 
        Gradient boosting iterates through a series of decision trees each time reducing log loss between the predicted and actual outcomes.
        This model has been fine-tuned resulting in the following parameters:
        """)
        s_mod.named_steps['xgbregressor']

if page == 'Fuel efficiency':

    st.sidebar.markdown('Input vehicle specs:')

    st.title("Fuel efficiency")
    st.markdown('Using regularized gradient boosting this model works on three regression problems to predict:')
    st.markdown("**Annual Fuel Costs | Fuel economy | CO2 emmissions**")
    st.markdown('Input vehicle engine parameters on the left hand sidebar.')

    @st.cache
    def load_data():
        df = pd.read_csv('https://raw.githubusercontent.com/jcolethornton/datasets/main/database.csv',
                        usecols=['Class', 'Drive',
                                'Engine Cylinders', 'Engine Displacement', 'Turbocharger',
                                'Supercharger', 'Fuel Type 1', 'Annual Fuel Cost (FT1)', 'Combined MPG (FT1)',
                                'Tailpipe CO2 in Grams/Mile (FT1)'])
        return df 
    df = load_data()

    # Remove special purpose class
    df = df.loc[~(df['Class'].str[:7]=='Special')]

    # Class of vehicle
    vehicle_class = {
    'Compact Cars': 'Compact car',
    'Large Cars': 'Large car',
    'Midsize Cars': 'Midsize car',
    'Midsize Station Wagons': 'Station Wagon - midsize',
    'Midsize-Large Station Wagons': 'Station Wagon - large',
    'Minicompact Cars': 'Compact car',
    'Minivan - 2WD': 'Minivan',
    'Minivan - 4WD': 'Minivan',
    'Small Pickup Trucks': 'Ute - small',
    'Small Pickup Trucks 2WD': 'Ute - small',
    'Small Pickup Trucks 4WD': 'Ute - small',
    'Small Sport Utility Vehicle 2WD': 'SUV - small',
    'Small Sport Utility Vehicle 4WD': 'SUV - small',
    'Small Station Wagons': 'Station Wagon - small',
    'Sport Utility Vehicle - 2WD': 'SUV',
    'Sport Utility Vehicle - 4WD': 'SUV',
    'Standard Pickup Trucks': 'Ute',
    'Standard Pickup Trucks 2WD': 'Ute',
    'Standard Pickup Trucks 4WD': 'Ute',
    'Standard Pickup Trucks/2wd': 'Ute',
    'Standard Sport Utility Vehicle 2WD': 'SUV',
    'Standard Sport Utility Vehicle 4WD': 'SUV',
    'Subcompact Cars': 'Compact cars',
    'Two Seaters' : 'Sport car',
    'Vans': 'Van',
    'Vans Passenger': 'Van',
    'Vans, Cargo Type': 'Van',
    'Vans, Passenger Type': 'Van'}

    df['Class'] = df['Class'].replace((vehicle_class))
    df = df.loc[~pd.isnull(df.Drive)]

    # rename columns to easier use
    df = df.rename({'Tailpipe CO2 in Grams/Mile (FT1)':'CO2 Grams/Mile',
                    'Combined MPG (FT1)': 'MPG',
                    'Fuel Type 1': 'Fueled by',
                    'Annual Fuel Cost (FT1)': 'Annual Fuel Cost'}, axis=1)

    # cylinders and displacment should only be in int format
    df['Engine Cylinders'] = df['Engine Cylinders'].fillna(0)
    df['Engine Cylinders'] = df['Engine Cylinders'].astype(int)
    df['Engine Displacement'] = df['Engine Displacement'].fillna(0)
    df['Engine Displacement'] = df['Engine Displacement'].astype(int)

    # Merge air intake systems
    df['Turbocharger'] = df['Turbocharger'].fillna("F")
    df['Supercharger'] = np.where(df['Supercharger'] == "S", "T", "F")
    df['Turbocharged'] = np.where((df['Turbocharger'] == "T") | (df['Supercharger'] == "T"), "Yes", "No")
    df = df.drop(['Turbocharger', 'Supercharger'], axis=1)

    # Metric conversions
    df['KMPL'] = df['MPG'] / 2.352
    df['CO2 Grams/KM'] = df['CO2 Grams/Mile'] / 1.609

    conversion = st.sidebar.selectbox('Imperial/Metric', ['Imperial', 'Metric'], index=1)

    # cut dataset to desired cols
    if conversion =='Imperial':
        X_cols = ['Fueled by', 'Class', 'Drive',
                'Engine Cylinders', 'Engine Displacement', 'Turbocharged']
        y_cols = ['Annual Fuel Cost', 'MPG', 'CO2 Grams/Mile']
    else:
        X_cols = ['Fueled by', 'Class', 'Drive',
                'Engine Cylinders', 'Engine Displacement', 'Turbocharged']
        y_cols = ['Annual Fuel Cost', 'KMPL', 'CO2 Grams/KM']
    
    df_2 = df.copy()

    # Prediction
    predict = st.sidebar.selectbox('Predict..', y_cols)
    
    drivetrain = sorted(df['Drive'].unique())
    any_drive  = ['Any']
    drivetrain = any_drive + drivetrain 

    v_class = st.sidebar.selectbox('Vehicle class', df['Class'].unique())
    drive = st.sidebar.selectbox('Drivetrain', drivetrain)

    # Drivetrain CC and cylinder selection
    if drive != 'Any':
        engine_c = st.sidebar.selectbox('Number of cylinders', sorted(df.loc[(df['Drive'] == drive) &\
            (df['Class'] == v_class)]['Engine Cylinders'].unique()), index=4)

        engine_d = st.sidebar.selectbox('CC displacement', sorted(df.loc[(df['Drive'] == drive) &\
             (df['Engine Cylinders'] == engine_c) &\
                 (df['Class'] == v_class)]['Engine Displacement'].unique()),index=2)
        
        df = df.loc[(df['Engine Cylinders'] == engine_c) &\
             (df['Engine Displacement'] == engine_d) &\
                  (df['Drive'] == drive) &\
                      (df['Class'] == v_class)]

    else:
        engine_c = st.sidebar.selectbox('Number of cylinders', sorted(df.loc[(df['Class'] == v_class)]['Engine Cylinders'].unique()),index=4)
        engine_d = st.sidebar.selectbox('CC displacement', sorted(df.loc[(df['Engine Cylinders'] == engine_c) &\
            (df['Class'] == v_class)]['Engine Displacement'].unique()),index=0)
        df = df.loc[(df['Engine Cylinders'] == engine_c) &\
             (df['Engine Displacement'] == engine_d) &\
                 (df['Class'] == v_class)]

    if len(df['Turbocharged'].unique()) == 2:
        turbo = st.sidebar.selectbox('Turbo', ['Yes', 'No'], index=1)
        df = df.loc[df['Turbocharged'] == turbo]
    else:
        turbo = st.sidebar.selectbox('Turbo', ["Not availble"])

    fuel_type = st.sidebar.selectbox('Fuel type', sorted(df['Fueled by'].unique())) 

    st.markdown(f'**Currently predicting:** {predict} for a {v_class} with\
                 {engine_c} cyclinders, {engine_d} CC displacement and {drive} drivetrain')
    
    # ml sets
    X = df_2[X_cols]
    y = df_2[predict]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=27)

    st.markdown("**Tune model parameters:**")
    estimators = st.slider('Number of esimators', min_value=10, max_value=1000, value=750)
    eta = st.slider('Learning rate', min_value=0.001, max_value=0.4, value=0.05)
    max_d = st.slider('Max depth', min_value=1, max_value=10, value=2)
    sample_level = st.slider('Sample by level', min_value=0.1, max_value=1.0, value=0.3)
    sample_tree = st.slider('Sample by tree', min_value=0.1, max_value=1.0, value=0.5)
    
    # ml parameters
    mod = xgb.XGBRegressor(max_depth=max_d,learning_rate=eta,n_estimators=estimators
                        , colsample_bylevel=sample_level, colsample_bytree=sample_tree)
    pipe = make_pipeline(OrdinalEncoder(), mod)
    
    pipe.fit(X_train, y_train)
    score = np.around(pipe.score(X_test, y_test),2)


    # fit on all
    pipe.fit(X, y)

    # prediction input
    df_pred = pd.DataFrame({
    'Fueled by': fuel_type,
    'Class': v_class,
    'Drive': drive,
    'Engine Cylinders': engine_c,
    'Engine Displacement': engine_d,
    'Turbocharged': turbo
    }, index=['Results'])

    result = np.around(pipe.predict(df_pred),2)
    if predict == 'Annual Fuel Cost':
        result = '$'+ str(result[0])
    else:
        result = str(result[0])
    
    # score
    pred_results = pd.DataFrame()
    pred_results['true'] = y
    pred_results['predicted'] = pipe.predict(X)
    pred_chart = px.scatter(pred_results, x='true', y='predicted', trendline='ols', trendline_color_override="red",
                              title=f'Model Score {score} R2')
    # results
    df_pred[predict] = result
    st.header(f'{predict}: {result}')
    st.table(df_pred)
    st.header(f'Score: {score} R2')
    st.write(pred_chart)
    
    # feature importance 
    feature_names = pipe.named_steps["ordinalencoder"].get_feature_names()
    feat = pd.DataFrame(
        {'Feature': X.columns,
         'Impact': pipe.steps[1][1].feature_importances_}).sort_values(by='Impact',
                                                                      ascending=False)
    feat = feat.head(5)
    feat['Impact'] = pd.Series(["{0:.2f}%".format(val * 100) for val in feat['Impact']], index=feat.index)
    
    st.subheader(f"Top 5 features used in predicting {predict}")
    st.table(feat)
    
    st.subheader(f"Correlation analysis using partial dependence")

    feature1 = st.selectbox('Cylinders / Displacment', ['Engine Cylinders','Engine Displacement'])
    core_feat = feat.loc[~(feat['Feature'].isin(['Engine Cylinders','Engine Displacement']))]
    feature2 = st.selectbox('Select second feature', sorted(core_feat['Feature'].unique()))

    # analyize feature datatypes and length
    OE1 = pd.DataFrame({
    feature1 : X_train[feature1],
    'Encoder' : pipe[0].transform(X_train)[feature1]})
    OE1 = OE1.drop_duplicates().reset_index().drop('index', axis=1)
    datatype1 = np.issubdtype(OE1[feature1].dtype, np.number)
    OE2 = pd.DataFrame({
    feature2 : X_train[feature2],
    'Encoder' : pipe[0].transform(X_train)[feature2]})
    OE2 = OE2.drop_duplicates().reset_index().drop('index', axis=1)
    datatype2 = np.issubdtype(OE2[feature2].dtype, np.number)

    # set gridpoints
    gridpoints = []
    gridtype   = []
    if datatype1:
        gridpoints.append(8)
        gridtype.append('percentile')
    else:
        gridpoints.append(len(OE1))
        gridtype.append('equal')
    if datatype2:
        gridpoints.append(8)
        gridtype.append('percentile')
    else:
        gridpoints.append(len(OE2))
        gridtype.append('equal')
    
    gbm_inter = pdp.pdp_interact(
            model=pipe[1], dataset=pipe[0].transform(X_train), model_features=pipe[0].get_feature_names(), 
            features=[feature1, feature2],num_grid_points=gridpoints, grid_types=gridtype)
    fig, axes = pdp.pdp_interact_plot(
        gbm_inter, [feature1, feature2], x_quantile=True, plot_type='grid', plot_pdp=True)
    axes['pdp_inter_ax']['_pdp_inter_ax'].set_yticklabels(OE2[feature2].tolist())
    
    st.write(fig)

    # Model params
    st.markdown("""
    **How this model works:** This model uses XGBoost Regression, an advanced regularized gradient boosting tool. 
    Gradient boosting iterates through a series of decision trees each time reducing log loss between the predicted and actual outcomes.
    This model is running live each time a parameter is changed. 
    Current parameters used in the model are:
    """)
    #st.markdown(pipe.named_steps)
    pipe.named_steps['xgbregressor']
    






    
    
    
    
    
        
        
        
        
        
