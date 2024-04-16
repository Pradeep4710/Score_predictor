import streamlit as st
import pickle
import pandas as pd
import numpy as np
import xgboost
from xgboost import XGBRegressor
import sklearn 

pipe = pickle.load(open('Test_Predictor.pkl','rb'))

teams = ['Australia', 'England', 'South Africa', 'Pakistan', 'West Indies', 'Bangladesh', 'Zimbabwe', 'New Zealand', 'India', 'Sri Lanka', 'ICC World XI', 'Ireland', 'Afghanistan']


cities = ['Galle International Stadium', "Lord's", 'Sydney Cricket Ground', 'Newlands', 'Basin Reserve', 'Adelaide Oval', 'Melbourne Cricket Ground', 'Sinhalese Sports Club Ground', 'Sheikh Zayed Stadium', 'Zahur Ahmed Chowdhury Stadium', 'Dubai International Cricket Stadium', 'Harare Sports Club', 'Kennington Oval', 'SuperSport Park', 'Brisbane Cricket Ground, Woolloongabba', 'Seddon Park', 'Shere Bangla National Stadium', 'Headingley', 'P Sara Oval', 'Kingsmead', 'Old Trafford', 'Edgbaston', 'Trent Bridge', 'Kensington Oval, Bridgetown', 'Western Australia Cricket Association Ground', 'Sabina Park, Kingston', 'Queens Sports Club', 'Eden Gardens', 'Shere Bangla National Stadium, Mirpur', 'Pallekele International Cricket Stadium', "St George's Park", 'The Wanderers Stadium', 'Hagley Oval', 'University Oval', 'Bellerive Oval', "Queen's Park Oval, Port of Spain", 'New Wanderers Stadium', 'Vidarbha Cricket Association Stadium, Jamtha', 'Sharjah Cricket Stadium', 'Rawalpindi Cricket Stadium', 'National Stadium', 'Eden Park', 'The Rose Bowl', 'Punjab Cricket Association Stadium, Mohali', 'Rajiv Gandhi International Stadium, Uppal', 'MA Chidambaram Stadium, Chepauk', 'Windsor Park, Roseau', 'Sir Vivian Richards Stadium, North Sound', 'National Stadium, Karachi', 'M Chinnaswamy Stadium', 'Sardar Patel Stadium, Motera', 'Wankhede Stadium', 'Basin Reserve, Wellington', 'McLean Park', 'Riverside Ground', 'Feroz Shah Kotla', 'Multan Cricket Stadium', 'Green Park', 'Hagley Oval, Christchurch', 'Sheikh Abu Naser Stadium', 'Warner Park, Basseterre', "Lord's, London", 'Zahur Ahmed Chowdhury Stadium, Chattogram', 'Perth Stadium', 'Trent Bridge, Nottingham', 'MA Chidambaram Stadium', 'W.A.C.A. Ground', 'Sophia Gardens', 'Sir Vivian Richards Stadium', 'Headingley, Leeds', 'Kennington Oval, London', 'Edgbaston, Birmingham', 'Queens Sports Club, Bulawayo', 'Gaddafi Stadium', "Antigua Recreation Ground, St John's", 'Antigua Recreation Ground', 'Bay Oval', 'Iqbal Stadium', 'Asgiriya Stadium', 'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium', 'Bankstown Oval, Sydney', 'Gaddafi Stadium, Lahore', 'Arun Jaitley Stadium', 'Providence Stadium', 'Arnos Vale Ground, Kingstown', 'Daren Sammy National Cricket Stadium, Gros Islet, St Lucia', 'County Ground, New Road, Worcester', 'Brisbane Cricket Ground', 'Saurashtra Cricket Association Stadium', 'Daren Sammy National Cricket Stadium, Gros Islet', 'Brisbane Cricket Ground, Woolloongabba, Brisbane', "National Cricket Stadium, St George's", "Sir Paul Getty's Ground", 'Sir Vivian Richards Stadium, North Sound, Antigua', 'JSCA International Stadium Complex', 'Punjab Cricket Association IS Bindra Stadium, Mohali', 'Bay Oval, Mount Maunganui', 'Chittagong Divisional Stadium', 'M.Chinnaswamy Stadium', 'Holkar Cricket Stadium', 'Kingsmead, Durban', 'Maharashtra Cricket Association Stadium', 'The Wanderers Stadium, Johannesburg', 'Sabina Park, Kingston, Jamaica', 'AMI Stadium', 'Newlands, Cape Town', 'Khan Shaheb Osman Ali Stadium', 'R Premadasa Stadium', 'Sir Vivian Richards Stadium, Antigua', 'SuperSport Park, Centurion']

st.title('Cricket Score Predictor')

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select batting team',sorted(teams))
with col2:
    bowling_team = st.selectbox('Select bowling team', sorted(teams))

city = st.selectbox('Select Venue',sorted(cities))

col3,col4,col5 = st.columns(3)

with col3:
    current_score = st.number_input('Current Score')
with col4:
    overs = st.number_input('Overs done(works for over>5)')
with col5:
    wickets = st.number_input('Wickets left')

last_five = st.number_input('Runs scored in last 5 overs')

if st.button('Predict Score'):
    balls_left = 120 - (overs*6)
    wickets_left = 10 -wickets
    crr = current_score/overs

    input_df = pd.DataFrame(
     {'batting_team': [batting_team], 'bowling_team': [bowling_team],'venue':city, 'cur_score': [current_score],'balls_left': [balls_left], 'wickets_left': [wickets], 'crr': [crr], 'last_five': [last_five]})
    result = pipe.predict(input_df)
    st.header("Predicted Score - " + str(int(result[0])))


