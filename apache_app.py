import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')


# App title
st.title("CRIMES AGAINST WOMEN IN INDIA")

data_url = ('https://raw.githubusercontent.com/apacheteam/Crime-Against-Women-PP22-C623-APACHE-TEAM/master/crimes_against_women_2001-2014.csv')
#data_url = (r'C:\Users\Tomi\datasets\crimes_against_women_2001-2014.csv')


def load_data():
    data = pd.read_csv(data_url)
    return data

# Load data
dataset = load_data()

# View dataset checkbox
if st.checkbox('View raw data'):
    # Inspect the raw data
    st.subheader('Dataset')
    st.write(dataset)
    st.write(dataset.shape)

# Polynomial regression function
def polynomial_reg(degree, x_train, y_train, x_test):
    poly_feat = PolynomialFeatures(degree)
    X_poly = poly_feat.fit_transform(x_train)
    x_test = poly_feat.fit_transform(x_test)

    # create and fit the polynomial regression model
    model = LinearRegression()
    poly_model = model.fit(X_poly, y_train)
    pred = poly_model.predict(x_test)
    return pred


# prediction
unique_states = dataset['state_ut'].unique()


def state_reg(df, state, crime, year):
    state_df = df[df['state_ut'] == state]
    # state_df = state_df.groupby('year', as_index=False)['rape', 'kidnapping_and_abduction', 'dowry_deaths',
    #                                                     'assault_on_women', 'insult_to_modesty',
    #                                                     'cruelty_by_husband_or_relatives',
    #                                                     'importation_of_girls'].sum()
    # Assuming state_df is your DataFrame and 'rap', 'kid', 'dowry_deaths' are the columns you want to select
    state_df = state_df.groupby('year', as_index=False)[['rape', 'kidnapping_and_abduction', 'dowry_deaths','assault_on_women', 'insult_to_modesty',
                                                         'cruelty_by_husband_or_relatives',
                                                         'importation_of_girls']].sum()
    state_df.fillna(0, inplace=True)

    state_df["t"] = np.arange(1, len(state_df) + 1)

    state_df["t_square"] = state_df["t"] * state_df["t"]
    state_df['log_crime'] = np.log(state_df[crime])
    x_train = state_df[['t', 't_square']]
    y_train = state_df[crime]
    if year <= 2023:
        yeartopred = state_df[['t', 't_square']][state_df['year'] == year]
    elif year == 2024:
        yeartopred1 = 15
        yeartopred2 = 15 ** 2
        yeartopred = np.array([[yeartopred1, yeartopred2]])
    elif year == 2025:
        yeartopred3 = 16
        yeartopred4 = 16 ** 2
        yeartopred = np.array([[yeartopred3, yeartopred4]])
    elif year == 2026:
        yeartopred3 = 17
        yeartopred4 = 17 ** 2
        yeartopred = np.array([[yeartopred3, yeartopred4]])

    # Quad = smf.ols(str(crime) + '~ t+ t_square', data = state_df).fit()
    pred_Quad = polynomial_reg(2, x_train, y_train, yeartopred)
    if pred_Quad <= 0:
        return 0
    else:
        return int(np.round(pred_Quad, 0))

# Prediction selectbox
# select state
state = st.sidebar.selectbox(
    'Select a state you\'re interested in: ',
    (unique_states)
)

# select crime
crime = st.sidebar.selectbox(
    'Crime: ', ('rape', 'kidnapping_and_abduction', 'dowry_deaths', 'assault_on_women',
    'insult_to_modesty', 'cruelty_by_husband_or_relatives', 'importation_of_girls')
)

# select year
year = st.sidebar.selectbox(
    'Year:', (2023, 2024, 2025,2026)
)

'Predict number of ', crime, 'cases in the year ', str(year), 'at ', state

# on_click of the PREDICT button
if st.button('PREDICT'):
    st.write('apache predicts', str(state_reg(dataset, state, crime, year)), crime, 'cases in ', state,
             'by the year', str(year))


