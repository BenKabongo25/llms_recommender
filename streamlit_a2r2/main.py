# Ben Kabongo - MIA Paris-Saclay x Onepoint
# NLP & RecSys - July 2024

# A2R2 streamlit app: Main

import streamlit as st

DATASETS = [
    "TripAdvisor",
    "BeerAdvocate",
    "RateBeer",
    "Other"
]

def draw_stars(rating):
    return ":star:" * int(rating), rating

st.subheader(":red[Attention] and :blue[Aspect]-based Rating and Review Prediction")
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    dataset = st.selectbox("Dataset:", DATASETS)

if dataset == "TripAdvisor":
    aspects = ["Service", "Cleanliness", "Value", "Sleep quality", "Rooms", "Location"]

elif dataset == "BeerAdvocate":
    aspects = []

elif dataset == "RateBeer":
    aspects = []

else:
    pass

with col2:
    user = st.selectbox("User:", ["User1", "User2", "User3"])
with col3:
    item = st.selectbox("Item:", ["Item1", "Item2", "Item3"])

# Peekassoh,100504,5.0,",“Unparalled service in the centre of the city.”,1279324800.0,
aspects_importance = [0.2, 0.2, 0.1, 0.2, 0.2, 0.1]
aspects_rating = [5.0, 5.0, 4.0, 5.0, 5.0, 5.0]
overall_rating = 5.0
review = """It is pretty evident from the get go that the staff at this hotel want to make 
your visit a memorable one. From check in to check out we were treated in a warm and friendly 
manner with people going out of their way to cater to our every need. That in itself makes me 
want to return. The hotel is within walking distance to the Heart of the City. It is extremely 
quiet considering the location. The rooms are spacious and super comfortable, with well 
appointed large bathrooms. We noticed refillable bottles of shampoo and soap, which indicated 
their efforts to minimize waste. They offer complementary coffee and pastries in the morning 
in the main lobby. In the evening they host a complementary wine tasting event which allows 
you to mingle with other guests and share dining and/or shopping tips. Very pleasant. The 
hotel restaurant is not outstanding but certainly adequate for a casual meal or Happy Hour. 
We enjoyed our meal there but if you are looking for a gourmet experience there are better 
choices close by. We will certainly make this our regular Seattle hotel based on location, 
comfort and service."""


st.divider()
with st.container():
    col1, col2, col3 = st.columns(3, vertical_alignment="top")
    with col1:
        st.write("**Aspects**")
    with col2:
        st.write("**Importance**")
    with col3:
        st.write("**Rating**")


for i, aspect in enumerate(aspects):
    with st.container():
        col1, col2, col3 = st.columns(3, vertical_alignment="top")
        with col1:
            st.write(aspect)
        with col2:
            st.progress(aspects_importance[i])
        with col3:
            st.write(*draw_stars(aspects_rating[i]))

st.divider()
with st.container():
    col1, col2 = st.columns([0.2, 0.8], vertical_alignment="top")
    with col1:
        st.write("**Overall**")
    with col2:
        st.write(*draw_stars(overall_rating))

with st.container():
    col1, col2 = st.columns([0.2, 0.8], vertical_alignment="top")
    with col1:
        st.write("**Review**")
    with col2:
        st.write(review)
