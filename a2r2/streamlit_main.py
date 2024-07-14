# Ben Kabongo - MIA Paris-Saclay x Onepoint
# NLP & RecSys - July 2024

# A2R2 streamlit app: Main

import os
import pandas as pd
import streamlit as st
import torch

from models import A2R2v1
from data import get_vocabularies


def draw_stars(rating):
    return rating, ":star:" * int(rating)

def write_ids(vocab):
    return list(map(lambda x: f"{x[0]}: {x[1]}", vocab._ids2elements.items()))

def get_id(app_id):
    return int(app_id.split(":")[0])

st.set_page_config(page_title="A2R2", page_icon=":star:")

class Args:
    def __init__(self):
        pass

args = Args()

DATASETS = [
    "TripAdvisor",
    #"BeerAdvocate",
    "RateBeer"
]

args.base_dir = "Datasets\\processed"

st.subheader(":red[Attention] and :blue[Aspect]-based Rating and Review Prediction")
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    dataset = st.selectbox("Dataset:", DATASETS)

args.dataset_dir = os.path.join(args.base_dir, dataset)
args.metadata_dir = os.path.join(args.dataset_dir, "samples", "metadata")
args.users_path = os.path.join(args.metadata_dir, "users.csv")
args.items_path = os.path.join(args.metadata_dir, "items.csv")
args.users_vocab_path = os.path.join(args.metadata_dir, "users_vocab.json")
args.items_vocab_path = os.path.join(args.metadata_dir, "items_vocab.json")

users_vocab, items_vocab = get_vocabularies(args)
n_users = len(users_vocab)
n_items = len(items_vocab)

args.do_classification = False
args.n_overall_ratings_classes = 1
args.n_aspect_ratings_classes = 1
args.embedding_dim = 32
args.padding_idx = 0
args.mlp_ratings_flag = True
args.mlp_aspect_shared_flag = False

if dataset == "TripAdvisor":
    args.aspects = ["Service", "Cleanliness", "Value", "Sleep quality", "Rooms", "Location"]
    args.model_path = os.path.join(args.dataset_dir, "exps", "a2r2v1", "model.pth")

elif dataset == "BeerAdvocate":
    args.aspects = ["appearance", "aroma", "taste", "palate"]
    args.model_path = os.path.join(
        args.dataset_dir, "exps", "a2r2v1_mlp_regression_separate_1720355804", "model.pth"
    )

elif dataset == "RateBeer":
    args.aspects = ["appearance", "aroma", "taste", "palate"]
    args.model_path = os.path.join(args.dataset_dir, "exps", "a2r2v1", "model.pth")

else:
    pass

model = A2R2v1(n_users, n_items, len(args.aspects), args)
model.load_state_dict(torch.load(args.model_path))
model.eval()
st.write("Model loaded")

with col2:
    user = get_id(st.selectbox("User:", write_ids(users_vocab)))
with col3:
    item = get_id(st.selectbox("Item:", write_ids(items_vocab)))

U_ids = torch.tensor([user])
I_ids = torch.tensor([item])

overall_rating, aspects_ratings, aspects_importance = model(U_ids, I_ids)
overall_rating = round(overall_rating.item(), 1)
aspects_rating = list(map(lambda x: round(x, 1), aspects_ratings.squeeze().tolist()))
aspects_importance = aspects_importance.squeeze().tolist()

review = """Loperam ipsum dolor sit amet, consectetur adipiscing elit.
Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut 
aliquip ex ea commodo consequat."""


st.divider()
with st.container():
    col1, col2, col3 = st.columns(3, vertical_alignment="top")
    with col1:
        st.write("**Aspects**")
    with col2:
        st.write("**Importance**")
    with col3:
        st.write("**Rating**")


for i, aspect in enumerate(args.aspects):
    with st.container():
        col1, col2, col3 = st.columns(3, vertical_alignment="center")
        with col1:
            st.write(aspect)
        with col2:
            st.progress(aspects_importance[i], text=str(aspects_importance[i]))
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
