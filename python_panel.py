from comet_ml import API
import streamlit as st

def banner():
    print("Picture")

def activities():
    print("Activities")

def opik_summary():
    print("Opik Summary")

def em_summary():
    print("Experiment Management")

banner()

cols = st.columns(2)

with cols[0]:
    activities()

with cols[1]:
    opik_summary()
    em_summary()
