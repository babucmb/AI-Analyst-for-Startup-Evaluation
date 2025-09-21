import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="AI Analyst for Startup Evaluation", layout="wide")

from src.ui.components import render_one_pager, render_radar_tab, render_qa_tab

st.title("AI Analyst for Startup Evaluation")

TAB_ONE, TAB_TWO, TAB_THREE = st.tabs(["One-Pager", "Radar Chart", "Q&A"])

with TAB_ONE:
	render_one_pager()
with TAB_TWO:
	render_radar_tab()
with TAB_THREE:
	render_qa_tab() 