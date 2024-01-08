import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pymc as pm
import arviz as az

# Custom hash function for xarray.Dataset
def hash_arviz_dataset(dataset):
    return None
# Function to load the model trace
@st.cache_data(hash_funcs={az.InferenceData: hash_arviz_dataset})
def load_trace():
    with pm.Model() as modelx:
        sigma = pm.HalfCauchy("sigma", beta=10)
        intercept = pm.HalfNormal("Intercept", sigma=20)
        Size = pm.Normal("Size", 0, sigma=20)
        Rooms = pm.Normal("Rooms", 0, sigma=10)
        
        # Dummy data, replaced later by actual user input
        size = pm.MutableData('size', np.log10([50]), dims="obs_idx")
        rooms = pm.MutableData('rooms', [2], dims="obs_idx")
        log_price = np.log10([300000])

        likelihood = pm.Normal("price", mu=intercept + Size * size + Rooms * rooms, sigma=sigma, observed=log_price, dims="obs_idx")

    return az.from_netcdf('modelx_trace.nc')

trace = load_trace()

# Predictive function
def predict_price_bayesian(size, rooms, trace):
    with pm.Model() as modelx:
        sigma = pm.HalfCauchy("sigma", beta=10)
        intercept = pm.HalfNormal("Intercept", sigma=20)
        Size = pm.Normal("Size", 0, sigma=20)
        Rooms = pm.Normal("Rooms", 0, sigma=10)
        
        size = pm.MutableData('size', np.log10([size]), dims="obs_idx")
        rooms = pm.MutableData('rooms', [rooms], dims="obs_idx")

        likelihood = pm.Normal("price", mu=intercept + Size * size + Rooms * rooms, sigma=sigma)

        # Set new data for prediction
        pm.set_data({'size': np.log10([size]), 'rooms': [rooms]})
        
        # Generate posterior predictive samples
        ppd_dv = pm.sample_posterior_predictive(trace, predictions=True, extend_inferencedata=True)

    pred_price = (10**ppd_dv.predictions["price"]).mean(dim=["chain", "draw"]).data.round(0)
    return pred_price

# Streamlit app interface
def main():
    st.title("Apartment Sales Price Predictor")

    # User input for apartment specifications
    with st.form(key='apartment_form'):
        size = st.text_input("Size of the apartment (in square meters):")
        rooms = st.number_input("Number of rooms:", min_value=1, max_value=10, step=1)
        submit_button = st.form_submit_button(label='Submit')

    if submit_button and size.isnumeric():
        size = float(size)
        rooms = int(rooms)
        ppd_dv = predict_price_bayesian(size, rooms, trace)

        # Visualization
        M = 1e6
        predictions = 10**(ppd_dv.predictions)
        price_predictions = predictions["price"]
        _, ax = plt.subplots(figsize=(9, 5))

        # Assuming each prediction corresponds to an apartment
        obs_idx = 0  # For example, the first prediction

        for k, threshold in enumerate(np.array([1, 3, 5, 7, 9])*M):  # Example thresholds
            probs_above_threshold = (price_predictions.sel(obs_idx=obs_idx) >= threshold).mean(dim=("chain", "draw"))

            ax.axvline(threshold, color=f"C{k}")
            _, pdf = az.kde(price_predictions.sel(obs_idx=obs_idx).stack(sample=("chain", "draw")).data)
            ax.text(
                x=threshold - 35,
                y=pdf.max() / 2,
                s=f">={threshold/M:.0f}M",
                color=f"C{k}",
                fontsize="16",
                fontweight="bold",
            )
            ax.text(
                x=threshold - 20,
                y=pdf.max() / 2.3,
                s=f"{probs_above_threshold.data:.0%}",
                color=f"C{k}",
                fontsize="16",
                fontweight="bold",
            )
        ax.set_title(f"Apartment {obs_idx}\nProbability to price more than thresholds", fontsize=16)
        ax.set(xlabel="Price", ylabel="Plausible values")
        st.pyplot(ax.figure)

# Run the app
if __name__ == "__main__":
    main()
