import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pymc as pm
import arviz as az




def load_trace():
    with pm.Model() as model:
        sigma = pm.HalfCauchy("sigma", beta=10)
        intercept = pm.HalfNormal("Intercept", sigma=20)
        Size = pm.Normal("Size", 0, sigma=20)
        Rooms = pm.Normal("Rooms", 0, sigma=10)
        
        # Dummy data, replaced later by actual user input
        size = pm.MutableData('size', np.log10([50]), dims="obs_idx")
        rooms = pm.MutableData('rooms', [2], dims="obs_idx")
        log_price = np.log10([300000])

        likelihood = pm.Normal("price", mu=intercept + Size * size + Rooms * rooms, sigma=sigma, observed=log_price, dims="obs_idx")
        idata =  az.from_netcdf('modelx_trace.nc')
        
    return idata, model


def predict_price(size, rooms):
    idata, model = load_trace()
    with model:
        pm.set_data({
        'size': np.log10(pd.Series([size])),
        'rooms': pd.Series([rooms]),
        })
    # Generate posterior predictive samples
        ppd_dv = pm.sample_posterior_predictive(idata, predictions=True, extend_inferencedata=True)
    return ppd_dv

def visulaize(predictions, user_price):
    M = 1e6
    obs_idx = 0 
    price_predictions = predictions["price"].sel(obs_idx=obs_idx)
    _, ax = plt.subplots(figsize=(9, 5))
    ax = az.plot_posterior(price_predictions, color="k", ax=ax)

     # Assuming each prediction corresponds to an apartment


    for k, threshold in enumerate(np.array([1, 3, 5, 7, 9,user_price/M])*M):  # Example thresholds
        probs_above_threshold = (price_predictions >= threshold).mean(dim=("chain", "draw"))

        ax.axvline(threshold, color=f"C{k}")
        _, pdf = az.kde(price_predictions.stack(sample=("chain", "draw")).data)
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
        ax.set_title(f"Apartment\nProbability to price more than thresholds", fontsize=16)
        ax.set(xlabel="Price", ylabel="Plausible values")
    st.pyplot(ax.figure)


# Streamlit app interface
def main():
    st.title("Apartment Sales Price Predictor")

    # User input for apartment specifications
    with st.form(key='apartment_form'):
        size = st.number_input("Size of the apartment (in square meters):", min_value=1, max_value = 200)
        rooms = st.number_input("Number of rooms:", min_value=1, max_value=5, step=1)
        user_price = st.number_input('what do you think?;')
        submit_button = st.form_submit_button(label='Submit')

    if submit_button:
        size = float(size)
        rooms = int(rooms)
        ppd_dv = predict_price(size, rooms)
    
        
        predictions = 10**(ppd_dv.predictions)
        price_predictions = predictions["price"]
        st.write(price_predictions.mean(dim=("chain", "draw")).data.round(0))

        visulaize(predictions, user_price)
# Run the app
if __name__ == "__main__":
    main()
