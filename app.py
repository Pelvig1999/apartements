# Import necessary libraries
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

# Mock predictive model function
def predict_price(rooms, size, latitude, longitude, build_year):
    # For demonstration purposes, we'll use a simple formula to 'predict' price
    # In reality, this should be replaced with calls to a trained model
    base_price = 300000  # base price in a currency of your choice
    price_per_room = 50000
    price_per_sqm = 3000
    age_factor = 0.99  # Depreciation factor per year

    # Calculate years since built
    years_since_built = pd.Timestamp.now().year - build_year.year

    # Generate a random factor to simulate prediction variability
    random_factor = random.uniform(0.9, 1.1)

    # Calculate the predicted price
    predicted_price = (base_price + 
                       (price_per_room * rooms) + 
                       (price_per_sqm * float(size)) * 
                       (age_factor ** years_since_built) * 
                       random_factor)
    
    return predicted_price


# Streamlit app
def main():
    st.title("Apartment Sales Price Predictor for the Copenhagen and Frederiksberg Municipality")
    st.markdown("""
        ## This webpage can help you get a more transparent insight into how you should price your apartment in correspondence to the market right now in Copenhagen.
        What you need to do is to input your apartment specifications below, and we will give you a rough estimation of how much you can get from selling your apartment right now.
    """)

    # User input for apartment specifications
    with st.form(key='apartment_form'):
        # Numeric input for the number of rooms
        rooms = st.number_input("How many rooms does your apartment have?", min_value=1, max_value=5, value=1, step=1)
        
        # Text input for the size of the apartment
        size = st.text_input("How large is your apartment (in square meters)?")

        # Text inputs for the latitude and longitude
        latitude = st.text_input("Latitude of your apartment:")
        longitude = st.text_input("Longitude of your apartment:")

        # Date input for the build year
        build_year = st.date_input("When was the building your apartment is located in built?")

        # Form submission button
        submit_button = st.form_submit_button(label='Submit')

    if submit_button and size.isnumeric():
        # Convert inputs and call the model
        size = float(size)
        latitude = float(latitude)
        longitude = float(longitude)
        predicted_price = predict_price(rooms, size, latitude, longitude, build_year)

        # Display the predicted price
        st.write(f"The predicted price of your apartment is: DKK{predicted_price:,.2f}")

        # Visualization (simple line plot with the predicted price)
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, predicted_price], 'o-')
        ax.set_title("Price Prediction Visualization")
        ax.set_ylabel("Price")
        ax.set_xticks([])
        ax.set_yticklabels(['', f'DKK{predicted_price:,.2f}'])
        st.pyplot(fig)
        # Disclaimer text under the visualization
        st.markdown("""
            ---
            **Disclaimer**: This price prediction is an *estimation*. We cannot guarantee that this is what your apartment will sell for. This is only an extra tool in your toolbox when you make the decision of selling. It gives you a more transparent look into how a real estate agent values your apartment and when they ask you how much you want from the sale, you have an indicator on what to answer. 

            Additionally, this visualization also gives you an indication of the possibility, in percentage, of your apartment price. Again, this is only an *estimation*, we cannot guarantee the selling price of your apartment.
            """)

    elif submit_button:
        st.error("Please make sure the size is a number.")

if __name__ == "__main__":
    main()
