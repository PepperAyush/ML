import streamlit as st # type: ignore
import pandas as pd # type: ignore
import pickle as pk


with open('model-3.pkl', 'rb') as file:
    model = pk.load(file)

st.header('Car Price Prediction ML Model')

car_names = ['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
             'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz',
             'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus',
             'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force',
             'Ambassador', 'Ashok', 'Isuzu', 'Opel']

fuel_types = ['Diesel', 'Petrol', 'LPG', 'CNG']

seller_types = ['Individual', 'Dealer', 'Trustmark Dealer']

transmission_types = ['Manual', 'Automatic']

owner_types = ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car']

name = st.selectbox('Select Car Brand', car_names)
year = st.slider('Car Manufactured Year', 1994, 2024)
km_driven = st.slider('No of kms Driven', 11, 200000)
fuel = st.selectbox('Fuel type', fuel_types)
seller_type = st.selectbox('Seller type', seller_types)
transmission = st.selectbox('Transmission type', transmission_types)
owner = st.selectbox('Owner type', owner_types)
mileage = st.slider('Car Mileage', 10, 40)
engine = st.slider('Engine CC', 700, 5000)
max_power = st.slider('Max Power', 0, 200)
seats = st.slider('No of Seats', 5, 10)

if st.button("Predict"):
    input_data_model = pd.DataFrame(
        [[name, year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats]],
        columns=['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats']
    )

    input_data_model['owner'].replace(owner_types, [1, 2, 3, 4, 5], inplace=True)
    input_data_model['fuel'].replace(fuel_types, [1, 2, 3, 4], inplace=True)
    input_data_model['seller_type'].replace(seller_types, [1, 2, 3], inplace=True)
    input_data_model['transmission'].replace(transmission_types, [1, 2], inplace=True)
    input_data_model['name'].replace(car_names, list(range(1, len(car_names) + 1)), inplace=True)

    car_price = model.predict(input_data_model)

    st.markdown(f'Car Price is going to be {car_price[0]:.2f}')
