import json
import pickle
import numpy as np

__locations = None
__data_columns = None
__model = None

def get_location_names():
    return __locations
def load_saved_artifacts():
    print("Started loading locations : ")
    global __locations
    global __data_columns
    with open("./artifacts/columns.json","r") as f:
        __data_columns = json.load(f)['data-columns']
        __locations = __data_columns[3:]

    global __model
    with open("./artifacts/bangalore home prices.pickle","rb") as f:
        __model = pickle.load(f)
    print("Locations with model loaded !!")

def get_estimated_price(location,sqft,bhk,bath):
    try:
        loc = __data_columns.index(location.lower())
    except:
        loc = -1
    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc >= 0:
        x[loc] = 1
    return round(__model.predict([x])[0],2)

def get_data_columns():
    return __data_columns

if __name__ == "__main__":
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price('1st phase jp nagar',1000,2,2))