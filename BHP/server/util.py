import json
import pickle
import numpy as np

__locations = None
__data_columns = None
__model = None


def get_location_names():
    load_saved_artifacts()
    return __locations


def get_estimated_price(location, sqft, bhk, bath):
    print('I am invoked...')
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk

    if loc_index >= 0:
        x[loc_index] = 1
    print(f'estimated price from util {round(__model.predict([x])[0], 2)}')
    return round(__model.predict([x])[0], 2)


def load_saved_artifacts():
    global __locations
    global __data_columns
    global __model

    with open("./artifacts/columns.json", "r") as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]

    with open("./artifacts/banglore_home_prices_model.pickle", "rb") as f:
        __model = pickle.load(f)


# if __name__ == "__main__":


