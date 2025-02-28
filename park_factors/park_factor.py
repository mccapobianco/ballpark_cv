from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import UnexpectedAlertPresentException
import time
import pickle
import numpy as np
from scipy.interpolate import interp1d
import os
import pandas as pd

with open(os.path.join(os.path.dirname(__file__), 'park_factor_components.pkl'), 'rb') as f:
    d = pickle.load(f)

    model = d['model']
    X_scaler = d['x_scaler']
    y_scaler = d['y_scaler']

def get_elevation_meters(coords):
    output = []
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # Optional: run in headless mode
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get('https://www.dcode.fr/earth-elevation')
        #textbox = driver.find_element(By.ID, 'earth_elevation_calculator_gps')
    i = 0
    while i < len(coords):
        try:
            lat, long = coords[i]
            textbox = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.ID, 'earth_elevation_calculator_gps')))
            fstring = f'({lat}, {long})'
            textbox.send_keys(fstring)
            button = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.ID, 'earth_elevation_calculator'))).find_element(By.TAG_NAME, 'button')
            #     button = driver.find_element(By.ID, 'earth_elevation_calculator')
            button.click()
            result = WebDriverWait(driver, 10).until(
                    EC.visibility_of_element_located((By.CLASS_NAME, 'result')))
            output.append(int(result.text))
            textbox.send_keys('\b'*len(fstring))
            i += 1
            time.sleep(10)
        except UnexpectedAlertPresentException:
            input('Unexpected error while determining park factor. Press enter to retry.')
    return output

feature_cols = ['lf_area', 'lc_area', 'cf_area', 'rc_area', 'rf_area', 'elevation_m']

def est_polar_area(f, a, b, sample_size=10000):
    x = np.linspace(a, b, sample_size)
    y = f(x)
    return sum(y**2)/2 * (b-a) / sample_size

def convert_to_polar_function(points):
    x,y = points[:,0], points[:,1]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    # Ensure theta is in [0, 2*pi] range
    theta = (theta + 2 * np.pi) % (2 * np.pi)

    # Sort by angle to ensure the function is well-defined
    sorted_indices = np.argsort(theta)
    theta_sorted = theta#[sorted_indices]
    r_sorted = r#[sorted_indices]
    # Interpolate r as a function of theta
    # Use 'linear', 'quadratic', 'cubic' or other methods for smoothness
    r_function = interp1d(theta_sorted, r_sorted, kind='linear', fill_value="extrapolate")
    return r_function 
     
def predict_park_factor(coord, points):
    features = {}
    polar_func = convert_to_polar_function(points)
    features['elevation_m'] = get_elevation_meters([coord])[0]
    features['lf_area'] = est_polar_area(polar_func, 0.25*np.pi, 0.35*np.pi)
    features['lc_area'] = est_polar_area(polar_func, 0.35*np.pi, 0.45*np.pi)
    features['cf_area'] = est_polar_area(polar_func, 0.45*np.pi, 0.55*np.pi)
    features['rc_area'] = est_polar_area(polar_func, 0.55*np.pi, 0.65*np.pi)
    features['rf_area'] = est_polar_area(polar_func, 0.65*np.pi, 0.75*np.pi)
    pre_feature_order = 'orientation,lf_depth,lc_depth,cf_depth,rc_depth,rf_depth,lf_area,lc_area,cf_area,rc_area,rf_area,total_area,avg_depth,temp_K,u_wind,v_wind,out_to_cf_wind,left_to_right_wind,elevation_m'.split(',')
    feature_order = ['lf_area', 'lc_area', 'cf_area', 'rc_area', 'rf_area', 'elevation_m']
    features = pd.DataFrame([{x:features.get(x,0) for x in pre_feature_order}])
    X = X_scaler.transform(features)
    X = X[:, [pre_feature_order.index(x) for x in feature_order]]

    return y_scaler.inverse_transform(model.predict(X))[0,0]
