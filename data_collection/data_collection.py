from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
import tempfile
from enum import Enum
import time
import matplotlib.pyplot as plt
import sys
import os
from cv2 import resize
import numpy as np
import labelme
import json
import pickle

class MapType(Enum):
    IMAGERY = 0, "Imagery"
    IMAGERY_HYBRID = 1, "Imagery Hybrid"
    STREETS = 2, "Streets"
    TOPOGRAPHIC = 3, "Topographic"
    NAVIGATION = 4, "Navigation"
    STREETS_NIGHT = 5, "Streets (Night)"
    TERRAIN_WITH_LABELS = 6, "Terrain with Labels"
    LIGHT_GRAY_CANVAS = 7, "Light Gray Canvas"
    DARK_GRAY_CANVAS = 8, "Dark Gray Canvas"
    OUTDOOR = 9, "Outdoor"
    OCEANS = 10, "Oceans"
    NATIONAL_GEOGRAPHIC_STYLE_MAP = 11, "National Geographic Style Map"
    OPENSTREETMAP = 12, "OpenStreetMap"
    CHARTED_TERRITORY_MAP = 13, "Charted Territory Map"
    COMMUNITY_MAP = 14, "Community Map"
    NAVIGATION_DARK = 15, "Navigation (Dark)"
    NEWSPAPER_MAP = 16, "Newspaper Map"
    HUMAN_GEOGRAPHY_MAP = 17, "Human Geography Map"
    HUMAN_GEOGRAPHY_DARK_MAP = 18, "Human Geography Dark Map"
    MODERN_ANTIQUE_MAP = 19, "Modern Antique Map"
    MID_CENTURY_MAP = 20, "Mid-Century Map"
    NOVA_MAP = 21, "Nova Map"
    COLORED_PENCIL_MAP = 22, "Colored Pencil Map"
    OUTLINE_MAP = 23, "Outline Map"
    FIREFLY_IMAGERY_HYBRID = 24, "Firefly Imagery Hybrid"
    NAIP_IMAGERY_HYBRID = 25, "NAIP Imagery Hybrid"
    USGS_NATIONAL_MAP = 26, "USGS National Map"
    USA_TOPO_MAPS = 27, "USA Topo Maps"
    BLUEPRINT = 28, "Blueprint"
    TOPOGRAPHIC_VECTOR = 29, "Topographic (Vector)"
    ENHANCED_CONTRAST_DARK_MAP = 30, "Enhanced Contrast Dark Map"
    ENVIRONMENT_MAP = 31, "Environment Map"
    NAVIGATION_PLACES = 32, "Navigation (Places)"
    NAVIGATION_DARK_PLACES = 33, "Navigation (Dark - Places)"
    ENHANCED_CONTRAST_MAP = 34, "Enhanced Contrast Map"

    def __init__(self, index, title):
        self.index = index
        self.title = title

def get_image_from_coordinate(coordinate, map_type=MapType.IMAGERY, zoom_level=0, 
                              output_type='ndarray'):
    valid_output_types = ['ndarray', 'png']
    if output_type not in valid_output_types:
        raise ValueError(f'Invalid value for `output_type`. Must be one of: {", ".join(valid_output_types)}')
    options = webdriver.ChromeOptions()
    options.add_argument("--window-size=3000,2000")
    options.add_argument("--headless")  # Optional: run in headless mode
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get('https://www.arcgis.com/home/webmap/viewer.html')
    popup_close = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.ID, 'dijit_form_Button_7_label'))
    )
    popup_close.click()
    for _ in range(12):
        zoom_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CLASS_NAME, 'esriSimpleSliderIncrementButton')))
        zoom_button.click()
    close_side = driver.find_element(By.CLASS_NAME, 'panel_collapse')
    close_side.click()
    searchbar = driver.find_element(By.CLASS_NAME, 'searchInput')
    searchbar.send_keys(', '.join([str(x) for x in coordinate]))
    search_button = driver.find_element(By.CLASS_NAME, 'searchButtonText')\
        .find_element(By.XPATH, './..')
    search_button.click()
    for _ in range(zoom_level):
        zoom_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CLASS_NAME, 'esriSimpleSliderIncrementButton')))
        zoom_button.click()
    for _ in range(-zoom_level):
        zoom_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CLASS_NAME, 'esriSimpleSliderDecrementButton')))
        zoom_button.click()
    basemap_button = driver.find_element(By.ID, 'webmap-basemap')
    basemap_button.click()
    
    map_options = []
    while len(map_options) != 35:
        map_options = driver.find_element(By.CLASS_NAME, 'galleryBackground')\
            .find_elements(By.XPATH, './div/div')
        
    map_options[map_type.index].click()
    search_close = driver.find_element(By.CLASS_NAME, 'titlePane')\
        .find_element(By.CLASS_NAME, 'close')
    search_close.click()
    measure_button = driver.find_element(By.ID, 'webmap-measure_label')
    measure_button.click()
    distance_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.CLASS_NAME, 'distanceIcon')))
    distance_button.click()
    unit_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.ID, 'dijit_form_DropDownButton_0')))
    unit_button.click()
    feet_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.ID, 'dijit_MenuItem_14_text')))
    feet_button.click()
    map_div = driver.find_element(By.ID, 'map_layers')
    actions = ActionChains(driver)
    width = map_div.size['width']
    height = map_div.size['height']
    actions.move_to_element_with_offset(map_div, -width/2, -height/2).click().perform()
    
    actions.move_to_element_with_offset(map_div, width/2-1, height/2-1).double_click().perform()
    measurement = driver.find_element(By.ID, 'dijit_layout_ContentPane_4').text
    screenshot_diagonal_feet = float(measurement.replace(" Feet (US)", "").replace(',',''))
    measure_close = driver.find_element(By.ID, "measureClose")
    measure_close.click()

    time.sleep(1)
    f = tempfile.NamedTemporaryFile('w+b', suffix='.png', delete=False)
    map_div.screenshot(f.name)
    driver.close()
    if output_type == 'ndarray':
        output = plt.imread(f.name)
    elif output_type == 'png':
        output = f.read()
    f.close()
    os.unlink(f.name)
    return output, screenshot_diagonal_feet

def resize_512(arr):
    x, y, z = arr.shape
    if x==y:
        return resize(arr, (512, 512))
    elif x>y:
        new_min = int(x/2 - y/2)
        new_max = new_min + y
        return resize(arr[new_min:new_max, :], (512, 512)) 
    elif x<y:
        new_min = int(y/2 - x/2)
        new_max = new_min + x
        return resize(arr[:,new_min:new_max], (512, 512)) 

def json_to_mask(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    mask, _ = labelme.utils.shapes_to_label(
        img_shape=(512, 512),
        shapes=data['shapes'],
        label_name_to_value={'_background_': 0, 'field':1, 'infield':2, 'mound':3},
    )
    return np.eye(4)[mask]

def json_to_pickle(json_file):
    output_file = json_file[:-4] + 'pkl'
    mask = json_to_mask(json_file)
    with open(output_file, 'wb') as f:
        pickle.dump(mask, f, protocol=pickle.HIGHEST_PROTOCOL)

def find_new_diagonal(old_diagonal_ft, old_dimensions_px, new_dimensions_px):
    x0, y0, _ = old_dimensions_px
    x1, y1, _ = new_dimensions_px
    old_diagonal_px = np.sqrt(x0**2 + y0**2)
    new_diagonal_px = np.sqrt(x1**2 + y1**2)
    return old_diagonal_ft / old_diagonal_px * new_diagonal_px

def get_image(coord, zoom_level=0, map_type=MapType.IMAGERY):
    img, screenshot_diagonal_feet = get_image_from_coordinate(coord, zoom_level=zoom_level, map_type=map_type)
    x, y, z = img.shape
    m = min(x,y)
    diagonal_ratio = m*np.sqrt(2) / np.sqrt(x**2 + y**2) 
    resized_diagonal = screenshot_diagonal_feet * diagonal_ratio
    png = resize_512(img)
    return png, resized_diagonal

def save_image(coord, ballpark, zoom_level=0, map_type=MapType.IMAGERY, save_metadata=False):
    img, screenshot_diagonal_feet = get_image_from_coordinate(coord, zoom_level=zoom_level, map_type=map_type)
    x, y, z = img.shape
    m = min(x,y)
    diagonal_ratio = m*np.sqrt(2) / np.sqrt(x**2 + y**2) 
    png = resize_512(img)
    output_path = os.path.join(os.path.dirname(__file__), f'new_images/{ballpark}.png')
    if os.path.exists(output_path):
        handled_existing = False
        while not handled_existing:
            opt = input('File exists. Press 1 to overwrite, 2 to cancel, and 3 to enter new ballpark.')
            if opt == '1':
                plt.imsave(output_path, png)
                handled_existing = True
                print('Saved.')
            elif opt == '2':
                print('Cancelled.')
                handled_existing = True
                return
            elif opt == '3':
                ballpark = input('New ballpark: ')
                handled_existing = not os.path.exists(output_path)
            else:
                print('Input not understood.')
    else:
        plt.imsave(output_path, png)
        print('Saved.')
    resized_diagonal = screenshot_diagonal_feet * diagonal_ratio
    field_json = {'filename': output_path,
                  'latitude': coord[0],
                  'longitude': coord[1],
                  'diagonal_length_ft': resized_diagonal,
                  'zoom_level':zoom_level, 
                  'map_type':map_type.title}
    if save_metadata:
        with open('metadata.json', 'r') as f:
            metadata = json.load(f)
        metadata[ballpark] = field_json
        with open('metadata.json', 'w') as f:
            json.dump(metadata, f)
            print('Saved metadata.')

if __name__ == '__main__':
    lat, long, ballpark = sys.argv[1:4]
    coord = float(lat), float(long)
    save_image(coord, ballpark)

