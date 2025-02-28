import json
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from scipy.ndimage import rotate
from scipy.spatial import ConvexHull
import os


def create_segments_from_png(img, model, scaled_input=False):
    if not scaled_input:
        img = img[..., :3] / 255
    img = np.transpose(img, (2,0,1))
    img_tensor = torch.tensor(img, dtype=torch.float32)
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor.to("cpu") 
    with torch.no_grad():  # No gradient calculation during inference
        output = model(img_tensor)  # Forward pass
        prediction = torch.argmax(output, dim=1).squeeze(0)  # Get the predicted class per pixel
    return prediction.numpy()
    


def label_to_shapes(label_array):
    
    shapes = []
    for label_id, label_name in enumerate(['background', 'field', 'infield', 'mound']):
        if label_id == 0:  # Skip the background label
            continue
        
        # Find contours for the current label
        mask = (label_array == label_id).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            points = contour.squeeze().tolist()  # Extract boundary points
            if len(points) < 3:  # Ignore degenerate shapes
                continue
            
            shapes.append({
                "label": label_name,
                "points": points,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
            })
    
    return shapes




def find_mound(segments, inplace=False):
    if not inplace:
        segments = segments.copy()
    infield_mask = (segments == 2).astype(int) + 1
    labeled_infield = label(infield_mask, connectivity=2)
    infield_center = find_center(segments > 1)
    regions = []
    for region in regionprops(labeled_infield):
        area = region.area
        perimeter = region.perimeter
        if 1000 > area > 5:
            circularity = 4 * np.pi * (area / (perimeter ** 2))  # Closer to 1 means more circular
            #TODO compare size to infield size, center to infield center, 
            #     set thresholds for three metrics
            mound_region = region
            mound_center = region.centroid
            infield_distance = np.linalg.norm(np.array(mound_center) - np.array(infield_center))
            regions.append((mound_center, mound_region, circularity, infield_distance))
    regions.sort(key=lambda x: x[3] + 20*abs(1-x[2]))
    region = regions[0][1]
    minr, minc, maxr, maxc = region.bbox
    segments[minr:maxr, minc:maxc][region.image] = 3
    if not inplace:
        return segments




#############################
### ORIENTATION FUNCTIONS ###
#############################


def find_center(a):
    return tuple(np.argwhere(a).mean(axis=0))

def find_angle_using_centers(center_1, center_2):
    x1, y1 = center_1
    x2, y2 = center_2
    # use x first to calculate angle because imshow uses (row,col)
    # and not Catesian coordinates
    angle_center_rad = np.arctan2(x1 - x2, y1 - y2)
    return angle_center_rad

def average_angles(angles, weights=None):
    if weights == None:
        weights = 1
    else:
        weights = np.array(weights)
    sin_sum = np.sum(weights * np.sin(angles))
    cos_sum = np.sum(weights * np.cos(angles))
    return np.arctan2(sin_sum, cos_sum)


def distance_to_border(start, theta, array_shape=(512, 512)):
    x0, y0 = start
    rows, cols = array_shape
    
    # Direction vector
    dx = np.cos(theta)
    dy = np.sin(theta)
    
    # Calculate intersection distances
    t_top = (0 - y0) / dy if dy != 0 else float('inf')  # Top edge (y = 0)
    t_bottom = (rows - 1 - y0) / dy if dy != 0 else float('inf')  # Bottom edge (y = rows - 1)
    t_left = (0 - x0) / dx if dx != 0 else float('inf')  # Left edge (x = 0)
    t_right = (cols - 1 - x0) / dx if dx != 0 else float('inf')  # Right edge (x = cols - 1)
    
    # Collect all distances
    distances = [t_top, t_bottom, t_left, t_right]
    
    # Filter for positive distances (moving forward only)
    positive_distances = [t for t in distances if t > 0]
    
    # Return the smallest positive distance
    return float(np.ceil(min(positive_distances)))

def find_furthest_points_angle(segments, window_width, center):
    # mound_center = find_center(segments==3)
    # center = int(mound_center[1])#segments.shape[1] // 2
    half_width = window_width // 2
    window = segments[:, center - half_width:center + half_width]
    points = np.argwhere(window)
    # Adjust coordinates to match the original array
    points[:, 1] += (center - half_width)
    hull = ConvexHull(points)
    outline = points[hull.vertices]
    # outline = label_to_shapes(segments)
    # outline = np.array(outline[0]['points'])
    
    
    max_distance = 0
    furthest_pair = None
    for i in range(len(outline)):
        for j in range(i + 1, len(outline)):
            dist = np.linalg.norm(outline[i] - outline[j])
            if dist > max_distance:
                max_distance = dist
                furthest_pair = (outline[i], outline[j])

    # Calculate angle using atan2
    p1, p2 = furthest_pair
    dy, dx = p2 - p1
    angle = np.arctan2(dy, dx)

    return angle


def find_mound_nearest_field(segments, angle, orientation):
    mound_center = find_center(segments==3)
    direction = np.array([np.sin(angle+orientation), np.cos(angle+orientation)])
    lower = 0.
    upper = distance_to_border(mound_center, angle+orientation)
    coord = np.array([np.nan, np.nan])
    last_coord = np.array([np.nan, np.nan])
    distance = np.inf
    while not np.all(coord == last_coord):
        last_coord = coord
        split = (upper + lower) / 2
        coord = np.round(mound_center + direction*split)
        distance = np.sqrt(np.sum((coord - mound_center)**2))
        x, y = coord.clip(0,511).astype(int)
        if segments[x,y] >= 2:
            lower = distance
        else:
            upper = distance
    return mound_center, (x,y)

def adjust_by_baseline(segments, angle, orientation):
    
    mound_center, xy1 = find_mound_nearest_field(segments, angle, orientation)
    mound_center, xy2 = find_mound_nearest_field(segments, -angle, orientation)
    x_new = (xy1[1] + xy2[1]) / 2
    y_new = (xy1[0] + xy2[0]) / 2
    #adjust for shallow angles
    if angle < 135*np.pi/180:
        new_angle = np.arctan2(xy1[0] - xy2[0], xy1[1] - xy2[1]) - np.pi/2
    else:
        new_angle = np.arctan2(mound_center[0]-y_new, mound_center[1]-x_new, )
    return new_angle

def clamp_angle(a, b, val):
    lower = a
    upper = b
    upper_adj = (b - a) % (np.pi*2)
    val_adj = (val - a) % (np.pi*2)
    if upper_adj > np.pi:
        lower = b
        upper = a
        upper_adj = (a - b) % (np.pi*2)
        val_adj = (val - b) % (np.pi*2)
    if val_adj <= upper_adj:
        return val
    elif val_adj >= np.pi - upper_adj/2:
        return lower
    else:
        return upper

def extract_field_shape(segments):
    shapes = label_to_shapes(segments>=1)
    field_shape = [np.array(s['points']) for s in shapes]
    field_shape = sorted(field_shape, key=lambda x: -len(x))[0]
    return field_shape

def find_orientation_from_segments(segments):
    #use a combo of methods to orient field
    segments = segments.copy()
    # remove small regions
    for region in regionprops(segments):
        area = region.area
        if area < 10:
            minr, minc, maxr, maxc = region.bbox
            segments[minr:maxr, minc:maxc][region.image] = 0

    #field-infield
    field_center = find_center(segments > 0)
    infield_center = find_center(segments > 1)
    mound_center = find_center(segments > 2)
    field_infield_rad = find_angle_using_centers(field_center, infield_center)

    #infield-mound
    infield_mound_rad = find_angle_using_centers(infield_center, mound_center)

    #field-mound
    field_mound_rad = find_angle_using_centers(field_center, mound_center)

    guesses = [field_infield_rad, infield_mound_rad, field_mound_rad ]

    current_guess = average_angles(guesses)
    
    # based on mound-baseline distances
    for angle in [115, 135, 160]:
        new_guess = current_guess
        for _ in range(100):
            next_guess = adjust_by_baseline(segments, angle*np.pi/180, new_guess)
            if abs(next_guess-new_guess) < 0.0001:
                break
            new_guess = next_guess
        guesses.append(new_guess)
    #based on furthest points
    rotated = rotate(segments, angle=90 + current_guess*180/np.pi, reshape=False)
    for i in range(2):
        center =  int(find_center(rotated==3)[1])
        fpa = find_furthest_points_angle(rotated>=i+1, 50//(i+1), center) - np.pi/4 - current_guess
        if np.dot(fpa, current_guess) < 0:
            fpa -= np.pi
        guesses.append(fpa)
    output = average_angles(guesses, weights=[2,3,2,1,1,1,0,0])
    return output

def orient_from_segments(segments, inplace=False, return_orientation=False):
    if not inplace:
        segments = segments.copy()
    orientation = find_orientation_from_segments(segments)
    rotation = 90 + orientation*180/np.pi
    segments = rotate(segments, angle=rotation, reshape=False)
    ret = []
    if not inplace:
        ret.append(segments)
    if return_orientation:
        ret.append(-np.radians(rotation))
    if not ret:
        ret = None
    elif len(ret) == 1:
        ret = ret[0]
    else:
        ret = tuple(ret)
    return ret

def rotate_around_mound(points, angle):
    new_points = points - np.array([[0,60.5]])
    new_points = np.matmul(np.array([[np.cos(angle),-np.sin(angle)],
            [np.sin(angle), np.cos(angle)]]), new_points.T).T
    
    return new_points + np.array([[0,60.5]])

def find_orientation_from_points(points):
    guess = 0
    new_points = points.copy()
    for _ in range(50):
        made_adjustment = False
        fair_territory = new_points[abs(new_points[:,0]) <= new_points[:,1], :]
        lf_corner = new_points[new_points[:,0].argmin(),:]
        lf_depth = fair_territory[fair_territory[:,0]<0,:]
        lf_depth = lf_depth[lf_depth[:,1].argmin(),:]
        if lf_depth[1] / lf_corner[1] < 0.99:
            guess += -np.pi/4 - np.arctan2(*lf_corner)
            made_adjustment = True
        rf_corner = new_points[new_points[:,0].argmax(),:]
        rf_depth = fair_territory[fair_territory[:,0]>0,:]
        rf_depth = rf_depth[rf_depth[:,1].argmin(),:]
        if rf_depth[1] / rf_corner[1] < 0.99:
            guess += np.pi/4 - np.arctan2(*rf_corner)
            made_adjustment = True
        if not made_adjustment:
           return guess
        new_points = rotate_around_mound(points, -guess)
    return guess
        
def orient_from_points(points, inplace=False, return_orientation=False):
    if not inplace:
        points = points.copy()
        
    orientation = -find_orientation_from_points(points)
    points = np.matmul(np.array([[np.cos(orientation),-np.sin(orientation)],
            [np.sin(orientation), np.cos(orientation)]]), points.T).T
    ret = []
    if not inplace:
        ret.append(points)
    if return_orientation:
        ret.append(-orientation)
    if not ret:
        ret = None
    elif len(ret) == 1:
        ret = ret[0]
    else:
        ret = tuple(ret)
    return ret



def segments_w_lines(segments, diagonal_ft):
    mound_y, mound_x = find_center(segments == 3)
    plate_x = mound_x
    plate_y = mound_y + 60.5 / diagonal_ft * (512*np.sqrt(2))
    plt.imshow(segments, cmap='tab20')
    plt.gca().plot([plate_x, plate_x+250], [plate_y, plate_y-250], c='black')
    plt.gca().plot([plate_x, plate_x-250], [plate_y, plate_y-250], c='black')




def convert_to_ft(segments, diagonal_ft, mound_center=None, aligned=True):
    if not aligned and mound_center is not None:
        Warning('`convert_to_ft`: manual `mound_center` input can have '\
                'unexpected behavior with `aligned=False`')
    segments = segments.copy()
    if not aligned:
        segments = orient_from_segments(segments)
    if mound_center is None:
        #reverse to convert to Cartesian
        mound_center = list(reversed(find_center(segments == 3)))
    field_shape = extract_field_shape(segments)
    scale_factor = diagonal_ft / (512*np.sqrt(2)) 
    field_shape = scale_factor * (field_shape - np.array(mound_center))
    field_shape[:,1] *= -1
    field_shape[:,1] += 60.5
    
    return field_shape


def predict_from_name(stadium_name, model=None, return_orientation=False):
    if model is None:
        device = torch.device('cpu')
        model  = smp.Unet(classes=4)
        model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), '../field_segmentation/model_12_28_2024.pth'), map_location=device, weights_only=True))
        model.to(device)
    with open(os.path.join(os.path.dirname(__file__), '../data_collection/metadata.json'),'r') as f:
        metadata = json.load(f)
    diagonal = metadata[stadium_name]['diagonal_length_ft']
    img = plt.imread(os.path.join(os.path.dirname(__file__), f"../data_collection/images/{stadium_name}.png"))
    segments = create_segments_from_png(img, model)
    segments_fixed = orient_from_segments(find_mound(segments), return_orientation=return_orientation)
    if return_orientation:
        segments_fixed, segments_orient = segments_fixed
    measurements = convert_to_ft(segments_fixed, metadata[stadium_name]['diagonal_length_ft'])
    measurements = orient_from_points(measurements, return_orientation=return_orientation)
    if return_orientation:
        measurements, points_orient = measurements
    if return_orientation:
        return measurements, (segments_orient+points_orient+2*np.pi) % (2*np.pi)
    return measurements



def predict_from_img(img, diagonal, model=None, return_orientation=False):
    if model is None:
        device = torch.device('cpu')
        model  = smp.Unet(classes=4)
        model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), '../field_segmentation/model_12_28_2024.pth'), map_location=device, weights_only=True))
        model.to(device)
    
    segments = create_segments_from_png(img, model)
    segments_fixed = orient_from_segments(find_mound(segments), return_orientation=return_orientation)
    if return_orientation:
        segments_fixed, segments_orient = segments_fixed
    measurements = convert_to_ft(segments_fixed, diagonal)
    measurements = orient_from_points(measurements, return_orientation=return_orientation)
    if return_orientation:
        measurements, points_orient = measurements
    if return_orientation:
        return measurements, (segments_orient+points_orient+2*np.pi) % (2*np.pi)
    return measurements