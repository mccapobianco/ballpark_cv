print('Loading...')
from data_collection.data_collection import get_image
from geometric_feature_extraction.feature_extraction import predict_from_img
from park_factors.park_factor import predict_park_factor, convert_to_polar_function
from visualization_tools.plot_field import plot_field
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
lat = float(input('Latitude: '))
long = float(input('Longitude: '))
coord = lat, long
print('Collecting information...')

img, diagonal = get_image(coord)
measurements, orientation = predict_from_img(img, diagonal, return_orientation=True)
park_factor = predict_park_factor(coord, measurements)

fig, ax = plt.subplots()
plot_field(measurements, ax)
polar = convert_to_polar_function(measurements)
angles = np.linspace(.25, .75, 5)*np.pi
depths = polar(angles)

for theta, r in zip(angles, depths):
    ax.text(np.cos(theta)*(r+30), np.sin(theta)*(r+30), f'{round(r)} ft', 
            ha='center', va='center', fontsize=11, weight='bold', rotation=(np.degrees(theta-np.pi/2)%360))

ax.text(0, (depths[2]+135)/2, f'Park Factor\n{round(park_factor)}', 
        ha='center', va='center', fontsize=16)

xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()

ax.add_patch(mpatches.Circle((xmin+20, ymin+50), 15, color='black', zorder=1))
ax.add_patch(mpatches.Circle((xmin+20, ymin+50), 9, color='white', zorder=3))
ax.add_patch(mpatches.Polygon([[xmin+15, ymin+50], [xmin+25, ymin+50], [xmin+20, ymin+5]], 
                              color='black', zorder=1))

ax.text(xmin+45, ymin+35, f"{abs(np.round(lat,4)):.4f}{chr(176)} {'NS'[lat<0]}\n{abs(np.round(long,4)):.4f}{chr(176)} {'EW'[long<0]}", 
        ha='left', va='center', fontsize=12, weight='bold')

ax.add_patch(mpatches.Circle((xmax-45, ymin+45), 40, ec='black', fc='white', linewidth=3, zorder=1))
ax.add_patch(mpatches.Circle((xmax-45, ymin+45), 5, ec='black', fc='white', linewidth=1, zorder=3))

orientation = -orientation

needle_pts = np.matmul(np.array([[np.cos(orientation),-np.sin(orientation)],
                            [np.sin(orientation), np.cos(orientation)]]), 
                    np.array([[5, 0],[-5, 0],[0,25], [0,30]]).T).T + [xmax-45, ymin+45]
needle = mpatches.Polygon(needle_pts[:3,:], color='black')
ax.add_patch(needle)
ax.text(needle_pts[3,0], needle_pts[3,1], 'N', ha='center', va='center', 
        weight='bold', fontsize=6, rotation=np.degrees(orientation))

ax.axis('off')
plt.show()