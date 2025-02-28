import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from matplotlib.patches import Arc

def smooth_field(measurements, s=1):
    x = measurements[:, 0]
    y = measurements[:, 1] 
    tck, u = splprep([x, y], s=s)
    u_new = np.linspace(0, 1, len(x))
    x_smooth, y_smooth = splev(u_new, tck)
    return np.column_stack((x_smooth, y_smooth))

#given two points, find where line between points intersects y=x
def intersect_y_equals_x(p1, p2):
    
    x1, y1 = p1
    x2, y2 = p2
    numerator = (x1*y2 - x2*y1) 
    denominator = (x1 - x2 - y1 + y2)
    if denominator == 0:
         return x1
    return numerator/denominator

def rotate_point_90_degrees(point, pivot):
            x, y = point
            px, py = pivot
            x_prime = x - px
            y_prime = y - py
            x_rotated = -y_prime
            y_rotated = x_prime
            x_new = x_rotated + px
            y_new = y_rotated + py
            return (x_new, y_new)

def plot_field(measurements, ax=None, c='black', lines=True, smoothing_factor=None, 
               boundarywidth=2, linewidth=1, zorder=-1):
    measurements = np.concatenate([measurements, measurements[[0],:]])
    if ax==None:
        ax = plt.gca()
    if smoothing_factor is not None:
        measurements = smooth_field(measurements, smoothing_factor)
    ax.plot(measurements[:,0], measurements[:,1],c=c, linewidth=boundarywidth, zorder=zorder)
    if lines:
        fair_territory = measurements[abs(measurements[:,0]) <= measurements[:,1], :]
        lf_fair = fair_territory[fair_territory[:,0].argmin(),:] 
        lf_index = (measurements==lf_fair).all(axis=1).astype(int).argmax()
        lf_index = (lf_index - 1) % len(measurements)
        lf_foul = measurements[lf_index, :]
        a = np.array([-1,1])
        lf_depth = -intersect_y_equals_x(lf_foul*a, lf_fair*a)
        rf_fair = fair_territory[fair_territory[:,0].argmax(),:] 
        rf_index = (measurements==rf_fair).all(axis=1).astype(int).argmax()
        rf_index = (rf_index - 1) % len(measurements)
        rf_foul = measurements[rf_index, :]
        rf_depth = intersect_y_equals_x(rf_foul, rf_fair)
        line_kwargs = {'color': c, 'linewidth': linewidth, 'zorder': zorder}
        #foul lines
        ax.plot([lf_depth,-3-8.5/12], [-lf_depth, 3+8.5/12], **line_kwargs)
        ax.plot([rf_depth,3+8.5/12], [rf_depth, 3+8.5/12], **line_kwargs)
        #plate
        ax.plot([0, 8.5/12, 8.5/12, -8.5/12, -8.5/12, 0], [0, 8.5/12, 17/12, 17/12, 8.5/12, 0], **line_kwargs)
        #outer baseline meets home circle
        bh_x = 3/np.sqrt(2) + 4*np.sqrt(5)
        bh_y = -3/np.sqrt(2) + 4*np.sqrt(5)
        #outer baseline meets back infield
        bi_x = (-121-6*np.sqrt(2) - np.sqrt(57487-1452*np.sqrt(2))) / 4
        bi_y = (121-6*np.sqrt(2) + np.sqrt(57487-1452*np.sqrt(2))) / 4
        #outer baseline
        ax.plot([bh_x, -bi_x], [bh_y, bi_y], **line_kwargs)
        ax.plot([-bh_x, bi_x], [bh_y, bi_y], **line_kwargs)
        #home circle
        ax.add_patch(Arc((0,0), 26, 26, theta1=np.degrees(np.arctan2(bh_y, -bh_x)), 
                         theta2=np.degrees(np.arctan2(bh_y,bh_x)), **line_kwargs))
        #back outfield
        ax.add_patch(Arc((0,60.5), 190, 190, theta1=np.degrees(np.arctan2(bi_y-60.5, -bi_x)), 
                         theta2=np.degrees(np.arctan2(bi_y-60.5,bi_x)), **line_kwargs))
        #mound
        ax.add_patch(Arc((0, 59), 18, 18, **line_kwargs))
        #rubber
        ax.plot([-1, 1, 1, -1, -1], [60.5, 60.5, 61, 61, 60.5], **line_kwargs)
        #inner baselines
        
        points = [(-bh_y, bh_x), (bh_y, bh_x)]
        for _ in range(6):
            points.append(rotate_point_90_degrees(points[-2], (0, 45*np.sqrt(2))))
        arc_center = (0, 0)
        for i in range(0,8,2):
            p0_x, p0_y = points[i%8]
            p1_x, p1_y = points[(i+1)%8]
            p2_x, p2_y = points[(i+2)%8]
            ax.plot([p1_x, p2_x], [p1_y, p2_y], **line_kwargs)
            cx, cy = arc_center
            ax.add_patch(Arc(arc_center, 26, 26, 
                            theta1=np.degrees(np.arctan2(p1_y-cy, p1_x-cx)), 
                            theta2=np.degrees(np.arctan2(p0_y-cy,p0_x-cx)), **line_kwargs))
            arc_center = rotate_point_90_degrees(arc_center, (0, 45*np.sqrt(2)))

        #boxes
        ax.plot([14.5/12, 14.5/12+4, 14.5/12+4, 14.5/12, 14.5/12],
            [-3+8.5/12, -3+8.5/12, 3+8.5/12, 3+8.5/12, -3+8.5/12], **line_kwargs)
        ax.plot([-14.5/12, -14.5/12-4, -14.5/12-4, -14.5/12, -14.5/12],
            [-3+8.5/12, -3+8.5/12, 3+8.5/12, 3+8.5/12, -3+8.5/12], **line_kwargs)
        ax.plot([-43/2/12, -43/2/12, 43/2/12, 43/2/12],
            [-3+8.5/12, -8, -8, -3+8.5/12], **line_kwargs)
        
        #bases
        ax.plot([90/np.sqrt(2), 90/np.sqrt(2)-18/12/np.sqrt(2), 90/np.sqrt(2)-36/12/np.sqrt(2), 90/np.sqrt(2)-18/12/np.sqrt(2)],
            [90/np.sqrt(2), 90/np.sqrt(2)+18/12/np.sqrt(2), 90/np.sqrt(2), 90/np.sqrt(2)-18/12/np.sqrt(2)], **line_kwargs)
        ax.plot([0, -18/12/np.sqrt(2), 0, 18/12/np.sqrt(2), 0],
            [90*np.sqrt(2)-18/12/np.sqrt(2), 90*np.sqrt(2), 90*np.sqrt(2)+18/12/np.sqrt(2), 90*np.sqrt(2), 90*np.sqrt(2)-18/12/np.sqrt(2)], **line_kwargs)
        ax.plot([-90/np.sqrt(2), -90/np.sqrt(2)+18/12/np.sqrt(2),-90/np.sqrt(2)+36/12/np.sqrt(2), -90/np.sqrt(2)+18/12/np.sqrt(2)],
            [90/np.sqrt(2), 90/np.sqrt(2)+18/12/np.sqrt(2), 90/np.sqrt(2), 90/np.sqrt(2)-18/12/np.sqrt(2)], **line_kwargs)
