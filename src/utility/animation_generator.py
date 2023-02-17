"""
animation functions are found here
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import animation

from src.simulator.config import *

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

veh_showing_route_step = 1
amin_frame_interval = 75
dpi = 100


# animation
def anim(frames_vehs):
    def init():
        for i in range(len(vehs)):
            vehs[i].set_data([frames_vehs[0][i].lng], [frames_vehs[0][i].lat])
            r1x = []
            r1y = []
            for leg in frames_vehs[0][i].route:
                for step in leg.steps:
                    geo = np.transpose(step.geo_pair)
                    r1x.extend(geo[0])
                    r1y.extend(geo[1])
            if i % veh_showing_route_step == 0:
                routes1[i].set_data(r1x, r1y)

        return vehs, routes1

    def animate(n):
        for i in range(len(vehs)):
            vehs[i].set_data([frames_vehs[n][i].lng], [frames_vehs[n][i].lat])
            r1x = []
            r1y = []
            for leg in frames_vehs[n][i].route:
                for step in leg.steps:
                    geo = np.transpose(step.geo_pair)
                    r1x.extend(geo[0])
                    r1y.extend(geo[1])
            if i % veh_showing_route_step == 0:
                routes1[i].set_data(r1x, r1y)
        return vehs, routes1

    fig = plt.figure(figsize=(MAP_WIDTH, MAP_HEIGHT))
    plt.xlim((Olng, Dlng))
    plt.ylim((Olat, Dlat))
    img = mpimg.imread(f'{root_path}/map.jpg')
    plt.imshow(img, extent=[Olng, Dlng, Olat, Dlat], aspect=(Dlng - Olng) / (Dlat - Olat) * MAP_HEIGHT / MAP_WIDTH)
    fig.subplots_adjust(left=0.00, bottom=0.00, right=1.00, top=1.00)
    vehs = []
    routes1 = []  # veh current route

    colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF']
    for i,v in enumerate(frames_vehs[0]):
        size = 6
        color = colors[i % len(colors)]  # assign a color based on the index
        vehs.append(plt.plot([], [], color=color, marker='o', markersize=size, alpha=1)[0])
        routes1.append(plt.plot([], [], linestyle='-', color=color, alpha=0.4)[0])
    anime = animation.FuncAnimation(fig, animate, init_func=init, frames=len(frames_vehs), interval=amin_frame_interval)
    # print('objective', objective, 'num of frames', len(frames_vehs))
    print('saving animation....')
    start_time = time.time()
    anime.save(f'{root_path}/media-gitignore/anim_new.mp4', dpi=dpi, fps=None, extra_args=['-vcodec', 'libx264'])
    print('...running time of encoding video : %.05f seconds' % (time.time() - start_time))
    return anime

