
import time
import datetime
import copy
import matplotlib.pyplot as plt
# start time of initialization
istime = time.time()
print('initializing the model...')
from tqdm import tqdm
from lib.Main import Model
from lib.Configure import T_TOTAL, INT_ASSIGN, IS_ANIMATION, DMD_SST, IS_ANALYSIS
from lib.Analysis import anim, print_results

if __name__ == '__main__':
    # frames record the states of the AMoD model for animation purpose
    frames = []
    frames_reqs = []
    # initialize the AMoD model
    model = Model()
    print('...running time of initialization: %.05f seconds' % (time.time() - istime))

    # start time of simulation
    stime = time.time()
    # dispatch the system for T_TOTAL seconds, at the interval of INT_ASSIGN
    for T in tqdm(range(30, T_TOTAL, INT_ASSIGN), desc='AMoD'):
        # start time of each episode
        estime = time.time()
        model.dispatch_at_time(T)
        if IS_ANIMATION:
            frames.append(copy.deepcopy(model.vehs))
            frames_reqs.append(copy.deepcopy(model.queue_))
        print('System situation at %s : %d reqs have been received, %d have been served (%.02f%%), %d are on board '
              '(%.02f%%), %d are being picked-up (%.02f%%), %d are unassigned (%.02f%%), %d are rejected (%.02f%%).'
              % (DMD_SST + datetime.timedelta(seconds=T),
                 model.N, len(model.reqs_served), (len(model.reqs_served) / model.N * 100),
                 len(model.reqs_serving), (len(model.reqs_serving) / model.N * 100),
                 len(model.reqs_picking), (len(model.reqs_picking) / model.N * 100),
                 len(model.reqs_unassigned), (len(model.reqs_unassigned) / model.N * 100),
                 len(model.rejs), (len(model.rejs) / model.N * 100)))
        print('...running time of simulation: %.02f seconds (last episode:%.02f)'
              % ((time.time() - stime), (time.time() - estime)))
        print('')

    print('End of this simulation.')
    # end time
    etime = time.time()
    # run time of this simulation
    runtime = etime - stime

    print('...running time of simulation: %.05f seconds' % runtime)

    # generate, show and save the animation of this simulation
    if IS_ANIMATION:
        print('...Outputing simulation video...')
        anime = anim(frames, frames_reqs)
        anime.save('output/anim.mp4', dpi=120, fps=None, extra_args=['-vcodec', 'libx264'])
        plt.show()

    # output the simulation results and save data
    if IS_ANALYSIS:
        print_results(model, runtime)
