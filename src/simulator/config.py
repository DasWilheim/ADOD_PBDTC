"""
constants are found here
"""
import argparse
import pickle
import os

parser = argparse.ArgumentParser()
parser.add_argument('--value', action='store', help='single value')
args = parser.parse_args()


# With this terminal command you can run multiple simulations with different values   for %v in (1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3.0) do python main.py --value %v
# Otherwise it will take an arbritrary value 
if args.value:
    value = float(args.value)
else:
    value = 1   #Default value 1 (no priority)
    
##################################################################################
# Data File Path
##################################################################################
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# map-data
PATH_TO_VEHICLE_STATIONS = f"{ROOT_PATH}/datalog-gitignore/map-data/stations-101.pickle"
PATH_TO_NETWORK_NODES = f"{ROOT_PATH}/datalog-gitignore/map-data/nodes.pickle"
PATH_TO_SHORTEST_PATH_TABLE = f"{ROOT_PATH}/datalog-gitignore/map-data/path-table.pickle"
PATH_TO_MEAN_TRAVEL_TIME_TABLE = f"{ROOT_PATH}/datalog-gitignore/map-data/mean-table.pickle"
PATH_TO_TRAVEL_DISTANCE_TABLE = f"{ROOT_PATH}/datalog-gitignore/map-data/dist-table.pickle"


# SIMULATION_DAYs = ["26"]
# TAXI_DATA_FILEs = [f"201605{day}-peak" for day in SIMULATION_DAYs]

SIMULATION_DAYs = ['05', '10', '15', '20', '25', '30', '35', '40', '45', '50', '55', '60', '65', '70', '75', '80', '85', '90', '95']
# SIMULATION_DAYs = ['40', '45', '50', '55', '60', '65', '70', '75', '80', '85', '90', '95']

# TAXI_DATA_FILEs = [f"20160525-flat{ratio}-{value}" for ratio in SIMULATION_DAYs]
TAXI_DATA_FILEs = ["aaaa"]

# PARTIAL_PATH_TO_TAXI_DATA = f"{ROOT_PATH}/src/data_functions/data_version_priority2/"
PARTIAL_PATH_TO_TAXI_DATA = f"{ROOT_PATH}/datalog-gitignore/taxi-data/"

# value-func-data
PARTIAL_PATH_TO_REPLAY_BUFFER_DATA = f"{ROOT_PATH}/datalog-gitignore/value-func-data/"

##################################################################################
# Mod System Config
##################################################################################
# dispatch_config
DISPATCHER = "OSP"        # 3 options: SBA Single-Request Batch Assignment, OSP-NR, OSP
REBALANCER = "NPO"        # 2 options: NONE, NPO Nearest Pending Order

# fleet_config:
FLEET_SIZE = [30]
VEH_CAPACITY = [3]
assert len(VEH_CAPACITY) == len(set(VEH_CAPACITY)), "VEH_CAPACITY contains duplicates"

# request_config:
REQUEST_DENSITY = 1    # <= 1
PRIORITY_RATIO = value


HEAT = False # if True, then packages become more lucrative (hotter) to pick up near their deadline
MAX_PICKUP_WAIT_TIME_MIN = [10]
MAX_PICKUP_WAIT_TIME_MIN_NON_PRIORITY = [PRIORITY_RATIO * MAX_PICKUP_WAIT_TIME_MIN[0]]
MAX_ONBOARD_DETOUR = 1   # < 2


##################################################################################
# Simulation Config
##################################################################################

SIMULATION_START_TIME = "2016-05-25 00:00:00"  # peak hour: 18:00:00 - 20:00:00
CYCLE_S = [30]
WARMUP_DURATION_MIN = 0       # 30 min
SIMULATION_DURATION_MIN = 1300  # <= 1370 min
WINDDOWN_DURATION_MIN = 50     # 39 min
DEBUG_PRINT = False
STORE_RESULTS = True        # store results in csv files

##################################################################################
# Value Function Config
##################################################################################

COLLECT_DATA = False
ENABLE_VALUE_FUNCTION = False
EVAL_NET_FILE_NAME = "FakeValueFunction"

ONLINE_TRAINING = False
if not ENABLE_VALUE_FUNCTION:
    ONLINE_TRAINING = False

##################################################################################
# Priority Function Config
##################################################################################

PRIORITY_CONTROL = True
PATH_TO_PLANE_DATA = f"{ROOT_PATH}/src/data_functions/data_plane/"
DESIRED_DELIVERY_TIME_TYPE_1 = 24.5    # minutes
SOFT_DEADLINE_TIME_TYPE_1 = 26    # minutes


##################################################################################
# Animation Config - Manhattan Map
##################################################################################
RENDER_VIDEO = False
# map width and height (km)

ZOOM_FACTOR = 1
MAP_WIDTH = 10.71 
MAP_HEIGHT = 20.85 
# coordinates
# (Olng, Olat) lower left corner
Olng = -74.0300
Olat = 40.6950
# (Olng, Olat) upper right corner
Dlng = -73.9030
Dlat = 40.8825


##################################################################################
# Config Change Functions
##################################################################################

def config_change_pr(pr):
    global PRIORITY_RATIO
    PRIORITY_RATIO = pr


def config_change_fs(fs):
    global FLEET_SIZE
    FLEET_SIZE[0] = fs


def config_change_vc(vc):
    global VEH_CAPACITY
    VEH_CAPACITY = vc


def config_change_wt(wt):
    global MAX_PICKUP_WAIT_TIME_MIN
    MAX_PICKUP_WAIT_TIME_MIN[0] = wt


def config_change_wp(wp):
    global MAX_PICKUP_WAIT_TIME_MIN_NON_PRIORITY
    MAX_PICKUP_WAIT_TIME_MIN_NON_PRIORITY[0] = wp


def config_change_bp(bp):
    global CYCLE_S
    CYCLE_S[0] = bp
