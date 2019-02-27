"""
defination of requests for the AMoD system
"""

import matplotlib.pyplot as plt
from lib.Configure import *


class Req(object):
    """
    Req is a class for requests
    Attributes:
        id: sequential unique id
        Tr: request time
        olng: origin longtitude
        olat: origin lngitude
        dlng: destination longtitude
        dlat: destination lngitude
        Ts: shortest travel time
        Cep: constraint - earliest pickup
        Clp: constraint - latest pickup
        Cld: constraint - latest dropoff
        Tp: pickup time
        Td: dropoff time
        D: detour factor
    """

    def __init__(self, osrm, id, Tr, olng=-0.162139, olat=51.490439, dlng=-0.104428, dlat=51.514180):
        self.id = id
        self.Tr = Tr
        self.olng = olng
        self.olat = olat
        self.dlng = dlng
        self.dlat = dlat
        self.Ts = osrm.get_duration(olng, olat, dlng, dlat)
        self.Cep = Tr
        self.Clp = Tr + MAX_WAIT
        # self.Cld = None
        self.Cld = Tr + MAX_DELAY if self.Ts > MAX_DELAY else Tr + MAX_DELAY / 2
        self.Tp = -1.0
        self.Td = -1.0
        self.D = 0.0
        self.served = False

    # return origin
    def get_origin(self):
        return self.olng, self.olat

    # return destination
    def get_destination(self):
        return self.dlng, self.dlat

    # visualize
    def draw(self):
        plt.plot(self.olng, self.olat, 'r', marker='+')
        plt.plot(self.dlng, self.dlat, 'r', marker='x')
        plt.plot([self.olng, self.dlng], [self.olat, self.dlat], 'r', linestyle='--', dashes=(0.5, 1.5))

    def __str__(self):
        str = "req %d from (%.7f, %.7f) to (%.7f, %.7f) at t = %.3f" % (
            self.id, self.olng, self.olat, self.dlng, self.dlat, self.Tr)
        str += "\n  latest pickup at t = %.3f, latest dropoff at t = %.3f" % (self.Clp, self.Cld)
        str += "\n  pickup at t = %.3f, dropoff at t = %.3f" % (self.Tp, self.Td)
        return str



