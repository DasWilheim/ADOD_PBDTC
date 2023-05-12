"""
definition of requests for the AMoD system
"""

from src.simulator.route_functions import *


class Req(object):
    """
    Req is a class for requests
    Attributes:
        id: sequential unique id
        Tr: request time
        onid: nearest origin node id in network
        dnid: nearest destination node id in network
        prio: priority of the request
        prio_ratio: priority ratio at the request moment
        heat: The heat of the request, closer to the deadline means hotter
        Ts: shortest travel time
        Ds: shortest travel distance
        Clp: constraint - latest pickup
        Cld: constraint - latest dropoff
        Clpn: constraint - latest pickup non-priority
        Cldn: constraint - latest dropoff non-priority
        start_time_window: start time window of a delivery window
        end_time_window: end time window of a delivery window
        Tp: actually pickup time
        Td: actually dropoff time
        D: detour factor
    """

    def __init__(self, id: int, Tr: int, onid: int, dnid: int, prio: int):#, start_time_window: str, stop_time_window: str):
        self.id = id
        self.status = OrderStatus.PENDING
        self.Tr = Tr
        self.onid = onid
        self.dnid = dnid
        self.prio = prio
        self.updated = False
        self.heat = 1.0
        self.Ts = get_duration_from_origin_to_dest(self.onid, self.dnid)
        self.Ds = get_distance_from_origin_to_dest(self.onid, self.dnid)
        if self.prio == 1 or PRIORITY_CONTROL:                   
            self.Clp = Tr + MAX_PICKUP_WAIT_TIME_MIN[0] * 60
            self.Cld = Tr + self.Ts + MAX_PICKUP_WAIT_TIME_MIN[0] * 60 * 2
        elif PRIORITY_CONTROL != True:    
            self.Clp = Tr + MAX_PICKUP_WAIT_TIME_MIN_NON_PRIORITY[0] * 60 
            self.Cld = Tr + self.Ts + MAX_PICKUP_WAIT_TIME_MIN_NON_PRIORITY[0] * 60 * 2 

        # self.Clp = Tr + min(MAX_PICKUP_WAIT_TIME_MIN[0] * 60, self.Ts * (2 - MAX_ONBOARD_DETOUR))
        # self.Cld = \
        #     Tr + self.Ts + min(MAX_PICKUP_WAIT_TIME_MIN[0] * 60 * 2, self.Clp - Tr + self.Ts * (MAX_ONBOARD_DETOUR - 1))

        self.Clp_backup = self.Clp
        self.Tp = -1.0
        self.Td = -1.0
        self.D = 0.0

    def update_pick_info(self, t: int):
        self.Tp = t
        #  DEBUG codes
        if self.status != OrderStatus.PICKING:
            print(f"[DEBUG1] req {self.id}, {self.status}, "
                  f"request time {self.Tr}, latest pickup {self.Clp}, pickup time {self.Tp}")

        assert (self.status == OrderStatus.PICKING)
        self.status = OrderStatus.ONBOARD

    def update_drop_info(self, t: int):
        self.Td = t
        self.D = (self.Td - self.Tp) / self.Ts
        assert(self.status == OrderStatus.ONBOARD)
        self.status = OrderStatus.COMPLETE
