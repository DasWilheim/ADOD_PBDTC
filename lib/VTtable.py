"""
compute all feasible veh-trip combinations for the vehicle fleet and requests in pool (queue + picking)
"""

import copy
import time
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

from lib.Configure import RIDESHARING_SIZE, CUTOFF_RTV, MODEE
from lib.Route import get_duration
from lib.ScheduleFinder import compute_schedule, test_constraints_get_cost, compute_schedule_cost


# build VT table for each veh(replan), respectively
def build_vt_table(vehs, reqs_new, reqs_old, T):
    veh_trip_edges = []

    # # parallel (not working properly)
    # trip_list_all = Parallel(n_jobs=-1)(delayed(feasible_trips_search)(veh, reqs_new, reqs_old, T) for veh in vehs)
    # for veh, (trip_list, schedule_list, cost_list) in zip(vehs, trip_list_all):
    #     for trips, schedules, costs in zip(trip_list, schedule_list, cost_list):
    #         for trip, schedule, cost in zip(trips, schedules, costs):
    #             veh_trip_edges.append((veh, trip, schedule, cost))

    # non-parallel
    for veh in tqdm(vehs, desc='VT Table'):
        feasible_trips_search(veh, reqs_new, reqs_old, T)
        for VTtable_k in veh.VTtable:
            for (trip, best_schedule, cost, all_schedules) in VTtable_k:
                veh_trip_edges.append((veh, trip, best_schedule, cost))
        # print('veh %d is finished' % veh.id)

    return veh_trip_edges


# search all feasible trips for a single vehicle, incrementally from the trip size of one
# a time consuming step
def feasible_trips_search(veh, reqs_new, reqs_old, T):
    start_time = time.time()
    rid_old = {req.id for req in reqs_old}

    # trips of size 1
    # add old trips (size 1)
    if MODEE == 'VT':
        veh.VTtable[0].clear()
    else:
        veh.VTtable[0] = get_old_trips(veh, rid_old, k=1)
    l_old_trips_k = len(veh.VTtable[0])  # number of old trips of size 1

    # add new trips (size 1)
    _schedule = []
    if not veh.idle:
        for leg in veh.route:
            if MODEE == 'VT':
                _schedule.append((leg.rid, leg.pod, leg.tnid, leg.ddl, leg.pf_path))
            else:  # 'VT_replan'
                if leg.rid in veh.onboard_rid:
                    _schedule.append((leg.rid, leg.pod, leg.tnid, leg.ddl, leg.pf_path))
    for req in reqs_new:
        # filter out the req which can not be served even when the veh is idle
        if get_duration(veh.nid, req.onid) + veh.t_to_nid + T > req.Clp:
            continue
        trip = tuple([req])  # trip is defined as tuple
        best_schedule, min_cost, all_schedules = compute_schedule(veh, trip, [], [_schedule])
        if best_schedule:
            veh.VTtable[0].append((trip, best_schedule, min_cost, all_schedules))
            # debug code
            assert {req.id for req in trip} <= {req.id for req in reqs_new}

    # trips of size k (k >= 2)
    for k in range(2, RIDESHARING_SIZE+1):
        # add old trips
        if MODEE == 'VT':
            veh.VTtable[k - 1].clear()
        else:
            veh.VTtable[k - 1] = get_old_trips(veh, rid_old, k)
        l_old_trips_k_1 = l_old_trips_k  # number of old trips of size k-1
        l_old_trips_k = len(veh.VTtable[k - 1])  # number of old trips of size k

        # add new trips
        l_all_trips_k_1 = len(veh.VTtable[k - 2])  # number of all trips of size k-1
        l_new_trips_k_1 = l_all_trips_k_1 - l_old_trips_k_1  # number of new trips of size k-1
        for i in range(1, l_new_trips_k_1 + 1):
            trip1 = veh.VTtable[k - 2][-i][0]  # a new trip of size k-1
            for j in range(i + 1, l_all_trips_k_1 + 1):
                trip2 = veh.VTtable[k - 2][-j][0]  # a trip of size k-1 different from trip1
                trip_k = tuple(sorted(set(trip1).union(set(trip2)), key=lambda r: r.id))
                if k > 2:
                    # check trip size is k
                    if len(trip_k) != k:
                        continue
                    # check trip is already computed
                    all_found_trip_k = [vt[0] for vt in veh.VTtable[k-1]]
                    if trip_k in all_found_trip_k:
                        continue
                    # check all subtrips are feasible
                    subtrips_check = True
                    for req in trip_k:
                        one_subtrip_of_trip_k = tuple(sorted((set(trip_k) - set([req])), key=lambda r: r.id))
                        all_found_trip_k_1 = [vt[0] for vt in veh.VTtable[k-2]]
                        if one_subtrip_of_trip_k not in all_found_trip_k_1:
                            subtrips_check = False
                            break
                    if not subtrips_check:
                        continue
                all_schedules_of_trip1 = veh.VTtable[k - 2][-i][3]
                all_schedules_of_trip2 = veh.VTtable[k - 2][-j][3]
                if len(all_schedules_of_trip1) <= len(all_schedules_of_trip2):
                    _trip = trip1  # subtrip of trip_k
                    _schedules = all_schedules_of_trip1  # all feasible schedules for trip1 of size k-1
                else:
                    _trip = trip2
                    _schedules = all_schedules_of_trip2

                best_schedule, min_cost, all_schedules = compute_schedule(veh, trip_k, _trip, _schedules)
                if best_schedule:
                    veh.VTtable[k - 1].append((trip_k, best_schedule, min_cost, all_schedules))

                    if not {req.id for req in trip_k} < {req.id for req in set(reqs_new).union(reqs_old)}:
                        print({req.id for req in trip_k}, {req.id for req in trip_k}-{req.id for req in set(reqs_new).union(reqs_old)})
                        print('veh', veh.id, ': size', k, 'add', [req.id for req in trip_k],  'schedules_num', len(all_schedules))
                    # debug code
                    assert {req.id for req in trip_k} < {req.id for req in set(reqs_new).union(reqs_old)}

                # if time.time() - start_time > CUTOFF_RTV:
                #     # print('veh', veh.id, ', trip size:', k, ', num of trips:', len(trip_list[k - 1]), '(time out)')
                #     return trip_list, schedule_list, cost_list
        if len(veh.VTtable[k - 1]) == 0:
            for k1 in range(k, RIDESHARING_SIZE):
                veh.VTtable[k1].clear()
            break


# find out which trips from last interval can be still considered feasible size k trips in current interval
def get_old_trips(veh, rid_old, k):
    old_VTtable_k = []

    new_pick_rid = set(veh.new_pick_rid)
    new_drop_rid = set(veh.new_drop_rid)
    new_both_rid = new_pick_rid.union(new_drop_rid)
    l_new_pick = len(veh.new_pick_rid)
    l_new_drop = len(veh.new_drop_rid)
    l_new_both = l_new_pick + l_new_drop
    assert l_new_both == len(new_both_rid)
    k = k + l_new_pick
    if k > RIDESHARING_SIZE:
        return old_VTtable_k

    if l_new_pick == 0:
        trip_id_sub = set()
        trip_id_sup = rid_old
    else:
        trip_id_sub = new_pick_rid
        trip_id_sup = rid_old.union(new_pick_rid)
        assert len(trip_id_sup) - len(rid_old) == l_new_pick

    veh_route = [(leg.rid, leg.pod) for leg in veh.route]
    veh_rid_enroute = {leg.rid for leg in veh.route}
    for (trip, best_schedule, cost, all_schedules) in veh.VTtable[k - 1]:
        trip_id = {req.id for req in trip}
        if trip_id_sub < trip_id < trip_id_sup:
            best_schedule = None
            min_cost = np.inf
            feasible_schedules = []
            if l_new_pick != 0:
                # remove picked req in trip
                trip_ = list(trip)
                for req in trip:
                    if req.id in new_pick_rid:
                        trip_.remove(req)
                trip_ = tuple(sorted(trip_, key=lambda r: r.id))
            else:
                trip_ = trip
            for schedule in all_schedules:
                if l_new_both != 0:
                    if {schedule[i][0] for i in range(l_new_both)} != new_both_rid:
                        continue
                    else:
                        assert sum([schedule[i][1] for i in range(l_new_both)]) == l_new_pick - 1 * l_new_drop
                        del schedule[0:l_new_both]

                # codes to fix bugs that caused by non-static travel times (starts)
                # (the same schedule (which is feasible) might not be feasible after veh moves along that schedule,
                #     if travel times are not static)
                trip_id_ = {req.id for req in trip_}
                trip_sche_ = [(rid, pod) for (rid, pod, tnid, ddl, pf_path) in schedule]
                if trip_sche_ == veh_route:
                    flag = True
                    c = compute_schedule_cost(veh, trip_, schedule)
                elif trip_id_ < veh_rid_enroute:
                    veh_partial_route = []
                    for (rid, pod) in veh_route:
                        if rid in trip_id_.union(set(veh.onboard_rid)):
                            veh_partial_route.append((rid, pod))
                    if trip_sche_ == veh_partial_route:
                        flag = True
                        c = compute_schedule_cost(veh, trip_, schedule)
                    else:
                        flag, c, viol = test_constraints_get_cost(veh, trip_, schedule, trip_[0], 0)
                else:
                    flag, c, viol = test_constraints_get_cost(veh, trip_, schedule, trip_[0], 0)
                # codes to fix bugs that caused by non-static travel times  (ends)

                if flag:
                    feasible_schedules.append(copy.deepcopy(schedule))
                    if c < min_cost:
                        best_schedule = copy.deepcopy(schedule)
                        min_cost = c
            if len(feasible_schedules) > 0:
                old_VTtable_k.append((trip_, best_schedule, min_cost, feasible_schedules))
                assert {req.id for req in trip_} < rid_old

    assert len([vt[0] for vt in old_VTtable_k]) == len(set([vt[0] for vt in old_VTtable_k]))
    return old_VTtable_k
