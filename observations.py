# observations.py
# By Sebastian Raaphorst, 2020. (Metric scoring by Bryan Miller, 2020.)

from defaults import *

from enum import IntEnum
from dataclasses import dataclass
from typing import List

import numpy as np


class Resource(IntEnum):
    """
    The resources on which an observation can be scheduled.
    """
    GN = 0
    GS = 1
    Both = 2


@dataclass
class Observation:
    """
    Encapsulates the data of a single observation for scheduling. This extracts the necessary data from the numpy
    arrays and puts it into this data structure for simpler access.
    """
    obs_idx: int
    name: str
    resource: Resource
    lb_time_constraint: int
    ub_time_constraint: int
    obs_time: float
    priority: float
    time_priorities: List[float]

    def __str__(self):
        return f'Observation {self.name}, resource={Resource(self.resource).name}, ' \
        f'obs_time={self.obs_time}, ub={self.ub_time_constraint} lb={self.lb_time_constraint}, priority={self.priority}'


class Observations:
    """
    The set of observations and the necessary information to perform the genetic algorithm (CFHT, 2008)
    to represent this as a scheduling problem.
    """

    def __init__(self):
        self.num_obs = 0
        self.name = np.empty((0,), dtype=str)
        self.resources = np.empty((0,), dtype=Resource)
        self.lb_time_constraints = np.empty((0,), dtype=int)
        self.ub_time_constraints = np.empty((0,), dtype=int)
        self.obs_time = np.empty((0,), dtype=float)
        self.priority = np.empty((0,), dtype=float)
        #self.time_priorities = np.empty((0,))
        self.time_priorities = []

        # This will be calculated after all the observations are inserted.
        self.completed = np.empty((0,), dtype=float)

    def add_obs(self, name: str, resource: Resource, obs_time: float,
                lb_time_constraint: float, ub_time_constraint: float,
                priority: float, time_priorities: List[float]) -> None:
        """
        Add an observation to the collection of observations.
        :param name: the name of the observation
        :param resource: the site(s) where the observation can be scheduled
        :param obs_time: the observation time length
        :param lb_time_constraint: a lower bound time constraint on when the observation can be scheduled (e.g. >= 5)
                                   None indicates a default of the earliest possible time
        :param ub_time_constraint: an upper bound time constraint on when the observation can be scheduled (e.g. <= 15)
                                   None indicates a default of the latest possible time
        :param priority: the metric score associated with the observation
        """
        self.name = np.append(self.name, name)
        self.resources = np.append(self.resources, resource)
        self.obs_time = np.append(self.obs_time, obs_time)
        self.lb_time_constraints = np.append(self.lb_time_constraints, lb_time_constraint)
        self.ub_time_constraints = np.append(self.ub_time_constraints, ub_time_constraint)
        self.priority = np.append(self.priority, priority)
        #self.time_priorities = np.append(self.time_priorities, time_priorities)
        self.time_priorities.append(time_priorities)
        self.num_obs += 1

    def __len__(self):
        return self.num_obs

    def __getitem__(self, item: int) -> Observation:
        return Observation(item, self.name[item], self.resources[item],
                           self.lb_time_constraints[item], self.ub_time_constraints[item],
                           self.obs_time[item], self.priority[item], self.time_priorities[item])


def print_observation(o: Observation):
    print(f"Observation {o.name}: resource={Resource(o.resource).name} obs_time={o.obs_time} "
          f"lb={o.lb_time_constraint} ub={o.ub_time_constraint} prio={o.priority}")


def print_observations(observations: Observations):
    for i in range(len(observations)):
        print_observation(observations[i])
