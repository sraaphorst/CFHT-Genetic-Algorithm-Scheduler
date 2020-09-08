# observations.py
# By Sebastian Raaphorst, 2020. (Metric scoring by Bryan Miller, 2020.)

from defaults import *

from enum import IntEnum
from dataclasses import dataclass
from random import randrange, random

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
    band: str
    resource: Resource
    lb_time_constraint: float
    ub_time_constraint: float
    obs_time: float
    used_time: float
    allocated_time: float
    priority: float

    def __str__(self):
        return f'Observation {self.obs_idx}, band={int(self.band)}, resource={Resource(self.resource).name}, ' \
        f'obs_time={self.obs_time}, ub={self.ub_time_constraint} lb={self.lb_time_constraint}, priority={self.priority}'


class Observations:
    """
    The set of observations and the necessary information to perform the genetic algorithm (CFHT, 2008)
    to represent this as a scheduling problem.
    """

    def __init__(self):
        self.num_obs = 0
        self.band = np.empty((0,), dtype=str)
        self.resources = np.empty((0,), dtype=Resource)
        self.lb_time_constraints = np.empty((0,), dtype=float)
        self.ub_time_constraints = np.empty((0,), dtype=float)
        self.used_time = np.empty((0,), dtype=float)
        self.allocated_time = np.empty((0,), dtype=float)
        self.obs_time = np.empty((0,), dtype=float)
        # self.visibility_nights = np.empty((0,), dtype=int)
        self.priority = np.empty((0,), dtype=float)

        # This will be calculated after all the observations are inserted.
        self.completed = np.empty((0,), dtype=float)

        self.params = {'1': {'m1': 1.406, 'b1': 2.0, 'm2': 0.50, 'b2': 0.5, 'xb': 0.8, 'xb0': 0.0, 'xc0': 0.0},
                       '2': {'m1': 1.406, 'b1': 1.0, 'm2': 0.50, 'b2': 0.5, 'xb': 0.8, 'xb0': 0.0, 'xc0': 0.0},
                       '3': {'m1': 1.406, 'b1': 0.0, 'm2': 0.50, 'b2': 0.5, 'xb': 0.8, 'xb0': 0.0, 'xc0': 0.0},
                       '4': {'m1': 0.00, 'b1': 0.0, 'm2': 0.00, 'b2': 0.0, 'xb': 0.8, 'xb0': 0.0, 'xc0': 0.0},
                       }

        # Now spread the metric to avoid band overlaps
        # m2 = {'3': 0.5, '2': 3.0, '1':10.0} # use with b1*r where r=3
        m2 = {'3': 1.0, '2': 6.0, '1': 20.0}  # use with b1 + 5.
        xb = 0.8
        b1 = 0.2
        for band in ['3', '2', '1']:
            b2 = b1 + 5. - m2[band]
            m1 = (m2[band] * xb + b2) / xb ** 2
            self.params[band]['m1'] = m1
            self.params[band]['m2'] = m2[band]
            self.params[band]['b1'] = b1
            self.params[band]['b2'] = b2
            self.params[band]['xb'] = xb
            b1 += m2[band] * 1.0 + b2

        # The band (grade) + rank factor table as per the paper.
        self.grade_plus_rank_table = {
            '1': {1: 10.0, 2: 9.8, 3: 9.6},
            '2': {1: 8.0, 2: 7.8},
            '3': {1: 5.0, 2: 4.8},
            '4': {1: 4.0, 2: 3.8}
        }

        # A sorted list of observations by metric score.
        self.sort = None

    def add_obs(self, band: str, resource: Resource, obs_time: float,
                lb_time_constraint: float = None, ub_time_constraint: float = None,
                allocated_time=None) -> None:
        """
        Add an observation to the collection of observations.
        :param band: the band of the observation: a string of 1, 2, or 3
        :param resource: the site(s) where the observation can be scheduled
        :param obs_time: the observation time length
        :param lb_time_constraint: a lower bound time constraint on when the observation can be scheduled (e.g. >= 5)
                                   None indicates a default of the earliest possible time
        :param ub_time_constraint: an upper bound time constraint on when the observation can be scheduled (e.g. <= 15)
                                   None indicates a default of the latest possible time
        :param allocated_time: the allocated time, where None indicates the obs_time
        """
        assert (allocated_time != 0)
        self.band = np.append(self.band, band)
        self.resources = np.append(self.resources, resource)
        self.obs_time = np.append(self.obs_time, obs_time)
        self.lb_time_constraints = np.append(self.lb_time_constraints, lb_time_constraint)
        self.ub_time_constraints = np.append(self.ub_time_constraints, ub_time_constraint)
        self.used_time = np.append(self.used_time, 0)
        self.allocated_time = np.append(self.allocated_time, obs_time if allocated_time is None else allocated_time)
        self.num_obs += 1

    def __len__(self):
        return self.num_obs

    def _calculate_completion(self) -> None:
        """
        Calculate the completion of the observations.
        This is a simplification that expects the observation to take up (at least) the entire allocated time.
        """
        time = (self.used_time + self.obs_time) / self.allocated_time
        self.completed = np.where(time > 1., 1., time)

    def calculate_priority(self) -> None:
        """
        Compute the priority as a function of completeness fraction and band for the objective function.

        Parameters
            completion: array/list of program completion fractions
            band: integer array of bands for each program

        By Bryan Miller, 2020.
        """
        # Calculate the completion for all observations.
        self._calculate_completion()

        nn = len(self.completed)
        metric = np.zeros(nn)
        for ii in range(nn):
            sband = str(self.band[ii])

            # If Band 3, then the Band 3 min fraction is used for xb
            if self.band[ii] == 3:
                xb = 0.8
            else:
                xb = self.params[sband]['xb']

            # Determine the intercept for the second piece (b2) so that the functions are continuous
            b2 = self.params[sband]['b2'] + self.params[sband]['xb0'] + self.params[sband]['b1']

            # Finally, calculate piecewise the metric.
            if self.completed[ii] == 0.:
                metric[ii] = 0.0
            elif self.completed[ii] < xb:
                metric[ii] = self.params[sband]['m1'] * self.completed[ii] ** 2 + self.params[sband]['b1']
            elif self.completed[ii] < 1.0:
                metric[ii] = self.params[sband]['m2'] * self.completed[ii] + b2
            else:
                metric[ii] = self.params[sband]['m2'] * 1.0 + b2 + self.params[sband]['xc0']
        self.priority = metric

    # def lookup_band_rank(self) -> np.ndarray:
    #     """
    #     This corresponds to the grade + rank formula in the paper, which is a hard-coded table.
    #     :return: an ndarray list of the values for all the observations
    #     """
    #     return np.array([self.grade_plus_rank_table[self.band[i]][self.rank[i]]
    #                      for i in range(self.num_obs)], dtype=float)

    # def calculate_visibility_linear_regression_slopes(self, start_index: int = 0) -> np.ndarray:
    #     """
    #     This calculates the target pressures / visibilities for the samples across the runs using the slope of the
    #     linear regression at the starting position of the visibility_nights for each observation.
    #
    #     :return: an ndarray list of slopes of all the visibility periods for each observation
    #     """
    #     slopes = np.empty((0,), dtype=float)
    #     for obs_vis_y in self.visibility_nights:
    #         obs_vis_coords = list(enumerate(obs_vis_y[start_index:], start=1))
    #
    #         # Divide the data into two lists: the x coordinates and the y coordinates.
    #         n = len(obs_vis_coords)
    #         x_coords, y_coords = list(zip(*obs_vis_coords))
    #         x_sum = sum(x_coords)
    #         x_sq_sum = sum([x * x for x in x_coords])
    #         x_sum_sq = x_sum * x_sum
    #         y_sum = sum(y_coords)
    #         product_sum = sum([x * y for x, y in obs_vis_coords])
    #         slope = (n * product_sum - x_sum * y_sum) / (n * x_sq_sum - x_sum_sq)
    #
    #         np.append(slopes, slope)
    #     return slopes

    def __getitem__(self, item: int) -> Observation:
        return Observation(item, self.band[item], self.resources[item],
                           self.lb_time_constraints[item], self.ub_time_constraints[item],
                           self.obs_time[item], self.used_time[item], self.allocated_time[item],
                           self.priority[item])


def print_observation(o: Observation):
    print(f"Observation {o.obs_idx}: band={o.band} resource={Resource(o.resource).name} obs_time={o.obs_time} "
          f"lb={o.lb_time_constraint} ub={o.ub_time_constraint} prio={o.priority}")


def print_observations(observations: Observations):
    for i in range(len(observations)):
        print_observation(observations[i])


def generate_random_observations(num: int,
                                 start_time: int = DEFAULT_START_TIME,
                                 stop_time: int = DEFAULT_STOP_TIME) -> Observations:
    observations = Observations()

    for _ in range(num):
        band = str(randrange(1, 4))
        resource = Resource(randrange(3))

        obs_time = randrange(DEFAULT_OBS_TIME_LOWER, DEFAULT_OBS_TIME_UPPER)

        lb_time_constraint = None
        ub_time_constraint = None
        while True:
            lb_time_constraint = randrange(start_time, stop_time) if random() < 0.2 else None
            ub_time_constraint = randrange(start_time, stop_time) if random() < 0.2 else None
            if lb_time_constraint is None or ub_time_constraint is None:
                break
            if lb_time_constraint < ub_time_constraint:
                break

        observations.add_obs(band, resource, obs_time, lb_time_constraint, ub_time_constraint)

    observations.calculate_priority()
    print_observations(observations)
    return observations
