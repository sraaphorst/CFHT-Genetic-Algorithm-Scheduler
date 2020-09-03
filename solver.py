# solver.py
# By Sebastian Raaphorst, 2020.

from typing import List, Tuple
import numpy as np
from observations import *

# A gene is an observation number, and a chromosome is a list of genes where order is important.
gene = int
chromosome = [int]


class Chromosome:
    """
    A chromosome for the genetic algorithm, which represents a schedule for a given site.
    """
    def __init__(self, observations: Observations, resource: Resource, start_time: float, stop_time: float):
        """
        A Chromosome for resource (the site).

        :param observations: the observations over which we are scheduling
        :param resource: the site (both is not allowed)
        :param start_time: the start time for the chromosome, derived from the GA
        :param stop_time: the stop time for the chromosome, derived from the GA
        """
        assert(resource != Resource.Both, "A chromosome is specific to a site.")
        self.observations = observations
        self.resource = resource
        self.start_time = start_time
        self.stop_time = stop_time

        # We need to maintain a list of what times are used up and what are free, so
        # we schedule this list with entries of (start_time, obs_num), and from there,
        # the free time to the next scheduled observation can be calculated.
        self.schedule = []

    def _get_gaps_in_range(self, obs_idx: int) -> List[float]:
        """
        Find the empty gaps in the range [lower_time, upper_time] that can accommodate the specified observation.
        The values returned are the earliest values in the gaps. If we had, for example, and obs of length 3, and
        we had scheduled:
        (obs1, 4), (obs2, 6), (obs3, 12)
        with length of obs1 being 2 (ending at 6),
            length of obs2 being 1 (ending at 7), and
            length of obs3 being 5 (ending at 17),
        with start time 0 and stop time 30, we would return:
        [0, 7, 17].

        :param obs_idx: the observation to consider
        :return: a list of the possible earliest start times for the observation in gaps as described above
        """
        obs = self.observations[obs_idx]
        lower_time = obs.lb_time_constraint if obs.lb_time_constraint is not None else self.start_time
        upper_time = obs.ub_time_constraint if obs.ub_time_constraint is not None else self.stop_time
        obs_time = obs.obs_time

        gap_start_times = []
        sorted_start_times = sorted(self.schedule)

        # Can we schedule at the start of the observation?
        # We can if one of the cases hold:
        # 1. There are no observations scheduled and this fits, i.e.:
        #       lower_time + obs_time < this chromosome's stop time
        # 2. The observation fits before the first currently scheduled observation, i.e.:
        #       lower_time + obs_time < first scheduled time
        first_scheduled_time = 0 if len(self.schedule) == 0 else sorted_start_times[0][0]
        if (len(self.schedule) == 0 and lower_time + obs_time < self.stop_time) or \
                lower_time + obs_time < first_scheduled_time:
            gap_start_times.append(lower_time)

        # Now check for all other unused intervals in the chromosome to see if this observation fits.
        for gene_idx, (start_time, curr_obs_idx) in enumerate(sorted_start_times):
            curr_obs = self.observations[curr_obs_idx]

            # Find the next valid gap.

            # First, determine the next free start time.
            # We look at the currently scheduled observation and determine when it ends, i.e.:
            # start_time + the length of time needed for that observation.
            # This is bounded by the lower_time on the new observation since we cannot schedule before that.
            next_start_time = max(start_time + curr_obs.obs_time, lower_time)

            # If this is not a valid start time, no subsequent start times will be valid by the timing constraints,
            # so just exit.
            if next_start_time > upper_time:
                break

            # Now we have two cases:
            # 1. curr_obs_idx is the last observation currently scheduled, in which case everything afterwards
            #    comprises one big gap; or
            # 2. There are more observations scheduled, in which case, the end of this interval is the start time
            #    of the next scheduled observation.
            next_end_time = self.stop_time if gene_idx == len(self.schedule) - 1 else sorted_start_times[gene_idx + 1]

            # If next_end_time - obs_time < lower_time, this would violate timing constraints, scheduling this
            # observation earlier than it is permitted.
            if next_end_time - obs_time < lower_time:
                continue

            # If the gap is big enough to accommodate this observation, add it to the list of gaps.
            if next_end_time - next_start_time >= obs_time:
                gap_start_times.append(next_start_time)

        return gap_start_times

    def determine_length(self) -> float:
        """
        Determine the amount of time currently used up in this chromosome.
        >= 85% is considered "optimal."
        :return: the length of time scheduled in this chromosome
        """
        return sum(self.observations[idx] for _, idx in self.schedule)

    def insert(self, obs_idx) -> bool:
        """
        Try to insert obs_idx into this chromosome in the earliest possible position. This fails if:
        1. The observation resource is not compatible with this resource.
        2. The timing constraints do not allow it to fit.
        3. There are no gaps big enough to accommodate it.
        Otherwise, it is scheduled.
        :param obs_idx: the index of the observation to try to schedule
        :return: True if we could schedule, and False otherwise
        """

        # Check site compatibility.
        obs = self.observations[obs_idx]
        if obs.resource != Resource.Both and obs.resource != self.resource:
            return False

        # Get the gap start times in this chromosome in which we can schedule the observation.
        gap_start_times = self._get_gaps_in_range(obs_idx)
        if len(gap_start_times) == 0:
            return False

        # Schedule the observation in the first gap.
        self.schedule.append((gap_start_times[0], obs_idx))
        return True


class GeneticAlgorithm:
    """
    A simplification prototype of the genetic algorithm listed in:

    Mahoney et al. "A genetic algorithm for ground-based telescope observation scheduling", SPIE, 2012.

    Instead of scheduling for one telescope, we must be able to schedule for two telescopes.
    We also simplify the algorithm by using a metric to score each observation instead of using the
    target pressure and the grade + rank table, as the metric is a surrogate for grade + rank.

    We treat the problem as a 0-1 knapsack problem for two knapsacks, with restrictions on which
    knapsack can contain which items (observations).
    """

    def __init__(self, observations: Observations, start_time: float = 0, stop_time: float = 30):
        """
        Initialize the problem by passing in the observations and the bounds on the time representations
        for observation scheduling.

        :param observations: the Observations object to be scheduled
        :param start_time: the start time, as a float (arbitrary)
        :param stop_time: the end time, as a float (arbitrary)
        """
        assert(len(observations) > 1, "Must have observations to schedule.")
        assert(start_time < stop_time, "The scheduling period must have some length.")

        self.observations = observations
        self.start_time = start_time
        self.stop_time = stop_time

    def _form_initial_population(self) -> Tuple[List[Chromosome], List[int]]:
        """
        We form the initial population of chromosomes by putting them at the earliest period that we can.
        GS is given slight priority over GN in such that a new chromosome where the observation can be scheduled
        at both is scheduled at GS.

        :return: A list of chromosomes and a list of unused observation indices.
        """
        chromosomes = []
        unused_genes = []

        for obs_idx in range(len(self.observations)):
            # We can only schedule the observation in a chromosome corresponding to its site.
            # Chromosome.insert handles this, so we don't have to worry about it here.
            scheduled = False

            for chromosome in chromosomes:
                if chromosome.insert(obs_idx):
                    scheduled = True
                    break
            if scheduled:
                break

            # Create a new chromosome and attempt to add it.
            chromosome = Chromosome(self.observations,
                                    Resource.GN if self.observations[obs_idx].resource == Resource.GN else Resource.GS,
                                    self.start_time,
                                    self.stop_time)
            scheduled = chromosome.insert(obs_idx)

            # Now if we could schedule, add the chromosome to its appropriate list.
            if scheduled:
                chromosomes.append(chromosome)
            else:
                unused_genes.append(obs_idx)

        return chromosomes, unused_genes
