# solver.py
# By Sebastian Raaphorst, 2020.

from typing import List, Tuple
from random import seed, sample
from math import ceil
from observations import *

# A gene is an observation number, and a chromosome is a list of genes where order is important.
gene = int
chromosome = [int]


class Chromosome:
    """
    A chromosome for the genetic algorithm, which represents a schedule for a given site.
    """
    def __init__(self, observations: Observations, resource: Resource,
                 start_time: int = DEFAULT_START_TIME, stop_time: int = DEFAULT_STOP_TIME):
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
        #print(f"Schedule: {self.schedule}")
        sorted_start_times = sorted(self.schedule)
        #print(f"Sorted start times: {sorted_start_times}")
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
            next_end_time = self.stop_time if gene_idx == len(self.schedule) - 1 else sorted_start_times[gene_idx + 1][0]

            # If next_end_time - obs_time < lower_time, this would violate timing constraints, scheduling this
            # observation earlier than it is permitted.
            if next_end_time - obs_time < lower_time:
                continue

            # If the gap is big enough to accommodate this observation, add it to the list of gaps.
            if next_end_time - next_start_time >= obs_time:
                gap_start_times.append(next_start_time)

        return gap_start_times

    def determine_capacity(self) -> float:
        """
        Determine the amount of time currently used up in this chromosome.
        >= 85% is considered "optimal."
        :return: the length of time scheduled in this chromosome
        """
        return sum(self.observations[idx] for _, idx in self.schedule)

    def determine_fitness(self) -> float:
        """
        Determine the value of the chromosome, which is just the sum of the metric over its observations.
        :return: the sum of the metric over the observations
        """
        return sum(self.observations[idx].priority for _, idx in self.schedule)

    def insert(self, obs_idx) -> bool:
        """
        Try to insert obs_idx into this chromosome in the earliest possible position. This fails if:
        1. The observation resource is not compatible with this resource.
        2. The timing constraints do not allow it to fit.
        3. There are no gaps big enough to accommodate it.
        4. The observation is already in this chromosome.
        Otherwise, it is scheduled.
        :param obs_idx: the index of the observation to try to schedule
        :return: True if we could schedule, and False otherwise
        """

        #print(f"\nTrying to schedule observation {obs_idx}:")
        #print_observation(self.observations[obs_idx])
        #print(f"into chromosome {Resource(self.resource).name} {self.schedule}")
        # Check site compatibility.
        #print(f"Fetching {obs_idx}...")
        obs = self.observations[obs_idx]
        if obs.resource != Resource.Both and obs.resource != self.resource:
            return False

        # Determine if this observation is already in this chromosome.
        observations = [c[1] for c in self.schedule]
        if obs_idx in observations:
            return False

        #print("Calculating gaps...")
        # Get the gap start times in this chromosome in which we can schedule the observation.
        gap_start_times = self._get_gaps_in_range(obs_idx)
        if len(gap_start_times) == 0:
            return False

        # Schedule the observation in the first gap and sort the gaps.
        self.schedule.append((gap_start_times[0], obs_idx))
        self.schedule = sorted(self.schedule)
        #print(f"New schedule: {self.schedule}")
        return True

    def remove(self, gene_idx):
        """
        Remove the gene at the specified position in the chromosome.
        :param gene_idx: the index of the gene
        """
        self.schedule.pop(gene_idx)

    def __len__(self) -> int:
        """
        Access the number of elements in this chromosome.
        :return: the number of elements in the chromosome
        """
        return len(self.schedule)

    def __getitem__(self, gene_idx) -> (float, int):
        """
        Get the start time and obs_idx at position gene_idx in this chromosome.
        :param gene_idx: a valid int index in the range [0, len(self)).
        :return: the start time and index of the observation
        """
        return self.schedule[gene_idx]

    def __str__(self) -> str:
        return f"{Resource(self.resource).name} {self.schedule}: {self.determine_fitness()}"


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

    def __init__(self, observations: Observations,
                 start_time: int = DEFAULT_START_TIME, stop_time: int = DEFAULT_STOP_TIME):
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
        self.chromosomes = []
        self.unused_genes = []

    def _form_initial_population(self):
        """
        We form the initial population of chromosomes by putting them at the earliest period that we can.
        GS is given slight priority over GN in such that a new chromosome where the observation can be scheduled
        at both is scheduled at GS.
        """
        for obs_idx in range(len(self.observations)):
            #print(f"SCHEDULING {obs_idx}")
            # We can only schedule the observation in a chromosome corresponding to its site.
            # Chromosome.insert handles this, so we don't have to worry about it here.
            scheduled = False

            for chromosome in self.chromosomes:
                if chromosome.insert(obs_idx):
                    scheduled = True
                    break
            if scheduled:
                continue

            # Create a new chromosome and attempt to add it.
            chromosome = Chromosome(self.observations,
                                    Resource.GN if self.observations[obs_idx].resource == Resource.GN else Resource.GS,
                                    self.start_time,
                                    self.stop_time)
            scheduled = chromosome.insert(obs_idx)

            # Now if we could schedule, add the chromosome to its appropriate list.
            if scheduled:
                self.chromosomes.append(chromosome)
            else:
                self.unused_genes.append(obs_idx)
            self._sort_chromosomes()

            #print("Current chromosomes:")
            #for c in self.chromosomes:
            #    print(f"{Resource(c.resource).name} {c.schedule}")
            #print("DONE.")

    def _sort_chromosomes(self):
        """
        Sort the chromosomes by non-increasing fitness.
        """
        self.chromosomes = sorted(self.chromosomes, key=lambda x: x.determine_fitness(), reverse=True)

    def _selection(self) -> Tuple[int, int]:
        """
        In selecting two chromosomes, we sort them, and then pick one from the top 25% and one at random.
        :return: the indices of the chromosomes after sorting
        """
        self._sort_chromosomes()
        c1_index = randrange(0, ceil(len(self.chromosomes) / 4))
        c2_index = -1
        while c2_index == -1 or c2_index == c1_index:
            c2_index = randrange(0, len(self.chromosomes))

        if self.chromosomes[c1_index].determine_fitness() > self.chromosomes[c2_index].determine_fitness():
            return c1_index, c2_index
        else:
            return c2_index, c1_index

    def _mate(self) -> bool:
        """
        Mate two chromosomes. This only works if:
        1. The genes in the chromosomes represent the same resource.
        2. The timing of the scheduling does not clash with overlaps (any overlaps are just dropped).
        I'm not sure if and how CFHT handles this.

        If a valid chromosome is found out of the two candidates, pick the higher fitness one and replace the lower
        fitness one in the chromosome list.

        :return: True if mating succeeded, and false otherwise.
        """

        # Selection of two chromosomes.
        c1_index, c2_index = self._selection()
        c1 = self.chromosomes[c1_index]
        c2 = self.chromosomes[c2_index]

        # If they are different sites, then fail.
        if c1.resource != c2.resource:
            return False

        # Pick random crossover points in c1 and c2. We want some of each chromosome, so pick between
        # [1, len - 1) instead of [0, len).
        if len(c1) == 1 or len(c2) == 1:
            return False

        c1_point = randrange(1, len(c1))
        c2_point = randrange(1, len(c2))

        # Mate to produce the first chromosome.
        c3 = Chromosome(self.observations, c1.resource, self.start_time, self.stop_time)
        for i in range(0, c1_point):
            c3.insert(c1[i][1])
        for i in range(c2_point, len(c2)):
            c3.insert(c2[i][1])

        c4 = Chromosome(self.observations, c1.resource, self.start_time, self.stop_time)
        for i in range(0, c2_point):
            c4.insert(c2[i][1])
        for i in range(c1_point, len(c1)):
            c4.insert(c1[i][1])

        # If we have improvement in one of the matings, then replace the lower-valued chromosome.
        max_c = c3 if c3.determine_fitness() > c4.determine_fitness() else c4
        if max_c.determine_fitness() > c2.determine_fitness():
            self.chromosomes[c2_index] = max_c
            self._sort_chromosomes()
            return True

        return False

    def _interleave(self) -> bool:
        """
        Perform the interleave operation between chromosomes.
        """

        # Selection of two chromosomes.
        c1_index, c2_index = self._selection()
        c1 = self.chromosomes[c1_index]
        c2 = self.chromosomes[c2_index]

        # If they are different sites, then fail.
        if c1.resource != c2.resource:
            return False

        # Interleave to produce the first chromosome.
        c3 = Chromosome(self.observations, c1.resource, self.start_time, self.stop_time)
        c4 = Chromosome(self.observations, c1.resource, self.start_time, self.stop_time)
        for i in range(min(len(c1), len(c2))):
            c3.insert(c1[i][1] if i % 2 == 0 else c2[i][1])
            c4.insert(c2[i][1] if i % 2 == 0 else c1[i][1])

        # If we have improvement in one of the crossovers, then replace the lower-valued chromosome.
        max_c = c3 if c3.determine_fitness() > c4.determine_fitness() else c4
        if max_c.determine_fitness() > c2.determine_fitness():
            self.chromosomes[c2_index] = max_c
            self._sort_chromosomes()
            return True

        return False

    def _mutation_swap(self) -> bool:
        """
        Swap two genes in the chromosome. This never changes the fitness of the chromosome, but can result
        in illegal chromosomes, in which case, we ignore the result.
        """

        # Select a chromosome to swap.
        c_idx = randrange(0, len(self.chromosomes))
        c = self.chromosomes[c_idx]

        if len(c) < 2:
            return False

        # Sample two positions to swap.
        pos1, pos2 = sample(range(0, len(c)), 2)
        pos1, pos2 = (pos1, pos2) if pos1 > pos2 else (pos2, pos1)

        new_c = Chromosome(self.observations, c.resource, self.start_time, self.stop_time)
        new_c.schedule = c.schedule[:]
        new_c.remove(pos1)
        new_c.remove(pos2)
        new_c.insert(c[pos2][1])
        new_c.insert(c[pos1][1])

        if new_c.determine_fitness() == c.determine_fitness:
            self.chromosomes[c_idx] = new_c
            return True

        return False

    def _mutation_mix(self) -> bool:
        """
        Try to replace a random number of observations in a randomly selected chromosome.
        """

        # Select a chromosome to mix up.
        c_idx = randrange(0, len(self.chromosomes))
        c = self.chromosomes[c_idx]

        if len(c) == 1:
            return False

        n = randrange(1, len(c))
        genes_to_drop = sorted(sample(range(0, len(c)), n), reverse=True)

        new_c = Chromosome(self.observations, c.resource, self.start_time, self.stop_time)
        new_c.schedule = c.schedule[:]
        for i in genes_to_drop:
            new_c.remove(i)

        genes_to_add = sample(range(0, len(self.observations)), n)
        for i in genes_to_add:
            new_c.insert(i)

        if new_c.determine_fitness() >= c.determine_fitness():
            self.chromosomes[c_idx] = new_c
            self._sort_chromosomes()
            return True

        return False

    @staticmethod
    def _print_best_fitness(c_gn: Chromosome, c_gs: Chromosome, i: int = None) -> None:
        print(f"Best fitnesses{f': {i}' if i is not None else ''}")
        if c_gn is not None:
            print(f"\tGN: {c_gn.determine_fitness()}\t{c_gn.schedule}")
        if c_gs is not None:
            print(f"\tGS: {c_gs.determine_fitness()}\t{c_gs.schedule}")

    def run(self, max_iterations: int = 1000) -> Tuple[Chromosome, Chromosome]:
        """
        Run the genetic algorithm prototype and return the best chromosomes for GN and GS.
        :param max_iterations: the maximum number of iterations to run
        :return: the best GN and GS chromosomes
        """
        # Initialize the chromosomes.
        self._form_initial_population()
        print("\n\n*** INITIAL CHROMOSOME POPULATION ***")
        for c in self.chromosomes:
            print(c)

        best_c_gn = None
        best_c_gs = None

        for i in range(len(self.chromosomes)):
            chromosome = self.chromosomes[i]
            if chromosome.resource == Resource.GN and best_c_gn is None:
                best_c_gn = chromosome
            if chromosome.resource == Resource.GS and best_c_gs is None:
                best_c_gs = chromosome
            if best_c_gn is not None and best_c_gs is not None:
                break

        self._print_best_fitness(best_c_gn, best_c_gs)
        for i in range(max_iterations):
            # Perform all the operations.
            self._mate()
            self._interleave()
            self._mutation_swap()
            self._mutation_mix()

            # See if we have a better best chromosome.
            chromosome = self.chromosomes[0]
            new_best = False
            if chromosome.resource == Resource.GN and chromosome.determine_fitness() > best_c_gn.determine_fitness():
                best_c_gn = self.chromosomes[0]
                new_best = True
            if chromosome.resource == Resource.GS and chromosome.determine_fitness() > best_c_gs.determine_fitness():
                best_c_gs = self.chromosomes[0]
                new_best = True
            if new_best:
                self._print_best_fitness(best_c_gn, best_c_gs, i)

        print("\n\nFINAL BEST FITNESSES:")
        self._print_best_fitness(best_c_gn, best_c_gs)

        return best_c_gn, best_c_gs


if __name__ == '__main__':
    seed(0)
    o = generate_random_observations(1000)
    ga = GeneticAlgorithm(o)
    ga.run()