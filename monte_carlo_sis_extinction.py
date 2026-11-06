def monte_carlo_sis_extinction(adjacency_matrix, infection_rates, curing_rates, NIIN, max_time_horizon=1,
                               iterations=10000, confidence_level=0.95, record_trajectory=False, plot_network=True,
                               show_progress=False, max_events_per_simulation=1000000, seed=None):
    """
    Function's Goal
    ---------------
    To estimate the extinction time and probability of extinction in a continuous-time, network-based 
    Susceptible–Infected–Susceptible (SIS) epidemic model using Monte Carlo simulation.
    
    SIS Epidemic Model Overview
    ---------------------------
    This function simulates the stochastic spread of an SIS epidemic model over a network containing N
    node(s), where N is finite. The SIS epidemic model assumes that a node is in one of two states: 
    
        0 : Susceptible-to-infection (S) (also known as healthy and susceptible), or
        1 : Infected (I)

    This function assumes a heterogeneous SIS epidemic model, where infection and curing rates are node 
    dependent. Infection rates are non-negative, while curing rates are strictly positive.

    An infected node i can infect each of its susceptible neighbours at rate infection_rates[i], weighted 
    by the corresponding adjacency matrix entry. An infected node i can be cured at rate curing_rates[i]. 
    Infection and curing event waiting times are conditionally independent given the current epidemic state.
    
    Cured node(s) can return to the infected state.

    The epidemic evolves as a continuous-time Markov chain (CTMC). At each step, independent exponentially
    distributed waiting times are generated for all possible node transitions. The next event is determined
    by selecting the node with the smallest waiting time using np.argmin. The possible state transitions are:

        Susceptible-to-infection -> Infected
        Infected                 -> Susceptible-to-infection

    Let T denote the random time until the epidemic reaches extinction, defined as the first time at which
    no infected node(s) remain in the network.

    The function performs multiple independent Monte Carlo simulations to estimate four principal extinction
    statistics:
        1. Probability of Extinction by Time H: P(T<=H).
        2. Unrestricted Expected Extinction time: E[T].
        3. Restricted Expected Extinction time: E[min(T,H)].
        4. Conditional Expected Extinction time: E[T|T<= H].

    Confidence intervals quantify Monte Carlo sampling uncertainty.

    Simulations are first observed up to a fixed time horizon H. Those that have not reached extinction by H
    are subsequently continued to estimate the unrestricted extinction time.

    Parameters
    ----------
    adjacency_matrix : array_like of shape (N, N)
        Non-negative adjacency matrix describing the network topology and infection interactions between nodes.
        The entry adjacency_matrix[i,j] represents the connection from node i to node j, where i not equal to j.
        For an undirected network, the matrix is symmetric. For a directed network, it may be asymmetric.

    infection_rates : array_like of shape (N,) or (1, N)
        Row vector containing the non-negative infection rate of each node. For a susceptible node i, the 
        infection rate is:

             sum^{N}_{j=1} adjacency_matrix[j,i] * infection_rates[j] * X[j],

        where X[j] equals 0, or 1 depending on whether node j is susceptible or infected, respectively. 
        Only infected neighbours contribute to the infection rate.

    curing_rates : array_like of shape (N,) or (1, N)
        Row vector containing strictly positive curing rates of each node. An infected node i cures at 
        rate curing_rates[i]. Once cured, node i can become infected again.

    NIIN : int
        Number of initially infected node(s). Must satisfy 1 <= NIIN <= N. At the beginning of each Monte Carlo
        simulation, NIIN distinct nodes are selected uniformly at random without replacement. All remaining 
        node(s) are initially susceptible.

    max_time_horizon : float, optional
        Positive observation time horizon H used to estimate the restricted and conditional extinction-time
        statistics and the probability of extinction by time H. Default is 1.

    iterations : int, optional
        Number of independent Monte Carlo simulations used to estimate the extinction-time statistics and 
        extinction probability. A larger number of iterations generally improves the precision of the Monte 
        Carlo estimates at the expense of additional computational time. Default is 10,000.

    confidence_level : float, optional
        Confidence level used to construct confidence intervals for the estimated quantities. Must satisfy 
        0 < confidence_level < 1. Default is 0.95, corresponding to 95% confidence intervals.

    record_trajectory : bool, optional
        If True, records and plots the number of infected nodes over time for the first Monte Carlo 
        simulation, up to the time horizon H or until an earlier stopping condition. Default is False.

    plot_network : bool, optional
        If True, displays the network topology using NetworkX, with node labels numbered from 1 to N. 
        Default is True.

    show_progress : bool, optional
        If True, displays a progress bar showing the completion of the Monte Carlo simulations. Requires
        the tqdm package. Default is False.

    max_events_per_simulation : int, optional
        Maximum number of epidemic transition events permitted in each Monte Carlo simulation. This parameter 
        provides a computational safeguard against excessively long simulations. If this limit is reached 
        before extinction, the corresponding unrestricted simulation is marked as incomplete.
        Default is 1,000,000.

    seed : int or None, optional
        Random seed used to initialise the NumPy random number generator, allowing reproducible Monte Carlo 
        simulations. If None, the random number generator is initialised without a user-specified seed. Default 
        is None.

    Returns
    -------
    results : dict
        Dictionary containing the estimated extinction-time statistics, extinction probability, confidence
        intervals, simulation completion information, and optional epidemic trajectory. The principal
        dictionary entries are:

            "mean_extinction_time" : float
                Estimated restricted mean extinction time - E[min(T,H)].

            "standard_error_time" : float
                Estimated standard error of the restricted mean extinction time.

            "confidence_interval_time" : tuple of float
                Approximate confidence interval for E[min(T,H)].

            "conditional_mean_extinction_time" : float
                Estimated conditional mean extinction time - E[T|T<=H].

            "conditional_standard_error" : float
                Estimated standard error of the conditional mean extinction time.

            "conditional_confidence_interval" : tuple of float
                Approximate confidence interval for E[T|T<=H].

            "unrestricted_mean_extinction_time" : float
                Estimated unrestricted mean extinction time - E[T].

            "unrestricted_standard_error" : float
                Estimated standard error of the unrestricted mean extinction time, when available.

            "unrestricted_confidence_interval" : tuple of float
                Approximate confidence interval for E[T], when available.

            "extinction_probability" : float
                Estimated probability that the epidemic becomes extinct by time H - P(T<=H).

            "confidence_interval_probability" : tuple of float
                Wilson score confidence interval for P(T<=H).

            "restricted_complete" : bool
                Indicates whether every simulation was successfully observed through extinction or the time
                horizon H.

            "unrestricted_complete" : bool
                Indicates whether all simulations eventually reached extinction without encountering a 
                stopping limit.

            "number_extinct" : int
                Number of Monte Carlo simulations that reached extinction by time H.

            "number_unrestricted_extinctions" : int
                Number of simulations that eventually reached extinction, including those continued beyond H.

            "number_incomplete_by_horizon" : int
                Number of simulations terminated before reaching extinction or the observation horizon H.

            "number_incomplete" : int
                Number of unrestricted simulations terminated because of computational or numerical 
                limitations.

            "number_infinite" : int
                Number of simulations that entered a non-extinct state with no possible further transitions, 
                resulting in an infinite extinction time.

            "iterations" : int
                Total number of requested Monte Carlo simulations.

            "trajectory" : dict or None
                Recorded epidemic trajectory from the first simulation when record_trajectory=True; otherwise 
                None. Contains time points, infection counts, extinction status, and stopping information.

    Notes
    -----
    1. The SIS model allows repeated infections because cured nodes immediately return to the susceptible
        state.

    2. The extinction state, in which all nodes are susceptible, is absorbing: once the epidemic reaches 
        extinction, no further infections can occur without external infection.

    3. Confidence intervals for mean extinction times use a normal approximation, whereas the extinction 
        probability confidence interval uses the Wilson score method.

    4. The unrestricted mean extinction time may be infinite when the epidemic can become trapped in a 
        non-extinct state, for example if infected nodes have zero curing rates and cannot otherwise recover.

    5. Statistics that cannot be reliably estimated because of incomplete simulations are returned as np.nan.

    6. The network graph is automatically visualised using NetworkX, with node labels numbered from 1 to N.
    """

    # ==========================================================
    # 1. REQUIRED LIBRARIES
    # ==========================================================
    
    import numbers
    from statistics import NormalDist
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    import textwrap

    # ==========================================================
    # 2. INPUT VALIDATION
    # ==========================================================

    def validate_nonnegative_array(value, name, shape=None):
        """
        Validate the input as a real-valued NumPy array containing only finite, non-negative values, and convert 
        it to float64.
        """
        
        # Convert the input to a NumPy array.
        try:
            array = np.asarray(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a real numeric array.") from exc

        # Validate the array shape, if specified.
        if shape is not None and array.shape != shape:
            raise ValueError(f"{name} must have shape {shape}.")

        # Ensure all elements have a real numeric data type.
        if not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.complexfloating):
            raise ValueError(f"{name} must contain real numeric values.")

        # Convert the array to double-precision floating-point values.
        try:
            array = array.astype(np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} cannot be converted to float64.") from exc

        # Ensure all elements are finite.
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain finite values.")

        # Ensure all elements are non-negative.
        if np.any(array < 0):
            raise ValueError(f"{name} must contain non-negative values.")

        return array

    # Validate the adjacency matrix.
    adjacency_matrix = validate_nonnegative_array(adjacency_matrix, "adjacency_matrix")

    # Ensure the adjacency_matrix is a non-empty square matrix.
    if adjacency_matrix.ndim != 2 or adjacency_matrix.shape[0] == 0 or adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
        raise ValueError("adjacency_matrix must be a non-empty square matrix.")

    # Validate infection and curing rates as finite, non-negative numerical arrays.
    infection_rates = validate_nonnegative_array(infection_rates, "infection_rates")
    curing_rates = validate_nonnegative_array(curing_rates, "curing_rates")

    # Ensure strictly positive curing rates.
    if np.any(curing_rates <=0):
        raise ValueError("curing_rates must contain strictly positive values.")

    # Determine the number of nodes and define the accepted rate vector shapes.
    number_of_nodes = adjacency_matrix.shape[0]
    N = number_of_nodes
    valid_shapes = {(number_of_nodes,), (1, number_of_nodes)}

    # Ensure infection_rates contains exactly one infection rate per node.
    if infection_rates.shape not in valid_shapes:
        raise ValueError(f"infection_rates must have shape ({number_of_nodes},) or (1, {number_of_nodes}).")

    # Ensure curing_rates contains exactly one curing rate per node.
    if curing_rates.shape not in valid_shapes:
        raise ValueError(f"curing_rates must have shape ({number_of_nodes},) or (1, {number_of_nodes}).")

    # Convert both rate vectors to one-dimensional arrays.
    infection_rates = infection_rates.ravel()
    curing_rates = curing_rates.ravel()

    # Ensure NIIN is an integer between 1 and N (inclusive), excluding Boolean values.
    if isinstance(NIIN, (bool, np.bool_)) or not isinstance(NIIN, numbers.Integral) or not 1 <= NIIN <= number_of_nodes:
        raise ValueError(f"NIIN must be an integer between 1 and {number_of_nodes}.")

    # Ensure max_time_horizon is a real number, excluding Boolean values.
    if isinstance(max_time_horizon, (bool, np.bool_)) or not isinstance(max_time_horizon, numbers.Real):
        raise ValueError("max_time_horizon must be a positive real number.")

    # Convert max_time_horizon to double-precision floating-point value.
    max_time_horizon = float(max_time_horizon)
    
    # Ensure max_time_horizon is finite and non-negative number.
    if not np.isfinite(max_time_horizon) or max_time_horizon <= 0:
        raise ValueError("max_time_horizon must be positive and finite.")

    # Ensure iterations is a positive integer within the int64 range, excluding Boolean values.
    if isinstance(iterations, (bool, np.bool_)) or not isinstance(iterations, numbers.Integral) or not 1 <= iterations <= np.iinfo(np.int64).max:
        raise ValueError("iterations must be a positive integer.")

    # Ensure max_events_per_simulation is a positive integer, excluding Boolean values.
    if (isinstance(max_events_per_simulation, (bool, np.bool_)) or not isinstance(max_events_per_simulation, numbers.Integral)
        or max_events_per_simulation < 1):
        raise ValueError("max_events_per_simulation must be a positive integer.")

    # Ensure seed is None or a non-negative integer within the int64 range, excluding Boolean values.
    if seed is not None:
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, numbers.Integral) or not 0 <= seed <= np.iinfo(np.int64).max:
            raise ValueError("seed must be None or a non-negative integer within the int64 range.")

    # Ensure confidence_level is a real number, excluding Boolean values.
    if isinstance(confidence_level, (bool, np.bool_)) or not isinstance(confidence_level, numbers.Real):
        raise ValueError("confidence_level must be a real number.")

    # Convert confidence_level to a double-precision floating-point value.
    confidence_level = float(confidence_level)

    # Ensure confidence_level is strictly between 0 and 1.
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be strictly between 0 and 1.")

    # Ensure record_trajectory is a Boolean value.
    if not isinstance(record_trajectory, (bool, np.bool_)):
        raise ValueError("record_trajectory must be True or False.")

    # Ensure plot_network is a Boolean value.
    if not isinstance(plot_network, (bool, np.bool_)):
        raise ValueError("plot_network must be True or False.")

    # Ensure show_progress is a Boolean value.
    if not isinstance(show_progress, (bool, np.bool_)):
        raise ValueError("show_progress must be True or False.")

    # Compute the critical z-value using the lower tail for numerical stability.
    z = -NormalDist().inv_cdf((1.0 - confidence_level) / 2.0)

    # Ensure the confidence_level is not too close 1.
    if not np.isfinite(z):
        raise ValueError("confidence_level is too close to 1.")

    # Initialise the random number generator.
    rng = np.random.default_rng(seed)

    # Define the observation horizon.
    H = max_time_horizon

    # ==========================================================
    # 3. PRINT SIMULATION PARAMETERS
    # ==========================================================

    # Define the report width and display the main simulation heading.
    separator_num = 100
    print("\n" + " CONTINUOUS-TIME SIS EPIDEMIC: MONTE CARLO SIMULATION ".center(separator_num, "="))

    # Create a reusable separator and display the simulation parameters heading.
    separator = "-" * separator_num
    print(f"\nSIMULATION PARAMETERS\n{separator}")

    # Format rate vectors without removing non-integer values.
    def format_vector(vector):
        return "[" + ", ".join(f"{x:g}" for x in vector) + "]"

    # Organise the simulation parameters into a dictionary.
    parameters = {
        "Number of Nodes": number_of_nodes,
        "Initially Infected Node(s) (Uniformly at Random)": NIIN,
        "Infection Rates Vector": format_vector(infection_rates),
        "Curing Rates Vector": format_vector(curing_rates),
        "Maximum Time Horizon (H)": H,
        "Monte Carlo Iterations": f"{iterations:,}",
        "Random Seed": seed,
        "Confidence Level": f"{confidence_level:.2%}",
        "Maximum Events per Simulation": f"{max_events_per_simulation:,}"
        }

    # Print parameters with automatic line wrapping and aligned continuation lines.
    label_width = 51
    value_indent = "   " + " " * label_width + " "

    for label, value in parameters.items():
        prefix = f"   {label + ':':<{label_width}} "

        wrapped = textwrap.fill(str(value), width=separator_num, initial_indent=prefix,
                                subsequent_indent=value_indent, break_long_words=False,
                                break_on_hyphens=False
                                )

        print(wrapped)

    # ==========================================================
    # 4. AUTOMATICALLY PLOT NETWORK GRAPH
    # ==========================================================

    if plot_network:
        
        print(f"\nNETWORK TOPOLOGY\n{separator}")
    
        # Determine whether the adjacency matrix represents an undirected graph.
        is_symmetric = np.array_equal(adjacency_matrix, adjacency_matrix.T)
    
        # Construct an undirected or directed graph based on matrix symmetry.
        if is_symmetric:
            G = nx.from_numpy_array(adjacency_matrix)
        else:
            G = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)
    
        # Convert node labels from 0,...,N-1 to 1,...,N.
        G = nx.relabel_nodes(G, {i: i + 1 for i in range(number_of_nodes)})
    
        fig, ax = plt.subplots(figsize=(9, 7))
    
        # Compute node positions using a reproducible force-directed layout.
        positions = nx.spring_layout(G, seed=42)
    
        # Draw the graph with node labels, edges, and arrows for directed graphs.
        nx.draw_networkx(G, pos=positions, ax=ax, with_labels=True, 
                         node_color="lightblue", node_size=700,
                         edge_color="gray", font_size=10, font_weight="bold",
                         arrows=G.is_directed())
    
        ax.set_title("Network Topology for the SIS Epidemic Model to Spread Over")
        ax.axis("off")
    
        plt.tight_layout()
        plt.show()
        plt.close(fig)

    # ==========================================================
    # 5. STATISTICAL HELPERS
    # ==========================================================

    def mean_and_ci(values):
        """
        Return the sample mean, standard error, and approximate normal confidence interval.
        """

        # Convert the input observations to a double-precision NumPy array.
        values = np.asarray(values, dtype=np.float64)

        # Return undefined statistics if no observations are available.
        if values.size == 0:
            return np.nan, np.nan, (np.nan, np.nan)

        mean = float(np.mean(values))

        # Return the mean without uncertainty estimates if only one observation is available.
        if values.size == 1:
            return mean, np.nan, (np.nan, np.nan)

        # Compute the standard error using the sample standard deviation (ddof=1).
        standard_error = float(np.std(values, ddof=1) / np.sqrt(values.size))

        # Compute the confidence interval margin of error using the standard normal critical value.
        margin = z * standard_error

        return mean, standard_error, (max(0.0, mean - margin), mean + margin)
                
    def wilson_interval(successes, total):
        """
        Compute the Wilson score confidence interval for a binomial proportion at the specified 
        confidence level.

        In the Monte Carlo SIS epidemic simulation, each independent simulation is treated as a 
        Bernoulli trial, where success corresponds to the epidemic reaching extinction within the 
        specified time horizon. Extinction occurs when no infected nodes remain, after which no 
        further infections are possible. If the true probability of extinction within the specified 
        time horizon is p, the total number of successful extinctions follows a binomial 
        distribution:

                S ~ Binomial(M,p),

        where S is the number of simulations reaching extinction within the time horizon and M is 
        the total number of independent simulations. The sample proportion is given by:

                p_hat = S / M and p_hat -> p as M -> infinity.

        By the Law of Large Numbers (LLN), p_hat converges to the true probability of extinction 
        within the specified time horizon, p, as M approaches infinity. By the Central Limit Theorem
        (CLT), the standardized sample proportion converges in distribution to a standard normal 
        random variable:

                (p_hat - p) / sqrt(p * (1 - p) / M) ->d N(0, 1) as M -> infinity,

        approximately for sufficiently large M, provided 0 < p < 1.

        The Wilson score confidence interval is obtained by inverting the corresponding normal 
        approximation score test and solving for the unknown probability p.
        """

        # Return undefined confidence bounds if no trials are available.
        if total == 0:
            return (np.nan, np.nan)

        # Compute the observed proportion of successful trials.
        p = successes / total

        # Compute the Wilson score denominator.
        denominator = 1 + (z**2 / total)

        # Compute the center of the Wilson score confidence interval.
        center = (p + z**2 / (2 * total)) / denominator

        # Compute the half-width of the Wilson score confidence interval.
        half_width = (z / denominator * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)))

        return max(0.0, center - half_width), min(1.0, center + half_width)

    # ==========================================================
    # 6. SIMULATION HELPERS
    # ==========================================================

    def calculate_rates(state):
        """
        Calculate transition rates for all nodes in the heterogeneous network-based SIS model.
        
        A susceptible node i becomes infected at total rate:

                sum_{j=1}^{N} a_ji * beta_j * X_j(t),

        where a_ji represents the directed connection from node j to node i, beta_j >= 0 is 
        the infection rate of node j, and X_j(t) is its state.

        An infected node i is cured at rate delta_i > 0.
        """

        # Initialise the transition-rate vector with zero rates.
        rates = np.zeros(N, dtype=float)

        # Calculate each node's rate according to its current state.
        for i in range(N):

            if state[i] == 0:
                # Susceptible node i: infection rate depends on the infectivity of its infected 
                # neighbours.
                rates[i] = np.sum(adjacency_matrix[:, i] * infection_rates * state)

            else:
                # Infected node i: cured at its node-dependent rate.
                rates[i] = curing_rates[i]

        # Ensure all calculated rates are finite.
        if not np.all(np.isfinite(rates)):
            raise ValueError("Transition rates must be finite.")

        # Ensure no calculated rate is negative.
        if np.any(rates < 0):
            raise ValueError("Transition rates must be non-negative.")

        return rates

    def simulate_phase(state, current_time, end_time, events_used, trajectory_times=None, trajectory_counts=None):
        """
        Simulate until extinction, the observation horizon, trapping, or a computational limit.
        """

        # Allocate an array to store the exponential waiting time for each node.
        waiting_times = np.empty(number_of_nodes, dtype=np.float64)

        # Continue simulating while at least one node remains infected.
        while np.any(state):

            # Stop if the specified time horizon has been reached.
            if end_time is not None and current_time >= end_time:
                return state, current_time, events_used, "horizon"

            # Compute transition rates and identify nodes with positive rates.
            rates = calculate_rates(state)
            positive = rates > 0

            # Stop if no further state transitions are possible.
            if not np.any(positive):
                return state, current_time, events_used, "trapped"

            # Initialise waiting times to infinity for nodes with zero transition rates.
            waiting_times.fill(np.inf)

            # Generate independent exponential waiting times for possible transitions.
            waiting_times[positive] = rng.exponential(size=np.count_nonzero(positive)) / rates[positive]

            # Identify the next transition event and its waiting time.
            event_node = int(np.argmin(waiting_times))
            waiting_time = float(waiting_times[event_node])

            # Stop at the time horizon if the next event would occur beyond it.
            if end_time is not None:
                
                # Compute the remaining simulation time before the horizon.
                remaining = end_time - current_time

                # Stop at the horizon if no transition occurs before it.
                if waiting_time > remaining:
                    return state, end_time, events_used, "horizon"

            # Stop if the maximum permitted number of transition events has been reached.
            if events_used >= max_events_per_simulation:
                return state, current_time, events_used, "event_limit"

            # Compute the absolute time of the next transition event.
            next_time = current_time + waiting_time

            # Stop if the next event time is non-finite or cannot advance numerically.
            if not np.isfinite(next_time) or next_time <= current_time:
                return state, current_time, events_used, "numerical_limit"

            # Advance the simulation clock to the next transition time.
            current_time = next_time

            # Toggle the selected node's infection state (0 -> 1 or 1 -> 0).
            state[event_node] = 1 - state[event_node]

            # Increment the total number of completed state transitions.
            events_used += 1

            # Record the trajectory if trajectory tracking is enabled.
            if trajectory_times is not None:
                trajectory_times.append(current_time)
                trajectory_counts.append(int(np.sum(state)))

        return state, current_time, events_used, "extinct"

    # ==========================================================
    # 7. INITIALISE MONTE CARLO STORAGE
    # ==========================================================

    # Initialise lists to store extinction-time observations.
    restricted_times = []
    conditional_times = []
    unrestricted_times = []

    # Initialise counters for extinction events.
    extinct_by_horizon_count = 0
    unrestricted_extinction_count = 0

    # Initialise counters for incomplete simulations and trapped states.
    incomplete_by_horizon = 0
    incomplete_unrestricted = 0
    trapped_count = 0

    # Initialise storage for the recorded simulation trajectory.
    trajectory = None

    # Enable a progress bar if requested.
    if show_progress:
        try:
            from tqdm import tqdm
        except ImportError as exc:
            raise ImportError("Install tqdm with: pip install tqdm") from exc

        # Create a progress bar for the Monte Carlo iterations.
        simulation_range = tqdm(range(iterations),desc="Simulating the SIS epidemic model spread")
    else:
        
        # Iterate without displaying a progress bar.
        simulation_range = range(iterations)

    # ==========================================================
    # 8. MONTE CARLO LOOP
    # ==========================================================

    # Perform the specified number of independent Monte Carlo simulations.
    for simulation_index in simulation_range:

        # Initialise all nodes in the susceptible state (state = 0).
        state = np.zeros(number_of_nodes, dtype=np.int8)

        # Select NIIN distinct nodes uniformly at random for initial infection.
        initially_infected = rng.choice(number_of_nodes, size=NIIN,replace=False)

        # Set the selected nodes to the infected state (state = 1).
        state[initially_infected] = 1

        # Initialise the simulation clock and transition counter.
        current_time = 0.0
        events_used = 0

        # Enable trajectory recording only for the first simulation, if requested.
        save_first = (record_trajectory and simulation_index == 0)

        # Initialise trajectory observations at time zero.
        times = [0.0] if save_first else None
        counts = [NIIN] if save_first else None

        # ------------------------------------------------------
        # Phase 1: Simulate up to the time horizon H.
        # ------------------------------------------------------

        # Simulate until extinction, the time horizon, or another stopping condition.
        state, current_time, events_used, reason = simulate_phase(
            state=state,
            current_time=current_time,
            end_time=H,
            events_used=events_used,
            trajectory_times=times,
            trajectory_counts=counts
            )

        # Determine whether all nodes are in the susceptible state.
        extinct_by_horizon = not np.any(state)

        # Identify simulations extinction outcome.
        horizon_observed = reason in {"extinct", "horizon", "trapped"}

        # Record extinction-time observations when the horizon outcome is known.
        if horizon_observed:

            if extinct_by_horizon:

                # Count extinctions occurring within the specified time horizon.
                extinct_by_horizon_count += 1

                # Record the observed extinction time for both estimators.
                restricted_times.append(current_time)
                conditional_times.append(current_time)
            else:

                # Record H when extinction has not occurred by the horizon.
                restricted_times.append(H)

        else:

            # Count simulations interrupted before the horizon outcome is known.
            incomplete_by_horizon += 1

        # ------------------------------------------------------
        # Record the first simulation trajectory.
        # ------------------------------------------------------

        if save_first:

            # Extend the trajectory to H if the epidemic has not become extinct.
            if horizon_observed and not extinct_by_horizon and times[-1] < H:
                times.append(H)
                counts.append(counts[-1])

            # Store the recorded trajectory and its termination information.
            trajectory = {
                "times": np.asarray(times),
                "infected_counts": np.asarray(counts),
                "extinct": extinct_by_horizon,
                "complete_to_horizon": horizon_observed,
                "stop_reason": reason
                }

        # ------------------------------------------------------
        # Phase 2: Continue beyond the time horizon H.
        # ------------------------------------------------------

        # Continue unrestricted simulation if the epidemic survives until H.
        if not extinct_by_horizon and reason == "horizon":

            state, current_time, events_used, reason = simulate_phase(
                state=state,
                current_time=H,
                end_time=None,
                events_used=events_used
                )

        # ------------------------------------------------------
        # Unrestricted outcome
        # ------------------------------------------------------

        # Record the extinction time if the disease-free state is reached.
        if not np.any(state):
            
            unrestricted_extinction_count += 1
            unrestricted_times.append(current_time)

        # Count simulations trapped in a non-extinct state.
        elif reason == "trapped":
            
            trapped_count += 1

        # Count simulations terminated without a determined extinction time.
        else:
            
            incomplete_unrestricted += 1

    # ==========================================================
    # 9. RESTRICTED STATISTICS
    # ==========================================================

    # Determine whether all horizon-restricted simulation outcomes are known.
    restricted_complete = (incomplete_by_horizon == 0)

    if restricted_complete:

        # Compute the restricted mean extinction time, standard error, and confidence interval.
        restricted_mean, restricted_se, restricted_ci = mean_and_ci(restricted_times)

        # Restrict the confidence interval bounds to the valid range [0, H].
        restricted_ci = (max(0.0, restricted_ci[0]) if np.isfinite(restricted_ci[0]) else np.nan,
                         min(H, restricted_ci[1]) if np.isfinite(restricted_ci[1]) else np.nan)

        # Estimate the probability of extinction within the time horizon H.
        extinction_probability = extinct_by_horizon_count / iterations

        # Compute the Wilson score confidence interval for the extinction probability.
        probability_ci = wilson_interval(extinct_by_horizon_count, iterations)

    else:

        # Mark restricted statistics as undefined when any horizon outcome is incomplete.
        restricted_mean = np.nan
        restricted_se = np.nan
        restricted_ci = (np.nan, np.nan)

        # Mark the extinction probability and its confidence interval as undefined.
        extinction_probability = np.nan
        probability_ci = (np.nan, np.nan)

    # ==========================================================
    # 10. CONDITIONAL STATISTICS
    # ==========================================================

    # Compute conditional statistics only if all horizon outcomes are known 
    # and at least one simulation reaches extinction within the time horizon.
    if restricted_complete and conditional_times:

        # Compute the conditional mean extinction time, standard error, and confidence interval.
        conditional_mean, conditional_se, conditional_ci = mean_and_ci(conditional_times)

    else:

        # Mark conditional statistics as undefined when the required conditions are not met.
        conditional_mean = np.nan
        conditional_se = np.nan
        conditional_ci = (np.nan, np.nan)

    # ==========================================================
    # 11. UNRESTRICTED STATISTICS
    # ==========================================================

    # Determine whether extinction times were observed for all simulations.
    unrestricted_complete = (unrestricted_extinction_count == iterations)

    if unrestricted_complete:

        # Compute the unrestricted mean extinction time, standard error,
        # and confidence interval.
        unrestricted_mean, unrestricted_se, unrestricted_ci = mean_and_ci(unrestricted_times)

    elif trapped_count > 0:

        # A trapped non-extinct state has T = infinity.
        unrestricted_mean = np.inf
        unrestricted_se = np.nan
        unrestricted_ci = (np.nan, np.nan)

    else:

        # Mark unrestricted statistics as undefined when some extinction times are incomplete.
        unrestricted_mean = np.nan
        unrestricted_se = np.nan
        unrestricted_ci = (np.nan, np.nan)

    # ==========================================================
    # 12. RETURNED RESULTS
    # ==========================================================

    results = {
        "mean_extinction_time": restricted_mean,
        "standard_error_time": restricted_se,
        "confidence_interval_time": restricted_ci,

        "conditional_mean_extinction_time": conditional_mean,
        "conditional_standard_error": conditional_se,
        "conditional_confidence_interval": conditional_ci,

        "unrestricted_mean_extinction_time": unrestricted_mean,
        "unrestricted_standard_error": unrestricted_se,
        "unrestricted_confidence_interval": unrestricted_ci,

        "extinction_probability": extinction_probability,
        "confidence_interval_probability": probability_ci,

        "restricted_complete": restricted_complete,
        "unrestricted_complete": unrestricted_complete,

        "number_extinct": extinct_by_horizon_count,
        "number_unrestricted_extinctions": unrestricted_extinction_count,
        "number_incomplete_by_horizon": incomplete_by_horizon,
        "number_incomplete": incomplete_unrestricted,
        "number_infinite": trapped_count,

        "iterations": iterations,
        "trajectory": trajectory
    }

    # ==========================================================
    # 13. EXTINCTION TIME STATISTICS REPORT
    # ==========================================================

    # Display the extinction time statistics heading and define T.
    print(f"\nEXTINCTION TIME STATISTICS\n{separator}")
    print("\n   Let T denote the time until the epidemic reaches extinction.")

    # Report the restricted mean extinction time, standard error, and confidence 
    # interval over the specified time horizon H.
    print(f"\n   1. RESTRICTED MEAN EXTINCTION TIME - E[min(T,{H})]")
    print("      Extinct trajectories contribute their extinction time.")
    print(f"      Trajectories surviving at time {H} only contribute {H}.")
    print(f"\n      {f'E[min(T,{H})]:':<48} {restricted_mean:.6f}")
    print(f"      {'Standard Error:':<48} {restricted_se:.6f}")
    print(f"      {f'{confidence_level:.1%} Confidence Interval:':<48} [{restricted_ci[0]:.6f}, {restricted_ci[1]:.6f}]")

    # Display a warning if the restricted statistics are incomplete.
    if not restricted_complete:
        print("      WARNING: Some trajectories stopped before H; restricted estimates are unavailable.")

    # Report the conditional mean extinction time, standard error, and confidence 
    # interval using only trajectories extinct by H.
    print(f"\n   2. CONDITIONAL MEAN EXTINCTION TIME - E[T|T<={H}]")
    print(f"      Includes only epidemics extinct by time {H}.")
    print(f"\n      {f'E[T|T<={H}]:':<48} {conditional_mean:.6f}")
    print(f"      {'Standard Error:':<48} {conditional_se:.6f}")
    print(f"      {f'{confidence_level:.1%} Confidence Interval:':<48} [{conditional_ci[0]:.6f}, {conditional_ci[1]:.6f}]")

    # Report the unrestricted mean extinction time, standard error, and confidence 
    # interval without imposing a time horizon.
    print("\n   3. MEAN EXTINCTION TIME - E[T]")
    print(f"\n      {'E[T]:':<48} {unrestricted_mean:.6f}")
    print(f"      {'Standard Error:':<48} {unrestricted_se:.6f}")
    print(f"      {f'{confidence_level:.1%} Confidence Interval:':<48} [{unrestricted_ci[0]:.6f}, {unrestricted_ci[1]:.6f}]")

    # Display warnings for trapped states or incomplete unrestricted simulations.
    if trapped_count > 0:
        print("      WARNING: At least one epidemic became trapped with infected nodes that cannot recover." )
    elif not unrestricted_complete:
        print("      WARNING: Some unrestricted trajectories did not complete.")

    # ==========================================================
    # 14. EXTINCTION PROBABILITY REPORT
    # ==========================================================

    # Report the estimated probability of extinction by time H and its Wilson
    # score confidence interval.
    print(f"\nEXTINCTION PROBABILITY\n{separator}")
    print(f"   The Probability of Extinction by Time {H} is defined as P(T<={H}).")
    print(f"\n   {f'Estimated P(T<={H}):':<51} {extinction_probability:.6f}")
    print(f"   {f'{confidence_level:.1%} Wilson Confidence Interval:':<51} [{probability_ci[0]:.6f}, {probability_ci[1]:.6f}]")

    # ==========================================================
    # 15. SIMULATION COMPLETION SUMMARY
    # ==========================================================

    # Report the total number of simulations, observed extinctions, incomplete 
    # trajectories, and trapped non-extinct states.
    print(f"\nSIMULATION COMPLETION SUMMARY\n{separator}")
    print(f"   {'Total Simulations:':<51} {iterations:,}")
    print(f"   {f'Extinct by Time {H}:':<51} {extinct_by_horizon_count:,}")
    print(f"   {'Extinct Eventually:':<51} {unrestricted_extinction_count:,}")
    print(f"   {f'Incomplete Before Time {H}:':<51} {incomplete_by_horizon:,}")
    print(f"   {'Incomplete Unrestricted Run(s):':<51} {incomplete_unrestricted:,}")
    print(f"   {'Trapped Non-Extinct Run(s):':<51} {trapped_count:,}")

    # ==========================================================
    # 16. OPTIONAL EPIDEMIC TRAJECTORY PLOT
    # ==========================================================

    # Display and plot the first simulated epidemic trajectory, if requested.
    if record_trajectory:

        # Retrieve the recorded transition times and infected node counts.
        times = trajectory["times"]
        counts = trajectory["infected_counts"]

        # Compute changes in the number of infected nodes between observations.
        changes = np.diff(counts)

        # Display the first simulated epidemic trajectory summary.
        print(f"\nFIRST SIMULATED EPIDEMIC TRAJECTORY\n{separator}")

        # Report the initial, maximum, and final numbers of infected nodes,
        # together with the total infection and curing events.
        print(f"   {'Initially Infected Node(s):':<51} {counts[0]}")
        print(f"   {'Maximum Infected Node(s):':<51} {np.max(counts)}")
        print(f"   {'Final Infected Node(s):':<51} {counts[-1]}")
        print(f"   {'New Infection Event(s):':<51} {np.count_nonzero(changes > 0)}")
        print(f"   {'Curing Event(s):':<51} {np.count_nonzero(changes < 0)}")
        print(f"   {f'Extinct by {H}:':<51} {trajectory['extinct']}")
        print(f"   {'Stop Reason:':<51} {trajectory['stop_reason'].title()}")

        # Report the extinction time, horizon completion, or premature termination.
        if trajectory["extinct"]:
            print(f"   {'Extinction Time:':<51} {times[-1]:.6f}")
        elif trajectory["complete_to_horizon"]:
            print(f"   {f'Observed Through by Time {H}:':<51} {H:.6f}")
        else:
            print(f"   Trajectory Stopped Before Time {H}.")

        # Format and display trajectory times with automatic line wrapping.
        trajectory_times = "[" + ", ".join(f"{t:.6f}" for t in times) + "]"
        prefix = f"   {'Trajectory Times:':<51}"
        value_indent = " " * len(prefix)
        print(textwrap.fill(trajectory_times, width=separator_num, initial_indent=prefix + " ",
                    subsequent_indent=value_indent + " ", break_long_words=False,
                    break_on_hyphens=False))

        # Format and display infected node counts with automatic line wrapping.
        infected_counts = "[" + ", ".join(str(c) for c in counts) + "]"
        prefix = f"   {'Infected Node Counts:':<51}"
        value_indent = " " * len(prefix)
        print(textwrap.fill(infected_counts, width=separator_num, initial_indent=prefix + " ",
                    subsequent_indent=value_indent + " ", break_long_words=False,
                    break_on_hyphens=False))

        print("\n\n")

        # Plot the continuous-time SIS epidemic trajectory as a step function, configure
        # its appearance, and display the figure.
        fig, ax = plt.subplots(figsize=(10,6))
        ax.step(times, counts, where="post", linewidth=2)
        ax.set_xlabel("Time (t)")
        ax.set_ylabel("Number of Infected Node(s)")
        ax.set_title("Continuous-Time SIS Epidemic Trajectory")
        ax.set_xlim(0, H)
        ax.set_ylim(0, number_of_nodes + 0.5)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        plt.close(fig)

    print('\n' + ' END OF FUNCTION OUTPUT '.center(100, '='))

    return results
