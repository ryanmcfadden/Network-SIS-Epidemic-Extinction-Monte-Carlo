# Monte Carlo Simulation of Extinction Times in Network-Based SIS Epidemics

## 1. Overview
This repository provides a Python implementation of a continuous-time, network-based stochastic Susceptible–Infected–Susceptible (SIS) epidemic model. During my PhD, I studied the SIS epidemic model on undirected networks. This implementation extends that setting to weighted and directed networks.

The primary objective is to estimate extinction-time statistics and the probability of extinction in a continuous-time, network-based SIS epidemic model using Monte Carlo simulation.

We consider a heterogeneous model that allows node-dependent infection and curing rates.

The implementation estimates four principal extinction statistics:
- Extinction probability by a specified observation horizon.
- Unrestricted expected extinction time.
- Restricted expected extinction time.
- Conditional expected extinction time.

We also calculate confidence intervals to quantify Monte Carlo sampling uncertainty.

## 2. SIS Epidemic Model Overview
The SIS epidemic model assumes that a node is in one of two states: 

- `0` — Susceptible-to-infection (also known as healthy and susceptible), or
- `1` — Infected.

The homogeneous (also called the classical) model assumes a constant infection rate $\beta\geq 0$ and a constant curing rate $\delta > 0$. An infected node can infect its neighbour(s) at rate $\beta$ and be cured at rate $\delta$, with infection and curing event waiting times conditionally independent given the current epidemic state. Note that if at least one node is infected and $\delta = 0$, the probability of extinction is zero, assuming each node has an infinite lifespan. 

The heterogeneous model assumes that either the infection rates are node-dependent, $\beta_{i} \geq 0$, and/or the curing rates are node-dependent, $\delta_{i} > 0$. An infected node $i$ can infect its neighbour(s) at rate $\beta_{i}$ and be cured at rate $\delta_{i}$, with infection and curing event waiting times conditionally independent given the current epidemic state.

The SIS epidemic model has a single absorbing state, corresponding to the configuration in which all nodes are susceptible.

### 2.1. Network Representation
We represent the network as a graph $G=(V,E)$, where $`V=\{1,\ldots,N\}`$ is the set of nodes and $E$ is the set of edges. The adjacency matrix $A=(a_{ij})_{i,j\in V}$ is an $(N\times N)$-dimensional matrix that stores the connections between nodes.

#### 2.1.1. Undirected Graph
For an undirected graph, the adjacency matrix is defined as follows, with its entries given by:

```math
a_{ij} =
\begin{cases}
1, & \{i,j\} \in E,\\
0, & \{i,j\} \notin E.
\end{cases}
```
The adjacency matrix is symmetric, so $a_{ij}=a_{ji}$ for all $i,j\in V$.

#### 2.1.2. Directed Graph
For a directed graph, an edge $i\to j$ means infection can pass from node $i$ to node $j$. The adjacency matrix is defined as follows, with its entries given by:

```math
a_{ij} =
\begin{cases}
1, & (i,j)\in E,\\
0, & \text{otherwise}.
\end{cases}
```

The reverse edge $j\to i$ is represented separately by $a_{ji}$, so $A$ can be asymmetric.

#### 2.1.3. Weighted Graph
A weighted graph assigns a non-negative weight $w_{ij}\geq 0$ to each connection, representing the strength or intensity of potential infection transmission between nodes. For a directed network, the adjacency matrix is defined as follows, with its entries given by:

```math
a_{ij} =
\begin{cases}
w_{ij}, & (i,j)\in E,\\
0, & \text{otherwise}.
\end{cases}
```

For an undirected weighted network, the same weight applies in both directions, so $a_{ij}=a_{ji}$. Setting every edge weight to $1$ gives the corresponding unweighted graph.

We assume that the network contains no self-loops, so $a_{ii}=0$ for all $i\in V$.

### 2.2. Exact Markovian Epidemic Model
Let $X_i(t)$ be the random variable representing the state of node $i$ at time $t \geq 0$, where

```math
X_i(t) =
\begin{cases}
0, & \text{if node } i \text{ is susceptible-to-infection at time } t, \\
1, & \text{if node } i \text{ is infected at time } t.
\end{cases}
```

The complete state of the epidemic at time $t$ is represented by the vector

```math
\overrightarrow{X} (t) = \left( X_1(t), \, X_2(t), \, \ldots \, ,\, X_N(t) \right).
```

The term _exact_ refers to explicitly tracking the state of each individual node, rather than representing the epidemic solely through aggregate counts of susceptible and infected nodes. Consequently, the model retains information about which nodes are susceptible and infected, allowing the network structure and heterogeneous transition rates to determine the epidemic's evolution.

The epidemic satisfies the Markov property: given the current states of all nodes in the network, its future evolution is independent of its past history.

### 2.3. Heterogeneous Transition Rates

A susceptible node can become infected and an infected node can become susceptible again. The transition rates for node $i$ are

```math
  X_{i} (t) = \begin{cases}
    0 \to 1, &\quad \text{at rate } \sum_{j=1}^{N} a_{ji}\,\beta_{j}\, X_j(t), \\
    1 \to 0, &\quad \text{at rate } \delta_i.
  \end{cases}
```

Here, $\beta_{j} \geq 0$ is the infection rate of node $j$, $\delta_{i} > 0$ is the curing rate of node $i$, and $X_{j}(t)$ is the state of node $j$ at time $t$.

## 3. Continuous-Time Monte Carlo Simulation
The epidemic evolves as a continuous-time Markov chain (CTMC), with infection and curing events governed by node-dependent transition rates.

The implementation uses the competing exponential clocks to simulate the epidemic's stochastic evolution.

At each simulation step:
  1. Calculate the infection rates of all susceptible nodes and the curing rates of all infected nodes, based on the current network state.
  2. Generate an independent exponential waiting time for each possible transition with a strictly positive rate.
  3. Identify the node with the smallest waiting time.
  4. Advance the simulation clock time by the selected waiting time.
  5. Update the state of the corresponding node from susceptible to infected or from infected to susceptible.
  6. Repeat the procedure until the epidemic becomes extinct or a specified stopping condition is reached.

For a possible transition with rate $r>0$, the waiting time $W$ follows an exponential distribution:

```math
W\sim\text{Exp} (r)
```

The waiting times are conditionally independent given the current epidemic state. After each event, the transition rates are recalculated to reflect the updated network configuration of susceptible and infected nodes.

The competing exponential clocks generate exact CTMC sample paths, subject to numerical precision and computational stopping limits.

By repeating the simulation independently across multiple Monte Carlo realisations, the implementation estimates the epidemic extinction probability and various extinction-time statistics, along with their corresponding confidence intervals.

## 4. Epidemic Extinction Time
The epidemic becomes extinct when all nodes in the network are susceptible.

Let $X_i(t)$ denote the state of node $i$ at time $t$, where $X_i(t)=1$ indicates that node $i$ is infected.

The epidemic extinction time is defined as
```math
  T=\inf\left\{t>0: \sum^{N}_{i=1} X_i(t) = 0 \right\}.
```

Since the network contains finitely many nodes and all curing rates are strictly positive, the SIS epidemic reaches extinction almost surely in finite time. Consequently, the unrestricted expected extinction time \(\mathbb{E}[T]\) is finite.

Let $H>0$ denote a specified observation time horizon. An epidemic may become extinct before, at, or after this horizon.

The implementation uses Monte Carlo simulation to estimate extinction-time statistics both within the observation horizon and without restricting extinction to that horizon.

## 5. Estimated Extinction Statistics
The implementation uses independent Monte Carlo realisations of the stochastic SIS epidemic model to estimate the extinction probability and several extinction-time statistics.

Let $T$ denote the epidemic extinction time, $H>0$ the observation time horizon, and $M$ the number of independent Monte Carlo realisations. 

Let $T_1,T_2,\ldots,T_M$ denote the corresponding extinction times.

### 5.1. Extinction Probability
The extinction probability by time $H$ is defined as
```math
\mathbb{P} (T\leq H).
```
Its Monte Carlo estimator is
```math
\widehat{\mathbb{P}(T\leq H)} = \frac{1}{M}\sum^{M}_{m=1} \mathbf{1}_{\{T_m\leq H\}},
```
where $`\mathbf{1}_{\{T_m\leq H\}}`$ equals $1$ if the epidemic becomes extinct by time $H$ in realisation $m$, and $0$ otherwise.

We calculate a Wilson score confidence interval to quantify the uncertainty in the estimated extinction probability.

### 5.2. Expected Extinction Time
The expected extinction time is defined as
```math
  \mathbb{E}[T].
```
It represents the expected time until the epidemic becomes extinct, without restricting extinction to the observation horizon $H$.

Its Monte Carlo estimator is
```math
  \widehat{\mathbb{E}[T]} = \frac{1}{M} \sum_{m=1}^{M} T_m.
```

### 5.3. Restricted Expected Extinction Time
The restricted mean extinction time is defined as
```math
\mathbb{E}[\min(T,H)].
```
It represents the expected extinction time truncated at the observation horizon $H$.

Its Monte Carlo estimator is
```math
\widehat{\mathbb{E}[\min(T,H)]} = \frac{1}{M}\sum_{m=1}^{M} \min(T_m,H).
```

Thus, realisations in which the epidemic remains active beyond $H$ contribute exactly $H$ to the estimate.

### 5.4. Conditional Expected Extinction Time
The conditional mean extinction time is defined as

```math
\mathbb{E}[T\mid T\leq H].
```

It represents the expected extinction time among epidemics that become extinct by the observation horizon $H$.

Its Monte Carlo estimator is

```math
\widehat{\mathbb{E}[T\mid T\leq H]} =
\frac{\sum_{m=1}^{M}T_m\,\mathbf{1}_{\{T_m\leq H\}}}
{\sum_{m=1}^{M}\mathbf{1}_{\{T_m\leq H\}}}.
```

This estimator is defined only when at least one Monte Carlo realisation becomes extinct by time $H$.

### 5.5. Confidence Intervals
The implementation calculates confidence intervals to quantify the statistical uncertainty arising from a finite number of independent Monte Carlo realisations.

Let $M$ denote the number of Monte Carlo realisations, and let $(1-\alpha)$ denote the specified confidence level, where $0<\alpha<1$. Define

```math
z_{\left(1-\frac{\alpha}{2}\right)} = \Phi^{-1} \left(1- \frac{\alpha}{2} \right),
```
where $\Phi^{-1} ( \cdot )$ is the inverse cumulative distribution function of the standard normal distribution.

#### 5.5.1. Wilson Score Confidence Interval

For the estimated extinction probability $`\widehat{p} = \widehat{\mathbb{P}} \left(T \leq H \right)`$, the implementation uses the Wilson score confidence interval, given by

```math
\frac{ \left( \widehat{p} + \frac{z^2}{2M} \right) \;\pm\; \left(z\sqrt{ \frac{\widehat{p}(1-\widehat{p})}{M} +\frac{z^2}{4M^2}} \right)}{1+\frac{z^2}{M}},
```
where $z = z_{\left(1-\frac{\alpha}{2}\right)} = \Phi^{-1} \left(1- \frac{\alpha}{2} \right)$.

The Wilson score interval is generally more reliable than the standard normal (Wald) interval for a probability, particularly when the estimated probability is close to $0$ or $1$.

#### 5.5.2. Normal-Approximation Confidence Intervals
For the estimated mean extinction-time statistics, the implementation uses normal-approximation confidence intervals.

Let $X_1,X_2,\ldots ,X_n$ denote the relevant observed values, with sample mean

```math
\overline{X}=\frac{1}{n}\sum_{m=1}^{n} X_m
```
and sample standard deviation

```math
s_\overline{X} = \sqrt{ \frac{1}{n-1} \sum_{m=1}^{n}(X_m - \overline{X})^2}.
```

The corresponding confidence interval is

```math
\boxed{\overline{X} \pm z_{\left(1-\frac{\alpha}{2}\right)} \frac{s_\overline{X}}{\sqrt{n}}}.
```

This formula is applied to the extinction-time statistics as follows:
- **Unrestricted Expected Extinction Time:** $X_m = T_m$, using $n=M$ completed realisations.
- **Restricted Expected Extinction Time:** $X_m = \min(T_m,H)$, using $n=M$ realisations.
- **Conditional Expected Extinction Time:** $X_m=T_m$ for realisations satisfying $T_m\leq H$, with $n$ equal to the number of such realisations.

For the conditional mean, a confidence interval can be calculated using this approach when at least two realisations become extinct by time $H$.

#### 5.5.3. Confidence Level and Monte Carlo Precision
The user specifies the confidence level $(1-\alpha)$. For example, a $95\%$ confidence level corresponds to

```math
\alpha=0.05, \qquad z_{\left(1- \frac{0.05}{2} \right)} = z_{\left(0.975 \right)} \approx 1.96.
```

Increasing the number of independent Monte Carlo realisations generally improves estimation precision and narrows the confidence intervals.

Under the usual finite-variance assumptions, the standard error of a sample mean decreases approximately at the rate

```math
s.e.(\overline{X}) = \frac{s_\overline{X}}{\sqrt{n}} = O\left( \frac{1}{\sqrt{n}} \right).
```
Thus, reducing the Monte Carlo standard error by approximately half generally requires four times as many realisations.

## 6. Requirements
The Monte Carlo simulation is implemented in Python and uses the following libraries:

- **NumPy:** For numerical computations, array operations and random number generation.
- **Matplotlib:** For visualising the network topology and simulated epidemic trajectories.
- **NetworkX:** For constructing and visualising directed, undirected and weighted networks.
- **tqdm (optional):** For displaying a progress bar during Monte Carlo simulations.

The Python standard library is also used for additional mathematical, statistical and formatting operations.

The required third-party libraries can be installed using:

```bash
pip install numpy matplotlib networkx tqdm
```

## 7. Acknowledgements

ChatGPT (OpenAI) was utilised to assist in improving the readability, robustness and computational efficiency of the Python implementation, as well as refining the documentation. The mathematical formulation, modelling decisions and interpretation of results remain the responsibility of the author.

