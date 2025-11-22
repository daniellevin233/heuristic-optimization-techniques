# Heuristic Optimization Techniques - Assignment 1

**Course:** TU Wien - Heuristic Optimization Techniques, WS 2025
**Deadline:** Sunday, 30.11.2025, 23:55
**Problem:** Selective Capacitated Fair Pickup and Delivery Problem (SCF-PDP)

## Overview

This assignment focuses on developing construction heuristics and local search-based metaheuristics for the SCF-PDP problem.

## Problem Description

The **Selective Capacitated Fair Pickup and Delivery Problem (SCF-PDP)** involves designing fair and feasible routes for a subset of customer requests. Each customer requires transportation of goods from a pickup location to a drop-off location.

### Key Characteristics

- **Graph Model:** Complete directed graph G = (V, A)
- **Nodes (V):** Vehicle depot + pickup and drop-off locations
- **Arcs (A):** Fastest travel routes between locations
- **Distance Calculation:** Euclidean distance rounded up: `⌈√((x_u - x_v)² + (y_u - y_v)²)⌉`

### Problem Parameters

- `n`: Number of customer requests
- `n_K`: Number of identical vehicles
- `C`: Maximum vehicle capacity
- `γ`: Minimum number of requests to be fulfilled (γ ≤ n)
- `ρ`: Fairness weight parameter
- `c_i`: Amount of goods for request i

### Objective Function

Minimize: `Σ d(R_k) + ρ · (1 - J(R))`

Where:
- `d(R_k)`: Total travel duration of vehicle k
- `J(R)`: Jain fairness measure = `(Σ d(R_k))² / (n · Σ d(R_k)²)`

## Assignment Tasks

### 1. Construction Heuristics
- **1.1** Develop a meaningful deterministic construction heuristic
- **1.2** Derive a randomized construction heuristic to be applied iteratively, yielding diverse promising solutions
- **1.3** Develop a Pilot or Beam Search based heuristic

### 2. Local Search Framework
- **2.1** Develop or use a framework for basic local search that handles:
  - Different neighborhood structures
  - Different step functions (first-improvement, best-improvement)

### 3. Neighborhood Structures
- **3.1** Develop at least three different meaningful neighborhood structures
- **3.2** Develop or use a Variable Neighborhood Descent (VND) framework using your neighborhood structures

### 4. Metaheuristics

#### 4.1 GRASP
Implement a Greedy Randomized Adaptive Search Procedure using:
- Your randomized construction heuristic
- An effective neighborhood structure with one step function or (a variant of) your VND

#### 4.2 Advanced Metaheuristic (Choose One)
Implement one of the following:
- **GVNS:** General Variable Neighborhood Search on top of VND
- **SA:** Simulated Annealing
- **TS:** Tabu Search
- **CC:** Configuration Checking

### 5. Delta-Evaluation
- Implement and use delta-evaluation
- Explain which steps use delta-evaluation and why it improves performance
- Analyze asymptotic runtime with and without delta-evaluation
- Identify preprocessing opportunities for efficiency

### 6. Parameter Tuning
- Perform experimental manual tuning of algorithmic parameters:
  - Degree of randomization
  - Neighborhood structure sizes
  - Probabilities for random step function
  - Cooling schedule
  - Tabu list length and variation
- Report impact of different settings on solution quality
- Use **training instances** for tuning
- Report results on **test instances**

### 7. Experimental Evaluation

#### Algorithm Comparisons
- **7.1** Compare deterministic and randomized construction heuristics and GRASP
- **7.2** Use deterministic construction solution to test:
  - Local search for at least 3 neighborhood structures with both step functions (≥6 variants)
  - VND
  - GVNS/SA/TS

#### Metrics to Report
- Running time
- Iterations
- Average final objective
- Objective over time
- Evolution of fairness and travel duration
- Run each algorithm multiple times to reduce statistical variance

#### Visualizations
- Prepare plots showing the above measures concisely

### 8. Report
Write a report containing:
- Description of algorithms
- Experimental results
- Conclusions and analysis

## Development Questions

### Solution Design
- How is a solution represented best?
- How do you generate different solutions?
- Which parts can be reasonably randomized?
- How to control the degree of randomization?
- Are random solutions sufficiently diverse?

### Algorithm Behavior
- Do you notice a cold start problem during greedy construction?
- Are requests roughly equally distributed across routes?
- What happens when manually changing ρ or γ to very high/low values?
- What hyperparameters did you choose (tabu list length, randomization factor)?
- Is every solution reachable via your neighborhood structures?

### Performance Analysis
- How many iterations to reach local optima?
- What does this say about your neighborhood structures?
- Does this change with different step functions?
- How does delta evaluation work for your neighborhoods?
- Can calculations be simplified using preprocessing?
- What is the time complexity to fully search one neighborhood?
- Does VND neighborhood order affect solution quality?

## Instance and Solution Format

### Instance File Format
```
n n_K C γ ρ
# demands
c1 c2 ... cn
# request locations
x_depot y_depot
x_pickup1 y_pickup1 
x_pickup2 y_pickup2 
... 
x_pickupn y_pickupn
---
x_dropoff1 y_dropoff1
x_dropoff2 y_dropoff2 
...
x_dropoffn y_dropoffn
```

### Solution File Format
```
instance_filename
R1,1 R1,2 ... R1,|R1|
R2,1 R2,2 ... R2,|R2|
...
RnK,1 RnK,2 ... RnK,|RnK|
```
For example:
```
example_instance
1 11 2 3 13 4 14 12
5 6 7 8 9 10 15 16 17 18 19 20
```

## Submission Requirements

Submit via TUWEL:
1. Complete report
2. Solutions for at least 3 competition instances
3. Source code (zip archive)
4. Best solutions for each instance and algorithm

### Constraints
- No multithreading/multiprocessing/GPU usage
- Use only single CPU threads
- Maximum runtime: 15 minutes per instance
- Report best solutions must match uploaded solutions

## Development Environment

### Allowed Tools
- Any programming language and development environment
- Open source frameworks (e.g., MHLib.jl)
- Suitable packages for graphs, visualization, etc.

### AC Group's Cluster
- Access: `ssh USERNAME@eowyn.ac.tuwien.ac.at` or `USERNAME@behemoth.ac.tuwien.ac.at`
- Available: Julia 1.11, gcc 7.5.0, Java openJDK 11, R 4.4.3, Python 3.13
- Submit jobs: `qsub -l h_rt=00:15:00 [QSUB_ARGS] COMMAND [CMD_ARGS]`
- Monitor jobs: `qstat`
- Delete jobs: `qdel <job_id>`

### Resources
- MHLib.jl: https://github.com/ac-tuwien/MHLib.jl
- Cluster info: https://www.ac.tuwien.ac.at/students/compute-cluster/

## Competition

- Upload solutions to competition server
- Solutions checked for correctness and entered in ranking table
- Ranking shows best three groups per instance & algorithm
- Ranking does not influence grade
- Best three groups win small prizes!

## Contact

For questions: heuopt@ac.tuwien.ac.at