# SCFPDP

We solves the **Selective Capacitated Fleet Pickup and Delivery Problem (SCFPDP)**.
The aim is to serve **at least γ requests**, not all.
Different vehicles can serve different subsets, and each request must respect pickup–delivery order and vehicle capacity.

## Objective

* Minimize total travel cost **+ fairness penalty**:

``Objective = Total distance + ρ · (1 – Jain’s Fairness Index)``

Fairness discourages solutions where one route is short and another is disproportionately long.

---

## Solution Structure (Built so far)

### **Solution**

- Represents multiple vehicle routes.
- Each route contains positions like: `[pickup_i, delivery_i, ..., pickup_j, delivery_j]`.

### **Route Constraints**

✔ Pickup before drop-off
✔ Capacity never exceeded
✔ Exactly one vehicle serves a request
✔ At least `γ` requests served across the fleet

---

## Construction Phase (#done)

| Step | Type                                  | Purpose                             |
| ---- | ------------------------------------- | ----------------------------------- |
| 1.1  | **Deterministic Greedy**        | Build a feasible baseline solution  |
| 1.2  | **Randomized Greedy**           | Generate diverse starting solutions |
| 1.3  | **Beam Search/Pilot Heuristic** | Look ahead, pick best future choice |

---

## Local Search Framework

We use reusable components for:

- **Neighborhood Operators**
- **Step Strategies** (move acceptance)

### Step Strategies Available

| Strategy                    | Meaning                          | Behavior           |
| --------------------------- | -------------------------------- | ------------------ |
| **First Improvement** | Stop at first better neighbor    | Fast, less optimal |
| **Best Improvement**  | Evaluate all neighbors pick best | Slower, better     |

---

## Neighborhood Operators for SCFPDP

Local search uses three deterministic neighborhoods, each producing feasible modifications while maintaining PDP constraints:

- pickup precedes drop-off
- vehicle capacity is never violated
- each request remains fully served by exactly one vehicle

---

### **1) INSERT**

**Action:** Move a pickup or drop-off to another valid position within the *same route*.

**Purpose:**

- Removes detours.
- Fixes stop ordering without changing vehicles.
- Improves route layout while keeping the same served requests.

**Example Effect:**

```
Before: depot → 3 → 8 → 4 → 12 → depot
After: depot → 3 → 4 → 8 → 12 → depot
```

---

### **2) SWAP**

**Action:** Swap two nodes (pickup or drop-off) within the same route.

**Purpose:**

- Quickly fixes unwanted ordering patterns.
- Less destructive than full re-insertion.
- Can correct inversions and reduce distance.

**Example Effect:**

```
Before: depot → 5 → 7 → 1 → 13 → depot
Swap(7,1)
After: depot → 5 → 1 → 7 → 13 → depot
```

---

### **3) RELOCATE**

**Action:** Move a pickup/drop-off to another position, *possibly to another route*.

**Purpose:**

- Changes which vehicle serves a request.
- Balances loads and route durations.
- Can eliminate overloads or empty routes.

**Example Effect:**

```
Vehicle 0: depot → 2 → 9 → depot
Vehicle 1: depot → 3 → 6 → depot

Relocate request 9 → Vehicle 1

Vehicle 0: depot → 2 → depot
Vehicle 1: depot → 3 → 6 → 9 → depot
```

---

### 📌 Summary Table of Neighborhoods

| Neighborhood       | Feasible move                             | Changes serving vehicle? | Use Case                        |
| ------------------ | ----------------------------------------- | ------------------------ | ------------------------------- |
| **Insert**   | Reposition pickup/drop in same route      | No                       | Removes detours, improves order |
| **Swap**     | Exchange two nodes in same route          | No                       | Fast ordering repair            |
| **Relocate** | Move request to another route or position | Yes                      | Balances load & capacity        |

---

# **✅** **1. Already built**

### **(A) Construction (starting solution)**

You already have:

* Greedy construction
* Randomized construction
* (Beam search is optional later)

**Purpose:**

Give the local search *something* to start from.

Local search cannot start from nothing.

---

### **(B) Neighborhoods (your move generators)**

You built:

1. **Insert**
2. **Swap**
3. **Relocate**

These define how your algorithm explores the space of feasible solutions.

This is your  *search language* .

---

### **(C) Step strategies**

You built:

* **First Improvement**
* **Best Improvement**

These define **how to select the next move** from all neighbors.

---

### **(D) VND (Variable Neighborhood Descent)**

You built:

* Insert → Swap → Relocate (fixed order)
* Restarts the loop when any improvement is found

VND is simply:

**Try smaller changes first → if stuck → try bigger ones.**

---

**At this stage ** **your core local search engine is done** **.**

---

# **✅** **2. What GRASP does in the big picture**

GRASP =

**Randomized construction → Local search → Repeat many times → Keep best**

That’s it.

Why?

Because randomized construction gives  *diverse starting points* .

Local search climbs to a local optimum.

Multiple runs = different local optima.

You pick the best.

GRASP is *NOT* replacing local search.

It *wraps around* it.


---

```
1. Construction Heuristics          (DONE)
   - Greedy
   - Randomized

2. Neighborhoods                    (DONE)
   - Insert, Swap, Relocate

3. Local Search                     (DONE)
   - First improvement
   - Best improvement
   - VND

4. Metaheuristics     (NEXT)
   - GRASP (mandatory)
   - SA / GVNS / Tabu (pick one)

5. Delta evaluation     (LATER)
   - Optimize neighborhood speed
   - Reduce repeated objective calculations

6. Parameter tuning     (LATER)
   - RCL size
   - Neighborhood order
   - Cooling schedule / tabu length

7. Experiments + plots  (LATER)
   - Compare LS / VND / GRASP / SA
```



# **✅** **4. next**

### **Step 1 — Validate your neighborhoods**

```
test_local_search_swap(instance)
test_local_search_insert(instance)
test_local_search_relocate(instance)
test_VND(instance)
```


* Does each move work?
* Does ordering remain valid?
* Does capacity remain valid?
* Does objective reduce?

You plot objective after each iteration (easy).

---

### **Step 2 — Compare FIRST vs BEST**

For all three neighborhoods:

* How many iterations?
* How fast?
* Which gives better objective?

This gives intuition about which step strategy to use in VND/GRASP.

---

### **Step 3 — Compare neighborhoods**

Run local search with:

* Insert only
* Swap only
* Relocate only
* Insert+Swap
* Insert+Swap+Relocate (VND)

This reveals:

* Which neighborhood is strongest?
* Which one is expensive?
* How many moves each enumerates?

---



### **Step 4 — Implement GRASP**

Use your randomized constructor + any local search:

```
for _ in range(20):
    sol = randomized_construct()
    sol = vnd(sol)
    best = min(best, sol)
```


This gives good global solutions.

---

### **Step 5 — Implement SA or Tabu**

(Not urgent. Start only after GRASP works.)

---

### **Step 6 — Delta evaluation**

This is just optimization:

* Avoid recalculating full route distance
* Store partial values
* Update objective incrementally in neighborhoods

This drastically speeds up VND.

---

### **Step 7 — Experiments**

You already have plotting tools.

You will plot:

* objective over time
* fairness evolution
* route lengths
* neighborhood effect
* GRASP convergence curve

---

# **💡 What you will actually experiment with**

### **You will experiment on:**

1. Neighborhoods (Insert vs Swap vs Relocate)
2. Step strategies (First vs Best)
3. VND vs Single-Neighborhood LS
4. Randomized construction vs Greedy construction
5. GRASP iterations vs runtime
6. SA / Tabu parameters
7. Delta evaluation effect (runtime comparison)

### **You will measure:**

* Objective value
* Fairness
* Travel distance
* Local search convergence speed
* Runtime per iteration

### **You will plot:**

* objective(t)
* fairness(t)
* distance(t)
* neighborhood acceptance ratio
* GRASP iteration curve

---
