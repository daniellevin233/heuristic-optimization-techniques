# SCFPDP

We solves the **Selective Capacitated Fleet Pickup and Delivery Problem (SCFPDP)**.
The aim is to serve **at least γ requests**, not all. 
Different vehicles can serve different subsets, and each request must respect pickup–delivery order and vehicle capacity.

## Objective

* [ ] Minimize total travel cost **+ fairness penalty**:

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

## Next: VND

### Plan

Fix a deterministic order:**INSERT → SWAP → RELOCATE**

- For each operator:
  1. Try to improve until no more improvement.
  2. Move to next operator.
  3. Restart sequence if improvement found.

> Stop when full cycle yields **no improvement**.

---
