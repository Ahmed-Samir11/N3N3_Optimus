---

# ☀️ Solar Microgrid Placement Optimization using QUBO

This project models and solves the **optimal placement of solar microgrids** using the **Quadratic Unconstrained Binary Optimization (QUBO)** model. It uses synthetic data to simulate site attributes and leverages the `autoqubo` package to build and solve a QUBO formulation that balances cost, energy potential, and population coverage under budgetary and capacity constraints.

## 📦 Features

* 🔧 Synthetic dataset generation for candidate sites  
* 🧠 Symbolic QUBO formulation using `autoqubo`  
* 🎯 Multi-objective optimization:  
  - Minimize installation cost  
  - Maximize population coverage  
  - Maximize energy potential  
* 📈 Constraints:
  - Total budget    
  - Minimum population served  
* ⚡ Solves the QUBO and outputs an optimal site selection summary  

## 🧮 QUBO Formulation

The optimization problem is encoded as the following QUBO function:

```
Q(x) = ∑ᵢ Cᵢ xᵢ - α ∑ᵢ Pᵢ xᵢ + θ (∑ᵢ Cᵢ xᵢ - B)² + μ (∑ᵢ xᵢ - K)² + λ (M - ∑ᵢ Pᵢ xᵢ)²
```

**Where:**

* `xᵢ ∈ {0, 1}` = 1 if site *i* is selected, 0 otherwise  
* `Cᵢ` = Installation cost at site *i*  
* `Pᵢ` = Population covered at site *i*  
* `B` = Budget cap  
* `K` = Maximum number of grids allowed  
* `M` = Minimum total population to be served  
* `α`, `θ`, `μ`, `λ` — Weighting/penalty parameters  

## 📁 Project Structure

```

QGridOpti/QEnergicbackend/
├── qubo.py              # Main optimization script
├── README.md            # Project overview and documentation

````

## 🔧 Installation

Ensure you have the latest version of Python and have installed the required packages:

```bash
pip install numpy pandas autoqubo
````

> ℹ️ This project uses the [`autoqubo`](https://github.com/FujitsuResearch/autoqubo) library by [Fujitsu Research](https://github.com/FujitsuResearch), an automated QUBO generator that converts high-level Python-based optimization problems into equivalent QUBO representations.

## 🚀 Usage

Run the script with:

```bash
python qubo.py
```

The script will print the selected microgrid sites and a summary including total cost, population coverage, and energy capacity.

## 📊 Sample Output

```
✅ Selected Microgrid Sites:
    Site_ID     Installation_Cost_USD  Population_Coverage  Energy_Capacity_kWh_day  X_coord  Y_coord
    Site_3      16000                 1300                 1350.00                  0.34     0.47
    ...

📊 Summary:
   - Total Installation Cost: $XXX,XXX
   - Total Population Covered: XX,XXX people
   - Total Energy Capacity: XXXX kWh/day
   - Execution Time: X.XX seconds
```

## 🔍 Parameters & Constraints

| Parameter        | Value   | Description                           |
| ---------------- | ------- | ------------------------------------- |
| `budget`         | 900,000 | Maximum allowed installation cost     |
| `max_grids`      | 10      | Maximum number of microgrids to place |
| `min_population` | 15,000  | Minimum population to be served       |
| `alpha`          | 0.1     | Weight for population coverage        |
| `gamma`          | 0.1     | Weight for energy capacity            |
| `theta`          | 1e-6    | Penalty for budget constraint         |
| `mu`             | 2       | Penalty for grid count constraint     |
| `lambda_`        | 1e-2    | Penalty for population constraint     |

## 📚 Background

This tool simulates the strategic placement of **off-grid solar microgrids**—a vital approach to rural electrification in underserved regions. By formulating the site selection as a QUBO, the model can be solved using quantum or quantum-inspired solvers.

## 🔗 See Also

* [`autoqubo`](https://github.com/FujitsuResearch/autoqubo) – An automated QUBO generator developed by [Fujitsu Research](https://github.com/FujitsuResearch). It converts high-level Python optimization problems into QUBO form using a novel data-driven translation method.

