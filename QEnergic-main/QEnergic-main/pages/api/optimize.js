// optimize.js
import { polygon as turfPolygon, booleanPointInPolygon, distance as turfDistance } from '@turf/turf';
import { ethiopiaSites } from '../../data_generator';

// Simulated annealing for QUBO
async function simulatedAnnealingQUBO(Q, n, candidates, budget, maxIter = 10000, T0 = 20, alpha = 0.999) {
  let x = Array(n).fill(0);
  let bestX = x.slice(), bestE = Infinity;
  let T = T0;
  const costPenalty = 10000 / budget;
  
  function energy(x) {
    let e = 0, cost = 0;
    for (let i = 0; i < n; i++) {
      if (x[i]) cost += candidates[i].cost;
      for (let j = 0; j < n; j++) e += Q[i][j] * x[i] * x[j];
    }
    if (cost > budget) e += costPenalty * (cost - budget) ** 2;
    return e;
  }

  for (let iter = 0; iter < maxIter; iter++) {
    let i = Math.floor(Math.random() * n);
    let x2 = x.slice(); x2[i] = 1 - x2[i];
    let e1 = energy(x), e2 = energy(x2);
    if (e2 < e1 || Math.random() < Math.exp((e1 - e2) / T)) {
      x = x2;
      if (e2 < bestE) { bestE = e2; bestX = x.slice(); }
    }
    T *= alpha;
    if (T < 1e-6) break;
  }
  return bestX;
}

// NAR greedy optimization
function narGreedyBudget(Q, candidates, budget) {
  const idxs = candidates.map((c, i) => i)
    .sort((a, b) => (candidates[b].energy / candidates[b].cost) - (candidates[a].energy / candidates[a].cost));
  let total = 0;
  let selected = [];
  for (let i of idxs) {
    if (total + candidates[i].cost <= budget) {
      selected.push(i);
      total += candidates[i].cost;
    }
  }
  return candidates.map((_, i) => selected.includes(i) ? 1 : 0);
}

// Gurobi QUBO solver
async function gurobiQUBOBudget(Q, candidates, budget) {
  try {
    const { spawnSync } = await import('child_process');
    const fs = await import('fs');
    const path = await import('path');
    const tmpDir = '/tmp';
    const quboPath = path.join(tmpDir, `qubo_gurobi_${Date.now()}.json`);
    const costsPath = path.join(tmpDir, `costs_gurobi_${Date.now()}.json`);
    fs.writeFileSync(quboPath, JSON.stringify({ Q }));
    fs.writeFileSync(costsPath, JSON.stringify(candidates.map(c => c.cost)));
    const pyPath = path.join(process.cwd(), 'gurobi_optimize.py');
    const result = spawnSync('python3', [pyPath, '--qubo_file', quboPath, '--budget', String(budget), '--costs_file', costsPath], { encoding: 'utf-8' });
    fs.unlinkSync(quboPath);
    fs.unlinkSync(costsPath);
    if (result.status !== 0) throw new Error(result.stderr || 'Gurobi failed');
    const out = result.stdout.trim().split('\n').pop();
    const parsed = JSON.parse(out);
    const x = candidates.map((_, i) => parsed.selected_indices.includes(i) ? 1 : 0);
    return x;
  } catch (e) {
    throw new Error('Gurobi optimization failed: ' + e.message);
  }
}

export default async function handler(req, res) {
  const { polygon, budget, popCenter, quboOnly, algo } = req.body;

  if (!polygon || polygon.length !== 4 || !budget || budget <= 0 || !popCenter) {
    return res.status(400).json({ error: 'Invalid input' });
  }

  // Close the polygon
  const coords = [...polygon.map(p => [p.lng, p.lat]), [polygon[0].lng, polygon[0].lat]];
  const turfPoly = turfPolygon([coords]);

  // Filter sites within polygon
  const candidates = ethiopiaSites.filter(site => 
    booleanPointInPolygon([site.X_coord, site.Y_coord], turfPoly)
  ).map(site => ({
    ...site,
    dist: turfDistance([site.X_coord, site.Y_coord], [popCenter.lng, popCenter.lat], { units: 'kilometers' })
  }));

  if (candidates.length === 0) {
    return res.status(200).json({ grids: [] });
  }

  // QUBO parameters (using the same values as the provided script)
  const K = req.body.K ?? Math.floor(budget / 15000); // default max sites
  const M = req.body.M ?? 0; // default min coverage

  // Build QUBO matrix
  const n = candidates.length;
  const Q = Array.from({ length: n }, (_, i) => Array(n).fill(0));
  const C = candidates.map(c => c.Installation_Cost_USD);
  const P = candidates.map(c => c.Population_Coverage);
  const E = candidates.map(c => c.Energy_Capacity_kWh_day);

  // Use the same QUBO parameters as the provided script
  const alpha = 1e-1;      // population coverage weight
  const gamma = 1e-1;      // energy capacity weight
  const theta = 1e-6;      // budget penalty
  const mu = 2;            // grid count penalty
  const lambda = 1e-2;     // minimum coverage penalty
  const max_grids = req.body.max_grids ?? 10;
  const min_population = req.body.min_population ?? 15000;

  for (let i = 0; i < n; i++) {
    // Objective function terms: cost - alpha*population - gamma*energy
    Q[i][i] += C[i] - alpha * P[i] - gamma * E[i];
    
    // Budget constraint: theta * (sum(costs * x) - budget)^2
    Q[i][i] += theta * (C[i] ** 2 - 2 * budget * C[i]);
    
    // Grid count constraint: mu * (sum(x) - max_grids)^2
    Q[i][i] += mu * (1 - 2 * max_grids);
    
    // Population constraint: lambda * (min_population - sum(population * x))^2
    Q[i][i] += lambda * (P[i] ** 2 - 2 * min_population * P[i]);
    
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      // Budget constraint cross terms
      Q[i][j] += 2 * theta * C[i] * C[j];
      // Grid count constraint cross terms
      Q[i][j] += 2 * mu;
      // Population constraint cross terms
      Q[i][j] += 2 * lambda * P[i] * P[j];
    }
  }
  
  // Constant offset terms
  Q[0][0] += theta * budget ** 2 + mu * max_grids ** 2 + lambda * min_population ** 2;

  // Return QUBO for quantum if requested
  if (quboOnly) {
    const maxCandidates = 1000;
    if (candidates.length > maxCandidates) {
      const step = Math.ceil(candidates.length / maxCandidates);
      const reducedCandidates = candidates.filter((_, i) => i % step === 0).slice(0, maxCandidates);
      const reducedQ = reducedCandidates.map((_, i) => 
        reducedCandidates.map((_, j) => Q[i * step]?.[j * step] ?? 0));
      return res.status(200).json({ Q: reducedQ, candidates: reducedCandidates, budget });
    }
    return res.status(200).json({ Q, candidates, budget });
  }

  // Run optimization
  let x;
  try {
    if (algo === 'nar') {
      x = narGreedyBudget(Q, candidates, budget);
    } else if (algo === 'gurobi') {
      x = await gurobiQUBOBudget(Q, candidates, budget);
    } else {
      x = await simulatedAnnealingQUBO(Q, candidates.length, candidates, budget);
    }
  } catch (e) {
    console.error('Optimization error:', e);
    return res.status(500).json({ error: e.message });
  }

  const grids = candidates.filter((c, i) => x[i]).map(c => ({
    lat: c.Y_coord,
    lng: c.X_coord,
    cost: c.Installation_Cost_USD,
    energy: c.Energy_Capacity_kWh_day,
    dist: c.dist,
    pop: c.Population_Coverage
  }));

  res.status(200).json({ grids });
}