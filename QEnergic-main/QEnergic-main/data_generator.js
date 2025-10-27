// data_generator.js
import { booleanPointInPolygon } from '@turf/turf';

// Ethiopia bounding box coordinates
const ETHIOPIA_BBOX = [32.997583, 3.397448, 47.982379, 14.894053];

export function generateDataset(name, numSites) {
  const siteIds = Array.from({ length: numSites }, (_, i) => `Site_${i + 1}`);
  const installCosts = Array.from({ length: numSites }, () => 
    Math.floor(Math.random() * 35000) + 15000); // $15k-$50k
  const populationCoverage = Array.from({ length: numSites }, () => 
    Math.floor(Math.random() * 1400) + 100); // 100-1500 people
  const solarPotential = Array.from({ length: numSites }, () => 
    parseFloat((Math.random() * 3 + 3.5).toFixed(2))); // 3.5-6.5 kWh/m2/day
  const energyCapacity = solarPotential.map((solar, i) => 
    parseFloat((solar * populationCoverage[i] * 0.3).toFixed(2))); // Energy capacity
  
  // Generate coordinates within Ethiopia bounding box
  const [minLng, minLat, maxLng, maxLat] = ETHIOPIA_BBOX;
  const coordinates = Array.from({ length: numSites }, () => [
    minLng + Math.random() * (maxLng - minLng),
    minLat + Math.random() * (maxLat - minLat)
  ]);

  const sites = siteIds.map((siteId, i) => ({
    Site_ID: siteId,
    Installation_Cost_USD: installCosts[i],
    Population_Coverage: populationCoverage[i],
    Solar_Potential_kWh_m2_day: solarPotential[i],
    Energy_Capacity_kWh_day: energyCapacity[i],
    X_coord: coordinates[i][0], // longitude
    Y_coord: coordinates[i][1]  // latitude
  }));

  return sites;
}

// Generate and export Ethiopia sites
const ethiopiaSites = generateDataset("Ethiopia_Offgrid_Potential", 50);

// Export both the function and the generated data
export { ethiopiaSites };
export default ethiopiaSites;