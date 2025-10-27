// MapOptimizer.jsx
import React, { useState } from 'react';
import { MapContainer, TileLayer, Marker, Polygon, Popup, useMapEvents } from 'react-leaflet';
import L from 'leaflet';
import axios from 'axios';
import africaCities from '../data/africa_cities.json';

// Leaflet marker icons fix
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.3/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.9.3/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.3/dist/images/marker-shadow.png',
});

function MarkerHandler({ markers, setMarkers, popCenter, setPopCenter, popMode }) {
  useMapEvents({
    click(e) {
      if (popMode) {
        setPopCenter({ lat: e.latlng.lat, lng: e.latlng.lng });
      } else if (markers.length < 4) {
        setMarkers(prev => [...prev, { lat: e.latlng.lat, lng: e.latlng.lng }]);
      }
    }
  });
  return null;
}

export default function MapOptimizer() {
  const [markers, setMarkers] = useState([]);
  const [budget, setBudget] = useState(100000); // Default $100k budget
  const [grids, setGrids] = useState([]);
  const [countryIdx, setCountryIdx] = useState(0);
  const [popCenter, setPopCenter] = useState(null);
  const [popMode, setPopMode] = useState(false);
  const [algo, setAlgo] = useState('classical');
  const [optTime, setOptTime] = useState(null);
  const [loading, setLoading] = useState(false);

  const selectedCountry = africaCities[countryIdx];

  const removeMarker = idx => {
    setMarkers(prev => prev.filter((_, i) => i !== idx));
  };

  const removePopCenter = () => setPopCenter(null);

  const solve = async (algorithm = 'classical') => {
    if (markers.length !== 4) {
      alert('Please place exactly 4 markers to define the area.');
      return;
    }
    if (!popCenter) {
      alert('Please mark the center of population on the map.');
      return;
    }

    setLoading(true);
    setAlgo(algorithm);
    setOptTime(null);

    try {
      const t0 = performance.now();
      const res = await fetch('/api/optimize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          polygon: markers,
          budget: Number(budget),
          popCenter,
          algo: algorithm === 'classical' ? undefined : algorithm
        })
      });

      const data = await res.json();
      if (data.error) throw new Error(data.error);

      setGrids(data.grids || []);
      setOptTime(performance.now() - t0);
    } catch (error) {
      alert(`Optimization failed: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="w-full max-w-4xl mx-auto p-4">
      <div className="mb-4 bg-white p-4 rounded-lg shadow">
        <label className="mr-2 font-medium">Country:</label>
        <select 
          value={countryIdx} 
          onChange={e => setCountryIdx(+e.target.value)} 
          className="border p-2 rounded mr-4"
        >
          {africaCities.map((city, i) => (
            <option value={i} key={city.name}>{city.name}</option>
          ))}
        </select>

        <button 
          onClick={() => setPopMode(m => !m)} 
          className={`px-4 py-2 rounded ${popMode ? 'bg-green-600 text-white' : 'bg-gray-200'}`}
        >
          {popMode ? 'Marking Population Center' : 'Mark Population Center'}
        </button>

        {popCenter && (
          <button 
            onClick={removePopCenter} 
            className="ml-2 px-4 py-2 rounded bg-red-200"
          >
            Remove Center
          </button>
        )}
      </div>

      <div className="mb-4 bg-white p-4 rounded-lg shadow">
        <div className="flex items-center space-x-4">
          <div>
            <label className="block font-medium mb-1">Budget ($):</label>
            <input 
              type="number" 
              min="10000" 
              step="1000"
              value={budget} 
              onChange={e => setBudget(e.target.value)} 
              className="border p-2 w-32 rounded"
            />
          </div>

          <button 
            onClick={() => solve('nar')} 
            disabled={loading}
            className="px-4 py-2 bg-red-600 text-white rounded disabled:opacity-50"
          >
            {loading ? 'Processing...' : 'Greedy (NAR)'}
          </button>

          <button 
            onClick={() => solve('classical')} 
            disabled={loading}
            className="px-4 py-2 bg-blue-600 text-white rounded disabled:opacity-50"
          >
            {loading ? 'Processing...' : 'Simulated Annealing'}
          </button>

          <button 
            onClick={() => solve('gurobi')} 
            disabled={loading}
            className="px-4 py-2 bg-green-700 text-white rounded disabled:opacity-50"
          >
            {loading ? 'Processing...' : 'Gurobi'}
          </button>
        </div>

        {optTime && (
          <div className="mt-2 text-green-700">
            Optimization time: {(optTime/1000).toFixed(2)} seconds ({algo})
          </div>
        )}
      </div>

      <div className="mb-4 rounded-lg overflow-hidden shadow-lg" style={{ height: '500px' }}>
        <MapContainer 
          center={[selectedCountry.lat, selectedCountry.lng]} 
          zoom={6} 
          style={{ height: '100%', width: '100%' }}
        >
          <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
          
          <MarkerHandler 
            markers={markers} 
            setMarkers={setMarkers} 
            popCenter={popCenter} 
            setPopCenter={setPopCenter} 
            popMode={popMode} 
          />

          {markers.map((m, i) => (
            <Marker 
              key={i} 
              position={[m.lat, m.lng]} 
              eventHandlers={{ contextmenu: () => removeMarker(i) }}
            >
              <Popup>Marker {i + 1} (Right-click to remove)</Popup>
            </Marker>
          ))}

          {markers.length === 4 && (
            <Polygon 
              positions={markers.map(m => [m.lat, m.lng])} 
              pathOptions={{ color: 'blue', fillOpacity: 0.2 }} 
            />
          )}

          {popCenter && (
            <Marker 
              position={[popCenter.lat, popCenter.lng]} 
              icon={L.divIcon({
                className: 'pop-center-marker', 
                html: '<div style="background:orange;width:24px;height:24px;border-radius:50%;border:3px solid white"></div>'
              })}
            >
              <Popup>Population Center</Popup>
            </Marker>
          )}

          {grids.map((g, i) => (
            <Marker 
              key={`grid-${i}`} 
              position={[g.lat, g.lng]} 
              icon={L.divIcon({
                className: 'grid-marker',
                html: `<div style="background:red;width:20px;height:20px;border-radius:50%;border:2px solid white">
                         <div style="color:white;font-size:10px;text-align:center;line-height:20px">${i+1}</div>
                       </div>`
              })}
            >
              <Popup>
                <div>
                  <strong>Site {i+1}</strong><br />
                  Cost: ${g.cost.toLocaleString()}<br />
                  Energy: {g.energy} kWh/day<br />
                  Population: {g.pop}<br />
                  Distance: {g.dist.toFixed(1)} km
                </div>
              </Popup>
            </Marker>
          ))}
        </MapContainer>
      </div>

      {grids.length > 0 && (
        <div className="bg-white p-4 rounded-lg shadow">
          <h2 className="text-xl font-semibold mb-3">Selected Sites ({grids.length})</h2>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-2 text-left">Site</th>
                  <th className="px-4 py-2 text-left">Latitude</th>
                  <th className="px-4 py-2 text-left">Longitude</th>
                  <th className="px-4 py-2 text-left">Cost ($)</th>
                  <th className="px-4 py-2 text-left">Energy (kWh/day)</th>
                  <th className="px-4 py-2 text-left">Population</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {grids.map((g, i) => (
                  <tr key={i}>
                    <td className="px-4 py-2">{i+1}</td>
                    <td className="px-4 py-2">{g.lat.toFixed(4)}</td>
                    <td className="px-4 py-2">{g.lng.toFixed(4)}</td>
                    <td className="px-4 py-2">{g.cost.toLocaleString()}</td>
                    <td className="px-4 py-2">{g.energy}</td>
                    <td className="px-4 py-2">{g.pop}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <div className="mt-4 text-sm text-gray-600">
        <p>1. Click to add exactly 4 markers to define the area (right-click to remove)</p>
        <p>2. Click "Mark Population Center" and place the population center marker</p>
        <p>3. Set your budget and click an optimization method</p>
      </div>
    </div>
  );
}