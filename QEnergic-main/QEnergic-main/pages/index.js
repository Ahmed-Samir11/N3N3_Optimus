import React from 'react';
import dynamic from 'next/dynamic';

// Dynamically import MapOptimizer without extension
const MapOptimizer = dynamic(() => import('../components/MapOptimizer'), { ssr: false });

export default function Home() {
  return (
    <div className="min-h-screen flex flex-col items-center p-4">
      <h1 className="text-3xl mb-4">Microgrid Placement Optimizer</h1>
      <MapOptimizer />
    </div>
  );
}