import { Canvas } from '@react-three/fiber';
import { FirstPersonControls } from './FirstPersonControls';
import { PointCloud } from './PointCloud';
import { CoordinateAxes } from './CoordinateAxes';
import { Lines3D } from './Lines3D';
import { useMemo } from 'react';
import { PointCloudData, LineData } from '@/hooks/usePointCloudLoader';

const generateSpherePoints = (numPoints: number, radius: number) => {
  const points = new Float32Array(numPoints * 3);
  const colors = new Float32Array(numPoints * 3);

  for (let i = 0; i < numPoints; i++) {
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.random() * Math.PI;

    const x = radius * Math.sin(phi) * Math.cos(theta);
    const y = radius * Math.sin(phi) * Math.sin(theta);
    const z = radius * Math.cos(phi);

    points[i * 3] = x;
    points[i * 3 + 1] = y;
    points[i * 3 + 2] = z;

    // Random vibrant colors
    colors[i * 3] = Math.random() * 0.5 + 0.5;
    colors[i * 3 + 1] = Math.random() * 0.5 + 0.5;
    colors[i * 3 + 2] = Math.random() * 0.5 + 0.5;
  }

  return { points, colors };
};

interface Scene3DProps {
  pointCloudData?: PointCloudData | null;
  linesData?: LineData[];
}

export const Scene3D = ({ pointCloudData, linesData = [] }: Scene3DProps) => {
  const defaultData = useMemo(() => generateSpherePoints(5000, 2.0), []);

  // Use loaded data if available, otherwise use default
  const displayData = pointCloudData || defaultData;

  return (
    <div className="w-full h-screen">
      <Canvas
        camera={{ position: [0, 0, 5], fov: 70, near: 0.01, far: 50 }}
        gl={{ antialias: true }}
      >
        <color attach="background" args={['#0a0e1a']} />
        
        {/* Lighting */}
        <ambientLight intensity={0.5} />
        <pointLight position={[10, 10, 10]} intensity={1} />
        
        {/* Controls */}
        <FirstPersonControls movementSpeed={3} />
        
        {/* Coordinate axes */}
        <CoordinateAxes length={1.5} />
        
        {/* Point cloud */}
        <PointCloud 
          points={displayData.points} 
          colors={displayData.colors} 
          size={0.03} 
        />
        
        {/* Lines from Python */}
        {linesData.length > 0 && <Lines3D lines={linesData} />}
        
        {/* Grid helper for reference */}
        <gridHelper args={[10, 10, '#00ffff', '#003333']} />
      </Canvas>
    </div>
  );
};
