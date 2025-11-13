import { Canvas } from '@react-three/fiber';
import { FirstPersonControls } from './FirstPersonControls';
import { PointCloud } from './PointCloud';
import { CoordinateAxes } from './CoordinateAxes';
import { useMemo } from 'react';

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

export const Scene3D = () => {
  const { points, colors } = useMemo(() => generateSpherePoints(5000, 2.0), []);

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
        <PointCloud points={points} colors={colors} size={0.03} />
        
        {/* Grid helper for reference */}
        <gridHelper args={[10, 10, '#00ffff', '#003333']} />
      </Canvas>
    </div>
  );
};
