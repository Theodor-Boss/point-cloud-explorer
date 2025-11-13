import { useRef, useMemo } from 'react';
import { Points, Point } from '@react-three/drei';
import * as THREE from 'three';

interface PointCloudProps {
  points: Float32Array;
  colors: Float32Array;
  size?: number;
}

export const PointCloud = ({ points, colors, size = 0.02 }: PointCloudProps) => {
  const pointsRef = useRef<THREE.Points>(null);

  const geometry = useMemo(() => {
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(points, 3));
    geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    return geom;
  }, [points, colors]);

  return (
    <points ref={pointsRef} geometry={geometry}>
      <pointsMaterial
        size={size}
        vertexColors
        sizeAttenuation={true}
        transparent={false}
        depthWrite={true}
      />
    </points>
  );
};
