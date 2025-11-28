import { useRef } from 'react';
import { useLoader } from '@react-three/fiber';
import * as THREE from 'three';

interface ImagePlaneProps {
  imageUrl: string;
  position?: [number, number, number];
  rotation?: [number, number, number];
  scale?: [number, number];
}

export const ImagePlane = ({ 
  imageUrl, 
  position = [0, 0, 0], 
  rotation = [0, 0, 0],
  scale = [1, 1]
}: ImagePlaneProps) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const texture = useLoader(THREE.TextureLoader, imageUrl);

  return (
    <mesh ref={meshRef} position={position} rotation={rotation}>
      <planeGeometry args={[scale[0], scale[1]]} />
      <meshBasicMaterial map={texture} side={THREE.DoubleSide} transparent />
    </mesh>
  );
};
