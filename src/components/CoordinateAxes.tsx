import { Line } from '@react-three/drei';

interface CoordinateAxesProps {
  length?: number;
}

export const CoordinateAxes = ({ length = 1.0 }: CoordinateAxesProps) => {
  return (
    <group>
      {/* X axis - Red */}
      <Line
        points={[[0, 0, 0], [length, 0, 0]]}
        color="red"
        lineWidth={2}
      />
      
      {/* Y axis - Green */}
      <Line
        points={[[0, 0, 0], [0, length, 0]]}
        color="green"
        lineWidth={2}
      />
      
      {/* Z axis - Blue */}
      <Line
        points={[[0, 0, 0], [0, 0, length]]}
        color="blue"
        lineWidth={2}
      />
    </group>
  );
};
