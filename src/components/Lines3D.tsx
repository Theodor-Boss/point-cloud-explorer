import { Line } from '@react-three/drei';
import { LineData } from '@/hooks/usePointCloudLoader';

interface Lines3DProps {
  lines: LineData[];
}

export const Lines3D = ({ lines }: Lines3DProps) => {
  return (
    <group>
      {lines.map((line, index) => (
        <Line
          key={index}
          points={[line.start, line.end]}
          color={`rgb(${line.color[0] * 255}, ${line.color[1] * 255}, ${line.color[2] * 255})`}
          lineWidth={2}
        />
      ))}
    </group>
  );
};
