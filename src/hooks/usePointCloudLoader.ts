import { useState, useCallback } from 'react';

export interface PointCloudData {
  points: Float32Array;
  colors: Float32Array;
  count: number;
}

export interface LineData {
  start: [number, number, number];
  end: [number, number, number];
  color: [number, number, number];
}

export const usePointCloudLoader = () => {
  const [pointCloudData, setPointCloudData] = useState<PointCloudData | null>(null);
  const [linesData, setLinesData] = useState<LineData[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadFromFile = useCallback(async (file: File) => {
    setIsLoading(true);
    setError(null);

    try {
      const text = await file.text();
      const data = JSON.parse(text);

      if (data.points && data.colors) {
        // Point cloud file
        const points = new Float32Array(data.points.flat());
        const colors = new Float32Array(data.colors.flat());
        
        setPointCloudData({
          points,
          colors,
          count: data.count || points.length / 3
        });
        
        console.log(`✓ Loaded ${data.count} points from ${file.name}`);
      } else if (data.lines) {
        // Lines file
        setLinesData(data.lines);
        console.log(`✓ Loaded ${data.lines.length} lines from ${file.name}`);
      } else {
        throw new Error('Invalid file format. Expected points/colors or lines data.');
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to load file';
      setError(message);
      console.error('Error loading file:', err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const loadFromUrl = useCallback(async (url: string) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();

      if (data.points && data.colors) {
        const points = new Float32Array(data.points.flat());
        const colors = new Float32Array(data.colors.flat());
        
        setPointCloudData({
          points,
          colors,
          count: data.count || points.length / 3
        });
        
        console.log(`✓ Loaded ${data.count} points from ${url}`);
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to load from URL';
      setError(message);
      console.error('Error loading from URL:', err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const clear = useCallback(() => {
    setPointCloudData(null);
    setLinesData([]);
    setError(null);
  }, []);

  return {
    pointCloudData,
    linesData,
    isLoading,
    error,
    loadFromFile,
    loadFromUrl,
    clear
  };
};
