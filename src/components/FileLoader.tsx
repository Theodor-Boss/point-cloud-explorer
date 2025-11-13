import { useCallback } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Upload, FileJson, AlertCircle } from 'lucide-react';
import { cn } from '@/lib/utils';

interface FileLoaderProps {
  onFileLoad: (file: File) => void;
  isLoading?: boolean;
  error?: string | null;
}

export const FileLoader = ({ onFileLoad, isLoading, error }: FileLoaderProps) => {
  const handleDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();

    const file = e.dataTransfer.files[0];
    if (file && file.name.endsWith('.json')) {
      onFileLoad(file);
    }
  }, [onFileLoad]);

  const handleDragOver = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      onFileLoad(file);
    }
  }, [onFileLoad]);

  return (
    <Card 
      className={cn(
        "p-6 bg-card/80 backdrop-blur-sm border-primary/20 transition-all",
        isLoading && "opacity-50 pointer-events-none"
      )}
      onDrop={handleDrop}
      onDragOver={handleDragOver}
    >
      <div className="flex flex-col items-center gap-4 text-center">
        <div className="p-4 rounded-full bg-primary/10">
          <FileJson className="w-8 h-8 text-primary" />
        </div>
        
        <div>
          <h3 className="text-lg font-semibold mb-1">Load Point Cloud</h3>
          <p className="text-sm text-muted-foreground">
            Drop a JSON file or click to browse
          </p>
        </div>

        <label className="cursor-pointer">
          <input
            type="file"
            accept=".json"
            onChange={handleFileInput}
            className="hidden"
            disabled={isLoading}
          />
          <Button variant="default" className="gap-2" disabled={isLoading}>
            <Upload className="w-4 h-4" />
            {isLoading ? 'Loading...' : 'Choose File'}
          </Button>
        </label>

        {error && (
          <div className="flex items-center gap-2 text-destructive text-sm">
            <AlertCircle className="w-4 h-4" />
            {error}
          </div>
        )}

        <div className="text-xs text-muted-foreground mt-2">
          <p>Expected format: JSON with points and colors arrays</p>
          <p className="mt-1">Use the Python export script to generate compatible files</p>
        </div>
      </div>
    </Card>
  );
};
