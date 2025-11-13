import { Scene3D } from '@/components/Scene3D';
import { ViewerUI } from '@/components/ViewerUI';
import { FileLoader } from '@/components/FileLoader';
import { usePointCloudLoader } from '@/hooks/usePointCloudLoader';
import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { X } from 'lucide-react';

const Index = () => {
  const { pointCloudData, linesData, isLoading, error, loadFromFile } = usePointCloudLoader();
  const [showLoader, setShowLoader] = useState(true);

  const hasData = pointCloudData !== null;

  return (
    <div className="relative w-full h-screen overflow-hidden bg-background">
      <Scene3D pointCloudData={pointCloudData} linesData={linesData} />
      <ViewerUI 
        pointCount={pointCloudData?.count || 5000}
        isLoaded={hasData}
      />
      
      {/* File loader overlay */}
      {showLoader && (
        <div className="fixed top-20 right-6 pointer-events-auto z-50">
          <div className="relative">
            <Button
              variant="ghost"
              size="icon"
              className="absolute -top-2 -right-2 h-6 w-6 rounded-full bg-card border border-border"
              onClick={() => setShowLoader(false)}
            >
              <X className="h-3 w-3" />
            </Button>
            <FileLoader 
              onFileLoad={loadFromFile}
              isLoading={isLoading}
              error={error}
            />
          </div>
        </div>
      )}
    </div>
  );
};

export default Index;
