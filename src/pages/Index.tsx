import { Scene3D } from '@/components/Scene3D';
import { ViewerUI } from '@/components/ViewerUI';
import { usePointCloudLoader } from '@/hooks/usePointCloudLoader';

const Index = () => {
  const { pointCloudData, linesData } = usePointCloudLoader();

  return (
    <div className="relative w-full h-screen overflow-hidden bg-background">
      <Scene3D pointCloudData={pointCloudData} linesData={linesData} />
      <ViewerUI 
        pointCount={pointCloudData?.count || 5000}
        isLoaded={pointCloudData !== null}
      />
    </div>
  );
};

export default Index;
