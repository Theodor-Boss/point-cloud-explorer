import { Scene3D } from '@/components/Scene3D';
import { ViewerUI } from '@/components/ViewerUI';

const Index = () => {
  return (
    <div className="relative w-full h-screen overflow-hidden bg-background">
      <Scene3D />
      <ViewerUI />
    </div>
  );
};

export default Index;
