import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

export const ViewerUI = () => {
  return (
    <div className="fixed inset-0 pointer-events-none">
      {/* Top bar */}
      <div className="absolute top-0 left-0 right-0 p-6 flex justify-between items-start">
        <div className="pointer-events-auto">
          <h1 className="text-3xl font-bold bg-gradient-primary bg-clip-text text-transparent mb-2">
            3D Point Cloud Viewer
          </h1>
          <p className="text-muted-foreground text-sm">
            First-person navigation • 5,000 points
          </p>
        </div>
        
        <Badge variant="secondary" className="pointer-events-auto backdrop-blur-sm bg-card/80 border-primary/20">
          <div className="w-2 h-2 rounded-full bg-primary animate-pulse mr-2" />
          Active
        </Badge>
      </div>

      {/* Controls overlay */}
      <div className="absolute bottom-6 left-6 pointer-events-auto">
        <Card className="p-4 bg-card/80 backdrop-blur-sm border-primary/20">
          <h3 className="text-sm font-semibold mb-3 text-foreground">Controls</h3>
          <div className="space-y-2 text-xs text-muted-foreground">
            <div className="flex items-center gap-3">
              <kbd className="px-2 py-1 bg-secondary rounded text-foreground font-mono">Click</kbd>
              <span>Lock pointer to look around</span>
            </div>
            <div className="flex items-center gap-3">
              <kbd className="px-2 py-1 bg-secondary rounded text-foreground font-mono">WASD</kbd>
              <span>Move forward/left/back/right</span>
            </div>
            <div className="flex items-center gap-3">
              <kbd className="px-2 py-1 bg-secondary rounded text-foreground font-mono">Space</kbd>
              <span>Move up</span>
            </div>
            <div className="flex items-center gap-3">
              <kbd className="px-2 py-1 bg-secondary rounded text-foreground font-mono">Shift</kbd>
              <span>Move down</span>
            </div>
            <div className="flex items-center gap-3">
              <kbd className="px-2 py-1 bg-secondary rounded text-foreground font-mono">ESC</kbd>
              <span>Exit pointer lock</span>
            </div>
          </div>
        </Card>
      </div>

      {/* Info overlay */}
      <div className="absolute bottom-6 right-6 pointer-events-auto">
        <Card className="p-4 bg-card/80 backdrop-blur-sm border-primary/20">
          <h3 className="text-sm font-semibold mb-2 text-foreground">Axes</h3>
          <div className="space-y-1 text-xs">
            <div className="flex items-center gap-2">
              <div className="w-3 h-0.5 bg-red-500" />
              <span className="text-muted-foreground">X-axis</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-0.5 bg-green-500" />
              <span className="text-muted-foreground">Y-axis</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-0.5 bg-blue-500" />
              <span className="text-muted-foreground">Z-axis</span>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
};
