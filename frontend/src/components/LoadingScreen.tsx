import { Loader2 } from "lucide-react";

export const LoadingScreen = () => {
  return (
    <div className="w-full max-w-2xl mx-auto px-4 py-12 animate-fade-in">
      <div className="flex flex-col items-center justify-center space-y-6">
        <Loader2 className="w-16 h-16 text-primary animate-spin" />
        
        <div className="text-center space-y-2">
          <h3 className="text-2xl font-semibold text-foreground">
            Analyzing Media
          </h3>
          <p className="text-muted-foreground">
            Our AI is processing your file to detect authenticity...
          </p>
        </div>
        
        <div className="w-full max-w-md h-1 bg-secondary rounded-full overflow-hidden">
          <div className="h-full bg-primary w-1/3 animate-pulse shadow-[0_0_10px_hsl(195_100%_50%/0.5)]" />
        </div>
      </div>
    </div>
  );
};
