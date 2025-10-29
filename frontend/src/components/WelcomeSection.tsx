import { useEffect, useState } from "react";

export const WelcomeSection = () => {
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    setVisible(true);
  }, []);

  return (
    <div className="flex flex-col items-center justify-center py-16 px-4">
      <div
        className={`text-center space-y-6 transition-all duration-1000 ${
          visible ? "opacity-100 translate-y-0" : "opacity-0 translate-y-10"
        }`}
      >
        <h1 className="text-5xl md:text-7xl font-bold text-primary tracking-tight" style={{ textShadow: '0 0 20px hsl(195 100% 50% / 0.8), 0 0 40px hsl(195 100% 50% / 0.4)' }}>
          Deep State
        </h1>
        
        <h2 className="text-2xl md:text-3xl font-light text-foreground tracking-wide">
          Authenticity Analyzer
        </h2>
        
        <p className="text-base md:text-lg text-muted-foreground max-w-2xl mx-auto mt-4">
          Advanced AI-powered deepfake detection system. Upload media files for instant authenticity verification.
        </p>
      </div>
    </div>
  );
};
