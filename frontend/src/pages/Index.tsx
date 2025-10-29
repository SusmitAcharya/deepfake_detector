import { useState } from "react";
import { WelcomeSection } from "@/components/WelcomeSection";
import { UploadArea } from "@/components/UploadArea";
import { LoadingScreen } from "@/components/LoadingScreen";
import { ResultDisplay } from "@/components/ResultDisplay";
import { Footer } from "@/components/Footer";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";

type AnalysisState = "upload" | "loading" | "results";

interface AnalysisResult {
  label: "deepfake" | "authentic";
  confidence: number;
}

const Index = () => {
  const [state, setState] = useState<AnalysisState>("upload");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [filePreviewUrl, setFilePreviewUrl] = useState<string | null>(null);
  const { toast } = useToast();

  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
    if (file.type.startsWith("image/")) {
      setFilePreviewUrl(URL.createObjectURL(file));
    }
  };

  const handleClearFile = () => {
    setSelectedFile(null);
    setFilePreviewUrl(null);
  };

  const handleAnalyze = async () => {
    if (!selectedFile) {
      toast({
        title: "No file selected",
        description: "Please upload an image or video to analyze.",
        variant: "destructive",
      });
      return;
    }

    setState("loading");

    try {
      // Create form data
      const formData = new FormData();
      formData.append("file", selectedFile);

      // TODO: Replace with your actual backend endpoint
      const response = await fetch("/api/analyze", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Analysis failed");
      }

      const data = await response.json();
      
      setResult({
        label: data.label,
        confidence: data.confidence,
      });
      
      setState("results");
      
      toast({
        title: "Analysis complete",
        description: "Your media has been analyzed successfully.",
      });
    } catch (error) {
      console.error("Analysis error:", error);
      
      // For demo purposes, show a mock result
      toast({
        title: "Demo Mode",
        description: "Showing mock results. Connect your backend to get real analysis.",
      });
      
      // Mock result for demonstration
      setTimeout(() => {
        setResult({
          label: Math.random() > 0.5 ? "authentic" : "deepfake",
          confidence: 0.85 + Math.random() * 0.14,
        });
        setState("results");
      }, 2000);
    }
  };

  const handleReset = () => {
    setState("upload");
    setSelectedFile(null);
    setResult(null);
    setFilePreviewUrl(null);
  };

  return (
    <div className="min-h-screen flex flex-col bg-background">
      <div className="flex-1 flex flex-col">
        <WelcomeSection />
        
        <main className="flex-1 pb-12">
          {state === "upload" && (
            <div className="space-y-8">
              <UploadArea
                onFileSelect={handleFileSelect}
                selectedFile={selectedFile}
                onClear={handleClearFile}
              />
              
              {selectedFile && (
                <div className="flex justify-center animate-fade-in">
                  <Button
                    variant="default"
                    size="xl"
                    onClick={handleAnalyze}
                  >
                    Analyze for Deepfake
                  </Button>
                </div>
              )}
            </div>
          )}
          
          {state === "loading" && <LoadingScreen />}
          
          {state === "results" && result && (
            <ResultDisplay
              label={result.label}
              confidence={result.confidence}
              filePreview={filePreviewUrl}
              fileName={selectedFile?.name || "Unknown"}
              onReset={handleReset}
            />
          )}
        </main>
      </div>
      
      <Footer />
    </div>
  );
};

export default Index;
