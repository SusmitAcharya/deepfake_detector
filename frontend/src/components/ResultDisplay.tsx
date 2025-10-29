import { CheckCircle2, AlertTriangle } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

interface ResultDisplayProps {
  label: "deepfake" | "authentic";
  confidence: number;
  filePreview: string | null;
  fileName: string;
  onReset: () => void;
}

export const ResultDisplay = ({
  label,
  confidence,
  filePreview,
  fileName,
  onReset,
}: ResultDisplayProps) => {
  const isAuthentic = label === "authentic";
  const confidencePercent = (confidence * 100).toFixed(1);

  return (
    <div className="w-full max-w-3xl mx-auto px-4 space-y-8 animate-slide-up">
      {/* File Preview */}
      {filePreview && (
        <Card className="overflow-hidden border border-border bg-card">
          <div className="relative">
            <img
              src={filePreview}
              alt="Analyzed media"
              className="w-full h-96 object-contain bg-black/20"
            />
            <div className="absolute top-4 left-4 bg-black/60 backdrop-blur-sm px-4 py-2 rounded-lg">
              <p className="text-sm text-foreground font-medium">{fileName}</p>
            </div>
          </div>
        </Card>
      )}

      {/* Results Card */}
      <Card className={`border p-8 ${
        isAuthentic
          ? "border-success/50 bg-success/5"
          : "border-destructive/50 bg-destructive/5"
      }`}>
        <div className="flex flex-col items-center space-y-6">
          {/* Icon */}
          {isAuthentic ? (
            <CheckCircle2 className="w-20 h-20 text-success" />
          ) : (
            <AlertTriangle className="w-20 h-20 text-destructive" />
          )}

          {/* Label */}
          <div className="text-center">
            <h2 className={`text-4xl md:text-5xl font-bold mb-2 ${
              isAuthentic ? "text-success" : "text-destructive"
            }`}>
              {isAuthentic ? "Authentic" : "Deepfake Detected"}
            </h2>
            <p className="text-muted-foreground text-lg">
              {isAuthentic
                ? "This media appears to be genuine"
                : "This media shows signs of manipulation"}
            </p>
          </div>

          {/* Confidence Score */}
          <div className="w-full max-w-md space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground font-medium">
                Confidence Level
              </span>
              <span className={`text-3xl font-bold ${
                isAuthentic ? "text-success" : "text-destructive"
              }`}>
                {confidencePercent}%
              </span>
            </div>
            
            <div className="w-full h-3 bg-secondary rounded-full overflow-hidden">
              <div
                className={`h-full transition-all duration-1000 ${
                  isAuthentic ? "bg-success" : "bg-destructive"
                }`}
                style={{ width: `${confidencePercent}%` }}
              />
            </div>
          </div>

          {/* Action Button */}
          <Button
            variant={isAuthentic ? "success" : "minimal"}
            size="xl"
            onClick={onReset}
            className="mt-4"
          >
            Upload Another File
          </Button>
        </div>
      </Card>
    </div>
  );
};
