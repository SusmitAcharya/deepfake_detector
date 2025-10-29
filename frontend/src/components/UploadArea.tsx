import { useState, useCallback } from "react";
import { Upload, FileVideo, FileImage, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";

interface UploadAreaProps {
  onFileSelect: (file: File) => void;
  selectedFile: File | null;
  onClear: () => void;
}

export const UploadArea = ({ onFileSelect, selectedFile, onClear }: UploadAreaProps) => {
  const [isDragging, setIsDragging] = useState(false);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setIsDragging(true);
    } else if (e.type === "dragleave") {
      setIsDragging(false);
    }
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragging(false);

      const files = e.dataTransfer.files;
      if (files && files.length > 0) {
        const file = files[0];
        if (file.type.startsWith("image/") || file.type.startsWith("video/")) {
          onFileSelect(file);
        }
      }
    },
    [onFileSelect]
  );

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      onFileSelect(files[0]);
    }
  };

  const getFileIcon = () => {
    if (!selectedFile) return <Upload className="w-16 h-16 text-primary" />;
    if (selectedFile.type.startsWith("video/")) return <FileVideo className="w-16 h-16 text-primary" />;
    return <FileImage className="w-16 h-16 text-primary" />;
  };

  const getPreviewUrl = () => {
    if (!selectedFile) return null;
    return URL.createObjectURL(selectedFile);
  };

  return (
    <div className="w-full max-w-2xl mx-auto px-4 animate-fade-in">
      <Card
        className={`relative border-2 border-dashed transition-all duration-300 overflow-hidden hover-glow ${
          isDragging
            ? "border-primary bg-primary/10"
            : "border-border hover:border-primary/70 bg-card"
        }`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <div className="p-12">
          {!selectedFile ? (
            <div className="flex flex-col items-center justify-center space-y-6">
              {getFileIcon()}
              <div className="text-center space-y-2">
                <h3 className="text-xl font-semibold text-foreground">
                  Drop your file here
                </h3>
                <p className="text-muted-foreground">
                  or click to browse (images and videos supported)
                </p>
              </div>
              <label htmlFor="file-upload">
                <Button variant="minimal" size="lg" asChild>
                  <span className="cursor-pointer">
                    Browse Files
                  </span>
                </Button>
              </label>
              <input
                id="file-upload"
                type="file"
                className="hidden"
                accept="image/*,video/*"
                onChange={handleFileInput}
              />
            </div>
          ) : (
            <div className="space-y-6">
              <div className="flex items-start justify-between">
                <div className="flex items-center space-x-4 flex-1">
                  {getFileIcon()}
                  <div className="flex-1 min-w-0">
                    <p className="text-lg font-medium text-foreground truncate">
                      {selectedFile.name}
                    </p>
                    <p className="text-sm text-muted-foreground">
                      {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                    </p>
                  </div>
                </div>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={onClear}
                  className="hover:bg-destructive/10 hover:text-destructive"
                >
                  <X className="w-5 h-5" />
                </Button>
              </div>
              
              {selectedFile.type.startsWith("image/") && (
                <div className="relative rounded-lg overflow-hidden border border-border">
                  <img
                    src={getPreviewUrl() || ""}
                    alt="Preview"
                    className="w-full h-64 object-contain bg-black/20"
                  />
                </div>
              )}
            </div>
          )}
        </div>
      </Card>
    </div>
  );
};
