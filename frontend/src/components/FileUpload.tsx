import { useCallback, useState, DragEvent, useRef } from "react";
import { Upload, FileImage } from "lucide-react";

interface FileUploadProps {
  onFilesSelected: (files: File[]) => void;
  isLoading: boolean;
}

export function FileUpload({ onFilesSelected, isLoading }: FileUploadProps) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFiles = useCallback(
    (fileList: FileList) => {
      const files = Array.from(fileList).filter(
        (f) => f.name.endsWith(".dcm") || f.name.endsWith(".dicom") || !f.name.includes(".")
      );
      if (files.length > 0) onFilesSelected(files);
    },
    [onFilesSelected]
  );

  const handleDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragging(false);
    handleFiles(e.dataTransfer.files);
  };

  return (
    <div
      className={`upload-zone ${dragging ? "dragging" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        multiple
        accept=".dcm,.dicom"
        className="hidden"
        onChange={(e) => e.target.files && handleFiles(e.target.files)}
      />
      <div className="flex flex-col items-center gap-3">
        {isLoading ? (
          <>
            <div className="w-10 h-10 border-2 border-primary border-t-transparent rounded-full animate-spin" />
            <p className="text-sm text-muted-foreground">Processing DICOM files…</p>
          </>
        ) : (
          <>
            <div className="w-14 h-14 rounded-xl bg-secondary flex items-center justify-center">
              <Upload className="w-6 h-6 text-primary" />
            </div>
            <div>
              <p className="text-sm font-medium text-foreground">
                Drop DICOM files here or click to browse
              </p>
              <p className="text-xs text-muted-foreground mt-1">
                Supports .dcm and .dicom files
              </p>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
