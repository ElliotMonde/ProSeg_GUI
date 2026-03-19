import { useState, useCallback } from "react";
import { FileUpload } from "@/components/FileUpload";
import { SliceViewer } from "@/components/SliceViewer";
import { SliceSlider } from "@/components/SliceSlider";
import { Volume3DViewer } from "@/components/Volume3DViewer";
import { ClassTogglePanel } from "@/components/ClassTogglePanel";
import { parseDicomFiles } from "@/lib/dicomParser";
import { runInference } from "@/lib/inferenceApi";
import { SEGMENTATION_CLASSES } from "@/types/segmentation";
import type { DicomSlice, SegmentationClass, SegmentationMask } from "@/types/segmentation";
import { Scan, Settings, Loader2, Layers, Box } from "lucide-react";
import { toast } from "sonner";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";

const Index = () => {
  const [slices, setSlices] = useState<DicomSlice[]>([]);
  const [currentSliceIdx, setCurrentSliceIdx] = useState(0);
  const [classes, setClasses] = useState<SegmentationClass[]>(SEGMENTATION_CLASSES);
  const [masks, setMasks] = useState<SegmentationMask[]>([]);
  const [overlayOpacity, setOverlayOpacity] = useState(0.45);
  const [imageOpacity, setImageOpacity] = useState(1.0);
  const [isLoading, setIsLoading] = useState(false);
  const [isInferring, setIsInferring] = useState(false);
  const [apiUrl, setApiUrl] = useState("http://localhost:8000/segment");
  const [showSettings, setShowSettings] = useState(false);
  const [showConfidences, setShowConfidences] = useState(false);
  const [showErrors, setShowErrors] = useState(true);
  const [rawFiles, setRawFiles] = useState<File[]>([]);
  const [viewMode, setViewMode] = useState<"2D" | "3D">("2D");

  const handleFilesSelected = useCallback(async (files: File[]) => {
    setIsLoading(true);
    try {
      const parsed = await parseDicomFiles(files);
      setSlices(parsed);
      setRawFiles(files);
      setCurrentSliceIdx(0);
      setMasks([]);
      setViewMode("2D");
      toast.success(`Loaded ${parsed.length} DICOM slices`);
    } catch (err) {
      toast.error(`Failed to parse DICOM: ${err instanceof Error ? err.message : "Unknown error"}`);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const handleRunInference = useCallback(async () => {
    if (rawFiles.length === 0) return;
    setIsInferring(true);
    try {
      const result = await runInference(rawFiles, apiUrl);
      setMasks(result.masks);
      toast.success("Segmentation complete!");
    } catch (err) {
      toast.error(`Inference failed: ${err instanceof Error ? err.message : "Unknown error"}`);
    } finally {
      setIsInferring(false);
    }
  }, [rawFiles, apiUrl]);

  const toggleClass = useCallback((id: number) => {
    setClasses((prev) =>
      prev.map((c) => (c.id === id ? { ...c, visible: !c.visible } : c))
    );
  }, []);

  const currentSlice = slices[currentSliceIdx];
  const currentMask = masks.find((m) => m.sliceIndex === currentSliceIdx);

  return (
    <div className="h-screen flex flex-col bg-background overflow-hidden">
      {/* Header */}
      <header className="flex items-center justify-between px-5 py-3 bg-card border-b border-border flex-shrink-0">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-primary/15 flex items-center justify-center">
            <Scan className="w-4 h-4 text-primary" />
          </div>
          <div>
            <h1 className="text-sm font-semibold text-foreground tracking-tight">
              ProSeg Viewer
            </h1>
            <p className="text-[10px] font-mono text-muted-foreground">
              YOLO11n Prostate MRI Segmentation
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {slices.length > 0 && (
            <span className="text-xs font-mono text-muted-foreground px-2 py-1 bg-secondary rounded">
              {slices.length} slices
            </span>
          )}
          <Tooltip>
            <TooltipTrigger asChild>
              <button
                onClick={() => setShowSettings(!showSettings)}
                className="p-2 rounded-md hover:bg-secondary transition-colors text-muted-foreground hover:text-foreground"
              >
                <Settings className="w-4 h-4" />
              </button>
            </TooltipTrigger>
            <TooltipContent side="bottom">
              API Settings
            </TooltipContent>
          </Tooltip>
        </div>
      </header>

      {/* Settings panel */}
      {showSettings && (
        <div className="px-5 py-3 bg-card border-b border-border flex items-center gap-3">
          <label className="text-xs font-mono text-muted-foreground">API URL:</label>
          <input
            type="text"
            value={apiUrl}
            onChange={(e) => setApiUrl(e.target.value)}
            className="flex-1 max-w-lg bg-secondary border border-border rounded px-3 py-1.5 text-xs font-mono text-foreground focus:outline-none focus:ring-1 focus:ring-primary"
          />
        </div>
      )}

      {/* Main content */}
      <div className="flex-1 flex min-h-0">
        {/* Viewer area */}
        <div className="flex-1 flex flex-col p-4 gap-3 min-w-0 relative">
          {currentSlice ? (
            <>
              {/* 3D/2D View Toggle */}
              <div className="absolute top-6 left-6 z-10 flex border border-border bg-background/80 backdrop-blur rounded-lg p-1 gap-1">
                <Tooltip>
                  <TooltipTrigger asChild>
                    <button
                      onClick={() => setViewMode("2D")}
                      className={`flex items-center gap-2 px-3 py-1.5 text-xs font-medium rounded-md transition-colors ${viewMode === "2D" ? "bg-primary text-primary-foreground shadow" : "text-muted-foreground hover:bg-secondary"
                        }`}
                    >
                      <Layers className="w-3.5 h-3.5" /> 2D Viewer
                    </button>
                  </TooltipTrigger>
                  <TooltipContent side="bottom">
                    View single DICOM slices
                  </TooltipContent>
                </Tooltip>

                <Tooltip>
                  <TooltipTrigger asChild>
                    <button
                      onClick={() => setViewMode("3D")}
                      className={`flex items-center gap-2 px-3 py-1.5 text-xs font-medium rounded-md transition-colors ${viewMode === "3D" ? "bg-primary text-primary-foreground shadow" : "text-muted-foreground hover:bg-secondary"
                        }`}
                    >
                      <Box className="w-3.5 h-3.5" /> 3D Volume
                    </button>
                  </TooltipTrigger>
                  <TooltipContent side="bottom">
                    View true 3D reconstructed volume
                  </TooltipContent>
                </Tooltip>
              </div>

              <div className="flex-1 min-h-0">
                {viewMode === "2D" ? (
                  <SliceViewer
                    slice={currentSlice}
                    mask={currentMask}
                    classes={classes}
                    overlayOpacity={overlayOpacity}
                    imageOpacity={imageOpacity}
                    showErrors={showErrors}
                  />
                ) : (
                  <Volume3DViewer
                    slices={slices}
                    masks={masks}
                    classes={classes}
                    overlayOpacity={overlayOpacity}
                    imageOpacity={imageOpacity}
                    showErrors={showErrors}
                  />
                )}
              </div>

              {viewMode === "2D" && (
                <SliceSlider
                  currentSlice={currentSliceIdx}
                  totalSlices={slices.length}
                  onChange={setCurrentSliceIdx}
                />
              )}
            </>
          ) : (
            <div className="flex-1 flex items-center justify-center">
              <div className="max-w-md w-full">
                <FileUpload onFilesSelected={handleFilesSelected} isLoading={isLoading} />
              </div>
            </div>
          )}
        </div>

        {/* Right panel */}
        {slices.length > 0 && (
          <div className="w-64 border-l border-border bg-card p-4 flex flex-col gap-4 overflow-y-auto flex-shrink-0">
            {/* Inference button */}
            <Tooltip>
              <TooltipTrigger asChild>
                <button
                  onClick={handleRunInference}
                  disabled={isInferring || slices.length === 0}
                  className="w-full py-2.5 px-4 bg-primary text-primary-foreground rounded-lg text-sm font-medium hover:bg-primary/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                >
                  {isInferring ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Running…
                    </>
                  ) : (
                    <>
                      <Scan className="w-4 h-4" />
                      Run Segmentation
                    </>
                  )}
                </button>
              </TooltipTrigger>
              <TooltipContent side="top">
                Run YOLO inference on all loaded DICOM slices
              </TooltipContent>
            </Tooltip>

            {/* Class toggles */}
            <ClassTogglePanel
              classes={classes}
              onToggle={toggleClass}
              opacity={overlayOpacity}
              onOpacityChange={setOverlayOpacity}
              imageOpacity={imageOpacity}
              onImageOpacityChange={setImageOpacity}
              showConfidences={showConfidences}
              onShowConfidencesChange={setShowConfidences}
              confidences={currentMask?.confidences}
              showErrors={showErrors}
              onShowErrorsChange={setShowErrors}
              errors={currentMask?.errors}
            />

            {/* Slice info */}
            {currentSlice && (
              <div className="panel-section space-y-2">
                <h3 className="text-xs font-mono font-semibold text-muted-foreground uppercase tracking-wider">
                  Slice Info
                </h3>
                <div className="space-y-1 text-xs font-mono">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">File</span>
                    <span className="text-foreground truncate ml-2 max-w-[120px]">
                      {currentSlice.fileName}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Size</span>
                    <span className="text-foreground">
                      {currentSlice.width}×{currentSlice.height}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Window</span>
                    <span className="text-foreground">
                      C:{currentSlice.windowCenter} W:{currentSlice.windowWidth}
                    </span>
                  </div>
                </div>
              </div>
            )}

            {/* Load new files */}
            <div className="mt-auto">
              <FileUpload onFilesSelected={handleFilesSelected} isLoading={isLoading} />
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Index;
