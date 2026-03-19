import type { SegmentationClass } from "@/types/segmentation";
import { Eye, EyeOff, AlertTriangle } from "lucide-react";
import { Switch } from "@/components/ui/switch";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";

interface ClassTogglePanelProps {
  classes: SegmentationClass[];
  onToggle: (id: number) => void;
  opacity: number;
  onOpacityChange: (value: number) => void;
  imageOpacity: number;
  onImageOpacityChange: (value: number) => void;
  showConfidences?: boolean;
  onShowConfidencesChange?: (value: boolean) => void;
  confidences?: Record<number, number>;
  showErrors?: boolean;
  onShowErrorsChange?: (value: boolean) => void;
  errors?: Record<number, boolean>;
}

export function ClassTogglePanel({
  classes,
  onToggle,
  opacity,
  onOpacityChange,
  imageOpacity,
  onImageOpacityChange,
  showConfidences = false,
  onShowConfidencesChange,
  confidences,
  showErrors = true,
  onShowErrorsChange,
  errors,
}: ClassTogglePanelProps) {
  return (
    <div className="panel-section space-y-3">
      <h3 className="text-xs font-mono font-semibold text-muted-foreground uppercase tracking-wider">
        Segmentation Classes
      </h3>
      <div className="space-y-1">
        {classes.map((cls) => {
          const hasError = errors && errors[cls.id];
          return (
            <Tooltip key={cls.id}>
              <TooltipTrigger asChild>
                <button
                  onClick={() => onToggle(cls.id)}
                  className="w-full flex items-center gap-3 px-3 py-2 rounded-md text-sm transition-colors hover:bg-secondary group relative"
                >
                  <span
                    className="class-indicator shrink-0 transition-all duration-200"
                    style={{
                      backgroundColor: cls.visible ? cls.color : "currentColor",
                      opacity: cls.visible ? 1 : 0.3,
                      filter: cls.visible ? "none" : "grayscale(100%)",
                    }}
                  />
                  <span
                    className={`flex-1 text-left flex items-center justify-between transition-all duration-200 ${cls.visible ? "text-foreground" : "text-muted-foreground opacity-50"
                      }`}
                  >
                    <span className="flex items-center gap-1.5">
                      {cls.name}
                      {hasError && (
                        <AlertTriangle className="w-3.5 h-3.5 text-destructive" />
                      )}
                    </span>
                    {showConfidences && confidences && confidences[cls.id] !== undefined && (
                      <span className="text-xs font-mono text-muted-foreground ml-2">
                        {(confidences[cls.id] * 100).toFixed(1)}%
                      </span>
                    )}
                  </span>
                  {cls.visible ? (
                    <Eye className="w-3.5 h-3.5 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity shrink-0" />
                  ) : (
                    <EyeOff className="w-3.5 h-3.5 text-muted-foreground shrink-0" />
                  )}
                </button>
              </TooltipTrigger>
              <TooltipContent side="left">
                Toggle visibility for {cls.name}
                {hasError && " (Error: Prediction occupies over 50% of image)"}
              </TooltipContent>
            </Tooltip>
          );
        })}
      </div>
      <div className="pt-2 border-t border-border space-y-4">
        {onShowConfidencesChange && (
          <Tooltip>
            <TooltipTrigger asChild>
              <div className="flex items-center justify-between">
                <label className="text-xs font-mono text-muted-foreground cursor-pointer" onClick={() => onShowConfidencesChange(!showConfidences)}>
                  Show Confidence Scores
                </label>
                <Switch
                  checked={showConfidences}
                  onCheckedChange={onShowConfidencesChange}
                />
              </div>
            </TooltipTrigger>
            <TooltipContent side="left">
              Display model confidence % for each detected slice
            </TooltipContent>
          </Tooltip>
        )}

        {onShowErrorsChange && (
          <Tooltip>
            <TooltipTrigger asChild>
              <div className="flex items-center justify-between">
                <label className="text-xs font-mono text-muted-foreground cursor-pointer" onClick={() => onShowErrorsChange(!showErrors)}>
                  Show Possible Errors
                </label>
                <Switch
                  checked={showErrors}
                  onCheckedChange={onShowErrorsChange}
                />
              </div>
            </TooltipTrigger>
            <TooltipContent side="left">
              Show/hide predictions occupying over 50% of the slice area
            </TooltipContent>
          </Tooltip>
        )}

        <div>
          <label className="text-xs font-mono text-muted-foreground block mb-2">
            Base Image Opacity: {Math.round(imageOpacity * 100)}%
          </label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={imageOpacity}
            onChange={(e) => onImageOpacityChange(parseFloat(e.target.value))}
            className="w-full accent-primary h-1"
          />
        </div>
        <div>
          <label className="text-xs font-mono text-muted-foreground block mb-2">
            Overlay Mask Opacity: {Math.round(opacity * 100)}%
          </label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={opacity}
            onChange={(e) => onOpacityChange(parseFloat(e.target.value))}
            className="w-full accent-primary h-1"
          />
        </div>
      </div>
    </div>
  );
}
