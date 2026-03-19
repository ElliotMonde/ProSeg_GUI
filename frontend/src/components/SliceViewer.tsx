import { useEffect, useRef, useMemo } from "react";
import type { DicomSlice, SegmentationClass, SegmentationMask } from "@/types/segmentation";

interface SliceViewerProps {
  slice: DicomSlice;
  mask?: SegmentationMask;
  classes: SegmentationClass[];
  overlayOpacity: number;
  imageOpacity?: number;
  showErrors?: boolean;
}

function parseHSL(hsl: string): [number, number, number] {
  const match = hsl.match(/hsl\((\d+),\s*(\d+)%?,\s*(\d+)%?\)/);
  if (!match) return [0, 0, 0];
  return [parseInt(match[1]), parseInt(match[2]), parseInt(match[3])];
}

function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  s /= 100;
  l /= 100;
  const k = (n: number) => (n + h / 30) % 12;
  const a = s * Math.min(l, 1 - l);
  const f = (n: number) => l - a * Math.max(-1, Math.min(k(n) - 3, 9 - k(n), 1));
  return [Math.round(f(0) * 255), Math.round(f(8) * 255), Math.round(f(4) * 255)];
}

export function SliceViewer({ slice, mask, classes, overlayOpacity, imageOpacity = 1.0, showErrors = true }: SliceViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const classColorMap = useMemo(() => {
    const map: Record<number, { rgb: [number, number, number]; visible: boolean }> = {};
    classes.forEach((cls) => {
      const [h, s, l] = parseHSL(cls.color);
      map[cls.id] = { rgb: hslToRgb(h, s, l), visible: cls.visible };
    });
    return map;
  }, [classes]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const { width, height, pixelData, windowCenter, windowWidth, rescaleSlope, rescaleIntercept } = slice;
    canvas.width = width;
    canvas.height = height;

    const imageData = ctx.createImageData(width, height);
    const data = imageData.data;

    const wMin = windowCenter - windowWidth / 2;
    const wMax = windowCenter + windowWidth / 2;

    // Render base MRI
    const baseAlpha = Math.max(0, Math.min(255, Math.round(imageOpacity * 255)));
    for (let i = 0; i < pixelData.length; i++) {
      const hu = pixelData[i] * rescaleSlope + rescaleIntercept;
      let gray = ((hu - wMin) / (wMax - wMin)) * 255;
      gray = Math.max(0, Math.min(255, gray));

      const idx = i * 4;
      data[idx] = gray;
      data[idx + 1] = gray;
      data[idx + 2] = gray;
      data[idx + 3] = baseAlpha;
    }

    // Overlay segmentation masks
    if (mask && mask.data.length === width * height) {
      for (let i = 0; i < mask.data.length; i++) {
        const classId = mask.data[i];
        if (classId === 0) continue;

        if (!showErrors && mask.errors && mask.errors[classId]) {
          continue;
        }

        const classInfo = classColorMap[classId];
        if (!classInfo || !classInfo.visible) continue;

        const idx = i * 4;
        const [r, g, b] = classInfo.rgb;
        const alpha = overlayOpacity;

        // Blend the colors natively with the new foreground alpha
        data[idx] = data[idx] * (1 - alpha) + r * alpha;
        data[idx + 1] = data[idx + 1] * (1 - alpha) + g * alpha;
        data[idx + 2] = data[idx + 2] * (1 - alpha) + b * alpha;

        // Ensure segmented structures remain fully opaque regardless of base image opacity
        data[idx + 3] = 255;
      }
    }

    ctx.putImageData(imageData, 0, 0);
  }, [slice, mask, classColorMap, overlayOpacity, imageOpacity, showErrors]);

  return (
    <div className="flex items-center justify-center w-full h-full bg-viewer-bg rounded-lg overflow-hidden">
      <canvas
        ref={canvasRef}
        className="viewer-canvas max-w-full max-h-full object-contain"
        style={{ imageRendering: "pixelated" }}
      />
    </div>
  );
}
