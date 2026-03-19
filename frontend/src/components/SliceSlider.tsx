interface SliceSliderProps {
  currentSlice: number;
  totalSlices: number;
  onChange: (index: number) => void;
}

export function SliceSlider({ currentSlice, totalSlices, onChange }: SliceSliderProps) {
  if (totalSlices <= 1) return null;

  return (
    <div className="flex items-center gap-3 px-4 py-3 bg-card rounded-lg border border-border">
      <span className="text-xs font-mono text-muted-foreground w-16">
        Slice
      </span>
      <input
        type="range"
        min="0"
        max={totalSlices - 1}
        value={currentSlice}
        onChange={(e) => onChange(parseInt(e.target.value))}
        className="flex-1 accent-primary h-1"
      />
      <span className="text-xs font-mono text-primary w-20 text-right">
        {currentSlice + 1} / {totalSlices}
      </span>
    </div>
  );
}
