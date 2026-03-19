import { useMemo } from "react";
import * as THREE from "three";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import type { DicomSlice, SegmentationClass, SegmentationMask } from "@/types/segmentation";

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

interface Volume3DViewerProps {
    slices: DicomSlice[];
    masks: SegmentationMask[];
    classes: SegmentationClass[];
    overlayOpacity: number;
    imageOpacity?: number;
    showErrors?: boolean;
}

export function Volume3DViewer({ slices, masks, classes, overlayOpacity, imageOpacity = 1.0, showErrors = true }: Volume3DViewerProps) {
    // Convert our class colors to RGB for binary masking overlays
    const classColorMap = useMemo(() => {
        const map: Record<number, { rgb: [number, number, number]; visible: boolean }> = {};
        classes.forEach((cls) => {
            const [h, s, l] = parseHSL(cls.color);
            map[cls.id] = { rgb: hslToRgb(h, s, l), visible: cls.visible };
        });
        return map;
    }, [classes]);

    // Separate textures into base images and independent mask matrices to allow volumetric rendering
    const textures = useMemo(() => {
        if (!slices.length) return [];

        return slices.map((slice, z) => {
            const { width, height, pixelData, windowCenter, windowWidth, rescaleSlope, rescaleIntercept } = slice;
            const mask = masks.find((m) => m.sliceIndex === z);
            let hasMask = false;

            const baseData = new Uint8Array(width * height * 4); // RGBA format
            const maskData = new Uint8Array(width * height * 4);
            const wMin = windowCenter - windowWidth / 2;
            const wMax = windowCenter + windowWidth / 2;

            for (let i = 0; i < pixelData.length; i++) {
                const hu = pixelData[i] * rescaleSlope + rescaleIntercept;
                let gray = ((hu - wMin) / (wMax - wMin)) * 255;
                gray = Math.max(0, Math.min(255, gray));

                // Fade mostly dark background space aggressively to see internal structures natively
                let alpha = gray > 40 ? Math.min(255, gray + 20) : 0;

                const idx = i * 4;
                baseData[idx] = gray;
                baseData[idx + 1] = gray;
                baseData[idx + 2] = gray;
                baseData[idx + 3] = alpha;

                maskData[idx] = 0;
                maskData[idx + 1] = 0;
                maskData[idx + 2] = 0;
                maskData[idx + 3] = 0;

                // Map separate mask layer cleanly independently
                if (mask && mask.data[i] !== 0) {
                    const classId = mask.data[i];

                    if (!showErrors && mask.errors && mask.errors[classId]) {
                        continue;
                    }

                    const classInfo = classColorMap[classId];
                    if (classInfo && classInfo.visible) {
                        const x = i % width;
                        const y = Math.floor(i / width);
                        // Calculate inner edge depth (0 = outer mask edge, 1 = inner mask boundary, 2 = mask core)
                        let depth = 2;

                        // Check immediate 4-way neighbors (outer edge)
                        if (
                            x <= 0 || mask.data[i - 1] !== classId ||
                            x >= width - 1 || mask.data[i + 1] !== classId ||
                            y <= 0 || mask.data[i - width] !== classId ||
                            y >= height - 1 || mask.data[i + width] !== classId
                        ) {
                            depth = 0;
                        }
                        // Check 2-pixels out (inner soft boundary)
                        else if (
                            x <= 1 || mask.data[i - 2] !== classId ||
                            x >= width - 2 || mask.data[i + 2] !== classId ||
                            y <= 1 || mask.data[i - width * 2] !== classId ||
                            y >= height - 2 || mask.data[i + width * 2] !== classId
                        ) {
                            depth = 1;
                        }

                        const [cr, cg, cb] = classInfo.rgb;

                        if (depth === 0) {
                            // Outer edge: highly transparent, slightly darker
                            maskData[idx] = cr * 0.5;
                            maskData[idx + 1] = cg * 0.5;
                            maskData[idx + 2] = cb * 0.5;
                            maskData[idx + 3] = 40;   // Fades out into the MRI volume
                        } else if (depth === 1) {
                            // Inner boundary: soft transition
                            maskData[idx] = cr * 0.8;
                            maskData[idx + 1] = cg * 0.8;
                            maskData[idx + 2] = cb * 0.8;
                            maskData[idx + 3] = 140;  // Medium opacity slope
                        } else {
                            // Core center: extremely solid and bright for maximum glossy light reflections
                            maskData[idx] = cr;
                            maskData[idx + 1] = cg;
                            maskData[idx + 2] = cb;
                            maskData[idx + 3] = 255;  // 100% thick center
                        }

                        hasMask = true;
                    }
                }
            }

            const baseTexture = new THREE.DataTexture(baseData, width, height, THREE.RGBAFormat);
            baseTexture.magFilter = THREE.NearestFilter;
            baseTexture.minFilter = THREE.NearestFilter;
            baseTexture.needsUpdate = true;

            let maskTexture = null;
            if (hasMask) {
                maskTexture = new THREE.DataTexture(maskData, width, height, THREE.RGBAFormat);
                maskTexture.magFilter = THREE.NearestFilter;
                maskTexture.minFilter = THREE.NearestFilter;
                maskTexture.needsUpdate = true;
            }

            return { baseTexture, maskTexture, hasMask };
        });
    }, [slices, masks, classColorMap, showErrors]); // Note: Optically dropping the sliders from this array drastically improves FPS!

    // Increase sub-planes for a tighter, more cohesive block mesh
    const PLANES_PER_SLICE = 12;

    return (
        <div className="w-full h-full bg-black/95 rounded-lg overflow-hidden relative cursor-move">
            <Canvas camera={{ position: [0, -4, 0], fov: 45 }}>
                <ambientLight intensity={2.0} />
                <spotLight position={[10, 20, 10]} angle={0.5} penumbra={1} intensity={5.0} castShadow />
                <directionalLight position={[-10, -20, -10]} intensity={2.0} color="#90b0d0" />
                <directionalLight position={[20, 0, 0]} intensity={1.5} color="#ffd0b0" />
                {textures.map((layer, i) => {
                    // Normalize physical stacking scale 
                    const sliceSpacing = 0.05;
                    const totalDepth = slices.length * sliceSpacing;
                    const zPos = - (totalDepth / 2) + (i * sliceSpacing);

                    const maskPlanes = [];
                    if (layer.hasMask && layer.maskTexture) {
                        const subSpacing = sliceSpacing / PLANES_PER_SLICE;
                        for (let j = 0; j < PLANES_PER_SLICE; j++) {
                            // Trim the extremely last interpolation frame so it doesn't pop out boundaries
                            if (i === slices.length - 1 && j > 0) break;

                            maskPlanes.push(
                                <mesh key={`mask-${i}-${j}`} position={[0, 0, zPos + (j * subSpacing)]}>
                                    <planeGeometry args={[2.5, 2.5]} />
                                    <meshPhongMaterial
                                        map={layer.maskTexture}
                                        transparent={true}
                                        opacity={overlayOpacity}
                                        alphaTest={0.001}
                                        shininess={60}
                                        side={2}
                                        depthWrite={true}
                                        depthTest={true}
                                    />
                                </mesh>
                            );
                        }
                    }

                    return (
                        <group key={`slice-${i}`}>
                            <mesh position={[0, 0, zPos]}>
                                <planeGeometry args={[2.5, 2.5]} />
                                <meshBasicMaterial
                                    map={layer.baseTexture}
                                    transparent={true}
                                    opacity={imageOpacity} // We use the material to slide opacity seamlessly
                                    side={2}
                                    depthWrite={false}
                                />
                            </mesh>
                            {maskPlanes}
                        </group>
                    );
                })}
                <OrbitControls enableZoom={true} enablePan={true} enableRotate={true} />
            </Canvas>
            <div className="absolute inset-x-0 bottom-4 pointer-events-none flex justify-center text-xs font-mono text-zinc-400">
                Drag to rotate. Scroll to zoom.
            </div>
        </div>
    );
}
