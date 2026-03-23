import dicomParser from "dicom-parser";
import type { DicomSlice } from "@/types/segmentation";

export function parseDicomFile(file: File): Promise<DicomSlice> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const arrayBuffer = reader.result as ArrayBuffer;
        const byteArray = new Uint8Array(arrayBuffer);
        const dataSet = dicomParser.parseDicom(byteArray);

        const rows = dataSet.uint16("x00280010") || 256;
        const cols = dataSet.uint16("x00280011") || 256;
        const bitsAllocated = dataSet.uint16("x00280100") || 16;
        const pixelRepresentation = dataSet.uint16("x00280103") || 0;
        const windowCenter = parseFloat(dataSet.string("x00281050") || "400");
        const windowWidth = parseFloat(dataSet.string("x00281051") || "800");
        const rescaleSlope = parseFloat(dataSet.string("x00281053") || "1");
        const rescaleIntercept = parseFloat(dataSet.string("x00281052") || "0");
        const instanceNumber = dataSet.intString("x00200013") || 0;

        const pixelDataElement = dataSet.elements.x7fe00010;
        if (!pixelDataElement) throw new Error("No pixel data found");

        const pixelData: number[] = [];

        if (bitsAllocated === 16) {
          const view = new DataView(
            byteArray.buffer,
            pixelDataElement.dataOffset,
            pixelDataElement.length
          );
          const numPixels = rows * cols;
          for (let i = 0; i < numPixels; i++) {
            const val = pixelRepresentation === 1
              ? view.getInt16(i * 2, true)
              : view.getUint16(i * 2, true);
            pixelData.push(val);
          }
        } else {
          for (let i = 0; i < rows * cols; i++) {
            pixelData.push(byteArray[pixelDataElement.dataOffset + i]);
          }
        }

        resolve({
          index: 0,
          fileName: file.name,
          pixelData,
          width: cols,
          height: rows,
          windowCenter,
          windowWidth,
          rescaleSlope,
          rescaleIntercept,
          instanceNumber,
        });
      } catch (err) {
        reject(err);
      }
    };
    reader.onerror = reject;
    reader.readAsArrayBuffer(file);
  });
}

export async function parseDicomFiles(files: File[]): Promise<DicomSlice[]> {
  const slices = await Promise.all(files.map(parseDicomFile));
  slices.sort((a, b) => a.instanceNumber - b.instanceNumber);
  return slices.map((s, i) => ({ ...s, index: i }));
}
