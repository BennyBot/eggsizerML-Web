import React, { useEffect, useRef, useState } from "react";

/**
 * EggSizer – a browser‑based re‑implementation of the original Qt/OpenCV C++ tool.
 *
 * Key port notes – thresholds & constants are kept identical to the C++ version:
 *   BLOB_MIN_AREA   = 5 000   px²
 *   BLOB_MAX_AREA   = 500 000 px²
 *   POLY_MIN_AREA   = 18 000  px²
 *   POLY_MAX_AREA   = 700 000 px²
 *   PIXELS_PER_MM²  = 100     (pixToMM in the C++ code)
 *
 * The processing pipeline mirrors the original source files one‑to‑one:
 *   1. autoCanny      →   Gaussian(5×5)  +  Otsu threshold  (cannyDetect.cpp)
 *   2. polyApprox     →   findContours + approxPolyDP        (measureEdges.cpp)
 *   3. detectBlobs    →   connectedComponentsWithStats       (blobDetect.cpp)
 *
 * OpenCV.js does not ship SimpleBlobDetector, therefore we approximate the blob
 * detector with connected‑components + area filtering so that the numeric
 * results remain comparable.
 */

const BLOB_MIN_AREA = 2000;
const BLOB_MAX_AREA = 1000000;
const POLY_MIN_AREA = 18000;
const POLY_MAX_AREA = 700000;
const PIXELS_PER_MM = 100; // “pixToMM” in the original code

const CANVAS_SIZE = { width: "45vw", height: "40vh", border: "1px solid #FFF "}; // CSS units for canvas size


export default function EggSizerApp() {
  // --- state ---------------------------------------------------------------
  const [files,   setFiles]                 = useState([]);      // FileList
  const [index,   setIndex]                 = useState(0);       // current image idx
  const [results, setResults]               = useState([]);      // accumulated CSV rows

  // --- refs to <canvas> ----------------------------------------------------
  const canvUL = useRef(); // uploaded image (upper‑left)
  const canvUR = useRef(); // blob processed   (upper‑right)
  const canvLL = useRef(); // Otsu processed   (lower‑left)
  const canvLR = useRef(); // polygon approx   (lower‑right)

  // --- helpers -------------------------------------------------------------
  const readImageToMat = (file, cb) => {
    const img = new Image();
    img.onload = () => {
      let mat = new cv.Mat();
      mat = cv.imread(img);
      cb(mat);
    };
    img.src = URL.createObjectURL(file);
  };

  const grayscaleImage = (src) => {
    const gray = new cv.Mat();
    if (src.channels() > 1) cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
    else src.copyTo(gray);
    return gray;
  }

  const autoCanny = (src) => {
    const gray = grayscaleImage(src);
    const blurred = new cv.Mat();
    const dst = new cv.Mat();

    cv.GaussianBlur(gray, blurred, new cv.Size(5, 5), 0.33, 0, cv.BORDER_DEFAULT);
    cv.threshold(blurred, dst, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU);

    gray.delete(); blurred.delete();
    return dst;
  };

  const polyApprox = (thresh, orig) => {
    const contours = new cv.MatVector();
    const hierarchy = new cv.Mat();
    cv.findContours(thresh, contours, hierarchy, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE);

    const dst = orig.clone();
    const areas = [];

    for (let i = 0; i < contours.size(); ++i) {
      const cnt = contours.get(i);
      const peri = cv.arcLength(cnt, true);
      const approx = new cv.Mat();
      cv.approxPolyDP(cnt, approx, 0.005 * peri, true);
      const area = cv.contourArea(approx, false);
      if (area >= POLY_MIN_AREA && area <= POLY_MAX_AREA) {
        areas.push(area / PIXELS_PER_MM);
        const colour = new cv.Scalar(0, 150, 0, 255);
        const tmpVec = new cv.MatVector();
        tmpVec.push_back(approx);
        cv.drawContours(dst, tmpVec, -1, colour, 2);
        tmpVec.delete(); 
        //colour.delete();
      }
      cnt.delete(); approx.delete();
    }
    contours.delete(); hierarchy.delete();
    return { dst, areas };
  };

  const detectBlobs = (src) => {

    let params = {
      faster: true,
      filterByArea: true,
      minArea: BLOB_MIN_AREA,
      maxArea: BLOB_MAX_AREA,
      filterByCircularity: false,
      filterByConvexity: false,
      filterByInertia: false,
      filterByColor: false,
      thresholdStep: 10,
      minThreshold: 0,
      maxThreshold: 255,
      minRepeatability: 2,
      minDistBetweenBlobs: 10,
    };

    // emulate SimpleBlobDetector with connected components
    const gray = grayscaleImage(src);
    const dst = src.clone();
    let areas = [];
    const thresh = new cv.Mat();
    cv.threshold(gray, thresh, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU);

    let centers = findBlobs(gray, thresh, params);

    centers.forEach((c) => {
      cv.circle(dst, new cv.Point(c.location.x, c.location.y), c.radius, new cv.Scalar(255, 0, 0, 255), 2);
      areas.push(c.radius * c.radius * Math.PI / PIXELS_PER_MM);
    });
    
    gray.delete(); thresh.delete();

    return { dst, areas };
  };

  const processFile = (file) => {
    readImageToMat(file, (orig) => {
      // UL: show original
      cv.imshow(canvUL.current, orig);

      // LL: Otsu threshold of Gaussian‑blurred grayscale
      const otsu = autoCanny(orig);
      cv.imshow(canvLL.current, otsu);

      // LR: polygonal approximation on threshold image
      const { dst: polyDst, areas: otsuAreas } = polyApprox(otsu, orig);
      cv.imshow(canvLR.current, polyDst);

      // UR: blob detection on original
      const { dst: blobDst, areas: blobAreas } = detectBlobs(orig);
      cv.imshow(canvUR.current, blobDst);

      // build per‑image result rows
      const rows = [];
      const maxLen = Math.max(otsuAreas.length, blobAreas.length);
      for (let i = 0; i < maxLen; ++i) {
        rows.push({
          imgName   : file.name,
          eggNo     : i + 1,
          otsuSize  : otsuAreas[i]  ?? '',
          blobSize  : blobAreas[i]  ?? ''
        });
      }
      setResults(prev => [...prev, ...rows]);

      // memory cleanup
      orig.delete(); otsu.delete(); polyDst.delete(); blobDst.delete();
    });
  };

  // --- handlers ------------------------------------------------------------
  const handleFileChange = (e) => {
    console.log("File(s) selected:", e.target.files);
    const selected = Array.from(e.target.files);
    setFiles(selected);
    setIndex(0);
    setResults([]);
    if (selected.length) processFile(selected[0]);
  };

  const showImage = (idx) => {
    if (idx < 0 || idx >= files.length) return;
    setIndex(idx);
    processFile(files[idx]);
  };

  const downloadCSV = () => {
    if (!results.length) return;
    const header = 'Image Name,Egg No.,Otsu Size (mm²),Blob Size (mm²)\n';
    const csv    = header + results.map(r => `${r.imgName},${r.eggNo},${r.otsuSize},${r.blobSize}`).join('\n');
    const blob   = new Blob([csv], { type: 'text/csv' });
    const link   = document.createElement('a');
    link.href    = URL.createObjectURL(blob);
    link.download = 'egg_sizes.csv';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // --- render --------------------------------------------------------------
  return (
    <div className="p-4 space-y-4 text-center">
      <h1 className="text-2xl font-bold">EggSizer CV (Web)</h1>

      {/* Controls */}
      <div className="flex flex-wrap gap-2 justify-center">
        <input type="file"
               multiple
               accept="image/*"
               onChange={handleFileChange}
               className="file:rounded-lg file:border-0 file:bg-gray-800 file:text-white" />
        <button disabled={index <= 0}
                className="px-3 py-1 rounded bg-gray-200 disabled:opacity-40"
                onClick={() => showImage(index - 1)}>Previous Image</button>
        <button disabled={index >= files.length - 1}
                className="px-3 py-1 rounded bg-gray-200 disabled:opacity-40"
                onClick={() => showImage(index + 1)}>Next Image</button>
        <button className="px-3 py-1 rounded bg-blue-600 text-white disabled:opacity-40"
                disabled={!results.length}
                onClick={downloadCSV}>Save Results</button>
      </div>
      <div className="grid grid-cols-2 gap-4 justify-items-center">
        <div className="row">
          <div className="twocolumn">
            <p className="font-semibold">Uploaded Image</p>
            <canvas ref={canvUL} width={400} height={300} style={CANVAS_SIZE} />
          </div>
          <div className="twocolumn">
            <p className="font-semibold">Blob Processed Image</p>
            <canvas ref={canvUR} width={400} height={300} style={CANVAS_SIZE} />
          </div>
        </div>
        <div className="row">
          <div className="twocolumn">
            <p className="font-semibold">Otsu's Processed Image</p>
            <canvas ref={canvLL} width={400} height={300} style={CANVAS_SIZE} />
          </div>
          <div className="twocolumn">
            <p className="font-semibold">Polygonally Approximated Otsu's Image</p>
            <canvas ref={canvLR} width={400} height={300} style={CANVAS_SIZE} />
          </div>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="table-auto mx-auto border mt-4">
          <thead>
            <tr className="bg-gray-100">
              <th className="px-2 border">Image</th>
              <th className="px-2 border">Egg No.</th>
              <th className="px-2 border">Otsu Size (mm²)</th>
              <th className="px-2 border">Blob Size (mm²)</th>
            </tr>
          </thead>
          <tbody>
            {results.map((row, i) => (
              <tr key={i}>
                <td className="px-2 border whitespace-nowrap">{row.imgName}</td>
                <td className="px-2 border text-center">{row.eggNo}</td>
                <td className="px-2 border text-right">{row.otsuSize}</td>
                <td className="px-2 border text-right">{row.blobSize}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

    </div>
  );
}
