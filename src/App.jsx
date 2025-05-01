import React, { useEffect, useRef, useState } from "react";

// -------------------------------------------------------------------------
// EggSizer CV (Web) – simplified 2‑panel layout + working results table
// -------------------------------------------------------------------------
// ‑ Shows only the **Blob** image (upper) and the **Polygon‑approx** image
//   (lower) on the left; the table lives on the right.
// ‑ Fixes the results table (Avg column, csv header, let declarations).
// ‑ Keeps dynamic loading of opencv.js and opencvblobdetector.js.
// -------------------------------------------------------------------------

const BLOB_MIN_AREA = 2000;
const BLOB_MAX_AREA = 1_000_000;
const POLY_MIN_AREA = 18_000;
const POLY_MAX_AREA = 700_000;
const PIXELS_PER_MM = 100;

const CANVAS_STYLE = { width: "45vw", height: "40vh", border: "1px solid #fff" };

export default function EggSizerApp() {
  /* -------------------- load OpenCV & blob detector once ---------------- */
  useEffect(() => {
    const core = document.createElement("script");
    core.src   = import.meta.env.BASE_URL + "opencv.js";
    core.async = true;
    core.onload = () => {
      cv.onRuntimeInitialized = () => {
        const blob = document.createElement("script");
        blob.src   = import.meta.env.BASE_URL + "opencvblobdetector.js";
        blob.async = true;
        document.body.appendChild(blob);
      };
    };
    document.body.appendChild(core);
  }, []);

  /* ------------------------------ state / refs -------------------------- */
  const [files,   setFiles]   = useState([]);
  const [index,   setIndex]   = useState(0);
  const [results, setResults] = useState([]);

  const canvBlob = useRef(); // blob‑processed image   (upper‑left)
  const canvPoly = useRef(); // polygon‑approx image   (lower‑left)

  /* ------------------------------ helpers ------------------------------- */
  const readImageToMat = (file, cb) => {
    const img = new Image();
    img.onload = () => cb(cv.imread(img));
    img.src = URL.createObjectURL(file);
  };

  const toGray = (src) => {
    const g = new cv.Mat();
    src.channels() > 1 ? cv.cvtColor(src, g, cv.COLOR_RGBA2GRAY) : src.copyTo(g);
    return g;
  };

  const autoCanny = (src) => {
    const gray = toGray(src);
    const blur = new cv.Mat();
    const dst  = new cv.Mat();
    cv.GaussianBlur(gray, blur, new cv.Size(5, 5), 0.33);
    cv.threshold(blur, dst, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU);
    gray.delete(); blur.delete();
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
    const gray = toGray(src);
    const bin  = new cv.Mat();
    cv.threshold(gray, bin, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU);
    const params = { faster:true, filterByArea:true, minArea:BLOB_MIN_AREA, maxArea:BLOB_MAX_AREA };
    const centers = findBlobs(gray, bin, params);
    const dst = src.clone();
    let areas = [];
    const thresh = new cv.Mat();
    cv.threshold(gray, thresh, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU);

    let centers = findBlobs(gray, thresh, params);

    centers.forEach((c) => {
      cv.circle(dst, new cv.Point(c.location.x, c.location.y), c.radius, new cv.Scalar(255, 0, 0, 255), 2);
      areas.push(c.radius * c.radius * Math.PI / PIXELS_PER_MM);
    });
    gray.delete(); bin.delete();
    return { dst, areas };
  };

  const processFile = (file) => {
    readImageToMat(file, (orig) => {
      const otsu = autoCanny(orig);
      const { dst: polyDst, areas: otsuA } = polyApprox(otsu, orig);
      const { dst: blobDst, areas: blobA } = detectBlobs(orig);
      cv.imshow(canvBlob.current, blobDst);
      cv.imshow(canvPoly.current, polyDst);

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

      setResults((prev) => [...prev, ...rows]);


      
      // memory cleanup
      orig.delete(); otsu.delete(); polyDst.delete(); blobDst.delete();
    });
  };

  /* ------------------------------ handlers ------------------------------ */
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

  /* ------------------------------ render -------------------------------- */
  return (
    <div className="p-4">
      <h1 className="text-2xl font-bold mb-4 text-center">EggSizer CV (Web)</h1>
      <div className="flex gap-2 justify-center mb-4">
        <input type="file" multiple accept="image/*" onChange={handleFileChange}
               className="file:rounded-lg file:border-0 file:bg-gray-800 file:text-white" />
      </div>

      {/* two‑column layout */}
      <div className="grid grid-cols-2 gap-4">
        {/* left column – the two canvases stacked */}
        <div className="space-y-4">
          <div>
            <p className="font-semibold mb-1">Blob Processed</p>
            <canvas ref={canvBlob} style={CANVAS_STYLE} />
          </div>
          <div>
            <p className="font-semibold mb-1">Polygon Approx</p>
            <canvas ref={canvPoly} style={CANVAS_STYLE} />
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
