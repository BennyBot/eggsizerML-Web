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
    let label = 0;
    for (let i = 0; i < contours.size(); ++i) {
      const cnt = contours.get(i);
      const approx = new cv.Mat();
      cv.approxPolyDP(cnt, approx, 0.005 * cv.arcLength(cnt, true), true);
      const area = cv.contourArea(approx);
      if (area >= POLY_MIN_AREA && area <= POLY_MAX_AREA) {
        label++;
        areas.push(area / PIXELS_PER_MM);
        const mv = new cv.MatVector(); mv.push_back(approx);
        cv.drawContours(dst, mv, -1, new cv.Scalar(0,150,0,255), 2);
        const m = cv.moments(cnt); const cx = m.m10/m.m00; const cy = m.m01/m.m00;
        cv.putText(dst, String(label), new cv.Point(cx, cy), cv.FONT_HERSHEY_SIMPLEX, 1, new cv.Scalar(0,150,0,255), 3);
        mv.delete();
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
    const areas = [];
    centers.forEach((c,i)=>{
      cv.circle(dst, new cv.Point(c.location.x, c.location.y), c.radius, new cv.Scalar(255,0,0,255), 2);
      cv.putText(dst,String(i+1), new cv.Point(c.location.x, c.location.y), cv.FONT_HERSHEY_SIMPLEX,1,new cv.Scalar(255,0,0,255),3);
      areas.push(Math.PI*c.radius*c.radius/PIXELS_PER_MM);
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
      const max = Math.max(otsuA.length, blobA.length);
      for(let i=0;i<max;i++){
        const o = otsuA[i]??""; const b = blobA[i]??"";
        const avg = o && b ? ((o+b)/2).toFixed(2):"";
        rows.push({img:file.name, id:i+1, otsu:o, blob:b, avg});
      }
      setResults(r=>[...r,...rows]);
      orig.delete(); otsu.delete(); polyDst.delete(); blobDst.delete();
    });
  };

  /* ------------------------------ handlers ------------------------------ */
  const handleFileChange = (e) => {
    const sel = Array.from(e.target.files);
    if (!sel.length) return;
    setFiles(sel); setIndex(0); setResults([]);
    processFile(sel[0]);
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

        {/* right column – results table */}
        <div className="overflow-x-auto h-[82vh]">
          <table className="table-auto w-full border">
            <thead className="sticky top-0 bg-gray-100">
              <tr>
                <th className="px-2 border">Image</th>
                <th className="px-2 border">Egg #</th>
                <th className="px-2 border">Otsu (mm²)</th>
                <th className="px-2 border">Blob (mm²)</th>
                <th className="px-2 border">Avg (mm²)</th>
              </tr>
            </thead>
            <tbody>
              {results.map((r,i)=>(
                <tr key={i}>
                  <td className="px-2 border whitespace-nowrap">{r.img}</td>
                  <td className="px-2 border text-center">{r.id}</td>
                  <td className="px-2 border text-right">{r.otsu}</td>
                  <td className="px-2 border text-right">{r.blob}</td>
                  <td className="px-2 border text-right">{r.avg}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
