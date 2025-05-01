import React, { useEffect, useRef, useState } from "react";

// -------------------------------------------------------------------------
// EggSizer CV (Web)
// • 2‑panel UI, prev/next navigation, interactive deletion.
// • When a row is deleted (via ✖ button or canvas click) the overlays on both
//   canvases are redrawn to match the current table.
// -------------------------------------------------------------------------

const BLOB_MIN_AREA = 2000;
const BLOB_MAX_AREA = 1_000_000;
const POLY_MIN_AREA = 18_000;
const POLY_MAX_AREA = 700_000;
const PIXELS_PER_MM = 100;
const CANVAS_STYLE = { width: "45vw", height: "40vh", border: "1px solid #fff" };

export default function EggSizerApp() {
  /* -------------------- dynamic OpenCV + blob detector ------------------ */
  useEffect(() => {
    const core = document.createElement("script");
    core.src = import.meta.env.BASE_URL + "opencv.js";
    core.async = true;
    core.onload = () => {
      cv.onRuntimeInitialized = () => {
        const blob = document.createElement("script");
        blob.src = import.meta.env.BASE_URL + "opencvblobdetector.js";
        blob.async = true;
        document.body.appendChild(blob);
      };
    };
    document.body.appendChild(core);
  }, []);

  /* --------------------------- state / refs ----------------------------- */
  const [files, setFiles]   = useState([]);
  const [idx,   setIdx]     = useState(0);
  const [rows,  setRows]    = useState([]);  // table rows

  const canvBlob = useRef();
  const canvPoly = useRef();

  // base images without overlays – so we can redraw after deletion
  const baseBlob = useRef(null);
  const basePoly = useRef(null);

  /* --------------------------- helpers ---------------------------------- */
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
    const eggs = [];
    let label = 0;
    for (let i = 0; i < contours.size(); ++i) {
      const cnt = contours.get(i);
      const approx = new cv.Mat();
      cv.approxPolyDP(cnt, approx, 0.005 * cv.arcLength(cnt, true), true);
      const area = cv.contourArea(approx);
      if (area >= POLY_MIN_AREA && area <= POLY_MAX_AREA) {
        label++;
        const m = cv.moments(cnt); const cx = m.m10/m.m00; const cy = m.m01/m.m00;
        eggs.push({ idx: label, center:{x:cx,y:cy}, area: area/PIXELS_PER_MM });
      }
      cnt.delete(); approx.delete();
    }
    contours.delete(); hierarchy.delete();
    return { dst, eggs };
  };

  const detectBlobs = (src) => {
    const gray = toGray(src);
    const bin  = new cv.Mat();
    cv.threshold(gray, bin, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU);
    const params = { faster:true, filterByArea:true, minArea:BLOB_MIN_AREA, maxArea:BLOB_MAX_AREA };
    const centers = findBlobs(gray, bin, params);
    const eggs = centers.map((c,i)=>({ idx:i+1, center:{x:c.location.x,y:c.location.y}, radius:c.radius, area: Math.PI*c.radius*c.radius/PIXELS_PER_MM }));
    gray.delete(); bin.delete();
    return eggs;
  };

  /* ------------------ draw helpers (overlay circles/labels) -------------- */
  const drawBlobOverlay = (dstMat, eggList) => {
    eggList.forEach(e=>{
      cv.circle(dstMat, new cv.Point(e.center.x, e.center.y), e.radius, new cv.Scalar(255,0,0,255), 2);
      cv.putText(dstMat, String(e.idx), new cv.Point(e.center.x, e.center.y), cv.FONT_HERSHEY_SIMPLEX, 1, new cv.Scalar(255,0,0,255), 3);
    });
  };
  const drawPolyOverlay = (dstMat, eggList) => {
    eggList.forEach(e=>{
      cv.putText(dstMat, String(e.idx), new cv.Point(e.center.x, e.center.y), cv.FONT_HERSHEY_SIMPLEX, 1, new cv.Scalar(0,150,0,255), 3);
    });
  };

  /* ------------------ process current image ----------------------------- */
  const processImage = () => {
    if (!files.length) return;
    const file = files[idx];
    const img = new Image();
    img.onload = () => {
      const orig = cv.imread(img);

      // polygon data
      const otsu = autoCanny(orig);
      const { dst: polyBase, eggs: polyEggs } = polyApprox(otsu, orig);

      // blob data
      const blobEggs = detectBlobs(orig);
      const blobBase = orig.clone();

      // merge by index
      const len = Math.max(polyEggs.length, blobEggs.length);
      const merged = [];
      for(let i=0;i<len;i++){
        const p = polyEggs[i]; const b = blobEggs[i];
        if(!p && !b) continue;
        const areaO = p?.area ?? "";
        const areaB = b?.area.toFixed(2) ?? "";
        const avg = areaO && areaB ? ((areaO+Number(areaB))/2).toFixed(2):"";
        merged.push({
          key:`${file.name}-${i+1}`,
          img:file.name,
          id:i+1,
          otsu:areaO,
          blob:areaB,
          avg,
          center:b?.center || p?.center,
          radius:b?.radius || 30 // default radius for hit test
        });
      }
      // draw overlays
      drawBlobOverlay(blobBase, merged);
      drawPolyOverlay(polyBase, merged);

      cv.imshow(canvBlob.current, blobBase);
      cv.imshow(canvPoly.current, polyBase);

      // keep bases for later redraw
      baseBlob.current && baseBlob.current.delete();
      basePoly.current && basePoly.current.delete();
      baseBlob.current = blobBase;
      basePoly.current = polyBase;

      setRows(merged);

      orig.delete(); otsu.delete();
    };
    img.src = URL.createObjectURL(file);
  };

  useEffect(processImage, [files, idx]);

  /* ------------------ deletion ----------------------------- */
  const redrawCanvases = (list) => {
    if(!baseBlob.current || !basePoly.current) return;
    const b = baseBlob.current.clone();
    const p = basePoly.current.clone();
    drawBlobOverlay(b, list);
    drawPolyOverlay(p, list);
    cv.imshow(canvBlob.current, b);
    cv.imshow(canvPoly.current, p);
    b.delete(); p.delete();
  };

  const removeRow = (key) => {
    const filtered = rows.filter(r=>r.key!==key);
    setRows(filtered);
    redrawCanvases(filtered);
  };

  const handleCanvasClick = (e) => {
    const rect = e.target.getBoundingClientRect();
    const x = e.clientX-rect.left, y = e.clientY-rect.top;
    const hit = rows.find(r=>{const dx=x-r.center.x,dy=y-r.center.y;return dx*dx+dy*dy<=r.radius*r.radius;});
    if(hit) removeRow(hit.key);
  };

  /* ------------------------------ render -------------------------------- */
  return (
    <div className="p-4">
      <h1 className="text-2xl font-bold mb-4 text-center">EggSizer CV (Web)</h1>
      <div className="flex gap-2 justify-center mb-4">
        <input type="file" multiple accept="image/*" onChange={(e)=>{setFiles(Array.from(e.target.files));setIdx(0);}}
               className="file:rounded-lg file:border-0 file:bg-gray-800 file:text-white" />
        <button onClick={()=>setIdx(i=>Math.max(0,i-1))} disabled={idx<=0} className="px-3 py-1 bg-gray-200 rounded disabled:opacity-40">Previous Image</button>
        <button onClick={()=>setIdx(i=>Math.min(files.length-1,i+1))} disabled={idx>=files.length-1} className="px-3 py-1 bg-gray-200 rounded disabled:opacity-40">Next Image</button>
      </div>

      <div className="grid grid-cols-2 gap-4">
        {/* left canvases */}
        <div className="space-y-4">
          <div>
            <p className="font-semibold mb-1">Blob Processed (click egg to remove)</p>
            <canvas ref={canvBlob} style={CANVAS_STYLE} onClick={handleCanvasClick} />
          </div>
          <div>
            <p className="font-semibold mb-1">Polygon Approx</p>
            <canvas ref={canvPoly} style={CANVAS_STYLE} />
          </div>
        </div>
        {/* right table */}
        <div className="overflow-x-auto h-[82vh]">
          <table className="table-auto w-full border text-sm">
            <thead className="sticky top-0 bg-gray-100">
              <tr><th className="px-2 border">Image</th><th className="px-2 border">Egg #</th><th className="px-2 border">Otsu (mm²)</th><th className="px-2 border">Blob (mm²)</th><th className="px-2 border">Avg (mm²)</th><th className="px-2 border"></th></tr>
            </thead>
            <tbody>
              {results.map(r=>(
                <tr key={r.key}>
                  <td className="px-2 border whitespace-nowrap">{r.img}</td>
                  <td className="px-2 border text-center">{r.id}</td>
                  <td className="px-2 border text-right">{r.otsu}</td>
                  <td className="px-2 border text-right">{r.blob}</td>
                  <td className="px-2 border text-right">{r.avg}</td>
                  <td className="px-2 border text-center"><button className="text-red-600" onClick={()=>removeRow(r.key)}>✖</button></td>
                </tr>))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
