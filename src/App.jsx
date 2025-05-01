import React, { useEffect, useRef, useState } from "react";

// -------------------------------------------------------------------------
// EggSizer CV (Web)
// • 2‑panel UI (blob + polygon) with Previous / Next navigation.
// • Interactive deletion from table **and** canvases.
// • **Export CSV** button downloads the current table.
// • CV scripts load async; UI stays disabled until both are ready.
// -------------------------------------------------------------------------

const BLOB_MIN_AREA = 2000;
const BLOB_MAX_AREA = 1_000_000;
const POLY_MIN_AREA = 18_000;
const POLY_MAX_AREA = 700_000;
const PIXELS_PER_MM = 100;
const CANVAS_STYLE  = { width: "45vw", height: "40vh", border: "1px solid #fff" };

export default function EggSizerApp() {
  /* -------------------- load OpenCV + blob detector -------------------- */
  const [cvReady, setCvReady] = useState(false);
  useEffect(() => {
    const core = document.createElement("script");
    core.src   = import.meta.env.BASE_URL + "opencv.js";
    core.async = true;
    core.onload = () => {
      cv.onRuntimeInitialized = () => {
        const blob = document.createElement("script");
        blob.src   = import.meta.env.BASE_URL + "opencvblobdetector.js";
        blob.async = true;
        blob.onload = () => setCvReady(true);
        document.body.appendChild(blob);
      };
    };
    document.body.appendChild(core);
  }, []);

  /* --------------------------- state / refs --------------------------- */
  const [files, setFiles] = useState([]);
  const [idx,   setIdx]   = useState(0);
  const [rows,  setRows]  = useState([]);   // table for current image

  const canvBlob = useRef();
  const canvPoly = useRef();
  const baseBlob = useRef(null);
  const basePoly = useRef(null);

  /* --------------------------- helpers -------------------------------- */
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
    const eggs = [];
    for (let i = 0; i < contours.size(); ++i) {
      const cnt = contours.get(i);
      const approx = new cv.Mat();
      cv.approxPolyDP(cnt, approx, 0.005 * cv.arcLength(cnt, true), true);
      const area = cv.contourArea(approx);
      if (area >= POLY_MIN_AREA && area <= POLY_MAX_AREA) {
        const m = cv.moments(cnt);
        eggs.push({ center:{x:m.m10/m.m00,y:m.m01/m.m00}, area: area/PIXELS_PER_MM });
      }
      cnt.delete(); approx.delete();
    }
    contours.delete(); hierarchy.delete();
    return eggs;
  };

  const detectBlobs = (src) => {
    const gray = toGray(src);
    const bin  = new cv.Mat();
    cv.threshold(gray, bin, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU);
    const params = { faster:true, filterByArea:true, minArea:BLOB_MIN_AREA, maxArea:BLOB_MAX_AREA };
    const centers = findBlobs(gray, bin, params);
    gray.delete(); bin.delete();
    return centers.map((c)=>({ center:{x:c.location.x,y:c.location.y}, radius:c.radius, area: Math.PI*c.radius*c.radius/PIXELS_PER_MM }));
  };

  /* ------------------ overlay helpers ------------------ */
  const drawBlobOverlay = (mat, list) => {
    list.forEach((e)=>{
      cv.circle(mat,new cv.Point(e.center.x,e.center.y),e.radius||30,new cv.Scalar(255,0,0,255),2);
      cv.putText(mat,String(e.id),new cv.Point(e.center.x,e.center.y),cv.FONT_HERSHEY_SIMPLEX,1,new cv.Scalar(255,0,0,255),3);
    });
  };
  const drawPolyOverlay = (mat,list)=>{
    list.forEach((e)=>{
      cv.putText(mat,String(e.id),new cv.Point(e.center.x,e.center.y),cv.FONT_HERSHEY_SIMPLEX,1,new cv.Scalar(0,150,0,255),3);
    });
  };

  /* ------------------ main processing ------------------ */
  const processCurrent = () => {
    if(!cvReady||!files.length) return;
    const file = files[idx];
    const img = new Image();
    img.onload = () => {
      const orig = cv.imread(img);
      const otsu = autoCanny(orig);
      const polyList = polyApprox(otsu, orig);
      const blobList = detectBlobs(orig);

      const len = Math.max(polyList.length, blobList.length);
      const merged=[];
      for(let i=0;i<len;i++){
        const p=polyList[i]; const b=blobList[i];
        const id=i+1;
        const areaO=p?.area||"";
        const areaB=b?.area.toFixed(2)||"";
        const avg = areaO&&areaB?((areaO+Number(areaB))/2).toFixed(2):"";
        merged.push({key:`${file.name}-${id}`,img:file.name,id,otsu:areaO,blob:areaB,avg,center:(b?.center)||(p?.center),radius:b?.radius||30});
      }

      const blobMat = orig.clone();
      const polyMat = orig.clone();
      drawBlobOverlay(blobMat, merged);
      drawPolyOverlay(polyMat, merged);
      cv.imshow(canvBlob.current, blobMat);
      cv.imshow(canvPoly.current, polyMat);
      baseBlob.current?.delete();
      basePoly.current?.delete();
      baseBlob.current = blobMat;
      basePoly.current = polyMat;
      setRows(merged);
      orig.delete(); otsu.delete();
    };
    img.src = URL.createObjectURL(file);
  };
  useEffect(processCurrent,[cvReady,files,idx]);

  /* ------------------ deletion + redraw ---------------- */
  const redraw = (list)=>{
    if(!baseBlob.current||!basePoly.current) return;
    const b=baseBlob.current.clone();
    const p=basePoly.current.clone();
    drawBlobOverlay(b,list); drawPolyOverlay(p,list);
    cv.imshow(canvBlob.current,b); cv.imshow(canvPoly.current,p);
    b.delete(); p.delete();
  };
  const removeRow=(key)=>{const f=rows.filter(r=>r.key!==key);setRows(f);redraw(f);}  ;
  const clickCanvas=(e)=>{
    const r=canvBlob.current.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top;
    const hit=rows.find(row=>{const dx=x-row.center.x,dy=y-row.center.y;return dx*dx+dy*dy<=row.radius*row.radius});
    if(hit) removeRow(hit.key);
  };

  /* ------------------ CSV export ---------------------- */
  const exportCSV=()=>{
    if(!rows.length) return;
    const header="Image,Egg,Otsu(mm²),Blob(mm²),Avg(mm²)\n";
    const body = rows.map(r=>`${r.img},${r.id},${r.otsu},${r.blob},${r.avg}`).join("\n");
    const blob = new Blob([header+body],{type:"text/csv"});
    const link=document.createElement("a");
    link.href=URL.createObjectURL(blob); link.download="egg_sizes.csv";
    link.click();
  };

  /* ------------------ UI -------------------------------- */
  return (
    <div className="p-4">
      <h1 className="text-2xl font-bold mb-4 text-center">EggSizer CV (Web)</h1>
      <div className="flex flex-wrap gap-2 justify-center mb-4">
        <input type="file" multiple accept="image/*" disabled={!cvReady} onChange={(e)=>{setFiles(Array.from(e.target.files));setIdx(0);}}
               className="file:rounded-lg file:border-0 file:bg-gray-800 file:text-white disabled:opacity-40" />
        <button onClick={()=>setIdx(i=>Math.max(0,i-1))} disabled={idx<=0} className="px-3 py-1 bg-gray-200 rounded disabled:opacity-40">Previous Image</button>
        <button onClick={()=>setIdx(i=>Math.min(files.length-1,i+1))} disabled={idx>=files.length-1} className="px-3 py-1 bg-gray-200 rounded disabled:opacity-40">Next Image</button>
        <button onClick={exportCSV} disabled={!rows.length} className="px-3 py-1 bg-blue-600 text-white rounded disabled:opacity-40">Export CSV</button>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-4">
          <div>
            <p className="font-semibold mb-1">Blob Processed (click egg to remove)</p>
            <canvas ref={canvBlob} style={CANVAS_STYLE} onClick={clickCanvas} />
          </div>
          <div>
            <p className="font-semibold mb-1">Polygon Approx</p>
            <canvas ref={canvPoly} style={CANVAS_STYLE} />
          </div>
        </div>
        <div className="overflow-x-auto h-[82vh]">
          <table className="table-auto w-full border text-sm">
            <thead className="sticky top-0 bg-gray-100">
              <tr><th className="px-2 border">Image</th><th className="px-2 border">Egg #</th><th className="px-2 border">Otsu</th><th className="px-2 border">Blob</th><th className="px-2 border">Avg</th><th className="px-2 border"></th></tr>
            </thead>
            <tbody>
              {rows.map(r=>(
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
      {!cvReady && <p className="text-center text-red-600 mt-4">Loading OpenCV …</p>}
    </div>
  );
}
