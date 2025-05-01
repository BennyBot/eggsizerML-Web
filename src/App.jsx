import React, { useEffect, useRef, useState } from "react";

// -----------------------------------------------------------------------------
// EggSizer CV (Web) – batch‑first workflow
// • When a file set is chosen we synchronously process **every** image, build one
//   combined table, and then show the first image.
// • Table rows = all eggs across all images (Image column keeps the filename).
// • Previous / Next only changes which image is drawn, table stays global.
// • Deleting a row removes it from the master table **and** redraws whichever
//   image is on screen if the egg belonged to it.
// • Export CSV dumps the full combined table.
// -----------------------------------------------------------------------------

const BLOB_MIN_AREA = 5000;
const BLOB_MAX_AREA = 500000;
const POLY_MIN_AREA = 18000;
const POLY_MAX_AREA = 700000;
const PIXELS_PER_MM = 100;
const CANVAS_STYLE  = { width: "100%", height: "40vh", border: "1px solid #fff" };

export default function EggSizerApp() {
  /* -------------------- load OpenCV + blob detector -------------------- */
  const [cvReady, setCvReady] = useState(false);
  useEffect(() => {
    const core = document.createElement("script");
    core.src = import.meta.env.BASE_URL + "opencv.js";
    core.async = true;
    core.onload = () => {
      cv.onRuntimeInitialized = () => {
        const blob = document.createElement("script");
        blob.src = import.meta.env.BASE_URL + "opencvblobdetector.js";
        blob.async = true;
        blob.onload = () => setCvReady(true);
        document.body.appendChild(blob);
      };
    };
    document.body.appendChild(core);
  }, []);

  /* --------------------------- state / refs --------------------------- */
  const [files, setFiles] = useState([]);   // File[]
  const [idx, setIdx] = useState(0);    // which image displayed
  const [rows, setRows] = useState([]);   // ALL egg rows across images
  const [bases, setBases] = useState({});   // {index:{blob:Mat,poly:Mat}}
  const [processing, setProcessing] = useState(false);

  const canvBlob = useRef();
  const canvPoly = useRef();

  /* --------------------------- helpers -------------------------------- */
  const toGray = (src) => {
    const g = new cv.Mat();
    src.channels() > 1 ? cv.cvtColor(src, g, cv.COLOR_RGBA2GRAY) : src.copyTo(g);
    return g;
  };


  const autoCanny = (src) => {
    const gray = toGray(src);
    const blur = new cv.Mat();
    const dst = new cv.Mat();
    cv.GaussianBlur(gray, blur, new cv.Size(5, 5), 0.33);
    cv.threshold(blur, dst, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU);
    gray.delete(); blur.delete();
    return dst;
  };


  const polyApprox = (thresh, orig) => {
    const contours = new cv.MatVector();
    const hierarchy = new cv.Mat();
    
    cv.findContours(thresh, contours, hierarchy, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE);
    
    const eggs=[];
    
    for(let i=0; i < contours.size(); i++){
      const cnt = contours.get(i);
      const approx = new cv.Mat();
      cv.approxPolyDP(cnt, approx, 0.005*cv.arcLength(cnt,true), true);
      const area = cv.contourArea(approx);
      const m = cv.moments(cnt);
      // extract vertices
      if(area < POLY_MIN_AREA || area > POLY_MAX_AREA) { cnt.delete(); approx.delete(); continue; }
      
      const pts=[];
      for(let j=0;j<approx.rows;j++) { 
        const pt = approx.intPtr(j); 
        pts.push({x: pt[0], y: pt[1]}); 
      }
      //console.log(`Number of vertices: ${pts.length}`);
      eggs.push(
        { 
          center:{
            x: m.m10/m.m00,
            y: m.m01/m.m00
          }, 
          area:area/PIXELS_PER_MM, 
          polyPts:pts 
        }
      );
      
      cnt.delete(); 
      approx.delete();
    }
    contours.delete(); hierarchy.delete();
    return eggs;
  };

  const detectBlobs = (src) => {
    const gray = toGray(src);
    const bin = new cv.Mat();
    cv.threshold(gray, bin, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU);
    const centers = findBlobs(gray, bin, {
      faster:true, 
      filterByArea:true, 
      minArea:BLOB_MIN_AREA,
      maxArea:BLOB_MAX_AREA
    });

    gray.delete(); bin.delete();

    return centers.map(
      c => (
        { 
          center: {
            x: c.location.x, 
            y:c.location.y
          }, 
          radius: c.radius, 
          area: Math.PI * c.radius * c.radius / PIXELS_PER_MM 
        }
      )
    );
  };

  /* overlays */
  const drawBlobOverlay=(mat,list)=>{
    list.forEach(e => {
      cv.circle(mat, new cv.Point(e.blobcenter.x, e.blobcenter.y), e.radius || 30, new cv.Scalar(255,0,0,255), 2);
      cv.putText(mat, String(e.id), new cv.Point(e.blobcenter.x,e.blobcenter.y), cv.FONT_HERSHEY_SIMPLEX, 1, new cv.Scalar(255,0,0,255), 3);
    });
  };


  const drawPolyOverlay=(mat,list)=>{
    list.forEach(e=>{
      if(e.polyPts){
        for(let i=0;i<e.polyPts.length;i++){
          const p1=e.polyPts[i]; const p2=e.polyPts[(i+1)%e.polyPts.length];
          cv.line(mat,new cv.Point(p1.x,p1.y),new cv.Point(p2.x,p2.y),new cv.Scalar(0,150,0,255),2);
        }
      }
      cv.putText(
        mat,
        String(e.id),
        new cv.Point(e.polycenter.x, e.polycenter.y),
        cv.FONT_HERSHEY_SIMPLEX,
        1,
        new cv.Scalar(0,150,0,255),
        3);
    });
  };


  /* ------------------ batch process all files -------------------------- */
  const processAll = async (fileList) => {
    setProcessing(true); 
    const newRows = []; 
    const newBases = {};

    for(let fi=0; fi<fileList.length; fi++) {
      const file = fileList[fi];
      await new Promise( (res) => {
        const img = new Image();

        img.onload = () => {
          const orig = cv.imread(img);
          const otsu = autoCanny(orig);
          const polyList = polyApprox(otsu,orig);
          const blobList = detectBlobs(orig);
          const len = Math.max(polyList.length,blobList.length);
          const combined = [];
          for(let i=0; i<len; i++){
            const p = polyList[i];
            const b = blobList[i];
            const id=i+1;
            const areaO = p?.area||"NONE";
            const areaB = b?.area.toFixed(2)||"NONE";
            const avg = areaO && areaB ? ((areaO+Number(areaB))/2).toFixed(2) : (areaO ? areaO : areaB);
            const row = {
              key: `${file.name}-${id}`,
              img: file.name,
              id,
              otsu: areaO, 
              blob: areaB, 
              polycenter: p?.center ?? {x:0, y:0},
              blobcenter: b?.center ?? {x:0, y:0},
              avg,
              radius: b?.radius || 30,
              imgIdx: fi,
              polyPts: p?.polyPts || [],
            };
            newRows.push(row);
            combined.push(row);
          }
          const blobMat = orig.clone();
          const polyMat = orig.clone();
          drawBlobOverlay(blobMat, combined); 
          drawPolyOverlay(polyMat, combined);
          newBases[fi] = {
            blob:blobMat, poly:polyMat, orig:orig
          };
          otsu.delete(); res();
        }; 
        img.src=URL.createObjectURL(file);
      });
    }
    setRows(newRows);
    setBases(newBases); 
    setProcessing(false);
  };

  /* ------------------ handle file selection --------------------------- */
  const onFileChange=(e)=>{const sel=Array.from(e.target.files); if(!sel.length) return; setFiles(sel); setIdx(0); setRows([]); setBases({}); processAll(sel);};

  /* ------------------ redraw current image ---------------------------- */
  useEffect(()=>{if(!(idx in bases)) return; cv.imshow(canvBlob.current,bases[idx].blob); cv.imshow(canvPoly.current,bases[idx].poly);},[idx,bases]);

  /* ------------------ deletion update --------------------------- */
  const removeRow=(key)=>{
    console.log(`Removing row ${key}`);
    // remove from rows
    const newRows = rows.filter(r => r.key !== key);
    setRows(newRows);

    // now, we need to update the bases using base.orig, and then redrawing the blob and poly without the deleted egg
    const base = bases[idx];
    if(!base) return;

    const orig = base.orig.clone();
    
    // we shouldn't need to redo the poly and blob detection, but we do need to remove the egg from the image
    // the row already contains the center and radius for blob, and the center and polyPts for poly
    const hit = newRows.find(r => r.key === key);
    if(!hit) {
      console.log("no hit");
      return;
    }

    bases[idx].blob.delete();
    bases[idx].poly.delete();

    bases[idx].blob = orig.clone();
    bases[idx].poly = orig.clone();

    const blobList = newRows.filter(r => r.imgIdx === idx).map(r => ({
      center: {x:r.blobcenter.x, y:r.blobcenter.y},
      radius: r.radius,
    }));
    const polyList = newRows.filter(r => r.imgIdx === idx).map(r => ({
      center: {x:r.polycenter.x, y:r.polycenter.y},
      polyPts: r.polyPts,
    }));

    drawBlobOverlay(bases[idx].blob, blobList);
    drawPolyOverlay(bases[idx].poly, polyList);

    cv.imshow(canvBlob.current, bases[idx].blob);
    cv.imshow(canvPoly.current, bases[idx].poly);
    // update the bases
    const newBases = {...bases};
    newBases[idx] = {
      blob: bases[idx].blob,
      poly: bases[idx].poly,
      orig: orig,
    };

    setBases(newBases);
  };


  const clickBlobCanvas = (e) => {
    const r = canvBlob.current.getBoundingClientRect();
    const x = e.clientX - r.left;
    const y = e.clientY - r.top;
    console.log(`Blob click at (${x},${y})`);
    const hit = rows.find(
      row => row.imgIdx === idx && ((x-row.blobcenter.x)**2 + (y-row.blobcenter.y)**2 <= row.radius**2)
    );
    console.log(`Hit: ${hit ?? "none"}`);
    if(hit) removeRow(hit.key);
  };

  const clickPolyCanvas = (e) => {
    const r = canvPoly.current.getBoundingClientRect();
    const x = e.clientX - r.left;
    const y = e.clientY - r.top;
    console.log(`Poly click at (${x},${y})`);
    const hit = rows.find(
      row => row.imgIdx === idx && ((x-row.polycenter.x)**2 + (y-row.polycenter.y)**2 <= row.radius**2)
    );
    console.log(`Hit: ${hit ?? "none"}`);
    if(hit) removeRow(hit.key);
  }
  /* ------------------ CSV export ---------------------- */
  const exportCSV=()=>{if(!rows.length) return;const header="Image,Egg,Otsu(mm²),Blob(mm²),Avg(mm²)\n";const body=rows.map(r=>`${r.img},${r.id},${r.otsu},${r.blob},${r.avg}`).join("\n");const b=new Blob([header+body],{type:"text/csv"});const a=document.createElement("a");a.href=URL.createObjectURL(b);a.download="egg_sizes.csv";a.click();};

  /* ------------------ UI -------------------------------- */
  return (
    <div className="p-4">
      <h1 className="text-2xl font-bold mb-4 text-center">EggSizerML (Web)</h1>
      <div className="flex flex-wrap gap-2 justify-center mb-4">
        <input type="file" multiple accept="image/*" disabled={!cvReady||processing} onChange={onFileChange}
               className="file:rounded-lg file:border-0 file:bg-gray-800 file:text-white disabled:opacity-40" />
        <button onClick={()=>setIdx(i=>Math.max(0,i-1))} disabled={idx<=0 || processing} className="px-3 py-1 bg-gray-200 rounded disabled:opacity-40">Previous Image</button>
        <button onClick={()=>setIdx(i=>Math.min(files.length-1,i+1))} disabled={idx>=files.length-1 || processing} className="px-3 py-1 bg-gray-200 rounded disabled:opacity-40">Next Image</button>
        <button onClick={exportCSV} disabled={!rows.length} className="px-3 py-1 bg-blue-600 text-white rounded disabled:opacity-40">Export CSV</button>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-4">
          <div><p className="font-semibold mb-1">Blob Processed (click egg to remove)</p><canvas ref={canvBlob} style={CANVAS_STYLE} onClick={clickBlobCanvas}/></div>
          <div><p className="font-semibold mb-1">Polygon Approx</p><canvas ref={canvPoly} style={CANVAS_STYLE} onClick={clickPolyCanvas}/></div>
        </div>
        <div className="overflow-x-auto h-[82vh] text-sm">
          <table className="table-auto w-full border"><thead className="sticky top-0 bg-gray-100"><tr><th className="px-2 border">Image</th><th className="px-2 border">Egg #</th><th className="px-2 border">Otsu</th><th className="px-2 border">Blob</th><th className="px-2 border">Avg</th><th className="px-2 border"></th></tr></thead><tbody>
            {rows.map(r=>(<tr key={r.key}><td className="px-2 border whitespace-nowrap">{r.img}</td><td className="px-2 border text-center">{r.id}</td><td className="px-2 border text-right">{r.otsu}</td><td className="px-2 border text-right">{r.blob}</td><td className="px-2 border text-right">{r.avg}</td><td className="px-2 border text-center"><button className="text-red-600" onClick={()=>removeRow(r.key)}>✖</button></td></tr>))}
          </tbody></table>
        </div>
      </div>
      {!cvReady && <p className="text-center text-red-600 mt-4">Loading OpenCV …</p>}
      {processing && <p className="text-center mt-2">Processing {files.length} images …</p>}
    </div>
  );
}
