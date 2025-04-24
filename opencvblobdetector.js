/*

Credits to Github User janpaul123 for porting OpenCV Blob Detector to JS.

https://gist.github.com/janpaul123/8b9061d1d093ec0b36dac2230434d34a

*/


function diff(v1, v2) {
    if (v1.x !== undefined) return { x: v1.x - v2.x, y: v1.y - v2.y };
    return v1.map((value, index) => value - v2[index]);
  }
  
  function norm(vector) {
    if (vector.x !== undefined) return norm([vector.x, vector.y]);
    return Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
  }
  
  const defaultParams = {
    thresholdStep: 10,
    minThreshold: 50,
    maxThreshold: 220,
    minRepeatability: 2,
    minDistBetweenBlobs: 10,
  
    filterByColor: true,
    blobColor: 0,
  
    filterByArea: true,
    minArea: 25,
    maxArea: 5000,
  
    filterByCircularity: false,
    minCircularity: 0.8,
    maxCircularity: 1000000,
  
    filterByInertia: true,
    //minInertiaRatio: 0.6,
    minInertiaRatio: 0.1,
    maxInertiaRatio: 1000000,
  
    filterByConvexity: true,
    //minConvexity: 0.8,
    minConvexity: 0.95,
    maxConvexity: 1000000,
  
    faster: false,
  };
  
  function findBlobs(image, binaryImage, params) {
    const contours = new cv.MatVector();
    const hierarchy = new cv.Mat();
    if (params.faster) {
      cv.findContours(binaryImage, contours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);
    } else {
      cv.findContours(binaryImage, contours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_NONE);
    }
    hierarchy.delete();
  
    const centers = [];
    const objectsToDelete = [];
    for (let i = 0; i < contours.size(); i++) {
      const contour = contours.get(i);
      objectsToDelete.push(contour);
      const area = cv.contourArea(contour);
  
      if (area == 0) continue;
  
      let center, moms;
      if (params.faster) {
        const { x, y, width, height } = cv.boundingRect(contour);
        center = {
          confidence: 1,
          location: { x: x + width / 2, y: y + height / 2 },
          radius: (width + height) / 4,
        };
      } else {
        moms = cv.moments(contour);
        center = {
          confidence: 1,
          location: { x: moms.m10 / moms.m00, y: moms.m01 / moms.m00 },
        };
      }
  
      if (params.filterByArea) {
        if (area < params.minArea || area >= params.maxArea) continue;
      }
  
      if (params.filterByCircularity) {
        const perimeter = cv.arcLength(contour, true);
        const ratio = 4 * cv.CV_PI * area / (perimeter * perimeter);
        if (ratio < params.minCircularity || ratio >= params.maxCircularity) continue;
      }
  
      if (params.filterByInertia) {
        if (params.faster) {
          throw new Error('Cannot both set params.faster and params.filterByInertia');
        }
  
        const denominator = Math.sqrt(
          Math.pow(2 * moms.mu11, 2) + Math.pow(moms.mu20 - moms.mu02, 2)
        );
        let ratio;
        if (denominator > 0.01) {
          const cosmin = (moms.mu20 - moms.mu02) / denominator;
          const sinmin = 2 * moms.mu11 / denominator;
          const cosmax = -cosmin;
          const sinmax = -sinmin;
  
          const imin =
            0.5 * (moms.mu20 + moms.mu02) -
            0.5 * (moms.mu20 - moms.mu02) * cosmin -
            moms.mu11 * sinmin;
          const imax =
            0.5 * (moms.mu20 + moms.mu02) -
            0.5 * (moms.mu20 - moms.mu02) * cosmax -
            moms.mu11 * sinmax;
          ratio = imin / imax;
        } else {
          ratio = 1;
        }
  
        if (ratio < params.minInertiaRatio || ratio >= params.maxInertiaRatio) continue;
  
        center.confidence = ratio * ratio;
      }
  
      if (params.filterByConvexity) {
        const hull = new cv.Mat();
        cv.convexHull(contour, hull);
        const hullArea = cv.contourArea(hull);
        const ratio = area / hullArea;
        hull.delete();
        if (ratio < params.minConvexity || ratio >= params.maxConvexity) continue;
      }
  
      if (params.filterByColor) {
        if (
          binaryImage.ucharAt(Math.round(center.location.y), Math.round(center.location.x)) !=
          params.blobColor
        )
          continue;
      }
  
      if (!params.faster) {
        const dists = [];
        for (let pointIdx = 0; pointIdx < contour.size().height; pointIdx++) {
          const pt = contour.intPtr(pointIdx);
          dists.push(norm(diff(center.location, { x: pt[0], y: pt[1] })));
        }
        dists.sort();
        center.radius =
          (dists[Math.floor((dists.length - 1) / 2)] + dists[Math.floor(dists.length / 2)]) / 2;
      }
  
      centers.push(center);
    }
    objectsToDelete.forEach(obj => obj.delete());
    contours.delete();
    return centers;
  }
