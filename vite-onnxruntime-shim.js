// Shim for onnxruntime-web
// This allows @xenova/transformers to import onnxruntime-web
// while we load it from CDN in index.html

// Access the global 'ort' object loaded from CDN
// Use a getter to ensure we get the latest reference
const getOrt = () => {
  if (typeof window !== 'undefined' && window.ort) {
    return window.ort;
  }
  // Return a proxy that will forward to ort once it's available
  return new Proxy({}, {
    get(target, prop) {
      if (typeof window !== 'undefined' && window.ort) {
        return window.ort[prop];
      }
      return undefined;
    }
  });
};

export default getOrt();

