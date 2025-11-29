import path from 'path';
import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import type { Plugin } from 'vite';

// Removed onnxRuntimeShimPlugin - now using installed package directly

/**
 * Plugin to fix Content-Type headers for proxied JSON responses
 * Hugging Face CDN returns text/plain for JSON files, which breaks transformers.js
 * This middleware intercepts responses AFTER the proxy and fixes the Content-Type
 * 
 * IMPORTANT: This runs AFTER the proxy middleware, so it can override headers
 * that the proxy sets from Hugging Face's response.
 */
const contentTypeFixPlugin = (): Plugin => {
  return {
    name: 'content-type-fix',
    configureServer(server) {
      // CRITICAL: This middleware MUST run BEFORE Vite's SPA fallback
      // to ensure /Xenova/ requests are handled by the proxy, not served as index.html
      server.middlewares.use((req, res, next) => {
        const url = req.url || '';
        
        // If this is a /Xenova/ request, ensure it goes through the proxy
        // Don't let Vite's SPA fallback serve index.html for these requests
        if (url.startsWith('/Xenova/')) {
          // Let the proxy handle it - don't serve index.html
          return next();
        }
        
        // Only fix Content-Type for proxied /Xenova/ JSON responses
        if (url.includes('/Xenova/') && url.endsWith('.json')) {
          // Store original methods
          const originalSetHeader = res.setHeader.bind(res);
          const originalWriteHead = res.writeHead.bind(res);
          const originalWrite = res.write.bind(res);
          const originalEnd = res.end.bind(res);
          
          let headersFixed = false;
          
          // Intercept setHeader calls
          res.setHeader = function(name: string, value: any) {
            if (name.toLowerCase() === 'content-type' && typeof value === 'string' && value.includes('text/plain')) {
              headersFixed = true;
              return originalSetHeader('content-type', 'application/json; charset=utf-8');
            }
            return originalSetHeader(name, value);
          };
          
          // Intercept writeHead
          res.writeHead = function(statusCode: number, statusMessage?: any, headers?: any) {
            if (headers && typeof headers === 'object') {
              const ct = headers['content-type'] || headers['Content-Type'];
              if (ct && typeof ct === 'string' && ct.includes('text/plain')) {
                headersFixed = true;
                headers['content-type'] = 'application/json; charset=utf-8';
                delete headers['Content-Type'];
              }
            }
            return originalWriteHead(statusCode, statusMessage, headers);
          };
          
          // Intercept write/end to fix headers if they weren't fixed yet
          res.write = function(chunk: any, encoding?: any) {
            if (!headersFixed && !res.headersSent) {
              const ct = res.getHeader('content-type');
              if (ct && typeof ct === 'string' && ct.includes('text/plain')) {
                headersFixed = true;
                res.setHeader('content-type', 'application/json; charset=utf-8');
              }
            }
            return originalWrite(chunk, encoding);
          };
          
          res.end = function(chunk?: any, encoding?: any) {
            if (!headersFixed && !res.headersSent) {
              const ct = res.getHeader('content-type');
              if (ct && typeof ct === 'string' && ct.includes('text/plain')) {
                headersFixed = true;
                res.setHeader('content-type', 'application/json; charset=utf-8');
              }
            }
            return originalEnd(chunk, encoding);
          };
        }
        
        next();
      });
    },
  };
};

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, '.', '');
  
  return {
    server: {
      port: 3001,
      host: '0.0.0.0',
      proxy: {
        /**
         * Proxy all model requests to Hugging Face CDN
         * 
         * How it works:
         * - transformers.js with remoteHost=localhost:3001 creates URLs like:
         *   http://localhost:3001/Xenova/whisper-tiny.en/resolve/main/tokenizer.json
         *   http://localhost:3001/onnx-community/whisper-tiny.en/resolve/main/tokenizer.json
         * - This proxy intercepts model paths and forwards them to Hugging Face
         * - Hugging Face CDN returns text/plain for JSON files, so we fix Content-Type
         */
        '^/(Xenova|onnx-community)/': {
          target: 'https://huggingface.co',
          changeOrigin: true,
          secure: true,
          followRedirects: true,
          ws: false, // Don't proxy WebSocket connections
          configure: (proxy, _options) => {
            proxy.on('proxyReq', (proxyReq, req, _res) => {
              try {
                // Remove cache headers to force fresh fetch
                proxyReq.removeHeader('if-none-match');
                proxyReq.removeHeader('if-modified-since');
                proxyReq.setHeader('cache-control', 'no-cache');
              } catch (e) {
                // Ignore header errors
              }
            });
            
            proxy.on('proxyRes', (proxyRes, req, res) => {
              try {
                // Add CORS headers
                if (res && !res.headersSent) {
                  res.setHeader('Access-Control-Allow-Origin', '*');
                  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
                  res.setHeader('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept');
                }
                
                // Fix Content-Type for .json files
                if (req.url && req.url.endsWith('.json') && !res.headersSent) {
                  res.setHeader('Content-Type', 'application/json; charset=utf-8');
                }
              } catch (e) {
                // Ignore header errors - don't crash the server
              }
            });
            
            proxy.on('error', (err, req, res) => {
              try {
                console.error('Proxy error:', err.message, 'URL:', req.url);
                if (res && !res.headersSent) {
                  res.statusCode = 500;
                  res.setHeader('Content-Type', 'application/json');
                  res.end(JSON.stringify({ error: 'Proxy error: ' + err.message }));
                }
              } catch (e) {
                // Don't crash if error handling fails
                console.error('Error handling proxy error:', e);
              }
            });
          },
        },
        '^/api/resolve-cache': {
          target: 'https://huggingface.co',
          changeOrigin: true,
          secure: true,
          followRedirects: true,
          ws: false, // Don't proxy WebSocket connections
          configure: (proxy, _options) => {
            proxy.on('proxyRes', (proxyRes, req, res) => {
              try {
                // Add CORS headers
                if (res && !res.headersSent) {
                  res.setHeader('Access-Control-Allow-Origin', '*');
                  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
                  res.setHeader('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept');
                }
                // Fix Content-Type for .json files
                if (req.url && req.url.endsWith('.json') && !res.headersSent) {
                  res.setHeader('Content-Type', 'application/json; charset=utf-8');
                }
              } catch (e) {
                // Ignore header errors - don't crash the server
              }
            });
            proxy.on('error', (err, req, res) => {
              try {
                console.error('Proxy error:', err.message, 'URL:', req.url);
                if (res && !res.headersSent) {
                  res.statusCode = 500;
                  res.setHeader('Content-Type', 'application/json');
                  res.end(JSON.stringify({ error: 'Proxy error: ' + err.message }));
                }
              } catch (e) {
                // Don't crash if error handling fails
                console.error('Error handling proxy error:', e);
              }
            });
          },
        },
      },
    },
    plugins: [react(), contentTypeFixPlugin()],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, '.'),
      },
      dedupe: ['onnxruntime-web'],
    },
    optimizeDeps: {
      exclude: ['@xenova/transformers', 'onnxruntime-web'],
      esbuildOptions: {
        target: 'esnext',
      },
    },
    ssr: {
      noExternal: ['@xenova/transformers'],
    },
    define: {
      global: 'globalThis',
      'process.env': '{}',
    },
    build: {
      target: 'esnext',
      rollupOptions: {
        external: (id) => {
          // Don't bundle onnxruntime-web - use CDN version instead
          // The CDN version is loaded in index.html
          return id === 'onnxruntime-web' || id.startsWith('onnxruntime-web/');
        },
        output: {
          globals: {
            'onnxruntime-web': 'ort'
          }
        }
      }
    },
    worker: {
      format: 'es',
    },
    assetsInclude: ['**/*.wasm'],
  };
});
