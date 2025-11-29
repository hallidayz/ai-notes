// HTTP interceptors for Hugging Face model requests
// This file is separate to avoid Fast Refresh issues with global prototype modifications

// Set up HTTP interceptors to rewrite Hugging Face requests to go through Vite proxy
if (typeof window !== 'undefined' && !(window as any).__transformersHttpIntercepted) {
    // Intercept fetch to rewrite Hugging Face requests to go through Vite proxy
    const originalFetch = window.fetch;
    window.fetch = async function(...args) {
        let url = args[0];
        let urlStr = '';
        
        // Handle different URL types
        if (typeof url === 'string') {
            urlStr = url;
        } else if (url instanceof URL) {
            urlStr = url.toString();
        } else if (url instanceof Request) {
            urlStr = url.url;
        } else {
            urlStr = String(url || '');
        }
        
        // Check if this is a Hugging Face request that needs to be proxied
        const isHuggingFaceRequest = urlStr && (
            urlStr.includes('huggingface.co') || 
            urlStr.includes('hf.co')
        );
        
        // Rewrite absolute Hugging Face URLs to relative URLs that go through our proxy
        if (isHuggingFaceRequest && urlStr.startsWith('http')) {
            try {
                const urlObj = new URL(urlStr);
                const path = urlObj.pathname;
                // Use path as-is (it already includes /Xenova/ or /onnx-community/...)
                let newUrl = path;
                // Add query string if present
                if (urlObj.search) {
                    newUrl += urlObj.search;
                }
                if (url instanceof Request) {
                    args[0] = new Request(newUrl, url);
                } else {
                    args[0] = newUrl;
                }
                urlStr = newUrl;
            } catch (e) {
                // Silently fail - use original URL
            }
        }
        
        // Intercept response to detect and replace HTML errors with JSON
        const response = await originalFetch.apply(this, args);
        
        // Check ALL model-related responses for HTML
        const isModelRequest = urlStr.endsWith('.json') || 
                              urlStr.includes('/resolve/') || 
                              urlStr.includes('/Xenova/') || 
                              urlStr.includes('/onnx-community/') ||
                              urlStr.includes('/api/resolve-cache') ||
                              urlStr.includes('huggingface.co') ||
                              urlStr.includes('hf.co');
        
        if (isModelRequest) {
            const contentType = response.headers.get('content-type') || '';
            // Check for HTML or error status
            if (contentType.includes('text/html') || response.status >= 400) {
                try {
                    const text = await response.clone().text();
                    if (text.trim().startsWith('<!DOCTYPE') || text.trim().startsWith('<html')) {
                        // Return a JSON error response instead of HTML
                        const errorJson = JSON.stringify({
                            error: 'Model download failed',
                            message: 'Received HTML error page instead of JSON. The model may be unavailable or the URL is incorrect.',
                            url: urlStr,
                            status: response.status
                        });
                        
                        // Create a new Response with JSON error
                        return new Response(errorJson, {
                            status: response.status >= 400 ? response.status : 500,
                            statusText: 'Model Download Error',
                            headers: {
                                'Content-Type': 'application/json; charset=utf-8',
                                'Content-Length': errorJson.length.toString()
                            }
                        });
                    }
                } catch (e) {
                    // If we can't read the response, return JSON error anyway
                    const errorJson = JSON.stringify({
                        error: 'Model download failed',
                        message: 'Unable to read response. The model may be unavailable.',
                        url: urlStr,
                        status: response.status
                    });
                    return new Response(errorJson, {
                        status: 500,
                        statusText: 'Model Download Error',
                        headers: {
                            'Content-Type': 'application/json; charset=utf-8'
                        }
                    });
                }
            }
        }
        
        return response;
    };
    
    // Also intercept XMLHttpRequest
    const originalXHROpen = XMLHttpRequest.prototype.open;
    const originalXHRSend = XMLHttpRequest.prototype.send;
    
    XMLHttpRequest.prototype.open = function(method: string, url: string | URL, ...rest: any[]) {
        let urlStr = url?.toString() || '';
        
        const isHuggingFaceRequest = urlStr && (
            urlStr.includes('huggingface.co') || 
            urlStr.includes('hf.co')
        );
        
        if (isHuggingFaceRequest && typeof url === 'string' && url.startsWith('http')) {
            try {
                const urlObj = new URL(url);
                const path = urlObj.pathname;
                // Use path as-is (it already includes /Xenova/ or /onnx-community/...)
                let newUrl = path;
                // Add query string if present
                if (urlObj.search) {
                    newUrl += urlObj.search;
                }
                url = newUrl;
            } catch (e) {
                // Silently fail - use original URL
            }
        }
        
        // Store URL and isHuggingFaceRequest flag on the XHR object for response interception
        (this as any).__interceptedUrl = urlStr;
        (this as any).__isHuggingFaceRequest = isHuggingFaceRequest;
        
        return originalXHROpen.apply(this, [method, url, ...rest]);
    };
    
    XMLHttpRequest.prototype.send = function(...args: any[]) {
        const xhr = this;
        const urlStr = (xhr as any).__interceptedUrl || '';
        const isHuggingFaceRequest = (xhr as any).__isHuggingFaceRequest || false;
        
        // Intercept response to detect HTML errors
        const checkForHtmlError = () => {
            if (xhr.readyState === 4 && (urlStr.endsWith('.json') || urlStr.includes('/resolve/') || urlStr.includes('/Xenova/') || urlStr.includes('/onnx-community/') || urlStr.includes('/api/resolve-cache'))) {
                const contentType = xhr.getResponseHeader('content-type') || '';
                if (contentType.includes('text/html') || xhr.status >= 400) {
                    const responseText = xhr.responseText || '';
                    if (responseText.trim().startsWith('<!DOCTYPE') || responseText.trim().startsWith('<html')) {
                        // Create JSON error response
                        const errorJson = JSON.stringify({
                            error: 'Model download failed',
                            message: 'Received HTML error page instead of JSON. The model may be unavailable or the URL is incorrect.',
                            url: urlStr,
                            status: xhr.status
                        });
                        // Override responseText to be valid JSON error
                        try {
                            Object.defineProperty(xhr, 'responseText', {
                                value: errorJson,
                                writable: false,
                                configurable: true
                            });
                            // Override response to be the JSON error
                            Object.defineProperty(xhr, 'response', {
                                value: JSON.parse(errorJson),
                                writable: false,
                                configurable: true
                            });
                        } catch (e) {
                            // Failed to override - let original error propagate
                        }
                    }
                }
            }
        };
        
        // Intercept onreadystatechange
        const originalOnReadyStateChange = xhr.onreadystatechange;
        xhr.onreadystatechange = function() {
            checkForHtmlError();
            if (originalOnReadyStateChange) {
                originalOnReadyStateChange.apply(xhr, arguments as any);
            }
        };
        
        // Also intercept addEventListener for 'readystatechange'
        const originalAddEventListener = xhr.addEventListener;
        xhr.addEventListener = function(type: string, listener: any, ...rest: any[]) {
            if (type === 'readystatechange' || type === 'load') {
                const wrappedListener = function(...args: any[]) {
                    checkForHtmlError();
                    if (listener) {
                        listener.apply(xhr, args);
                    }
                };
                return originalAddEventListener.apply(xhr, [type, wrappedListener, ...rest]);
            }
            return originalAddEventListener.apply(xhr, [type, listener, ...rest]);
        };
        
        return originalXHRSend.apply(this, args);
    };
    
    (window as any).__transformersHttpIntercepted = true;
}

