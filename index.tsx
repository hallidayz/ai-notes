import React from 'react';
import { createRoot } from 'react-dom/client';
import App from './src/app/App';
import { OAuthCallback } from './src/components/OAuthCallback';

// Check if this is an OAuth callback route
const path = window.location.pathname;
const googleCallbackMatch = path.match(/^\/oauth\/google\/callback/);
const outlookCallbackMatch = path.match(/^\/oauth\/outlook\/callback/);

let ComponentToRender: React.ReactElement;

if (googleCallbackMatch) {
    ComponentToRender = <OAuthCallback provider="google" />;
} else if (outlookCallbackMatch) {
    ComponentToRender = <OAuthCallback provider="outlook" />;
} else {
    ComponentToRender = <App />;
}

const root = createRoot(document.getElementById('root')!);
root.render(ComponentToRender);
