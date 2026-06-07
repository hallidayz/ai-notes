export function oauthSuccessPage(origin: string, provider: string, payload: Record<string, unknown>): string {
  return `<!DOCTYPE html>
<html><body>
  <script>
    window.opener?.postMessage({
      type: 'OAUTH_AUTH_SUCCESS',
      provider: ${JSON.stringify(provider)},
      tokens: ${JSON.stringify(payload)}
    }, ${JSON.stringify(origin)});
    window.close();
  </script>
  <p>Authentication successful. You can close this window.</p>
</body></html>`;
}

export function oauthErrorPage(origin: string, provider: string, message: string): string {
  return `<!DOCTYPE html>
<html><body>
  <script>
    window.opener?.postMessage({
      type: 'OAUTH_AUTH_ERROR',
      provider: ${JSON.stringify(provider)},
      error: ${JSON.stringify(message)}
    }, ${JSON.stringify(origin)});
    window.close();
  </script>
  <p>Authentication failed: ${message}</p>
</body></html>`;
}
