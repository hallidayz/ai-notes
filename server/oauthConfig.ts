import fs from 'fs/promises';
import path from 'path';

export interface ProviderCredentials {
  clientId: string;
  clientSecret: string;
}

export interface OAuthConfigFile {
  google?: ProviderCredentials;
  microsoft?: ProviderCredentials;
  notion?: ProviderCredentials;
}

const CONFIG_PATH = path.join(process.cwd(), 'local_storage', 'oauth_config.json');

let cachedConfig: OAuthConfigFile | null = null;

export async function loadOAuthConfig(): Promise<OAuthConfigFile> {
  if (cachedConfig) return cachedConfig;
  try {
    const raw = await fs.readFile(CONFIG_PATH, 'utf-8');
    cachedConfig = JSON.parse(raw) as OAuthConfigFile;
  } catch {
    cachedConfig = {};
  }
  return cachedConfig;
}

export async function saveOAuthConfig(config: OAuthConfigFile): Promise<void> {
  await fs.mkdir(path.dirname(CONFIG_PATH), { recursive: true });
  await fs.writeFile(CONFIG_PATH, JSON.stringify(config, null, 2));
  cachedConfig = config;
}

export function getGoogleCredentials(config?: OAuthConfigFile): ProviderCredentials | null {
  const clientId = process.env.GOOGLE_CLIENT_ID || config?.google?.clientId;
  const clientSecret = process.env.GOOGLE_CLIENT_SECRET || config?.google?.clientSecret;
  if (!clientId || !clientSecret) return null;
  return { clientId, clientSecret };
}

export function getMicrosoftCredentials(config?: OAuthConfigFile): ProviderCredentials | null {
  const clientId = process.env.MICROSOFT_CLIENT_ID || config?.microsoft?.clientId;
  const clientSecret = process.env.MICROSOFT_CLIENT_SECRET || config?.microsoft?.clientSecret;
  if (!clientId || !clientSecret) return null;
  return { clientId, clientSecret };
}

export function getNotionCredentials(config?: OAuthConfigFile): ProviderCredentials | null {
  const clientId = process.env.NOTION_CLIENT_ID || config?.notion?.clientId;
  const clientSecret = process.env.NOTION_CLIENT_SECRET || config?.notion?.clientSecret;
  if (!clientId || !clientSecret) return null;
  return { clientId, clientSecret };
}

export function maskClientId(clientId: string): string {
  if (clientId.length <= 8) return '••••••••';
  return `${clientId.slice(0, 6)}…${clientId.slice(-4)}`;
}

export async function getOAuthStatus() {
  const config = await loadOAuthConfig();
  const google = getGoogleCredentials(config);
  const microsoft = getMicrosoftCredentials(config);
  const notion = getNotionCredentials(config);

  return {
    google: {
      configured: Boolean(google),
      clientId: google ? maskClientId(google.clientId) : null,
      source: process.env.GOOGLE_CLIENT_ID ? 'env' : config.google ? 'settings' : null,
    },
    microsoft: {
      configured: Boolean(microsoft),
      clientId: microsoft ? maskClientId(microsoft.clientId) : null,
      source: process.env.MICROSOFT_CLIENT_ID ? 'env' : config.microsoft ? 'settings' : null,
    },
    notion: {
      configured: Boolean(notion),
      clientId: notion ? maskClientId(notion.clientId) : null,
      source: process.env.NOTION_CLIENT_ID ? 'env' : config.notion ? 'settings' : null,
    },
  };
}
