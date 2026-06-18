import React, { useState, useEffect } from 'react';
import { motion } from 'motion/react';
import { Settings as SettingsIcon, Save, Database, Cloud } from 'lucide-react';
import { StorageType } from '../types';
import { FileSystemDirectoryHandle } from '../services/storageProvider';

interface SettingsProps {
  onBack: () => void;
  showStatus: (msg: string, type: 'success' | 'error' | 'info') => void;
  storageType: StorageType;
  onStorageTypeChange: (type: StorageType) => void;
  storageFolder: string;
  onStorageFolderChange: (folder: string, dirHandle?: FileSystemDirectoryHandle) => void;
}

export const Settings: React.FC<SettingsProps> = ({ onBack, showStatus, storageType, onStorageTypeChange, storageFolder, onStorageFolderChange }) => {
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [oauthSettings, setOauthSettings] = useState({
    googleClientId: '',
    googleClientSecret: '',
    microsoftClientId: '',
    microsoftClientSecret: '',
    notionClientId: '',
    notionClientSecret: '',
    nylasClientId: '',
    nylasApiKey: '',
    nylasApiUri: 'https://api.us.nylas.com',
  });

  useEffect(() => {
    const fetchSettings = async () => {
      try {
        const response = await fetch('/api/settings/oauth');
        if (response.ok) {
          const data = await response.json();
          setOauthSettings({
            googleClientId: data.googleClientId || '',
            googleClientSecret: data.googleClientSecret || '',
            microsoftClientId: data.microsoftClientId || '',
            microsoftClientSecret: data.microsoftClientSecret || '',
            notionClientId: data.notionClientId || '',
            notionClientSecret: data.notionClientSecret || '',
            nylasClientId: data.nylasClientId || '',
            nylasApiKey: data.nylasApiKey || '',
            nylasApiUri: data.nylasApiUri || 'https://api.us.nylas.com',
          });
        }
      } catch (error) {
        console.error('Failed to fetch OAuth settings', error);
      } finally {
        setIsLoading(false);
      }
    };
    fetchSettings();
  }, []);

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSaving(true);
    try {
      const response = await fetch('/api/settings/oauth', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(oauthSettings),
      });
      if (response.ok) {
        showStatus('Settings saved successfully', 'success');
      } else {
        throw new Error('Failed to save');
      }
    } catch (error) {
      console.error('Failed to save OAuth settings', error);
      showStatus('Failed to save settings', 'error');
    } finally {
      setIsSaving(false);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setOauthSettings(prev => ({
      ...prev,
      [e.target.name]: e.target.value
    }));
  };

  const handleSelectFolder = async (e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      if ('showDirectoryPicker' in window) {
        const dirHandle = await (window as { showDirectoryPicker?: (options?: { startIn: string }) => Promise<FileSystemDirectoryHandle> }).showDirectoryPicker?.({ startIn: 'downloads' });
        if (dirHandle) {
          onStorageFolderChange(dirHandle.name, dirHandle);
          showStatus(`Storage folder set to ${dirHandle.name}`, 'success');
        }
      } else {
        showStatus('Directory picker not supported in this browser', 'error');
      }
    } catch (err) {
      console.error('Error selecting folder:', err);
    }
  };

  return (
    <div className="settings-page p-6 max-w-3xl mx-auto">
      <div className="flex items-center gap-4 mb-8">
        <button onClick={onBack} className="btn-secondary">
          &larr; Back
        </button>
        <h2 className="text-2xl font-bold flex items-center gap-2">
          <SettingsIcon className="w-6 h-6" />
          App Settings
        </h2>
      </div>

      {isLoading ? (
        <div className="loading"><div className="spinner"></div>Loading settings...</div>
      ) : (
        <form onSubmit={handleSave} className="space-y-8">
          <motion.div 
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="card p-6"
          >
            <h3 className="text-xl font-bold mb-4 border-b pb-2">Storage Location</h3>
            <p className="text-sm text-gray-500 mb-6">
              Choose where your data is stored.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div 
                className={`p-4 rounded-lg border-2 cursor-pointer transition-all ${storageType === StorageType.BROWSER ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20' : 'border-gray-200 dark:border-gray-700 hover:border-blue-300'}`}
                onClick={() => onStorageTypeChange(StorageType.BROWSER)}
              >
                <div className="flex items-center gap-3 mb-2">
                  <Database className={`w-5 h-5 ${storageType === StorageType.BROWSER ? 'text-blue-500' : 'text-gray-400'}`} />
                  <span className="font-bold">Browser (IndexedDB)</span>
                </div>
                <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
                  Fastest, fully offline, data stays in this browser. This does require the files to be written to a location on the device, by default, make it the downloads folder and allow the user to change it based on what they want to use.
                </p>
                {storageType === StorageType.BROWSER && (
                  <div className="mt-auto pt-4 border-t border-gray-100 dark:border-gray-800 flex items-center justify-between">
                    <span className="text-xs font-medium text-gray-500">Folder: {storageFolder}</span>
                    <button 
                      type="button"
                      onClick={handleSelectFolder}
                      className="text-xs bg-gray-100 hover:bg-gray-200 dark:bg-gray-800 dark:hover:bg-gray-700 px-3 py-1.5 rounded-md transition-colors"
                    >
                      Change Folder
                    </button>
                  </div>
                )}
              </div>
              
              <div 
                className={`p-4 rounded-lg border-2 cursor-pointer transition-all ${storageType === StorageType.SERVER ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20' : 'border-gray-200 dark:border-gray-700 hover:border-blue-300'}`}
                onClick={() => onStorageTypeChange(StorageType.SERVER)}
              >
                <div className="flex items-center gap-3 mb-2">
                  <Cloud className={`w-5 h-5 ${storageType === StorageType.SERVER ? 'text-blue-500' : 'text-gray-400'}`} />
                  <span className="font-bold">Server (Cloud)</span>
                </div>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  Sync across devices, data stored on our secure server.
                </p>
              </div>
            </div>
          </motion.div>

          <motion.div 
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="card p-6"
          >
            <h3 className="text-xl font-bold mb-4 border-b pb-2">OAuth Integrations</h3>
            <p className="text-sm text-gray-500 mb-6">
              Configure your own API keys for calendar integrations. These keys are stored securely on your local server instance and are not part of the source code.
            </p>

            <div className="space-y-6">
              {/* Nylas Settings */}
              <div className="bg-gray-50 dark:bg-gray-800/50 p-4 rounded-lg border border-gray-100 dark:border-gray-700">
                <h4 className="font-semibold mb-3 flex items-center gap-2">
                  <div className="w-2 h-2 bg-green-500 rounded-full" />
                  Nylas (Unified API)
                </h4>
                <p className="text-xs text-gray-500 mb-4">
                  Use Nylas for a "One-Time" setup that syncs Google, Outlook, and Apple calendars automatically in the background.
                </p>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="input-label">Client ID</label>
                    <input
                      type="text"
                      name="nylasClientId"
                      value={oauthSettings.nylasClientId}
                      onChange={handleChange}
                      placeholder="Enter Nylas Client ID"
                      className="w-full"
                    />
                  </div>
                  <div>
                    <label className="input-label">API Key</label>
                    <input
                      type="password"
                      name="nylasApiKey"
                      value={oauthSettings.nylasApiKey}
                      onChange={handleChange}
                      placeholder="Enter Nylas API Key"
                      className="w-full"
                    />
                  </div>
                  <div className="md:col-span-2">
                    <label className="input-label">API URI</label>
                    <input
                      type="text"
                      name="nylasApiUri"
                      value={oauthSettings.nylasApiUri}
                      onChange={handleChange}
                      placeholder="https://api.us.nylas.com"
                      className="w-full"
                    />
                  </div>
                </div>
              </div>

              {/* Google Settings */}
              <div className="bg-gray-50 dark:bg-gray-800/50 p-4 rounded-lg border border-gray-100 dark:border-gray-700">
                <h4 className="font-semibold mb-3 flex items-center gap-2">
                  <div className="w-2 h-2 bg-blue-500 rounded-full" />
                  Google Calendar
                </h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="input-label">Client ID</label>
                    <input
                      type="text"
                      name="googleClientId"
                      value={oauthSettings.googleClientId}
                      onChange={handleChange}
                      placeholder="Enter Google Client ID"
                      className="w-full"
                    />
                  </div>
                  <div>
                    <label className="input-label">Client Secret</label>
                    <input
                      type="password"
                      name="googleClientSecret"
                      value={oauthSettings.googleClientSecret}
                      onChange={handleChange}
                      placeholder="Enter Google Client Secret"
                      className="w-full"
                    />
                  </div>
                </div>
              </div>

              {/* Microsoft Settings */}
              <div className="bg-gray-50 dark:bg-gray-800/50 p-4 rounded-lg border border-gray-100 dark:border-gray-700">
                <h4 className="font-semibold mb-3 flex items-center gap-2">
                  <div className="w-2 h-2 bg-blue-600 rounded-full" />
                  Microsoft Outlook
                </h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="input-label">Client ID</label>
                    <input
                      type="text"
                      name="microsoftClientId"
                      value={oauthSettings.microsoftClientId}
                      onChange={handleChange}
                      placeholder="Enter Microsoft Client ID"
                      className="w-full"
                    />
                  </div>
                  <div>
                    <label className="input-label">Client Secret</label>
                    <input
                      type="password"
                      name="microsoftClientSecret"
                      value={oauthSettings.microsoftClientSecret}
                      onChange={handleChange}
                      placeholder="Enter Microsoft Client Secret"
                      className="w-full"
                    />
                  </div>
                </div>
              </div>

              {/* Notion Settings */}
              <div className="bg-gray-50 dark:bg-gray-800/50 p-4 rounded-lg border border-gray-100 dark:border-gray-700">
                <h4 className="font-semibold mb-3 flex items-center gap-2">
                  <div className="w-2 h-2 bg-black dark:bg-white rounded-full" />
                  Notion
                </h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="input-label">Client ID</label>
                    <input
                      type="text"
                      name="notionClientId"
                      value={oauthSettings.notionClientId}
                      onChange={handleChange}
                      placeholder="Enter Notion Client ID"
                      className="w-full"
                    />
                  </div>
                  <div>
                    <label className="input-label">Client Secret</label>
                    <input
                      type="password"
                      name="notionClientSecret"
                      value={oauthSettings.notionClientSecret}
                      onChange={handleChange}
                      placeholder="Enter Notion Client Secret"
                      className="w-full"
                    />
                  </div>
                </div>
              </div>
            </div>
          </motion.div>

          <div className="flex justify-end">
            <button 
              type="submit" 
              className="btn-primary flex items-center gap-2 px-6 py-3"
              disabled={isSaving}
            >
              <Save className="w-4 h-4" />
              {isSaving ? 'Saving...' : 'Save Settings'}
            </button>
          </div>
        </form>
      )}
    </div>
  );
};
