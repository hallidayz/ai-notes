
import React, { useState, useEffect } from 'react';
import { Check, Download, Loader2, Info, ChevronLeft } from 'lucide-react';
import { LocalModel, ModelConfig, StorageProvider } from '../types';
import { onDeviceAIService } from '../services/onDeviceAIService';

interface LocalModelsProps {
    storageProvider: StorageProvider;
    onBack: () => void;
    showStatus: (msg: string, type: 'success' | 'error' | 'info') => void;
}

const AVAILABLE_MODELS: LocalModel[] = [
    // Transcription Models
    {
        id: 'whisper-tiny-en',
        name: 'Whisper Tiny (English)',
        parameters: '39M',
        provider: 'OpenAI',
        type: 'transcription',
        huggingFacePath: 'Xenova/whisper-tiny.en'
    },
    {
        id: 'whisper-base-en',
        name: 'Whisper Base (English)',
        parameters: '74M',
        provider: 'OpenAI',
        type: 'transcription',
        huggingFacePath: 'Xenova/whisper-base.en'
    },
    // Analysis Models (from screenshot & standard)
    {
        id: 'lfm2-350m',
        name: 'LFM2-350M',
        parameters: '350M',
        provider: 'LiquidAI',
        type: 'analysis',
        huggingFacePath: 'Xenova/flan-t5-small' // Placeholder if real path unknown
    },
    {
        id: 'lfm2-700m',
        name: 'LFM2-700M',
        parameters: '700M',
        provider: 'LiquidAI',
        type: 'analysis',
        huggingFacePath: 'Xenova/flan-t5-base' // Placeholder
    },
    {
        id: 'qwen3-0.6b',
        name: 'Qwen3-0.6B',
        parameters: '0.6B',
        provider: 'Qwen',
        type: 'analysis',
        huggingFacePath: 'Xenova/qwen-0.5b-instruct' // Placeholder
    },
    {
        id: 'flan-t5-small',
        name: 'Flan-T5 Small',
        parameters: '60M',
        provider: 'Google',
        type: 'analysis',
        huggingFacePath: 'Xenova/flan-t5-small'
    },
    {
        id: 'phi-1_5',
        name: 'Phi-1.5',
        parameters: '1.3B',
        provider: 'Microsoft',
        type: 'analysis',
        huggingFacePath: 'Xenova/phi-1_5'
    }
];

export const LocalModels: React.FC<LocalModelsProps> = ({ storageProvider, onBack, showStatus }) => {
    const [config, setConfig] = useState<ModelConfig>({
        transcriptionModelId: 'whisper-tiny-en',
        analysisModelId: 'flan-t5-small'
    });
    const [downloadingId, setDownloadingId] = useState<string | null>(null);
    const [downloadProgress, setDownloadProgress] = useState(0);
    const [downloadedModels, setDownloadedModels] = useState<Set<string>>(new Set(['whisper-tiny-en', 'flan-t5-small']));

    useEffect(() => {
        const loadConfig = async () => {
            try {
                const savedConfig = await storageProvider.getConfig('model_config');
                if (savedConfig) {
                    setConfig(savedConfig);
                }
                
                const savedDownloaded = await storageProvider.getConfig('downloaded_models');
                if (savedDownloaded) {
                    setDownloadedModels(new Set(savedDownloaded));
                }
            } catch (e) {
                console.error("Failed to load model config", e);
            }
        };
        loadConfig();
    }, [storageProvider]);

    const saveConfig = async (newConfig: ModelConfig) => {
        setConfig(newConfig);
        try {
            await storageProvider.saveConfig('model_config', newConfig);
            onDeviceAIService.updateConfig(newConfig);
        } catch (e) {
            console.error("Failed to save model config", e);
        }
    };

    const handleDownload = async (model: LocalModel) => {
        setDownloadingId(model.id);
        setDownloadProgress(0);
        
        try {
            await onDeviceAIService.preloadModel(model.huggingFacePath, model.type, (p) => {
                if (p.status === 'progress') {
                    setDownloadProgress(p.progress || 0);
                }
            });
            
            const newDownloaded = new Set(downloadedModels);
            newDownloaded.add(model.id);
            setDownloadedModels(newDownloaded);
            
            await storageProvider.saveConfig('downloaded_models', Array.from(newDownloaded));
            
            showStatus(`${model.name} downloaded successfully.`, 'success');
        } catch (e) {
            console.error("Download failed", e);
            showStatus(`Failed to download ${model.name}.`, 'error');
        } finally {
            setDownloadingId(null);
        }
    };

    const handleSelect = (model: LocalModel) => {
        if (!downloadedModels.has(model.id)) {
            handleDownload(model);
            return;
        }

        if (model.type === 'transcription') {
            saveConfig({ ...config, transcriptionModelId: model.id });
        } else {
            saveConfig({ ...config, analysisModelId: model.id });
        }
    };

    const renderModelCard = (model: LocalModel) => {
        const isSelected = config.transcriptionModelId === model.id || config.analysisModelId === model.id;
        const isDownloaded = downloadedModels.has(model.id);
        const isDownloading = downloadingId === model.id;

        return (
            <div 
                key={model.id}
                className={`flex items-center p-4 border-b border-black/5 hover:bg-black/5 transition-colors cursor-pointer ${isSelected ? 'bg-black/5' : ''}`}
                onClick={() => handleSelect(model)}
            >
                <div className="w-10 h-10 rounded-lg bg-white border border-black/10 flex items-center justify-center mr-4">
                    {isSelected ? (
                        <div className="w-6 h-6 rounded-full bg-black flex items-center justify-center">
                            <Check className="text-white w-4 h-4" />
                        </div>
                    ) : (
                        <div className="w-6 h-6 rounded-full border-2 border-black/20" />
                    )}
                </div>
                
                <div className="flex-1">
                    <div className="flex items-center">
                        <h3 className="font-medium text-sm">{model.name}</h3>
                        {isDownloaded && !isSelected && (
                            <Check className="w-3 h-3 text-green-500 ml-2" />
                        )}
                    </div>
                    <p className="text-xs text-gray-500">
                        {model.parameters} parameters • {model.provider}
                    </p>
                    {isDownloaded && (
                        <div className="flex items-center mt-1">
                            <Check className="w-3 h-3 text-green-500 mr-1" />
                            <span className="text-[10px] text-green-600 font-medium">Downloaded</span>
                        </div>
                    )}
                </div>

                <div className="flex items-center">
                    {isDownloading ? (
                        <div className="flex flex-col items-end">
                            <Loader2 className="w-5 h-5 animate-spin text-black mb-1" />
                            <span className="text-[10px] font-mono">{Math.round(downloadProgress)}%</span>
                        </div>
                    ) : !isDownloaded ? (
                        <Download className="w-5 h-5 text-gray-400" />
                    ) : null}
                </div>
            </div>
        );
    };

    return (
        <div className="flex flex-col h-full bg-white">
            <div className="p-4 border-b border-black/10 flex items-center sticky top-0 bg-white z-10">
                <button onClick={onBack} className="mr-4 p-1 hover:bg-black/5 rounded-full transition-colors">
                    <ChevronLeft className="w-6 h-6" />
                </button>
                <h2 className="text-xl font-bold">Local models</h2>
            </div>

            <div className="flex-1 overflow-y-auto">
                <div className="p-4 bg-blue-50 border-b border-blue-100 flex items-start">
                    <Info className="w-5 h-5 text-blue-500 mr-3 mt-0.5 flex-shrink-0" />
                    <p className="text-xs text-blue-700 leading-relaxed">
                        These models run entirely on your device. Downloading them may take a few moments depending on your connection. Once downloaded, they work fully offline.
                    </p>
                </div>

                <div className="px-4 py-2 bg-gray-50 text-[10px] font-bold text-gray-400 uppercase tracking-wider">
                    Transcription Models
                </div>
                {AVAILABLE_MODELS.filter(m => m.type === 'transcription').map(renderModelCard)}

                <div className="px-4 py-2 bg-gray-50 text-[10px] font-bold text-gray-400 uppercase tracking-wider mt-4">
                    Analysis Models
                </div>
                {AVAILABLE_MODELS.filter(m => m.type === 'analysis').map(renderModelCard)}
            </div>
        </div>
    );
};
