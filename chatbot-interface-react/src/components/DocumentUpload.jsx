import React, { useState, useRef } from 'react';
import { Upload, X, File, Plus, Check } from 'lucide-react';
import { uploadDocument } from '../services/api';

/**
 * Enhanced document upload component with progress indicator and metadata options
 * @param {Object} props
 * @param {Function} props.onUploadSuccess - Callback when upload is successful
 * @param {Function} props.onUploadError - Callback when upload fails
 */
const DocumentUpload = ({ onUploadSuccess, onUploadError }) => {
    const [selectedFile, setSelectedFile] = useState(null);
    const [isUploading, setIsUploading] = useState(false);
    const [uploadProgress, setUploadProgress] = useState(0);
    const [description, setDescription] = useState('');
    const [showAdvanced, setShowAdvanced] = useState(false);
    const [chunkSize, setChunkSize] = useState('');
    const [chunkOverlap, setChunkOverlap] = useState('');
    const [extractImages, setExtractImages] = useState(true);
    const fileInputRef = useRef(null);

    // Handle file selection
    const handleFileChange = (e) => {
        const file = e.target.files[0];
        if (!file) return;

        // Validate file is PDF
        if (file.type !== 'application/pdf') {
            onUploadError('Veuillez sélectionner un fichier PDF.');
            return;
        }

        setSelectedFile(file);
    };

    // Simulate progress updates (since fetch doesn't expose progress)
    const simulateProgress = () => {
        setUploadProgress(0);
        const interval = setInterval(() => {
            setUploadProgress(prev => {
                if (prev >= 90) {
                    clearInterval(interval);
                    return prev;
                }
                return prev + 10;
            });
        }, 300);
        return interval;
    };

    // Handle document upload
    const handleUpload = async () => {
        if (!selectedFile) return;

        setIsUploading(true);
        const progressInterval = simulateProgress();

        try {
            // Prepare advanced options
            const chunkSizeNum = chunkSize ? parseInt(chunkSize, 10) : null;
            const chunkOverlapNum = chunkOverlap ? parseInt(chunkOverlap, 10) : null;

            // Upload document with metadata
            await uploadDocument(
                selectedFile,
                description || null,
                chunkSizeNum,
                chunkOverlapNum,
                extractImages
            );

            // Complete progress
            setUploadProgress(100);

            // Reset form and notify success
            setTimeout(() => {
                if (onUploadSuccess) {
                    onUploadSuccess(selectedFile.name);
                }
                resetForm();
            }, 500);
        } catch (error) {
            if (onUploadError) {
                onUploadError(error.message || 'Erreur lors de l\'upload du document');
            }
        } finally {
            clearInterval(progressInterval);
            setIsUploading(false);
        }
    };

    // Reset the form
    const resetForm = () => {
        setSelectedFile(null);
        setDescription('');
        setChunkSize('');
        setChunkOverlap('');
        setExtractImages(true);
        setUploadProgress(0);
        if (fileInputRef.current) {
            fileInputRef.current.value = '';
        }
    };

    return (
        <div className="bg-gray-800 rounded-lg p-4">
            <h3 className="text-lg font-medium mb-4">Ajouter un document</h3>

            {/* File selection */}
            <div className="mb-4">
                {!selectedFile ? (
                    <label
                        htmlFor="document-upload-input"
                        className="flex flex-col items-center justify-center border-2 border-dashed border-gray-600 rounded-lg py-6 px-4 cursor-pointer hover:bg-gray-700 transition-colors"
                    >
                        <Upload size={32} className="text-blue-400 mb-2" />
                        <span className="text-sm text-gray-300">Cliquez pour sélectionner un fichier PDF</span>
                        <span className="text-xs text-gray-500 mt-1">Format accepté: PDF</span>
                    </label>
                ) : (
                    <div className="flex items-start bg-gray-700 rounded-lg p-3">
                        <File size={24} className="text-blue-400 mr-3 mt-1 flex-shrink-0" />
                        <div className="flex-1">
                            <div className="flex justify-between">
                                <span className="text-sm font-medium">{selectedFile.name}</span>
                                <button
                                    onClick={resetForm}
                                    className="text-gray-400 hover:text-white"
                                    disabled={isUploading}
                                >
                                    <X size={18} />
                                </button>
                            </div>
                            <span className="text-xs text-gray-400">
                                {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                            </span>
                        </div>
                    </div>
                )}
                <input
                    id="document-upload-input"
                    ref={fileInputRef}
                    type="file"
                    accept=".pdf"
                    onChange={handleFileChange}
                    className="hidden"
                    disabled={isUploading}
                />
            </div>

            {/* Document description */}
            {selectedFile && (
                <>
                    <div className="mb-4">
                        <label htmlFor="document-description" className="block text-sm font-medium mb-1">
                            Description du document (optionnel)
                        </label>
                        <textarea
                            id="document-description"
                            value={description}
                            onChange={(e) => setDescription(e.target.value)}
                            placeholder="Ajoutez une description pour faciliter la recherche..."
                            className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white focus:outline-none focus:ring-1 focus:ring-blue-500"
                            rows="2"
                            disabled={isUploading}
                        />
                    </div>

                    {/* Advanced options toggle */}
                    <div className="mb-4">
                        <button
                            type="button"
                            onClick={() => setShowAdvanced(!showAdvanced)}
                            className="flex items-center text-sm text-gray-400 hover:text-white"
                            disabled={isUploading}
                        >
                            <Plus size={16} className="mr-1" />
                            Options avancées
                        </button>
                    </div>

                    {/* Advanced options */}
                    {showAdvanced && (
                        <div className="mb-4 pl-2 border-l border-gray-700 space-y-3">
                            <div>
                                <label htmlFor="chunk-size" className="block text-sm font-medium mb-1">
                                    Taille des chunks
                                </label>
                                <input
                                    id="chunk-size"
                                    type="number"
                                    value={chunkSize}
                                    onChange={(e) => setChunkSize(e.target.value)}
                                    placeholder="Par défaut: 1000"
                                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white focus:outline-none focus:ring-1 focus:ring-blue-500"
                                    disabled={isUploading}
                                />
                            </div>

                            <div>
                                <label htmlFor="chunk-overlap" className="block text-sm font-medium mb-1">
                                    Chevauchement des chunks
                                </label>
                                <input
                                    id="chunk-overlap"
                                    type="number"
                                    value={chunkOverlap}
                                    onChange={(e) => setChunkOverlap(e.target.value)}
                                    placeholder="Par défaut: 200"
                                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white focus:outline-none focus:ring-1 focus:ring-blue-500"
                                    disabled={isUploading}
                                />
                            </div>

                            <div className="flex items-center">
                                <input
                                    id="extract-images"
                                    type="checkbox"
                                    checked={extractImages}
                                    onChange={(e) => setExtractImages(e.target.checked)}
                                    className="w-4 h-4 text-blue-600 bg-gray-700 border-gray-600 focus:ring-blue-500"
                                    disabled={isUploading}
                                />
                                <label htmlFor="extract-images" className="ml-2 text-sm font-medium">
                                    Extraire les images du PDF
                                </label>
                            </div>
                        </div>
                    )}

                    {/* Upload progress */}
                    {isUploading && (
                        <div className="mb-4">
                            <div className="w-full bg-gray-700 rounded-full h-2.5">
                                <div
                                    className="bg-blue-600 h-2.5 rounded-full"
                                    style={{ width: `${uploadProgress}%` }}
                                ></div>
                            </div>
                            <div className="flex justify-between mt-1">
                                <span className="text-xs text-gray-400">Chargement...</span>
                                <span className="text-xs text-gray-400">{uploadProgress}%</span>
                            </div>
                        </div>
                    )}

                    {/* Upload button */}
                    <div className="flex justify-end">
                        <button
                            onClick={handleUpload}
                            disabled={!selectedFile || isUploading}
                            className={`flex items-center gap-2 rounded-lg px-4 py-2
                ${!selectedFile || isUploading
                                    ? 'bg-gray-700 text-gray-400 cursor-not-allowed'
                                    : 'bg-blue-600 hover:bg-blue-700 text-white'
                                }`}
                        >
                            {uploadProgress === 100 ? <Check size={18} /> : <Upload size={18} />}
                            {uploadProgress === 100 ? 'Document ajouté' : 'Ajouter à la bibliothèque'}
                        </button>
                    </div>
                </>
            )}
        </div>
    );
};

export default DocumentUpload; 