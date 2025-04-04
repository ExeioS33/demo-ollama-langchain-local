import React, { useState, useRef } from 'react';
import { X, Image as ImageIcon } from 'lucide-react';

/**
 * Component for uploading and previewing images before sending to the API
 * @param {Object} props
 * @param {Function} props.onImageSelect - Callback when image is selected
 * @param {Function} props.onImageRemove - Callback when image is removed
 */
const ImageUploadPreview = ({ onImageSelect, onImageRemove }) => {
    const [previewUrl, setPreviewUrl] = useState(null);
    const fileInputRef = useRef(null);

    const handleFileChange = (e) => {
        const file = e.target.files[0];
        if (!file) {
            return;
        }

        if (file.type.startsWith('image/')) {
            const reader = new FileReader();
            reader.onloadend = () => {
                setPreviewUrl(reader.result);
                onImageSelect(file);
            };
            reader.readAsDataURL(file);
        } else {
            alert('Veuillez sélectionner une image');
        }
    };

    const handleRemoveImage = () => {
        if (previewUrl) {
            URL.revokeObjectURL(previewUrl);
        }

        setPreviewUrl(null);

        // Reset the file input
        if (fileInputRef.current) {
            fileInputRef.current.value = '';
        }

        // Call parent callback
        if (onImageRemove) {
            onImageRemove();
        }
    };

    return (
        <div className="mt-3">
            {previewUrl ? (
                <div className="relative">
                    <img
                        src={previewUrl}
                        alt="Preview"
                        className="rounded-lg max-h-60 max-w-full object-contain bg-gray-800"
                    />
                    <button
                        onClick={handleRemoveImage}
                        className="absolute top-2 right-2 bg-gray-900 bg-opacity-70 rounded-full p-1 text-white hover:bg-red-600 transition-colors"
                    >
                        <X size={16} />
                    </button>
                </div>
            ) : (
                <label
                    htmlFor="image-upload-preview"
                    className="flex items-center gap-2 text-gray-400 hover:text-white cursor-pointer"
                >
                    <ImageIcon size={20} />
                    <span>Ajouter une image</span>
                </label>
            )}
            <input
                id="image-upload-preview"
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                className="hidden"
            />
        </div>
    );
};

export default ImageUploadPreview; 