/**
 * API Service for RAG System
 * Handles all interactions with the backend API
 */

// Default API URL, can be overridden by environment variables
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Common fetch options for all requests
const commonFetchOptions = {
    credentials: 'include',
    headers: {
        'Accept': 'application/json'
    }
};

/**
 * Send a text query to the RAG system
 * @param {string} query - The user's query text
 * @param {number} topK - Number of results to retrieve (default: 3)
 * @param {boolean} imageOnly - Whether to return only image results (default: false)
 * @returns {Promise} - The response with answer and sources
 */
export const sendTextQuery = async (query, topK = 3, imageOnly = false) => {
    try {
        const response = await fetch(`${API_BASE_URL}/query/text`, {
            method: 'POST',
            ...commonFetchOptions,
            headers: {
                ...commonFetchOptions.headers,
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query,
                top_k: topK,
                image_only: imageOnly,
            }),
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `Failed to send text query (${response.status})`);
        }

        return await response.json();
    } catch (error) {
        console.error('Error sending text query:', error);
        throw error;
    }
};

/**
 * Upload a document to the RAG system
 * @param {File} file - The document file to upload
 * @param {string} description - Optional description for the document
 * @param {number} chunkSize - Optional chunk size for document processing
 * @param {number} chunkOverlap - Optional chunk overlap for document processing
 * @param {boolean} extractImages - Whether to extract images from PDF (default: true)
 * @returns {Promise} - The response with document IDs
 */
export const uploadDocument = async (file, description = null, chunkSize = null, chunkOverlap = null, extractImages = true) => {
    try {
        const formData = new FormData();
        formData.append('document', file);

        if (description) formData.append('description', description);
        if (chunkSize) formData.append('chunk_size', chunkSize);
        if (chunkOverlap) formData.append('chunk_overlap', chunkOverlap);
        formData.append('extract_images', extractImages);

        const response = await fetch(`${API_BASE_URL}/add/document`, {
            method: 'POST',
            ...commonFetchOptions,
            body: formData,
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `Failed to upload document (${response.status})`);
        }

        return await response.json();
    } catch (error) {
        console.error('Error uploading document:', error);
        throw error;
    }
};

/**
 * Send an image query to the RAG system
 * @param {File} imageFile - The image file to query
 * @param {string} query - The text query to accompany the image
 * @param {number} topK - Number of results to retrieve
 * @returns {Promise} - The response with answer and sources
 */
export const sendImageQuery = async (imageFile, query = 'Décris cette image', topK = 3) => {
    try {
        const formData = new FormData();
        formData.append('image', imageFile);
        formData.append('query', query);
        formData.append('top_k', topK);

        const response = await fetch(`${API_BASE_URL}/query/image`, {
            method: 'POST',
            ...commonFetchOptions,
            body: formData,
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `Failed to send image query (${response.status})`);
        }

        return await response.json();
    } catch (error) {
        console.error('Error sending image query:', error);
        throw error;
    }
};

/**
 * Send a combined text and image query to the RAG system
 * @param {string} query - The text query
 * @param {File} imageFile - Optional image file
 * @param {boolean} referenceImage - Whether to reference indexed images
 * @param {number} topK - Number of results to retrieve
 * @returns {Promise} - The response with answer and sources
 */
export const sendCombinedQuery = async (query, imageFile = null, referenceImage = false, topK = 3) => {
    try {
        const formData = new FormData();
        formData.append('query', query);
        if (imageFile) formData.append('image', imageFile);
        formData.append('reference_image', referenceImage);
        formData.append('top_k', topK);

        const response = await fetch(`${API_BASE_URL}/query/combined`, {
            method: 'POST',
            ...commonFetchOptions,
            body: formData,
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `Failed to send combined query (${response.status})`);
        }

        return await response.json();
    } catch (error) {
        console.error('Error sending combined query:', error);
        throw error;
    }
}; 