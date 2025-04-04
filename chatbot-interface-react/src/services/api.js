/**
 * API Service for RAG System
 * Handles all interactions with the backend API
 */

// Default API URL, can be overridden by environment variables
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Enable this for development if the API is not available
const USE_MOCK_DATA = true;

// Common fetch options for all requests
const commonFetchOptions = {
    // Remove credentials to avoid CORS preflight issues
    // credentials: 'include',
    headers: {
        'Accept': 'application/json'
    }
};

/**
 * Mock responses for development when API is not available
 */
const mockResponses = {
    textQuery: {
        answer: "Voici une réponse simulée du système RAG. Cette fonctionnalité est actuellement en mode développement et les réponses réelles seront disponibles une fois l'API connectée.",
        sources: [
            {
                title: "Document exemple",
                text: "Ceci est un exemple de source qui serait retournée par le système RAG.",
                page_number: 1,
                is_image: false
            },
            {
                title: "Image exemple",
                text: "Description d'une image qui pourrait être retournée comme source.",
                is_image: true
            }
        ]
    },
    documentUpload: {
        ids: ["mock-id-1", "mock-id-2"],
        message: "Document ajouté avec succès (simulation)."
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
        // Return mock data if API is not available
        if (USE_MOCK_DATA) {
            console.log('Using mock data for text query:', query);
            // Add a delay to simulate network request
            await new Promise(resolve => setTimeout(resolve, 1000));
            return mockResponses.textQuery;
        }

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
        // Return mock data on error if enabled
        if (USE_MOCK_DATA) {
            console.log('Falling back to mock data due to error');
            return mockResponses.textQuery;
        }
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
        // Return mock data if API is not available
        if (USE_MOCK_DATA) {
            console.log('Using mock data for document upload:', file.name);
            // Add a delay to simulate network request
            await new Promise(resolve => setTimeout(resolve, 2000));
            return mockResponses.documentUpload;
        }

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
        // Return mock data on error if enabled
        if (USE_MOCK_DATA) {
            console.log('Falling back to mock data due to error');
            return mockResponses.documentUpload;
        }
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
        // Return mock data if API is not available
        if (USE_MOCK_DATA) {
            console.log('Using mock data for image query:', imageFile.name);
            // Add a delay to simulate network request
            await new Promise(resolve => setTimeout(resolve, 1500));
            return mockResponses.textQuery;
        }

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
        // Return mock data on error if enabled
        if (USE_MOCK_DATA) {
            console.log('Falling back to mock data due to error');
            return mockResponses.textQuery;
        }
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
        // Return mock data if API is not available
        if (USE_MOCK_DATA) {
            console.log('Using mock data for combined query:', query);
            // Add a delay to simulate network request
            await new Promise(resolve => setTimeout(resolve, 2000));
            return mockResponses.textQuery;
        }

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
        // Return mock data on error if enabled
        if (USE_MOCK_DATA) {
            console.log('Falling back to mock data due to error');
            return mockResponses.textQuery;
        }
        throw error;
    }
}; 