import React, { useState, useEffect, useRef } from 'react';
import { Send, Paperclip, Upload, InfoIcon } from 'lucide-react';
import ImageUploadPreview from './ImageUploadPreview';
import LoadingIndicator from './LoadingIndicator';
import DocumentUpload from './DocumentUpload';
import Modal from './Modal';
import MarkdownMessage from './MarkdownMessage';
import { sendTextQuery, uploadDocument, sendImageQuery, sendCombinedQuery } from '../services/api';

// Local logo path
const CF_LOGO_PATH = '/images/groupecf_logo.jpeg';

// Add this style to the top of the file, after the imports
const styles = document.createElement('style');
styles.textContent = `
  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(-20px); }
    to { opacity: 1; transform: translateY(0); }
  }
  
  .animate-fade-in {
    animation: fadeIn 0.3s ease-out forwards;
  }
`;
document.head.appendChild(styles);

// Toast notification component
const ToastNotification = ({ message, onClose, isError = false }) => {
    useEffect(() => {
        const timer = setTimeout(() => {
            onClose();
        }, 8000); // Extended display time to 8 seconds

        return () => clearTimeout(timer);
    }, [onClose]);

    return (
        <div className={`fixed top-4 right-4 ${isError ? 'bg-red-500' : 'bg-green-500'} text-white px-5 py-4 rounded-lg shadow-xl flex items-center max-w-md animate-fade-in`}>
            <div className="mr-3">
                {isError ? (
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <circle cx="12" cy="12" r="10"></circle>
                        <line x1="15" y1="9" x2="9" y2="15"></line>
                        <line x1="9" y1="9" x2="15" y2="15"></line>
                    </svg>
                ) : (
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
                        <polyline points="22 4 12 14.01 9 11.01"></polyline>
                    </svg>
                )}
            </div>
            <div className="flex-1">
                <span className="font-medium">{message}</span>
            </div>
            <button onClick={onClose} className="ml-3 text-white hover:text-gray-200">
                <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <line x1="18" y1="6" x2="6" y2="18"></line>
                    <line x1="6" y1="6" x2="18" y2="18"></line>
                </svg>
            </button>
        </div>
    );
};

/**
 * Main chat interface component for the RAG system
 * Handles all user interactions, API calls, and UI states
 */
const ChatInterface = () => {
    const [messages, setMessages] = useState([
        { id: 1, content: "Bonjour! Comment puis-je vous aider aujourd'hui. Je m'occupes de répondre à vos demandes RH", isUser: false, sources: [] }
    ]);
    const [inputMessage, setInputMessage] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [toast, setToast] = useState({ message: '', visible: false, isError: false });
    const [selectedFile, setSelectedFile] = useState(null);
    const [selectedImage, setSelectedImage] = useState(null);
    const [showAboutModal, setShowAboutModal] = useState(false);
    const [showUploadModal, setShowUploadModal] = useState(false);
    const messagesEndRef = useRef(null);
    const imageInputRef = useRef(null);
    const [activeChat] = useState('Session 1');

    // Set document title
    useEffect(() => {
        document.title = "CFChat";

        // Set favicon
        const link = document.querySelector("link[rel~='icon']") || document.createElement('link');
        link.type = 'image/svg+xml';
        link.rel = 'icon';
        link.href = CF_LOGO_PATH;
        document.getElementsByTagName('head')[0].appendChild(link);
    }, []);

    // Auto scroll to bottom when messages change
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    // Display toast message
    const showToastMessage = (message, isError = false) => {
        setToast({ message, isError, visible: true });
    };

    // Handle sending a message
    const handleSendMessage = async () => {
        if ((inputMessage.trim() === '' && !selectedImage) || isLoading) return;

        const messageId = Date.now();
        const messageText = inputMessage.trim();

        // Create user message
        const userMessage = {
            id: messageId,
            content: messageText || 'Image envoyée',
            isUser: true
        };

        // Add user message to chat
        setMessages(prev => [...prev, userMessage]);

        // Clear input and image
        setInputMessage('');

        // Start loading
        setIsLoading(true);

        try {
            let response;

            // Determine if this is a text, image, or combined query
            if (selectedImage && messageText) {
                // Combined text and image query
                response = await sendCombinedQuery(messageText, selectedImage);
            } else if (selectedImage) {
                // Image-only query
                response = await sendImageQuery(selectedImage);
            } else {
                // Text-only query
                response = await sendTextQuery(messageText);
            }

            // Add bot response to chat
            const botMessage = {
                id: Date.now(),
                content: response.answer,
                isUser: false,
                sources: response.sources || []
            };

            setMessages(prev => [...prev, botMessage]);
        } catch (error) {
            console.error('Error sending message:', error);
            showToastMessage(`Erreur: ${error.message}`, true);
        } finally {
            setIsLoading(false);
            setSelectedImage(null);
        }
    };

    // Handle document file upload success
    const handleUploadSuccess = (fileName) => {
        showToastMessage(`Le document "${fileName}" a été ajouté à la bibliothèque`);

        // Close the modal
        setShowUploadModal(false);
    };

    // Handle document upload error
    const handleUploadError = (errorMessage) => {
        showToastMessage(errorMessage, true);
    };

    // Handle quick document upload (without metadata)
    const handleQuickFileUpload = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        if (file.type === 'application/pdf') {
            setSelectedFile(file);
            setIsLoading(true);

            try {
                await uploadDocument(file);
                showToastMessage(`Le document "${file.name}" a été ajouté à la bibliothèque`);
            } catch (error) {
                console.error('Error uploading document:', error);
                showToastMessage('Erreur lors de l\'ajout du document', true);
            } finally {
                setIsLoading(false);
                setSelectedFile(null);
            }
        } else {
            showToastMessage('Veuillez sélectionner un fichier PDF', true);
        }
    };

    // This function is currently not used directly but kept for future implementation
    // eslint-disable-next-line no-unused-vars
    const handleImageQuery = async (file) => {
        if (!file) return;

        setIsLoading(true);

        try {
            const response = await sendImageQuery(file);

            // Add bot response to chat
            const botResponse = {
                id: Date.now() + 1,
                content: response.answer,
                isUser: false,
                sources: response.sources
            };

            setMessages(prev => [...prev, botResponse]);
        } catch (error) {
            console.error('Error sending image query:', error);
            showToastMessage('Erreur lors de l\'analyse de l\'image', true);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="flex h-screen bg-gray-900 text-white">
            {/* Sidebar */}
            <div className="w-64 flex flex-col bg-gray-900 border-r border-gray-800">
                {/* Logo and Brand */}
                <div className="p-4 flex items-center justify-center gap-2 border-b border-gray-800">
                    <img
                        src={CF_LOGO_PATH}
                        alt="CF Logo"
                        className="h-16 w-16 rounded-full"
                    />
                </div>

                {/* Simplified sidebar with only two buttons */}
                <div className="flex-1 flex flex-col p-4 gap-3">
                    {/* Enrichir la bibliothèque button */}
                    <button
                        onClick={() => setShowUploadModal(true)}
                        className="w-full flex items-center justify-center gap-2 bg-indigo-600 hover:bg-indigo-700 text-white py-3 px-4 rounded-lg transition-colors cursor-pointer"
                    >
                        <Upload size={18} />
                        <span>Enrichir la bibliothèque</span>
                    </button>

                    {/* A propos button */}
                    <button
                        className="w-full flex items-center justify-center gap-2 bg-gray-800 hover:bg-gray-700 text-white py-3 px-4 rounded-lg transition-colors"
                        onClick={() => {
                            const aboutMessage = {
                                id: Date.now(),
                                content: "# À propos de CFChat\n\nCFChat est un assistant conversationnel basé sur la **Récupération Augmentée de Génération (RAG)**. Il vous permet d'interagir avec vos documents et d'obtenir des réponses précises basées sur leurs contenus.\n\n## Fonctionnalités\n\n- Interrogation de documents textuels\n- Analyse d'images et de contenus multimodaux\n- Support des formats PDF avec extraction intelligente\n- Interface conversationnelle intuitive",
                                isUser: false,
                                sources: []
                            };
                            setMessages(prev => [...prev, aboutMessage]);
                        }}
                    >
                        <InfoIcon size={18} />
                        <span>A propos</span>
                    </button>
                </div>
            </div>

            {/* Main chat area */}
            <div className="flex-1 flex flex-col">
                {/* Chat header with CF logo and title */}
                <div className="p-4 border-b border-gray-800 flex justify-between items-center">
                    <div className="flex items-center gap-3">
                        <img
                            src={CF_LOGO_PATH}
                            alt="CF Logo"
                            className="h-8 w-8 rounded-full"
                        />
                        <h2 className="text-xl font-bold">CFChat</h2>
                    </div>
                    <div className="text-sm text-gray-400">
                        {activeChat} • {new Date().toLocaleDateString()} {new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </div>
                </div>

                {/* Messages area */}
                <div className="flex-1 overflow-y-auto p-6">
                    {messages.map(message => (
                        <MarkdownMessage
                            key={message.id}
                            isUser={message.isUser}
                            content={message.content}
                            sources={message.sources}
                        />
                    ))}
                    {isLoading && <LoadingIndicator />}
                    <div ref={messagesEndRef} />
                </div>

                {/* Input area */}
                <div className="border-t border-gray-800 p-4">
                    <form
                        className="flex flex-col gap-2"
                        onSubmit={(e) => {
                            e.preventDefault();
                            handleSendMessage();
                        }}
                    >
                        <div className="flex items-center gap-2">
                            <div className="flex items-center gap-2">
                                <label htmlFor="file-upload" className="cursor-pointer text-gray-400 hover:text-white">
                                    <Paperclip size={20} />
                                </label>
                                <input
                                    id="file-upload"
                                    type="file"
                                    accept=".pdf"
                                    onChange={handleQuickFileUpload}
                                    className="hidden"
                                />
                            </div>

                            <input
                                type="text"
                                value={inputMessage}
                                onChange={(e) => setInputMessage(e.target.value)}
                                placeholder="Send a message"
                                className="flex-1 bg-gray-800 rounded-lg border border-gray-700 py-3 px-4 focus:outline-none focus:border-blue-500 text-white"
                                disabled={isLoading}
                            />

                            <button
                                type="submit"
                                disabled={!inputMessage.trim() || isLoading}
                                className={`rounded-lg p-3 ${inputMessage.trim() && !isLoading
                                    ? 'bg-blue-600 hover:bg-blue-700'
                                    : 'bg-gray-800 cursor-not-allowed'
                                    } transition-colors`}
                            >
                                <Send size={20} />
                            </button>
                        </div>

                        {/* Image upload button */}
                        <button
                            type="button"
                            onClick={() => imageInputRef.current.click()}
                            className="hover:bg-gray-700 rounded p-2"
                            title="Ajouter une image à la question"
                        >
                            <Paperclip size={20} />
                        </button>

                        {/* Hidden file input for image upload */}
                        <input
                            type="file"
                            accept="image/*"
                            onChange={(e) => {
                                if (e.target.files[0]) {
                                    setSelectedImage(e.target.files[0]);
                                }
                            }}
                            ref={imageInputRef}
                            className="hidden"
                        />

                        {/* Image preview (if image is selected) */}
                        {selectedImage && (
                            <div className="relative mt-2">
                                <img
                                    src={URL.createObjectURL(selectedImage)}
                                    alt="Preview"
                                    className="max-h-32 rounded-md"
                                />
                                <button
                                    type="button"
                                    onClick={() => setSelectedImage(null)}
                                    className="absolute top-1 right-1 bg-red-500 rounded-full p-1"
                                    title="Supprimer l'image"
                                >
                                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                        <line x1="18" y1="6" x2="6" y2="18"></line>
                                        <line x1="6" y1="6" x2="18" y2="18"></line>
                                    </svg>
                                </button>
                            </div>
                        )}

                        {/* Display selected file name if any */}
                        {selectedFile && (
                            <div className="mt-2 text-sm text-gray-400">
                                Fichier sélectionné: {selectedFile.name}
                            </div>
                        )}
                    </form>
                </div>
            </div>

            {/* Toast notification */}
            {toast.visible && (
                <ToastNotification
                    message={toast.message}
                    isError={toast.isError}
                    onClose={() => setToast({ ...toast, visible: false })}
                />
            )}

            {/* Document Upload Modal */}
            <Modal
                isOpen={showUploadModal}
                onClose={() => setShowUploadModal(false)}
                title="Enrichir la bibliothèque"
                size="large"
            >
                <DocumentUpload
                    onUploadSuccess={handleUploadSuccess}
                    onUploadError={handleUploadError}
                />
            </Modal>
        </div>
    );
};

export default ChatInterface; 