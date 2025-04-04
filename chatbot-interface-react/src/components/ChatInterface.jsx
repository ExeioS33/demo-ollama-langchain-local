import React, { useState, useEffect } from 'react';
import { Send, Paperclip, Image, Upload, X, InfoIcon } from 'lucide-react';

// Message component to display chat messages
const Message = ({ isUser, content }) => {
    return (
        <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-6`}>
            <div
                className={`max-w-[80%] rounded-lg px-4 py-3 ${isUser
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-800 text-white'
                    }`}
            >
                {content}
            </div>
        </div>
    );
};

// Toast notification component
const ToastNotification = ({ message, onClose }) => {
    useEffect(() => {
        const timer = setTimeout(() => {
            onClose();
        }, 5000);

        return () => clearTimeout(timer);
    }, [onClose]);

    return (
        <div className="fixed top-4 right-4 bg-green-500 text-white px-4 py-3 rounded-lg shadow-lg flex items-center">
            <span>{message}</span>
            <button onClick={onClose} className="ml-3 text-white">
                <X size={18} />
            </button>
        </div>
    );
};

const ChatInterface = () => {
    const [messages, setMessages] = useState([
        { id: 1, content: "Bonjour! Comment puis-je vous aider aujourd'hui?", isUser: false }
    ]);
    const [inputMessage, setInputMessage] = useState('');
    const [selectedFile, setSelectedFile] = useState(null);
    const [activeChat, setActiveChat] = useState('Session 1');
    const [showToast, setShowToast] = useState(false);

    // Set document title
    useEffect(() => {
        document.title = "CFChat";
    }, []);

    // Handle sending a message
    const handleSendMessage = (e) => {
        e.preventDefault();
        if (inputMessage.trim() === '') return;

        const newMessage = {
            id: Date.now(),
            content: inputMessage,
            isUser: true
        };

        setMessages([...messages, newMessage]);
        setInputMessage('');

        // Simulate bot response
        setTimeout(() => {
            const botResponse = {
                id: Date.now() + 1,
                content: "Je suis en train de traiter votre requête...",
                isUser: false
            };
            setMessages(prev => [...prev, botResponse]);
        }, 1000);
    };

    // Handle file upload
    const handleFileUpload = (e) => {
        const file = e.target.files[0];
        if (file && file.type === 'application/pdf') {
            setSelectedFile(file);
            // Simulate success notification
            setTimeout(() => {
                setShowToast(true);

                // Also add a message to the chat
                const uploadMessage = {
                    id: Date.now(),
                    content: `Le document "${file.name}" a été ajouté à la bibliothèque.`,
                    isUser: false
                };
                setMessages(prev => [...prev, uploadMessage]);
            }, 1500);
        } else if (file) {
            alert("Veuillez sélectionner un fichier PDF.");
        }
    };

    return (
        <div className="flex h-screen bg-gray-900 text-white">
            {/* Sidebar */}
            <div className="w-64 flex flex-col bg-gray-900 border-r border-gray-800">
                {/* Logo and Brand */}
                <div className="p-4 flex items-center justify-center gap-2 border-b border-gray-800">
                    <img
                        src="https://media.licdn.com/dms/image/v2/D4E0BAQHfnhLIsEztTQ/company-logo_200_200/company-logo_200_200/0/1721126603488/groupecf_logo?e=2147483647&v=beta&t=HGdMzdb9xrP0sOORtLTJtATA_irZt2Hgj_sUZWiSJNc"
                        alt="CF Logo"
                        className="h-16 w-16 rounded-full"
                    />
                </div>

                {/* Simplified sidebar with only two buttons */}
                <div className="flex-1 flex flex-col p-4 gap-3">
                    {/* Enrichir la bibliothèque button */}
                    <label
                        htmlFor="document-upload"
                        className="w-full flex items-center justify-center gap-2 bg-indigo-600 hover:bg-indigo-700 text-white py-3 px-4 rounded-lg transition-colors cursor-pointer"
                    >
                        <Upload size={18} />
                        <span>Enrichir la bibliothèque</span>
                    </label>
                    <input
                        id="document-upload"
                        type="file"
                        accept=".pdf"
                        onChange={handleFileUpload}
                        className="hidden"
                    />

                    {/* A propos button */}
                    <button
                        className="w-full flex items-center justify-center gap-2 bg-gray-800 hover:bg-gray-700 text-white py-3 px-4 rounded-lg transition-colors"
                        onClick={() => {
                            const aboutMessage = {
                                id: Date.now(),
                                content: "CFChat est un assistant conversationnel basé sur la Récupération Augmentée de Génération (RAG). Il vous permet d'interagir avec vos documents et d'obtenir des réponses précises basées sur leurs contenus.",
                                isUser: false
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
                            src="https://media.licdn.com/dms/image/v2/D4E0BAQHfnhLIsEztTQ/company-logo_200_200/company-logo_200_200/0/1721126603488/groupecf_logo?e=2147483647&v=beta&t=HGdMzdb9xrP0sOORtLTJtATA_irZt2Hgj_sUZWiSJNc"
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
                        <Message
                            key={message.id}
                            isUser={message.isUser}
                            content={message.content}
                        />
                    ))}
                </div>

                {/* Input area */}
                <div className="border-t border-gray-800 p-4">
                    <form onSubmit={handleSendMessage} className="flex items-center gap-2">
                        <div className="flex items-center gap-2">
                            <label htmlFor="file-upload" className="cursor-pointer text-gray-400 hover:text-white">
                                <Paperclip size={20} />
                            </label>
                            <input
                                id="file-upload"
                                type="file"
                                accept=".pdf"
                                onChange={handleFileUpload}
                                className="hidden"
                            />
                            <label htmlFor="image-upload" className="cursor-pointer text-gray-400 hover:text-white">
                                <Image size={20} />
                            </label>
                            <input
                                id="image-upload"
                                type="file"
                                accept="image/*"
                                className="hidden"
                            />
                        </div>

                        <input
                            type="text"
                            value={inputMessage}
                            onChange={(e) => setInputMessage(e.target.value)}
                            placeholder="Send a message"
                            className="flex-1 bg-gray-800 rounded-lg border border-gray-700 py-3 px-4 focus:outline-none focus:border-blue-500 text-white"
                        />

                        <button
                            type="submit"
                            disabled={!inputMessage.trim()}
                            className={`rounded-lg p-3 ${inputMessage.trim() ? 'bg-blue-600 hover:bg-blue-700' : 'bg-gray-800 cursor-not-allowed'} transition-colors`}
                        >
                            <Send size={20} />
                        </button>
                    </form>

                    {/* Display selected file name if any */}
                    {selectedFile && (
                        <div className="mt-2 text-sm text-gray-400">
                            Fichier sélectionné: {selectedFile.name}
                        </div>
                    )}
                </div>
            </div>

            {/* Toast notification */}
            {showToast && (
                <ToastNotification
                    message="Document ajouté à la bibliothèque"
                    onClose={() => setShowToast(false)}
                />
            )}
        </div>
    );
};

export default ChatInterface; 