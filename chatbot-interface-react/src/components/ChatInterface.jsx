import React, { useState } from 'react';
import { Send, PlusCircle, Paperclip, Image } from 'lucide-react';

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

// Sidebar chat item component
const ChatItem = ({ active, title, onClick }) => {
    return (
        <div
            onClick={onClick}
            className={`px-4 py-3 rounded-lg mb-1 cursor-pointer ${active ? 'bg-gray-800' : 'hover:bg-gray-800'}`}
        >
            <div className="flex items-center">
                <span className="text-sm text-gray-300">{title}</span>
            </div>
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

    // Chat history
    const chatHistory = [
        { id: 1, title: 'Today' },
        { id: 2, title: 'What can RAG do for me?', isSection: false },
        { id: 3, title: 'How does Reachats work?', isSection: false },
        { id: 4, title: 'Create a design system', isSection: false },
        { id: 5, title: 'How does Multimodal RAG work?', isSection: false },
        { id: 6, title: 'Last 7 days', isSection: true },
        { id: 7, title: 'Storybook and RAGblocks', isSection: false },
        { id: 8, title: 'Who is able to use Reachats?', isSection: false },
    ];

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
                {/* New chat button */}
                <div className="p-4">
                    <button
                        className="w-full flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-700 text-white py-2 px-4 rounded-lg transition-colors"
                        onClick={() => {
                            setMessages([{ id: 1, content: "Bonjour! Comment puis-je vous aider aujourd'hui?", isUser: false }]);
                            setActiveChat('New Session');
                        }}
                    >
                        <PlusCircle size={18} />
                        <span>New chat</span>
                    </button>
                </div>

                {/* Chat history */}
                <div className="flex-1 overflow-y-auto p-2">
                    {chatHistory.map(chat =>
                        chat.isSection ? (
                            <div key={chat.id} className="text-xs text-gray-500 font-medium px-4 py-2 uppercase">
                                {chat.title}
                            </div>
                        ) : (
                            <ChatItem
                                key={chat.id}
                                title={chat.title}
                                active={activeChat === chat.title}
                                onClick={() => setActiveChat(chat.title)}
                            />
                        )
                    )}
                </div>
            </div>

            {/* Main chat area */}
            <div className="flex-1 flex flex-col">
                {/* Chat header */}
                <div className="p-4 border-b border-gray-800 flex justify-between items-center">
                    <h2 className="text-xl font-medium">{activeChat}</h2>
                    <div className="text-sm text-gray-400">
                        {new Date().toLocaleDateString()} {new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
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
        </div>
    );
};

export default ChatInterface; 