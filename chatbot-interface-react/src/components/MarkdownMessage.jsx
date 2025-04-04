import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import SourcesDisplay from './SourcesDisplay';

/**
 * Message component that supports markdown rendering
 * @param {Object} props
 * @param {boolean} props.isUser - Whether the message is from the user
 * @param {string} props.content - The message content
 * @param {Array} props.sources - Optional sources data for bot responses
 */
const MarkdownMessage = ({ isUser, content, sources }) => {
    // CSS classes for message container based on sender
    const containerClass = `flex ${isUser ? 'justify-end' : 'justify-start'} mb-6`;
    const messageClass = `max-w-[80%] rounded-lg px-4 py-3 ${isUser ? 'bg-blue-600 text-white' : 'bg-gray-800 text-white'
        }`;

    // Custom components for markdown rendering
    const components = {
        // Style links 
        a: ({ node, children, ...props }) => (
            <a
                {...props}
                className="text-blue-300 hover:text-blue-200 underline"
                target="_blank"
                rel="noopener noreferrer"
            >
                {children}
            </a>
        ),
        // Style code blocks
        code: ({ node, inline, children, ...props }) => (
            inline ?
                <code className="bg-gray-700 px-1 rounded" {...props}>{children}</code> :
                <code className="block bg-gray-700 p-2 rounded-md overflow-x-auto my-2" {...props}>{children}</code>
        ),
        // Style headings
        h1: ({ node, children, ...props }) =>
            <h1 className="text-xl font-bold mt-4 mb-2" {...props}>{children}</h1>,
        h2: ({ node, children, ...props }) =>
            <h2 className="text-lg font-bold mt-3 mb-2" {...props}>{children}</h2>,
        h3: ({ node, children, ...props }) =>
            <h3 className="text-md font-bold mt-2 mb-1" {...props}>{children}</h3>,
        // Style lists
        ul: ({ node, children, ...props }) =>
            <ul className="list-disc pl-5 my-2" {...props}>{children}</ul>,
        ol: ({ node, children, ...props }) =>
            <ol className="list-decimal pl-5 my-2" {...props}>{children}</ol>,
        // Style blockquotes
        blockquote: ({ node, children, ...props }) =>
            <blockquote className="border-l-4 border-gray-600 pl-3 italic my-2" {...props}>{children}</blockquote>,
    };

    return (
        <div className={containerClass}>
            <div className={messageClass}>
                <div className="whitespace-pre-wrap">
                    <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        components={components}
                    >
                        {content}
                    </ReactMarkdown>
                </div>

                {!isUser && sources && sources.length > 0 && (
                    <SourcesDisplay sources={sources} />
                )}
            </div>
        </div>
    );
};

export default MarkdownMessage; 