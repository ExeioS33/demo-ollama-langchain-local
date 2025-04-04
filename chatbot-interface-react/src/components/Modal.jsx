import React, { useEffect } from 'react';
import { X } from 'lucide-react';

/**
 * Reusable Modal component for displaying content in an overlay
 * @param {Object} props
 * @param {boolean} props.isOpen - Whether the modal is visible
 * @param {Function} props.onClose - Function to call when modal is closed
 * @param {string} props.title - Modal title text
 * @param {React.ReactNode} props.children - Modal content
 * @param {string} props.size - Modal size (small, medium, large)
 */
const Modal = ({ isOpen, onClose, title, children, size = 'medium' }) => {
    // Handle ESC key to close modal
    useEffect(() => {
        const handleEsc = (e) => {
            if (e.key === 'Escape' && isOpen) {
                onClose();
            }
        };

        window.addEventListener('keydown', handleEsc);

        // Prevent scrolling when modal is open
        if (isOpen) {
            document.body.style.overflow = 'hidden';
        } else {
            document.body.style.overflow = 'unset';
        }

        return () => {
            window.removeEventListener('keydown', handleEsc);
            document.body.style.overflow = 'unset';
        };
    }, [isOpen, onClose]);

    // Don't render if not open
    if (!isOpen) return null;

    // Determine width based on size prop
    const getWidthClass = () => {
        switch (size) {
            case 'small': return 'max-w-md';
            case 'large': return 'max-w-4xl';
            case 'medium':
            default: return 'max-w-2xl';
        }
    };

    // Handle click on backdrop to close
    const handleBackdropClick = (e) => {
        if (e.target === e.currentTarget) {
            onClose();
        }
    };

    return (
        <div
            className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50 p-4"
            onClick={handleBackdropClick}
        >
            <div
                className={`bg-gray-900 rounded-xl shadow-xl w-full ${getWidthClass()} max-h-[90vh] flex flex-col overflow-hidden`}
                role="dialog"
                aria-modal="true"
                aria-labelledby="modal-title"
            >
                {/* Modal Header */}
                <div className="px-6 py-4 border-b border-gray-800 flex items-center justify-between">
                    <h3 id="modal-title" className="text-xl font-semibold">
                        {title}
                    </h3>
                    <button
                        onClick={onClose}
                        className="text-gray-400 hover:text-white p-1 rounded-full hover:bg-gray-800 transition-colors"
                        aria-label="Close"
                    >
                        <X size={20} />
                    </button>
                </div>

                {/* Modal Content */}
                <div className="p-6 flex-1 overflow-y-auto">
                    {children}
                </div>
            </div>
        </div>
    );
};

export default Modal; 