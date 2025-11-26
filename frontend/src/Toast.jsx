import React, { useEffect, useState } from 'react';

const Toast = ({ message, type, onClose }) => {
  const [isVisible, setIsVisible] = useState(true);

  useEffect(() => {
    const timer = setTimeout(() => {
      setIsVisible(false);
      if (onClose) {
        onClose();
      }
    }, 3000); // Hide after 3 seconds

    return () => clearTimeout(timer);
  }, [onClose]);

  const bgColor = type === 'error' ? 'bg-red-500' : 'bg-green-500';

  if (!isVisible) return null;

  return (
    <div className={`fixed top-4 right-4 p-4 rounded-md shadow-lg text-white ${bgColor}`}>
      {message}
      <button onClick={() => setIsVisible(false)} className="ml-4 font-bold">
        X
      </button>
    </div>
  );
};

export default Toast; 