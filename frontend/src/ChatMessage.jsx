import React from 'react';

// A component to display a single chat message
function ChatMessage({ sender, text, type, isError }) {
  const isBot = sender === 'bot';

  // Avatar component to show a simple icon for the user or bot
  const Avatar = ({ isBot }) => (
    <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${isBot ? 'bg-gray-600' : 'bg-blue-500'}`}>
      <span className="text-white font-bold text-sm">
        {isBot ? 'AI' : 'You'}
      </span>
    </div>
  );

  return (
    <div className={`flex items-start gap-4 ${isBot ? '' : 'justify-end'}`}>
      {/* Show avatar on the left for the bot */}
      {isBot && <Avatar isBot={isBot} />}

      <div className={`flex flex-col ${isBot ? 'items-start' : 'items-end'}`}>
        <span className="font-bold text-sm text-gray-700 dark:text-gray-300 mb-1">
          {isBot ? 'AI Anime Assistant' : 'You'}
        </span>
        <div
          className={`relative max-w-lg px-4 py-3 rounded-2xl shadow ${
            isBot
              ? (isError ? 'bg-red-700 text-white' : 'bg-white dark:bg-gray-700 text-gray-800 dark:text-gray-200 rounded-tl-none')
              : 'bg-blue-500 text-white rounded-br-none'
          }`}
        >
          {/* Simple loading dots animation */}
          {type === 'loading' ? (
            <div className="flex items-center space-x-1">
              <span className="w-2 h-2 bg-gray-400 rounded-full animate-pulse [animation-delay:-0.3s]"></span>
              <span className="w-2 h-2 bg-gray-400 rounded-full animate-pulse [animation-delay:-0.15s]"></span>
              <span className="w-2 h-2 bg-gray-400 rounded-full animate-pulse"></span>
            </div>
          ) : (
            <p className="whitespace-pre-wrap">{text}</p>
          )}
        </div>
      </div>
      
      {/* Show avatar on the right for the user */}
      {!isBot && <Avatar isBot={isBot} />}
    </div>
  );
}

export default ChatMessage;

