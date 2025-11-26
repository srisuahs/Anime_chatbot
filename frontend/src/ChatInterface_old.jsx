import React, { useState, useEffect, useRef } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from './AuthContext';
import ChatMessage from './ChatMessage';
import AnimeRecommendationCard from './AnimeRecommendationCard';

const ChatInterface = () => {
  const { token, logout } = useAuth();
  const [messages, setMessages] = useState([
    { sender: 'bot', text: "Hello! I'm your AI Anime Assistant. How can I help you today?" }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [recommendations, setRecommendations] = useState(null);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, recommendations]);

  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = { sender: 'user', text: input };
    setMessages(prev => [...prev, userMessage]);
    const originalInput = input;
    setInput('');
    setIsLoading(true);
    setRecommendations(null);

    try {
      const response = await fetch('http://127.0.0.1:8000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ message: originalInput })
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to get response');
      }

      const data = await response.json();
      const botMessage = { sender: 'bot', text: data.response, action_type: data.action_type };
      setMessages(prev => [...prev, botMessage]);

      if (data.recommendations && data.recommendations.length > 0) {
        setRecommendations(data.recommendations);
      }
    } catch (error) {
      console.error('Chat error:', error);
      const errorMessage = { sender: 'bot', text: `Sorry, something went wrong: ${error.message}`, type: 'error' };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleAnimeAction = async (animeId, action) => {
    try {
      const response = await fetch(`http://127.0.0.1:8000/mal/anime/${animeId}/status?status=${action}`, {
        method: 'PUT',
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (response.ok) {
        const successMsg = { sender: 'bot', text: `✅ Added to your ${action === 'watching' ? 'Watching' : 'Plan to Watch'} list!` };
        setMessages(prev => [...prev, successMsg]);
      }
    } catch (error) {
      console.error('Action failed:', error);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gray-900 text-white">
      <div className="flex items-center justify-between bg-gray-800 p-4 shadow-md border-b border-gray-700">
        <h1 className="text-2xl font-bold text-indigo-400">🎌 AI Anime Assistant</h1>
        <div className="flex gap-3">
          <Link to="/schedule" className="px-4 py-2 bg-indigo-600 rounded-lg hover:bg-indigo-700 transition">
            📅 Schedule
          </Link>
          <Link to="/settings" className="px-4 py-2 bg-gray-700 rounded-lg hover:bg-gray-600 transition">
            ⚙️ Settings
          </Link>
          <button onClick={logout} className="px-4 py-2 bg-red-600 rounded-lg hover:bg-red-700 transition">
            🚪 Logout
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {messages.map((msg, idx) => (
          <ChatMessage key={idx} sender={msg.sender} text={msg.text} type={msg.type} />
        ))}
        {isLoading && <ChatMessage sender="bot" text="⏳ Typing..." type="loading" />}
        
        {recommendations && recommendations.length > 0 && (
          <div className="mt-8">
            <p className="text-lg font-bold text-indigo-400 mb-4">📺 Recommended Anime:</p>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {recommendations.map(anime => (
                <AnimeRecommendationCard 
                  key={anime.anime_id}
                  anime={anime}
                  onAction={handleAnimeAction}
                />
              ))}
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      <form onSubmit={handleSend} className="p-4 bg-gray-800 border-t border-gray-700">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Tell me about your anime..."
            className="flex-1 px-4 py-3 bg-gray-700 text-white rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={isLoading || !input.trim()}
            className="px-6 py-3 bg-indigo-600 rounded-lg hover:bg-indigo-700 disabled:bg-gray-600 disabled:cursor-not-allowed transition"
          >
            Send
          </button>
        </div>
      </form>
    </div>
  );
};

export default ChatInterface;
