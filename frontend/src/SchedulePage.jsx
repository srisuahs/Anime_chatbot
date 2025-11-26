import React, { useState, useEffect } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from './AuthContext';
import Toast from './Toast';
import EpisodeSchedule from './EpisodeSchedule';

const SchedulePage = () => {
  const { token } = useAuth();
  const navigate = useNavigate();
  const [watchingList, setWatchingList] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showEnglish, setShowEnglish] = useState(true);
  const [toast, setToast] = useState({ message: '', type: '', isVisible: false });
  const [activeTab, setActiveTab] = useState('watching'); // 'watching' or 'schedule'

  useEffect(() => {
    if (!token) {
      navigate('/');
      return;
    }
    fetchWatchingList();
  }, [token, navigate]);

  const fetchWatchingList = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch('http://127.0.0.1:8000/mal/my-anime-list/watching', {
        headers: { 'Authorization': `Bearer ${token}` }
      });

      if (!response.ok) {
        const errorData = await response.json();
        if (response.status === 404) {
          throw new Error('Please connect your MyAnimeList account in Settings.');
        }
        throw new Error(errorData.detail || 'Failed to fetch anime list');
      }

      const data = await response.json();
      setWatchingList(data.data || []);
    } catch (err) {
      setError(err.message);
      setToast({
        message: `❌ Error: ${err.message}`,
        type: 'error',
        isVisible: true
      });
    } finally {
      setIsLoading(false);
    }
  };

  if (isLoading) {
    return (
      <div className="flex flex-col items-center justify-center h-screen bg-gray-900 text-white">
        <div className="text-4xl mb-4">📺</div>
        <p className="text-lg">Loading your anime list...</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Tabs */}
      <div className="sticky top-0 bg-gray-800 border-b border-gray-700 z-40">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex gap-4">
            <button
              onClick={() => setActiveTab('watching')}
              className={`py-4 px-6 font-semibold border-b-2 transition ${
                activeTab === 'watching'
                  ? 'border-indigo-500 text-indigo-400'
                  : 'border-transparent text-gray-400 hover:text-white'
              }`}
            >
              📺 My Watching List
            </button>
            <button
              onClick={() => setActiveTab('schedule')}
              className={`py-4 px-6 font-semibold border-b-2 transition ${
                activeTab === 'schedule'
                  ? 'border-indigo-500 text-indigo-400'
                  : 'border-transparent text-gray-400 hover:text-white'
              }`}
            >
              📅 Episode Schedule
            </button>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="p-6">
        {activeTab === 'watching' ? (
          // Watching List Tab
          <div className="max-w-7xl mx-auto">
            <div className="flex justify-between items-center mb-8 bg-gray-800 p-6 rounded-lg border border-gray-700 shadow-lg">
              <div>
                <h1 className="text-4xl font-bold text-indigo-400 mb-2">📺 My Watching List</h1>
                <p className="text-gray-400">Total anime: {watchingList.length}</p>
              </div>
              <div className="flex gap-3">
                <button
                  onClick={() => setShowEnglish(!showEnglish)}
                  className="px-4 py-2 bg-gray-700 rounded-lg hover:bg-indigo-600 transition font-semibold"
                >
                  {showEnglish ? '日本語' : 'English'}
                </button>
                <button
                  onClick={fetchWatchingList}
                  className="px-4 py-2 bg-indigo-600 rounded-lg hover:bg-indigo-700 transition font-semibold"
                >
                  🔄 Refresh
                </button>
              </div>
            </div>

            {error && (
              <div className="bg-red-900 border border-red-700 text-red-200 px-6 py-4 rounded-lg mb-6">
                <p className="font-semibold">❌ {error}</p>
                <button
                  onClick={fetchWatchingList}
                  className="mt-3 px-4 py-2 bg-red-700 rounded hover:bg-red-600 transition"
                >
                  Try Again
                </button>
              </div>
            )}

            {watchingList.length > 0 ? (
              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-6">
                {watchingList.map(item => {
                  const anime = item.node;
                  const listStatus = item.list_status;
                  const altTitles = anime.alternative_titles || {};
                  const displayTitle = showEnglish && altTitles.en ? altTitles.en : anime.title;
                  const japaneseTitle = altTitles.ja || anime.title;
                  const imageUrl = anime.main_picture?.large || anime.main_picture?.medium;
                  const progress = anime.num_episodes 
                    ? Math.round((listStatus.num_episodes_watched / anime.num_episodes) * 100) 
                    : 0;

                  return (
                    <div
                      key={anime.id}
                      className="bg-gray-800 rounded-lg overflow-hidden shadow-lg hover:shadow-2xl transition-all duration-300 border border-gray-700 hover:border-indigo-500 group"
                    >
                      <div className="relative h-72 bg-gray-900 overflow-hidden">
                        {imageUrl ? (
                          <>
                            <img
                              src={imageUrl}
                              alt={anime.title}
                              className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-300"
                            />
                            <div className="absolute inset-0 bg-black bg-opacity-40 group-hover:bg-opacity-30 transition-all"></div>
                          </>
                        ) : (
                          <div className="w-full h-full bg-gradient-to-br from-gray-700 to-gray-900 flex items-center justify-center">
                            <span className="text-5xl">🎬</span>
                          </div>
                        )}
                        
                        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black to-transparent p-3">
                          <div className="w-full bg-gray-700 rounded-full h-2 mb-2">
                            <div
                              className="bg-gradient-to-r from-green-500 to-green-600 h-2 rounded-full transition-all"
                              style={{ width: `${progress}%` }}
                            ></div>
                          </div>
                          <p className="text-xs text-gray-300">
                            {listStatus.num_episodes_watched} / {anime.num_episodes || '?'} episodes
                          </p>
                        </div>

                        {listStatus.score > 0 && (
                          <div className="absolute top-3 right-3 bg-gradient-to-r from-yellow-500 to-yellow-600 text-black px-3 py-1 rounded-full font-bold shadow-lg text-sm">
                            ⭐ {listStatus.score}
                          </div>
                        )}
                      </div>

                      <div className="p-4">
                        <h3 className="text-lg font-bold text-white line-clamp-2 mb-2">{displayTitle}</h3>
                        
                        {showEnglish && japaneseTitle && japaneseTitle !== displayTitle && (
                          <p className="text-xs text-gray-400 line-clamp-1 mb-3">{japaneseTitle}</p>
                        )}

                        <div className="space-y-2 text-sm text-gray-400">
                          {listStatus.start_date && (
                            <p>📅 Started: <span className="text-gray-300">{listStatus.start_date}</span></p>
                          )}
                          {anime.num_episodes && (
                            <p>🎬 Total: <span className="text-gray-300">{anime.num_episodes} episodes</span></p>
                          )}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="text-center py-20">
                <div className="text-6xl mb-4">📭</div>
                <p className="text-xl text-gray-400 mb-6">Your watching list is empty!</p>
                <Link
                  to="/"
                  className="inline-block px-6 py-3 bg-indigo-600 rounded-lg hover:bg-indigo-700 transition font-semibold"
                >
                  💬 Start adding anime in chat
                </Link>
              </div>
            )}
          </div>
        ) : (
          // Schedule Tab
          <EpisodeSchedule />
        )}
      </div>

      {toast.isVisible && (
        <Toast
          message={toast.message}
          type={toast.type}
          onClose={() => setToast({ ...toast, isVisible: false })}
        />
      )}
    </div>
  );
};

export default SchedulePage;
