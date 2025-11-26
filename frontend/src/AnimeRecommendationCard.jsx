import React, { useState } from 'react';

const AnimeRecommendationCard = ({ anime, onAction }) => {
  const [showEnglish, setShowEnglish] = useState(true);
  const [actionTaken, setActionTaken] = useState(null);

  const handleAction = async (action) => {
    await onAction(anime.anime_id, action);
    setActionTaken(action);
  };

  const displayTitle = showEnglish && anime.english_title ? anime.english_title : anime.title;
  const japaneseTitle = anime.japanese_title ? anime.japanese_title : anime.title;

  return (
    <div className="bg-gray-800 rounded-lg overflow-hidden shadow-lg hover:shadow-2xl transition-all duration-300 border border-gray-700 hover:border-indigo-500">
      <div className="relative h-64 bg-gray-900 overflow-hidden group">
        {anime.image_url ? (
          <>
            <img 
              src={anime.image_url} 
              alt={anime.title}
              className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-300"
            />
            <div className="absolute inset-0 bg-black bg-opacity-30 group-hover:bg-opacity-20 transition-all"></div>
          </>
        ) : (
          <div className="w-full h-full bg-gradient-to-br from-gray-700 to-gray-900 flex items-center justify-center">
            <span className="text-4xl">🎬</span>
          </div>
        )}
        
        {anime.score && (
          <div className="absolute top-3 right-3 bg-gradient-to-r from-yellow-500 to-yellow-600 text-black px-3 py-1 rounded-full font-bold shadow-lg">
            ⭐ {anime.score.toFixed(1)}
          </div>
        )}
      </div>

      <div className="p-4">
        <div className="flex justify-between items-start gap-2 mb-3">
          <div className="flex-1">
            <h3 className="text-lg font-bold text-white line-clamp-2">{displayTitle}</h3>
            {showEnglish && anime.japanese_title && (
              <p className="text-xs text-gray-400 line-clamp-1">{japaneseTitle}</p>
            )}
          </div>
          {anime.english_title && anime.japanese_title && (
            <button
              onClick={() => setShowEnglish(!showEnglish)}
              className="flex-shrink-0 text-xs bg-gray-700 px-2 py-1 rounded hover:bg-indigo-600 transition font-semibold"
              title={showEnglish ? 'Switch to Japanese' : 'Switch to English'}
            >
              {showEnglish ? 'JA' : 'EN'}
            </button>
          )}
        </div>

        {anime.synopsis && (
          <p className="text-xs text-gray-400 mb-4 line-clamp-3 leading-relaxed">
            {anime.synopsis}
          </p>
        )}

        {!actionTaken ? (
          <div className="flex gap-2">
            <button
              onClick={() => handleAction('watching')}
              className="flex-1 py-2 px-3 bg-gradient-to-r from-green-600 to-green-700 rounded-lg hover:from-green-700 hover:to-green-800 text-sm font-semibold transition shadow-md hover:shadow-lg"
            >
              ▶ Watching
            </button>
            <button
              onClick={() => handleAction('plan_to_watch')}
              className="flex-1 py-2 px-3 bg-gradient-to-r from-blue-600 to-blue-700 rounded-lg hover:from-blue-700 hover:to-blue-800 text-sm font-semibold transition shadow-md hover:shadow-lg"
            >
              📋 Plan
            </button>
          </div>
        ) : (
          <div className="py-2 px-3 bg-gradient-to-r from-gray-700 to-gray-800 rounded-lg text-center text-sm font-semibold text-green-400 border border-green-500">
            ✅ Added
          </div>
        )}
      </div>
    </div>
  );
};

export default AnimeRecommendationCard;
