import React, { useState, useEffect } from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { useAuth } from "./AuthContext";

const EpisodeSchedule = () => {
  const { token } = useAuth();
  const [currentDate, setCurrentDate] = useState(new Date());
  const [schedule, setSchedule] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [currentTime, setCurrentTime] = useState(new Date());

  // Update current time every second (reactive)
  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date());
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  // Fetch watching anime and filter by selected day
  useEffect(() => {
    if (token) {
      loadScheduleForDay(currentDate);
    }
  }, [currentDate, token]);

  const loadScheduleForDay = async (date) => {
    setIsLoading(true);
    try {
      // Get day of week (0=Sunday, 1=Monday, etc.)
      const dayOfWeek = date.getDay();
      const dayName = ["sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday"][dayOfWeek];

      // Fetch user's watching list with broadcast info
      const response = await fetch(
        `http://127.0.0.1:8000/mal/my-anime-list/watching?fields=list_status,broadcast,start_date,main_picture,num_episodes`,
        {
          headers: { Authorization: `Bearer ${token}` },
        }
      );

      if (response.ok) {
        const data = await response.json();
        
        // Filter anime that air on this day
        const airingToday = data.data
          .filter((item) => {
            const broadcast = item.node.broadcast;
            return (
              broadcast &&
              broadcast.day_of_the_week &&
              broadcast.day_of_the_week.toLowerCase() === dayName
            );
          })
          .map((item) => {
            const anime = item.node;
            const broadcast = anime.broadcast;
            return {
              id: anime.id,
              title: anime.title,
              englishTitle: anime.alternative_titles?.en || "",
              image: anime.main_picture?.large || anime.main_picture?.medium,
              time: broadcast.start_time || "Unknown",
              status: item.list_status,
              episodes: anime.num_episodes || "?",
            };
          })
          .sort((a, b) => {
            // Sort by time
            const timeA = parseInt(a.time.replace(":", ""));
            const timeB = parseInt(b.time.replace(":", ""));
            return timeA - timeB;
          });

        setSchedule(airingToday);
      }
    } catch (error) {
      console.error("Error loading schedule:", error);
      setSchedule([]);
    } finally {
      setIsLoading(false);
    }
  };

  const goToPreviousDay = () => {
    const newDate = new Date(currentDate);
    newDate.setDate(newDate.getDate() - 1);
    setCurrentDate(newDate);
  };

  const goToNextDay = () => {
    const newDate = new Date(currentDate);
    newDate.setDate(newDate.getDate() + 1);
    setCurrentDate(newDate);
  };

  const formatDate = (date) => {
    const options = { weekday: "short", month: "short", day: "numeric" };
    return date.toLocaleDateString("en-US", options);
  };

  const formatTime = (time24) => {
    if (!time24 || time24 === "Unknown") return "Unknown";
    const [hours, minutes] = time24.split(":");
    return `${String(hours).padStart(2, "0")}:${minutes}`;
  };

  // Convert JST time to IST time
  const convertJSTtoIST = (jstTimeStr) => {
    if (!jstTimeStr || jstTimeStr === "Unknown") return "Unknown";
    
    const [hours, minutes] = jstTimeStr.split(":").map(Number);
    // JST is UTC+9, IST is UTC+5:30
    // Difference is 3.5 hours (JST ahead of IST)
    let istHours = hours - 3;
    let istMinutes = minutes - 30;
    
    if (istMinutes < 0) {
      istHours -= 1;
      istMinutes += 60;
    }
    
    if (istHours < 0) {
      istHours += 24;
    }
    
    return `${String(istHours).padStart(2, "0")}:${String(istMinutes).padStart(2, "0")}`;
  };

  const getCurrentTimeIST = () => {
    return currentTime.toLocaleTimeString("en-IN", {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: true,
    });
  };

  const dayOfWeek = currentDate.toLocaleDateString("en-US", { weekday: "long" });
  const dateStr = formatDate(currentDate);

  return (
    <div className="bg-gray-900 min-h-screen p-6">
      {/* Header with Time */}
      <div className="flex justify-between items-center mb-8">
        <h2 className="text-3xl font-bold text-yellow-400">Estimated Schedule</h2>
        <div className="bg-white text-black px-4 py-2 rounded-full font-bold text-sm">
          {getCurrentTimeIST()}
        </div>
      </div>

      {/* Date Navigation */}
      <div className="flex items-center justify-between mb-8 overflow-x-auto">
        <button
          onClick={goToPreviousDay}
          className="p-2 hover:bg-gray-800 rounded-lg transition"
        >
          <ChevronLeft size={24} className="text-white" />
        </button>

        <div className="flex gap-3 overflow-x-auto pb-2">
          {[-3, -2, -1, 0, 1, 2, 3].map((offset) => {
            const date = new Date(currentDate);
            date.setDate(date.getDate() + offset);
            const isToday = offset === 0;
            const dayShort = date.toLocaleDateString("en-US", {
              weekday: "short",
            });
            const day = date.getDate();

            return (
              <button
                key={offset}
                onClick={() => {
                  const newDate = new Date(currentDate);
                  newDate.setDate(newDate.getDate() + offset);
                  setCurrentDate(newDate);
                }}
                className={`px-4 py-3 rounded-lg font-semibold whitespace-nowrap transition ${
                  isToday
                    ? "bg-yellow-400 text-black"
                    : "bg-gray-700 text-white hover:bg-gray-600"
                }`}
              >
                <div>{dayShort}</div>
                <div className="text-sm">Nov {day}</div>
              </button>
            );
          })}
        </div>

        <button
          onClick={goToNextDay}
          className="p-2 hover:bg-gray-800 rounded-lg transition"
        >
          <ChevronRight size={24} className="text-white" />
        </button>
      </div>

      {/* Schedule Content */}
      <div className="bg-gray-800 rounded-lg p-6">
        <h3 className="text-xl font-bold text-white mb-6">
          {dayOfWeek}, Nov {currentDate.getDate()}
        </h3>

        {isLoading ? (
          <div className="text-center py-12 text-gray-400">
            <p>Loading schedule...</p>
          </div>
        ) : schedule.length === 0 ? (
          <div className="text-center py-12 text-gray-400">
            <p className="text-lg">No anime airing today</p>
            <p className="text-sm mt-2">Check another day for upcoming episodes</p>
          </div>
        ) : (
          <div className="space-y-4">
            {schedule.map((anime) => (
              <div
                key={anime.id}
                className="flex items-center gap-4 bg-gray-700 rounded-lg p-4 hover:bg-gray-600 transition"
              >
                {/* Anime Image */}
                {anime.image && (
                  <img
                    src={anime.image}
                    alt={anime.title}
                    className="w-16 h-24 object-cover rounded-lg flex-shrink-0"
                  />
                )}

                {/* Anime Info */}
                <div className="flex-1 min-w-0">
                  <h4 className="text-lg font-bold text-white truncate">
                    {anime.title}
                  </h4>
                  {anime.englishTitle && (
                    <p className="text-sm text-gray-300 truncate">
                      {anime.englishTitle}
                    </p>
                  )}
                  <p className="text-xs text-gray-400 mt-1">
                    Episode {(anime.status?.num_episodes_watched || 0) + 1} / {anime.episodes}
                  </p>
                </div>

                {/* Air Times (JST and IST) */}
                <div className="text-right flex-shrink-0">
                  <p className="text-2xl font-bold text-indigo-400">
                    {formatTime(anime.time)}
                  </p>
                  <p className="text-xs text-gray-400">JST</p>
                  <p className="text-lg font-bold text-yellow-400 mt-1">
                    {convertJSTtoIST(anime.time)}
                  </p>
                  <p className="text-xs text-gray-400">IST</p>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Footer Info */}
      <div className="mt-6 text-center text-gray-400 text-sm">
        <p>Times shown: JST (Japan Standard Time) and IST (Indian Standard Time)</p>
        <p>Your current time (IST): {getCurrentTimeIST()}</p>
      </div>
    </div>
  );
};

export default EpisodeSchedule;
