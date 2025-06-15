import React, { useState, useEffect, useRef} from 'react';
import { BrowserRouter as Router, Routes, Route, Link,  useNavigate, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, PieChart, Pie, Cell, ResponsiveContainer } from 'recharts';
import { FiUpload, FiFileText, FiAlertTriangle, FiCheckCircle } from 'react-icons/fi';
import './App.css';
// тут находится импорт функции sendMessage
import { sendMessage } from './api/chat'; // Add this import

function App() {
  const [isHovering, setIsHovering] = useState(false);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
  
  // Custom cursor effect
  useEffect(() => {
    const cursor = document.createElement('div');
    cursor.className = 'custom-cursor';
    document.body.appendChild(cursor);
    
    const follower = document.createElement('div');
    follower.className = 'cursor-follower';
    document.body.appendChild(follower);
    
    const moveCursor = (e) => {
      setMousePos({ x: e.clientX, y: e.clientY });
      cursor.style.left = `${e.clientX}px`;
      cursor.style.top = `${e.clientY}px`;
      
      setTimeout(() => {
        follower.style.left = `${e.clientX}px`;
        follower.style.top = `${e.clientY}px`;
      }, 100);
    };
    
    document.addEventListener('mousemove', moveCursor);
    
    return () => {
      document.removeEventListener('mousemove', moveCursor);
      document.body.removeChild(cursor);
      document.body.removeChild(follower);
    };
  }, []);

  // Create particles
  useEffect(() => {
    const particlesContainer = document.querySelector('.particles');
    if (!particlesContainer) return;
    
    const particleCount = 30;
    
    for (let i = 0; i < particleCount; i++) {
      const particle = document.createElement('div');
      particle.className = 'particle';
      
      const size = Math.random() * 10 + 5;
      const posX = Math.random() * 100;
      const duration = Math.random() * 10 + 5;
      const delay = Math.random() * 5;
      
      particle.style.width = `${size}px`;
      particle.style.height = `${size}px`;
      particle.style.left = `${posX}%`;
      particle.style.animationDuration = `${duration}s`;
      particle.style.animationDelay = `${delay}s`;
      particle.style.opacity = Math.random() * 0.5 + 0.1;
      
      particlesContainer.appendChild(particle);
    }
  }, []);

  return (
    <Router>
      <div className="app">
        <div className="particles"></div>
        
        <header className="navbar">
          <div className="logo">Connect</div>
          <nav className="nav-links">
            {['Главная', 'Чат', 'Стажировки', 'Расписание', 'Мероприятия', 'Профиль', 'Аналитика', 'Проверка'].map((item) => (
              <Link
                key={item}
                to={item === 'Главная' ? '/' : `/${item.toLowerCase()}`}
                onMouseEnter={() => setIsHovering(true)}
                onMouseLeave={() => setIsHovering(false)}
                className="nav-link"
              >
                {item}
              </Link>
            ))}
          </nav>
         
        </header>

        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/чат" element={<ChatPage />} />
          <Route path="/стажировки" element={<InternshipsPage />} />
          <Route path="/расписание" element={<SchedulePage />} />
          <Route path="/мероприятия" element={<EventsPage />} />
          <Route path="/профиль" element={<ProfilePage />} />
          <Route path="/auth" element={<AuthPage />} />
          <Route path="/аналитика" element={<AnalyticsPage />} />
          <Route path="/Проверка" element={<PlagiarismCheckPage />} />

        </Routes>
      </div>
    </Router>
  );
}



function HomePage() {
  const [isHovering, setIsHovering] = useState(false);

  return (
    <main className="main-content">
      <div className="text-content">
        <h1>
          Будущее <span className="gray">в</span> <br />
          ИИ <span className="black">Образовании</span>
        </h1>

        <p className="mission">
          Наша миссия — поддержка студентов на всём пути обучения с помощью ИИ.
        </p>
        
        <div className="cta-boxes">
          <Link 
            to="/чат" 
            className="box migrate"
            onMouseEnter={() => setIsHovering(true)}
            onMouseLeave={() => setIsHovering(false)}
          >
            <h3>Написать в Чат</h3>
      
          </Link>

          <Link 
            to="/auth" 
            className="box migrate"
            onMouseEnter={() => setIsHovering(true)}
            onMouseLeave={() => setIsHovering(false)}
          >
            <h3>Присоединиться</h3>
            <p>→</p>
          </Link>
          
        </div>
      </div>

      <div className="sphere large-sphere"></div>
      <div className="sphere small-sphere"></div>
      <div className="sphere tiny-sphere"></div>
    </main>
  );
}


const ChatPage = () => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [history, setHistory] = useState([
    'Как подготовиться к собеседованию?',
    'Какие технологии сейчас востребованы?',
    'Какой лучший фреймворк для React?'
  ]);
  const [isHovering, setIsHovering] = useState(false);
  const [conversationId, setConversationId] = useState(null);

  useEffect(() => {
    const storedId = localStorage.getItem('conversation_id');
    if (storedId) {
      setConversationId(storedId);
    }
  }, []);

  // Send a message and include the conversation ID if it exists
  const handleSendMessage = async () => {
      if (inputValue.trim()) {
        const newMessage = {
          id: Date.now(),
          text: inputValue,
          timestamp: new Date().toLocaleTimeString(),
          isUser: true
        };

        setMessages([...messages, newMessage]);
        setHistory([...history, inputValue]);
        setInputValue('');

        try {
          const response = await sendMessage(inputValue, history, conversationId);
          const botResponse = {
            id: Date.now() + 1,
            text: response.answer || "Ошибка получения ответа от сервера.",
            timestamp: new Date().toLocaleTimeString(),
            isUser: false
          };
          setMessages(prev => [...prev, botResponse]);
        } catch (error) {
          setMessages(prev => [
            ...prev,
            {
              id: Date.now() + 2,
              text: "Ошибка соединения с сервером.",
              timestamp: new Date().toLocaleTimeString(),
              isUser: false
            }
          ]);
        }
      }
    };

  const loadFromHistory = (text) => {
    const newMessage = {
      id: Date.now(),
      text: text,
      timestamp: new Date().toLocaleTimeString(),
      isUser: true
    };
    setMessages([...messages, newMessage]);
    
    setTimeout(() => {
      const botResponse = {
        id: Date.now() + 1,
        text: `Ответ на ваш предыдущий запрос "${text}"`,
        timestamp: new Date().toLocaleTimeString(),
        isUser: false
      };
      setMessages(prev => [...prev, botResponse]);
    }, 1000);
  };

  return (
    <div className="chat-page">
      <div className="chat-history">
        <h3>История запросов</h3>
        <div className="history-list">
          {history.map((item, index) => (
            <div 
              key={index} 
              className="history-item"
              onClick={() => loadFromHistory(item)}
              onMouseEnter={() => setIsHovering(true)}
              onMouseLeave={() => setIsHovering(false)}
            >
              {item.length > 30 ? `${item.substring(0, 30)}...` : item}
            </div>
          ))}
        </div>
      </div>
      
      <div className="chat-container">
        <div className="chat-messages">
          {messages.length === 0 ? (
            <div className="empty-chat">
              <h3>Чат с ИИ-ассистентом</h3>
              <p>Задайте вопрос или выберите из истории запросов</p>
            </div>
          ) : (
            messages.map((message) => (
              <div 
                key={message.id} 
                className={`message ${message.isUser ? 'user' : 'bot'}`}
              >
                <div className="message-content">
                  {message.text}
                  <span className="message-time">{message.timestamp}</span>
                </div>
              </div>
            ))
          )}
        </div>
        
        <div className="chat-input">
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder="Введите ваш вопрос..."
            onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
          />
          <button 
            onClick={handleSendMessage}
            onMouseEnter={() => setIsHovering(true)}
            onMouseLeave={() => setIsHovering(false)}
          >
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M22 2L11 13" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M22 2L15 22L11 13L2 9L22 2Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
};



const InternshipsPage = () => {
  const [selectedInternship, setSelectedInternship] = useState(null);
  const [coverLetter, setCoverLetter] = useState('');
  const [showResults, setShowResults] = useState(false);
  const [applications, setApplications] = useState([]);
  const [isGenerating, setIsGenerating] = useState(false);

  const internships = [
    {
      id: 1,
      title: "Frontend-разработчик (React)",
      company: "TechCorp",
      logo: "🏢",
      location: "Алматы / Оффлайн",
      description: "Разработка пользовательских интерфейсов для корпоративных решений. Вы будете работать в команде опытных разработчиков над созданием современных веб-приложений.",
      skills: "JavaScript, React, Redux, HTML/CSS",
      spots: 3,
      salary: "от 250 000₸",
      duration: "1 год+"
    },
    {
      id: 2,
      title: "Data Science стажер",
      company: "DataPro",
      logo: "📊",
      location: "Астана",
      description: "Анализ больших данных и построение ML-моделей для решения бизнес-задач. Работа с реальными проектами под руководством ментора.",
      skills: "Python, Pandas, NumPy, SQL",
      spots: 2,
      salary: "от 300 000₸",
      duration: "1 год+"
    },
    {
      id: 3,
      title: "Backend-разработчик (Node.js)",
      company: "WebSolutions",
      logo: "🔧",
      location: "Удаленно",
      description: "Разработка API и серверной логики для веб-приложений. Идеальная возможность для тех, кто хочет углубиться в backend-разработку.",
      skills: "Node.js, Express, MongoDB, REST API",
      spots: 5,
      salary: "от 200 000₸",
      duration: "3-4 курс университета"
    }
  ];

  const internshipResults = [
    { id: 1, status: "Рассматривается", feedback: "Ваша заявка находится на рассмотрении. Результаты будут через 1-2 недели." },
    { id: 2, status: "Принято", feedback: "Поздравляем! Вы приняты на стажировку. HR-менеджер свяжется с вами в течение 3 рабочих дней." },
    { id: 3, status: "Отклонено", feedback: "К сожалению, ваша заявка отклонена. Попробуйте податься на другие позиции, соответствующие вашему опыту." }
  ];

  const handleApply = (internship) => {
    setSelectedInternship(internship);
    setCoverLetter(`Я хочу подать заявку на стажировку "${internship.title}" в компании ${internship.company}, потому что...`);
  };

  const generateCoverLetter = () => {
    setIsGenerating(true);
    setTimeout(() => {
      setCoverLetter(`Уважаемая команда ${selectedInternship.company}!

Меня зовут [Ваше имя], и я хотел бы подать заявку на стажировку "${selectedInternship.title}".

Мои навыки в области ${selectedInternship.skills.split(', ')[0]} и мой интерес к ${selectedInternship.description.split('. ')[0].toLowerCase()} делают меня отличным кандидатом на эту позицию.

Во время моего обучения/работы я [опишите ваш соответствующий опыт]. Я уверен, что смогу внести значительный вклад в ваш проект.

Буду рад возможности обсудить, как мои навыки и опыт могут быть полезны вашей команде.

С уважением,
[Ваше имя]
[Контактная информация]`);
      setIsGenerating(false);
    }, 1500);
  };

  const submitApplication = () => {
    const newApplication = {
      internshipId: selectedInternship.id,
      title: selectedInternship.title,
      company: selectedInternship.company,
      status: "Рассматривается",
      date: new Date().toLocaleDateString('ru-RU'),
      coverLetter: coverLetter
    };
    
    setApplications([...applications, newApplication]);
    alert(`Заявка на стажировку "${selectedInternship.title}" успешно отправлена!`);
    setSelectedInternship(null);
    setCoverLetter('');
  };

  return (
    <div className="internships-page">
      <div className="internships-header">
        <h1>Доступные стажировки</h1>
        <button 
          className="results-btn"
          onClick={() => setShowResults(!showResults)}
        >
          {showResults ? "← Вернуться к стажировкам" : "Мои результаты"}
        </button>
      </div>

      {showResults ? (
        <div className="results-section">
          <h2 className="section-title">Результаты моих заявок</h2>
          {applications.length === 0 ? (
            <div className="empty-results">
              <p>У вас пока нет отправленных заявок</p>
              <button 
                className="primary-btn"
                onClick={() => setShowResults(false)}
              >
                Посмотреть стажировки
              </button>
            </div>
          ) : (
            <div className="results-list">
              {applications.map((app, index) => {
                const result = internshipResults.find(r => r.id === app.internshipId) || {};
                return (
                  <div key={index} className={`result-card ${result.status.toLowerCase()}`}>
                    <div className="result-header">
                      <h3>{app.title}</h3>
                      <span className="company">{app.company}</span>
                      <div className={`status-badge ${result.status.toLowerCase()}`}>
                        {result.status}
                      </div>
                    </div>
                    <div className="result-content">
                      <p className="feedback">{result.feedback}</p>
                      <p className="date">Дата подачи: {app.date}</p>
                      <button 
                        className="details-btn"
                        onClick={() => {
                          setSelectedInternship(internships.find(i => i.id === app.internshipId));
                          setCoverLetter(app.coverLetter);
                        }}
                      >
                        Посмотреть заявку
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      ) : (
        <div className="internships-container">
          <div className="internships-list">
            {internships.map(internship => (
              <div key={internship.id} className="internship-card">
                <div className="internship-header">
                  <div className="company-logo">{internship.logo}</div>
                  <div className="internship-info">
                    <h2>{internship.title}</h2>
                    <span className="company">{internship.company}</span>
                    <span className="location">{internship.location}</span>
                  </div>
                </div>
                
                <div className="internship-details">
                  <p className="description">{internship.description}</p>
                  
                  <div className="details-grid">
                    <div className="detail-item">
                      <span className="detail-label">Требуемые навыки:</span>
                      <span className="detail-value">{internship.skills}</span>
                    </div>
                    <div className="detail-item">
                      <span className="detail-label">Зарплата:</span>
                      <span className="detail-value salary">{internship.salary}</span>
                    </div>
                    <div className="detail-item">
                      <span className="detail-label">Опыт:</span>
                      <span className="detail-value">{internship.duration}</span>
                    </div>
                  </div>
                </div>
                
                <div className="internship-footer">
                  <div className="spots-available">
                    <span className="spots-label">Доступно мест:</span>
                    <span className="spots-count">{internship.spots}</span>
                  </div>
                  <button 
                    className="apply-btn"
                    onClick={() => handleApply(internship)}
                  >
                    Подать заявку
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {selectedInternship && (
        <div className="application-modal">
          <div className="modal-overlay" onClick={() => setSelectedInternship(null)}></div>
          <div className="modal-content">
            <button 
              className="close-modal"
              onClick={() => setSelectedInternship(null)}
            >
              &times;
            </button>
            
            <div className="modal-header">
              <h2>Заявка на стажировку</h2>
              <div className="internship-title">{selectedInternship.title} в {selectedInternship.company}</div>
            </div>
            
            <div className="application-form">
              <div className="form-section">
                <h3 className="section-title">Сопроводительное письмо</h3>
                <p className="section-description">Расскажите, почему вы хотите пройти эту стажировку и чем можете быть полезны</p>
                
                <div className="cover-letter-editor">
                  <textarea
                    value={coverLetter}
                    onChange={(e) => setCoverLetter(e.target.value)}
                    placeholder="Напишите ваше сопроводительное письмо здесь..."
                    className="letter-textarea"
                  />
                  <div className="editor-toolbar">
                    <button 
                      className={`generate-btn ${isGenerating ? 'generating' : ''}`}
                      onClick={generateCoverLetter}
                      disabled={isGenerating}
                    >
                      {isGenerating ? (
                        <>
                          <span className="spinner"></span>
                          Генерация...
                        </>
                      ) : (
                        <>
                          <span className="icon">✨</span>
                          Сгенерировать письмо
                        </>
                      )}
                    </button>
                    <div className="word-count">{coverLetter.length}/1000 символов</div>
                  </div>
                </div>
              </div>
              
              <div className="form-actions">
                <button 
                  className="submit-btn"
                  onClick={submitApplication}
                  disabled={coverLetter.length < 50}
                >
                  Отправить заявку
                </button>
                <button 
                  className="cancel-btn"
                  onClick={() => setSelectedInternship(null)}
                >
                  Сохранить черновик
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

const SchedulePage = () => {
  // Состояние для текущей недели
  const [currentWeek, setCurrentWeek] = useState(new Date());
  const [schedule, setSchedule] = useState([]);
  const [isLoading, setIsLoading] = useState(true);

  // Загрузка расписания (в реальном приложении - API запрос)
  useEffect(() => {
    // Имитация загрузки данных
    setTimeout(() => {
      setSchedule(generateMockSchedule());
      setIsLoading(false);
    }, 800);
  }, [currentWeek]);

  // Генерация мокового расписания
  const generateMockSchedule = () => {
    const days = ['Понедельник', 'Вторник', 'Среда', 'Четверг', 'Пятница', 'Суббота'];
    const eventTypes = ['Лекция', 'Практика', 'Консультация', 'Самостоятельная работа'];
    const courses = ['React Advanced', 'Node.js Basics', 'Алгоритмы и структуры данных', 'Базы данных'];
    
    return days.map(day => {
      const eventsCount = Math.floor(Math.random() * 3) + 1; // 1-3 события в день
      const dayEvents = [];
      
      for (let i = 0; i < eventsCount; i++) {
        const hour = 9 + Math.floor(Math.random() * 8); // С 9 до 17
        const minutes = Math.random() > 0.5 ? 0 : 30;
        const duration = [60, 90, 120][Math.floor(Math.random() * 3)]; // 1, 1.5 или 2 часа
        
        dayEvents.push({
          id: `${day}-${i}`,
          title: courses[Math.floor(Math.random() * courses.length)],
          type: eventTypes[Math.floor(Math.random() * eventTypes.length)],
          time: `${hour}:${minutes === 0 ? '00' : minutes}`,
          duration,
          location: ['Zoom', 'Аудитория 304', 'Аудитория 412', 'Онлайн'][Math.floor(Math.random() * 4)],
          instructor: ['Иванов А.А.', 'Петрова С.И.', 'Сидоров В.В.'][Math.floor(Math.random() * 3)],
          isCompleted: Math.random() > 0.7,
          isImportant: Math.random() > 0.8
        });
      }
      
      // Сортируем события по времени
      dayEvents.sort((a, b) => a.time.localeCompare(b.time));
      
      return {
        dayName: day,
        date: new Date(), // В реальном приложении - дата дня недели
        events: dayEvents
      };
    });
  };

  // Навигация по неделям
  const prevWeek = () => {
    setCurrentWeek(new Date(currentWeek.setDate(currentWeek.getDate() - 7)));
    setIsLoading(true);
  };

  const nextWeek = () => {
    setCurrentWeek(new Date(currentWeek.setDate(currentWeek.getDate() + 7)));
    setIsLoading(true);
  };

  const goToToday = () => {
    setCurrentWeek(new Date());
    setIsLoading(true);
  };

  // Форматирование даты
  const formatDateRange = (date) => {
    const start = new Date(date);
    start.setDate(start.getDate() - start.getDay() + 1); // Начало недели (понедельник)
    
    const end = new Date(start);
    end.setDate(start.getDate() + 5); // Конец недели (пятница)
    
    return `${start.toLocaleDateString('ru-RU', { day: 'numeric', month: 'long' })} - ${end.toLocaleDateString('ru-RU', { day: 'numeric', month: 'long' })}`;
  };

  return (
    <div className="schedule-page">
      <div className="schedule-header">
        <h1>Мое расписание</h1>
        <div className="week-navigation">
          <button onClick={prevWeek} className="nav-button">
            &lt; Предыдущая
          </button>
          <div className="current-week">
            {formatDateRange(currentWeek)}
            <button onClick={goToToday} className="today-button">
              Сегодня
            </button>
          </div>
          <button onClick={nextWeek} className="nav-button">
            Следующая &gt;
          </button>
        </div>
      </div>

      {isLoading ? (
        <div className="loading-spinner">
          <div className="spinner"></div>
          <p>Загрузка расписания...</p>
        </div>
      ) : (
        <div className="schedule-grid">
          {schedule.map((daySchedule) => (
            <div key={daySchedule.dayName} className="day-column">
              <div className="day-header">
                <h3>{daySchedule.dayName}</h3>
                <span className="day-date">
                  {daySchedule.date.toLocaleDateString('ru-RU', { day: 'numeric', month: 'short' })}
                </span>
              </div>
              
              <div className="events-list">
                {daySchedule.events.length > 0 ? (
                  daySchedule.events.map((event) => (
                    <div 
                      key={event.id} 
                      className={`event-card ${event.isCompleted ? 'completed' : ''} ${event.isImportant ? 'important' : ''}`}
                    >
                      <div className="event-time">
                        {event.time} - {calculateEndTime(event.time, event.duration)}
                      </div>
                      <h4 className="event-title">{event.title}</h4>
                      <div className="event-details">
                        <span className={`event-type ${event.type.toLowerCase()}`}>
                          {event.type}
                        </span>
                        <span className="event-location">
                          {event.location}
                        </span>
                      </div>
                      <div className="event-instructor">
                        Преподаватель: {event.instructor}
                      </div>
                      {event.isImportant && (
                        <div className="important-badge">Важно!</div>
                      )}
                    </div>
                  ))
                ) : (
                  <div className="no-events">
                    Нет занятий
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

const calculateEndTime = (startTime, duration) => {
  const [hours, minutes] = startTime.split(':').map(Number);
  const totalMinutes = hours * 60 + minutes + duration;
  const endHours = Math.floor(totalMinutes / 60);
  const endMinutes = totalMinutes % 60;
  return `${endHours}:${endMinutes < 10 ? '0' + endMinutes : endMinutes}`;
};

const EventsPage = () => {
  const [activeTab, setActiveTab] = useState('upcoming');
  const [registeredEvents, setRegisteredEvents] = useState([]);
  const [expandedEvent, setExpandedEvent] = useState(null);

  const eventsData = [
    {
      id: 1,
      title: "День открытых клубов",
      date: new Date(),
      time: "10:00-16:00",
      location: "Главный холл",
      description: "Знакомство со всеми студенческими клубами университета. IT-клуб, Дебаты, Фотоклуб и другие.",
      type: "club",
      emoji: "🎭",
      seats: { total: 150, booked: 112 },
      deadline: "до 18:00",
      color: "#6366F1"
    },
    {
      id: 2,
      title: "Регистрация на сессию",
      date: new Date(Date.now() + 86400000),
      time: "09:00-17:00",
      location: "Личный кабинет",
      description: "Обязательная регистрация на летнюю экзаменационную сессию для всех курсов.",
      type: "exam",
      emoji: "📚",
      seats: { total: 500, booked: 327 },
      deadline: "до 23:59",
      color: "#EC4899"
    },
    {
      id: 3,
      title: "Мастер-класс IELTS 9.0",
      date: new Date(Date.now() + 172800000),
      time: "15:00-17:30",
      location: "Коворкинг №3",
      description: "Разбор стратегий сдачи от студента, получившего максимальный балл. Включает Q&A сессию.",
      type: "masterclass",
      emoji: "🎯",
      seats: { total: 30, booked: 28 },
      deadline: "до 12:00",
      color: "#F59E0B"
    },
    {
      id: 4,
      title: "Хакатон 'Код будущего'",
      date: new Date("2023-05-21"),
      time: "10:00-22:00",
      location: "IT-центр",
      description: "24-часовое соревнование по разработке IT-решений. Призы от технологических партнеров.",
      type: "hackathon",
      emoji: "💻",
      seats: { total: 50, booked: 49 },
      deadline: "до 20:00 20 мая",
      color: "#10B981"
    }
  ];

  const handleRegister = (id) => {
    setRegisteredEvents([...registeredEvents, id]);
    setExpandedEvent(null);
  };

  const handleExpand = (id) => {
    setExpandedEvent(expandedEvent === id ? null : id);
  };

  const upcomingEvents = eventsData.filter(e => !registeredEvents.includes(e.id));
  const myEvents = eventsData.filter(e => registeredEvents.includes(e.id));

  return (
    <div className="events-container-2">
      <header className="events-header-2">
        <motion.h1 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          Календарь событий
        </motion.h1>
        
        <div className="tabs-2">
          {['upcoming', 'my'].map(tab => (
            <button
              key={tab}
              className={`tab-2 ${activeTab === tab ? 'active-2' : ''}`}
              onClick={() => setActiveTab(tab)}
            >
              {tab === 'upcoming' ? 'Предстоящие' : 'Мои события'}
              {tab === 'my' && myEvents.length > 0 && (
                <span className="badge-2">{myEvents.length}</span>
              )}
            </button>
          ))}
        </div>
      </header>

      <div className="events-grid-2">

      {(activeTab === 'upcoming' ? upcomingEvents : myEvents).map(event => (
          <div 
            key={event.id}
            className={`event-card-2 ${expandedEvent === event.id ? 'expanded-2' : ''}`}
            style={{ borderLeft: `4px solid ${event.color}` }}
          > <div className="event-header-2" onClick={() => handleExpand(event.id)}>
          <div className="event-emoji-2">{event.emoji}</div>
          <div className="event-main-2">
            <h3>{event.title}</h3>
            <div className="event-meta-2">
              <span>{event.date.toLocaleDateString('ru-RU', { weekday: 'short', day: 'numeric', month: 'short' })}</span>
              <span>{event.time}</span>
              <span>{event.location}</span>
            </div>
          </div>
          <div className={`event-status-2 ${event.seats.booked >= event.seats.total ? 'full-2' : ''}`}>
            {event.seats.booked >= event.seats.total ? 'Мест нет' : `${event.seats.total - event.seats.booked} мест`}
          </div>
        </div>
     <AnimatePresence>
              {expandedEvent === event.id && (
                <motion.div
                  className="event-details-2"
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  transition={{ duration: 0.2 }}
                >
                  <p>{event.description}</p>
                  <div className="event-footer-2">
                    <span className="deadline-2">⏳ {event.deadline}</span>
                    {activeTab === 'upcoming' && (
                      <button
                        className={`register-btn-2 ${event.seats.booked >= event.seats.total ? 'disabled-2' : ''}`}
                        onClick={(e) => {
                          e.stopPropagation();
                          handleRegister(event.id);
                        }}
                        disabled={event.seats.booked >= event.seats.total}
                      >
                        {registeredEvents.includes(event.id) ? 'Зарегистрирован' : 'Записаться'}
                      </button>
                    )}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        ))}
      </div>
    </div>
  );
};


const AuthPage = () => {
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [name, setName] = useState('');
  const [error, setError] = useState('');
  const navigate = useNavigate();

  // Предустановленные пользователи
  const predefinedUsers = [
    { 
      email: 'student@edu.ru', 
      password: 'student123', 
      role: 'student',
      name: 'Смирнов Алексей',
      faculty: 'Факультет компьютерных наук',
      course: '3 курс',
      gpa: '4.7'
    },
    { 
      email: 'teacher@edu.ru', 
      password: 'teacher123', 
      role: 'teacher',
      name: 'Петрова Мария Ивановна',
      department: 'Кафедра программной инженерии',
      position: 'Доцент'
    },
    { 
      email: 'admin@edu.ru', 
      password: 'admin123', 
      role: 'admin',
      name: 'Козлов Дмитрий Сергеевич',
      position: 'Системный администратор'
    },
    { 
      email: 'applicant@edu.ru', 
      password: 'applicant123', 
      role: 'applicant',
      name: 'Новикова Екатерина',
      status: 'На рассмотрении'
    }
  ];


   const handleSubmit = (e) => {
    e.preventDefault();
    setError('');

    if (isLogin) {
      const user = predefinedUsers.find(
        user => user.email === email && user.password === password
      );

      if (user) {
        // Сохраняем полные данные пользователя
        localStorage.setItem('user', JSON.stringify(user));
        // Перенаправляем с указанием роли
        navigate(`/profile?role=${user.role}`);
      } else {
        setError('Неверный email или пароль');
      }
    } else {
      // Логика регистрации остается прежней
      if (predefinedUsers.some(user => user.email === email)) {
        setError('Пользователь с таким email уже существует');
      } else {
        setError('Регистрация новых пользователей временно недоступна');
      }
    }
  };


  const toggleAuthMode = () => {
    setIsLogin(!isLogin);
    setError('');
  };

  return (
    <motion.div 
      className="auth-container"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      <div className="auth-card">
        <div className="auth-header">
          <h2>{isLogin ? 'Вход в аккаунт' : 'Регистрация'}</h2>
          <div className="auth-switch">
            {isLogin ? 'Нет аккаунта?' : 'Уже есть аккаунт?'}
            <button onClick={toggleAuthMode}>
              {isLogin ? 'Зарегистрироваться' : 'Войти'}
            </button>
          </div>
        </div>

        <form onSubmit={handleSubmit}>
          {!isLogin && (
            <div className="form-group">
              <label>ФИО</label>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Иванов Иван Иванович"
                required
              />
            </div>
          )}

          <div className="form-group">
            <label>Email</label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="example@edu.ru"
              required
            />
          </div>

          <div className="form-group">
            <label>Пароль</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••"
              required
              minLength="6"
            />
          </div>

          {error && <div className="error-message">{error}</div>}

          <button type="submit" className="submit-button">
            {isLogin ? 'Войти' : 'Зарегистрироваться'}
          </button>
        </form>

        
      </div>
    </motion.div>
  );
};



<Route path="/auth" element={<AuthPage />} />



const ProfilePage = () => {
  const location = useLocation();
  const searchParams = new URLSearchParams(location.search);
  const roleFromURL = searchParams.get('role');
  
  // Получаем данные пользователя из localStorage
  const storedUser = JSON.parse(localStorage.getItem('user'));
  
  // Определяем роль: сначала из URL, потом из сохраненных данных
  const role = roleFromURL || storedUser?.role || 'student';
  
  // Уникальные данные для каждой роли
  const roleProfiles = {
    student: {
      name: storedUser?.name || 'Смирнов Алексей',
      email: storedUser?.email || 'smirnov@university.edu',
      faculty: storedUser?.faculty || 'Факультет компьютерных наук',
      course: storedUser?.course || '3 курс',
      gpa: storedUser?.gpa || '4.7',
      skills: ['JavaScript', 'React', 'Python', 'SQL'],
      bio: 'Увлекаюсь веб-разработкой и анализом данных. Ищу стажировку в IT-компании.',
      avatar: null,
      resume: null
    },
    teacher: {
      name: storedUser?.name || 'Петрова Мария Ивановна',
      email: storedUser?.email || 'petrova@university.edu',
      department: storedUser?.department || 'Кафедра программной инженерии',
      position: storedUser?.position || 'Доцент',
      degree: 'Кандидат технических наук',
      courses: ['Веб-технологии', 'Базы данных', 'Машинное обучение'],
      bio: 'Преподаю с 2015 года. Научные интересы: искусственный интеллект и компьютерное зрение.',
      avatar: null
    },
    admin: {
      name: storedUser?.name || 'Козлов Дмитрий Сергеевич',
      email: storedUser?.email || 'admin@university.edu',
      position: storedUser?.position || 'Системный администратор',
      accessLevel: 'Полный доступ',
      lastLogin: new Date().toLocaleString(),
      bio: 'Отвечаю за работу образовательного портала и смежных систем.',
      avatar: null
    },
    applicant: {
      name: storedUser?.name || 'Новикова Екатерина',
      email: storedUser?.email || 'novikova@gmail.com',
      status: storedUser?.status || 'На рассмотрении',
      appliedFaculty: 'Факультет искусственного интеллекта',
      examResults: 'Математика: 92, Информатика: 88, Русский язык: 85',
      motivationLetter: 'Мечтаю разрабатывать системы компьютерного зрения для медицины.',
      documents: null,
      avatar: null
    }
  };

  // Используем данные для текущей роли
  const [user, setUser] = useState(roleProfiles[role]);
  const [editMode, setEditMode] = useState(false);
  const fileInputRef = useRef(null);
  const avatarInputRef = useRef(null);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setUser(prev => ({ ...prev, [name]: value }));
  };

  const handleAvatarUpload = (e) => {
    const file = e.target.files[0];
    if (file?.type.match('image.*')) {
      const reader = new FileReader();
      reader.onload = (event) => {
        setUser(prev => ({ ...prev, avatar: event.target.result }));
      };
      reader.readAsDataURL(file);
    }
  };

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file?.type === 'application/pdf') {
      if (role === 'student') {
        setUser(prev => ({ ...prev, resume: file }));
      } else {
        setUser(prev => ({ ...prev, documents: file }));
      }
    } else {
      alert('Пожалуйста, загрузите PDF-файл');
    }
  };

  // Рендер специфичных полей
  const renderRoleFields = () => {
    switch (role) {
      case 'student':
        return (
          <>
            <div className="info-item">
              <label>Факультет</label>
              {editMode ? (
                <input
                  type="text"
                  name="faculty"
                  value={user.faculty}
                  onChange={handleInputChange}
                />
              ) : (
                <p>{user.faculty}</p>
              )}
            </div>
            <div className="info-item">
              <label>Курс</label>
              {editMode ? (
                <select
                  name="course"
                  value={user.course}
                  onChange={handleInputChange}
                >
                  {[1, 2, 3, 4].map(n => <option key={n} value={`${n} курс`}>{n} курс</option>)}
                </select>
              ) : (
                <p>{user.course}</p>
              )}
            </div>
            <div className="info-item">
              <label>Средний балл</label>
              {editMode ? (
                <input
                  type="text"
                  name="gpa"
                  value={user.gpa}
                  onChange={handleInputChange}
                />
              ) : (
                <p>{user.gpa}</p>
              )}
            </div>
          </>
        );

      case 'teacher':
        return (
          <>
            <div className="info-item">
              <label>Кафедра</label>
              {editMode ? (
                <input
                  type="text"
                  name="department"
                  value={user.department}
                  onChange={handleInputChange}
                />
              ) : (
                <p>{user.department}</p>
              )}
            </div>
            <div className="info-item">
              <label>Должность</label>
              <p>{user.position}</p>
            </div>
            <div className="info-item">
              <label>Ученая степень</label>
              <p>{user.degree}</p>
            </div>
          </>
        );

      case 'admin':
        return (
          <>
            <div className="info-item">
              <label>Должность</label>
              <p>{user.position}</p>
            </div>
            <div className="info-item">
              <label>Уровень доступа</label>
              <p>{user.accessLevel}</p>
            </div>
            <div className="info-item">
              <label>Последний вход</label>
              <p>{user.lastLogin}</p>
            </div>
          </>
        );

      case 'applicant':
        return (
          <>
            <div className="info-item">
              <label>Статус заявки</label>
              <p className={`status-${user.status.replace(' ', '')}`}>{user.status}</p>
            </div>
            <div className="info-item">
              <label>Факультет</label>
              <p>{user.appliedFaculty}</p>
            </div>
            <div className="info-item">
              <label>Результаты ЕГЭ</label>
              <p>{user.examResults}</p>
            </div>
          </>
        );

      default:
        return null;
    }
  };

  return (
    <motion.div
      className={`profile-container ${role}`}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      <div className="profile-header">
        <h1>
          {role === 'student' && 'Профиль студента'}
          {role === 'teacher' && 'Профиль преподавателя'}
          {role === 'admin' && 'Административная панель'}
          {role === 'applicant' && 'Моя заявка'}
        </h1>
        
        {role !== 'admin' && (
          <button
            className={`edit-button ${editMode ? 'save-button' : ''}`}
            onClick={() => setEditMode(!editMode)}
          >
            {editMode ? 'Сохранить' : 'Редактировать'}
          </button>
        )}
      </div>

      <div className="profile-content">
        <div className="profile-main">
          <div className="avatar-section">
            <div
              className="avatar-wrapper"
              onClick={() => editMode && avatarInputRef.current?.click()}
            >
              {user.avatar ? (
                <img src={user.avatar} alt="Аватар" className="avatar" />
              ) : (
                <div className="avatar-placeholder">
                  {user.name.split(' ').map(n => n[0]).join('')}
                </div>
              )}
              {editMode && (
                <>
                  <input
                    type="file"
                    ref={avatarInputRef}
                    onChange={handleAvatarUpload}
                    accept="image/*"
                    style={{ display: 'none' }}
                  />
                  <div className="avatar-overlay">Изменить</div>
                </>
              )}
            </div>
            <h2>{user.name}</h2>
            <p className="role-badge">
              {role === 'student' && 'Студент'}
              {role === 'teacher' && 'Преподаватель'}
              {role === 'admin' && 'Администратор'}
              {role === 'applicant' && 'Абитуриент'}
            </p>
          </div>

          <div className="info-section">
            <div className="personal-info">
              <h2>{role === 'applicant' ? 'Информация о заявке' : 'Личная информация'}</h2>
              <div className="info-grid">
                <div className="info-item">
                  <label>Email</label>
                  {editMode ? (
                    <input
                      type="email"
                      name="email"
                      value={user.email}
                      onChange={handleInputChange}
                    />
                  ) : (
                    <p>{user.email}</p>
                  )}
                </div>

                {renderRoleFields()}
              </div>
            </div>

            <div className="bio-section">
              <h2>{role === 'applicant' ? 'Мотивационное письмо' : 'О себе'}</h2>
              {editMode ? (
                <textarea
                  name="bio"
                  value={user.bio || user.motivationLetter}
                  onChange={handleInputChange}
                  className="bio-textarea"
                />
              ) : (
                <p className="bio-text">{user.bio || user.motivationLetter}</p>
              )}
            </div>
          </div>
        </div>

        {/* Секция навыков (студент) */}
        {role === 'student' && (
          <div className="skills-section">
            <h2>Технические навыки</h2>
            <div className="skills-tags">
              {user.skills.map((skill, index) => (
                <span key={index} className="skill-tag">{skill}</span>
              ))}
              {editMode && (
                <button className="add-skill-button">+ Добавить навык</button>
              )}
            </div>
          </div>
        )}

        {/* Секция курсов (преподаватель) */}
        {role === 'teacher' && (
          <div className="courses-section">
            <h2>Преподаваемые дисциплины</h2>
            <ul className="courses-list">
              {user.courses.map((course, index) => (
                <li key={index}>
                  {course}
                  {editMode && <button className="remove-course">×</button>}
                </li>
              ))}
              {editMode && (
                <button className="add-course-button">+ Добавить курс</button>
              )}
            </ul>
          </div>
        )}

        {/* Секция документов */}
        {(role === 'student' || role === 'applicant') && (
          <div className="documents-section">
            <h2>{role === 'student' ? 'Резюме' : 'Прикрепленные документы'}</h2>
            <div className="document-upload">
              {(role === 'student' ? user.resume : user.documents) ? (
                <div className="document-preview">
                  <span className="document-icon">📄</span>
                  <div className="document-details">
                    <span className="document-name">
                      {(role === 'student' ? user.resume : user.documents)?.name}
                    </span>
                    <span className="document-size">
                      {((role === 'student' ? user.resume : user.documents)?.size / 1024).toFixed(1)} KB
                    </span>
                  </div>
                  <div className="document-actions">
                    <a
                      href={URL.createObjectURL(role === 'student' ? user.resume : user.documents)}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="view-button"
                    >
                      Просмотр
                    </a>
                    <button
                      onClick={() => role === 'student' 
                        ? setUser(p => ({ ...p, resume: null })) 
                        : setUser(p => ({ ...p, documents: null }))
                      }
                      className="remove-button"
                    >
                      Удалить
                    </button>
                  </div>
                </div>
              ) : (
                <motion.div
                  className="upload-area"
                  onClick={() => fileInputRef.current?.click()}
                  whileHover={{ scale: 1.02 }}
                >
                  <input
                    type="file"
                    ref={fileInputRef}
                    onChange={handleFileUpload}
                    accept=".pdf"
                    style={{ display: 'none' }}
                  />
                  <span className="upload-icon">📤</span>
                  <p>
                    {role === 'student' 
                      ? 'Загрузите ваше резюме' 
                      : 'Загрузите документы для поступления'}
                  </p>
                  <span className="file-requirements">PDF, не более 5MB</span>
                </motion.div>
              )}
            </div>
          </div>
        )}
      </div>
    </motion.div>
  );
};

















const AnalyticsPage = () => {
  const navigate = useNavigate();
  const [students, setStudents] = useState([]);
  const [filteredStudents, setFilteredStudents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filters, setFilters] = useState({
    course: '',
    faculty: '',
    gpaMin: '',
    gpaMax: ''
  });

  // Проверка роли преподавателя
  useEffect(() => {
    const user = JSON.parse(localStorage.getItem('user'));
    if (user?.role !== 'teacher') {
      navigate('/profile');
    }
  }, [navigate]);

  // Загрузка данных студентов (в реальном приложении - API запрос)
  useEffect(() => {
    // Имитация загрузки данных
    setTimeout(() => {
      const mockStudents = [
        { id: 1, name: 'Иванов Алексей', faculty: 'ФКН', course: 3, gpa: 4.8, skills: ['JS', 'React'], status: 'активный' },
        { id: 2, name: 'Петрова Мария', faculty: 'ФКН', course: 2, gpa: 4.5, skills: ['Python', 'Django'], status: 'активный' },
        { id: 3, name: 'Сидоров Дмитрий', faculty: 'ФЭН', course: 4, gpa: 3.9, skills: ['Java', 'Spring'], status: 'активный' },
        { id: 4, name: 'Козлова Анна', faculty: 'ФКН', course: 3, gpa: 4.2, skills: ['C++', 'Algorithms'], status: 'академический отпуск' },
        { id: 5, name: 'Михайлов Иван', faculty: 'ФЭН', course: 2, gpa: 3.7, skills: ['PHP', 'Laravel'], status: 'активный' },
      ];
      setStudents(mockStudents);
      setFilteredStudents(mockStudents);
      setLoading(false);
    }, 800);
  }, []);

  // Применение фильтров
  useEffect(() => {
    let result = [...students];
    
    if (filters.course) {
      result = result.filter(s => s.course === parseInt(filters.course));
    }
    
    if (filters.faculty) {
      result = result.filter(s => s.faculty.toLowerCase().includes(filters.faculty.toLowerCase()));
    }
    
    if (filters.gpaMin) {
      result = result.filter(s => s.gpa >= parseFloat(filters.gpaMin));
    }
    
    if (filters.gpaMax) {
      result = result.filter(s => s.gpa <= parseFloat(filters.gpaMax));
    }
    
    setFilteredStudents(result);
  }, [filters, students]);

  const handleFilterChange = (e) => {
    const { name, value } = e.target;
    setFilters(prev => ({ ...prev, [name]: value }));
  };

  // Данные для графика успеваемости
  const chartData = [
    { name: 'ФКН', 'Средний балл': 4.3, 'Кол-во студентов': 12 },
    { name: 'ФЭН', 'Средний балл': 3.9, 'Кол-во студентов': 8 },
    { name: 'ФГН', 'Средний балл': 4.1, 'Кол-во студентов': 5 },
  ];

  if (loading) {
    return <div className="loading">Загрузка данных...</div>;
  }

  return (
    <div className="analytics-page">
      <header className="analytics-header">
        <h1>Аналитика студентов</h1>
        <button 
          className="back-button"
          onClick={() => navigate('/profile')}
        >
          Назад к профилю
        </button>
      </header>

      <div className="analytics-content">
        <section className="filters-section">
          <h2>Фильтры</h2>
          <div className="filter-grid">
            <div className="filter-item">
              <label>Курс</label>
              <select name="course" value={filters.course} onChange={handleFilterChange}>
                <option value="">Все курсы</option>
                <option value="1">1 курс</option>
                <option value="2">2 курс</option>
                <option value="3">3 курс</option>
                <option value="4">4 курс</option>
              </select>
            </div>
            
            <div className="filter-item">
              <label>Факультет</label>
              <input 
                type="text" 
                name="faculty" 
                placeholder="Введите факультет"
                value={filters.faculty}
                onChange={handleFilterChange}
              />
            </div>
            
            <div className="filter-item">
              <label>GPA от</label>
              <input 
                type="number" 
                name="gpaMin" 
                min="0" 
                max="5" 
                step="0.1"
                placeholder="3.0"
                value={filters.gpaMin}
                onChange={handleFilterChange}
              />
            </div>
            
            <div className="filter-item">
              <label>GPA до</label>
              <input 
                type="number" 
                name="gpaMax" 
                min="0" 
                max="5" 
                step="0.1"
                placeholder="5.0"
                value={filters.gpaMax}
                onChange={handleFilterChange}
              />
            </div>
          </div>
        </section>

        <section className="stats-section">
          <div className="stat-card">
            <h3>Всего студентов</h3>
            <p>{students.length}</p>
          </div>
          <div className="stat-card">
            <h3>Средний GPA</h3>
            <p>{(students.reduce((sum, s) => sum + s.gpa, 0) / students.length).toFixed(2)}</p>
          </div>
          <div className="stat-card">
            <h3>Активных</h3>
            <p>{students.filter(s => s.status === 'активный').length}</p>
          </div>
        </section>

        <section className="charts-section">
          <div className="chart-container">
            <h3>Успеваемость по факультетам</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="Средний балл" fill="#4f46e5" />
                <Bar dataKey="Кол-во студентов" fill="#10b981" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>

        <section className="students-table-section">
          <h2>Список студентов ({filteredStudents.length})</h2>
          <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th>ФИО</th>
                  <th>Факультет</th>
                  <th>Курс</th>
                  <th>GPA</th>
                  <th>Навыки</th>
                  <th>Статус</th>
                </tr>
              </thead>
              <tbody>
                {filteredStudents.map(student => (
                  <tr key={student.id}>
                    <td>{student.name}</td>
                    <td>{student.faculty}</td>
                    <td>{student.course}</td>
                    <td className={student.gpa >= 4.5 ? 'high-gpa' : student.gpa < 3.5 ? 'low-gpa' : ''}>
                      {student.gpa}
                    </td>
                    <td>{student.skills.join(', ')}</td>
                    <td className={`status-${student.status.replace(' ', '-')}`}>
                      {student.status}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </div>
    </div>
  );
};


function PlagiarismCheckPage() {
  // Локальная база студенческих работ
  const worksDatabase = [
    {
      id: 1,
      title: "Анализ алгоритмов сортировки",
      author: "Иванов Иван",
      date: "2023-10-15",
      content: "Алгоритмы сортировки являются важной частью программирования. Быстрая сортировка эффективна для больших данных."
    },
    {
      id: 2,
      title: "Сравнение React и Vue",
      author: "Петрова Мария",
      date: "2023-11-02",
      content: "React и Vue - популярные фронтенд-фреймворки. Vue проще для новичков, а React имеет больше вакансий."
    },
    {
      id: 3,
      title: "Исследование нейросетей",
      author: "Сидоров Алексей",
      date: "2023-09-10",
      content: "Нейросети revolutionизируют обработку данных. Сверточные сети особенно эффективны для image recognition."
    }
  ];

  // Генерация хеша для текста
  const generateHash = (text) => {
    const normalizedText = text
      .toLowerCase()
      .replace(/[^\w\sа-яё]/gi, '')
      .replace(/\s+/g, ' '); 


    const words = normalizedText.split(' ');
    const shingles = [];
    
    for (let i = 0; i <= words.length - 3; i++) {
      shingles.push(words.slice(i, i + 3).join(' '));
    }

    return shingles.join('|');
  };


  const calculateSimilarity = (hash1, hash2) => {
    if (!hash1 || !hash2) return 0;
    
    const shingles1 = hash1.split('|');
    const shingles2 = hash2.split('|');
    
    const intersection = shingles1.filter(shingle => 
      shingles2.includes(shingle)
    ).length;
    
    const union = shingles1.length + shingles2.length - intersection;
    
    return Math.round((intersection / union) * 100);
  };

  // Поиск похожих работ
  const checkAgainstDatabase = (text) => {
    const textHash = generateHash(text);
    const results = [];

    // Генерируем хеши для всех работ в базе
    const dbWithHashes = worksDatabase.map(work => ({
      ...work,
      hash: generateHash(work.content)
    }));

    dbWithHashes.forEach(item => {
      const similarity = calculateSimilarity(textHash, item.hash);
      if (similarity > 30) { // Порог срабатывания 30%
        results.push({
          ...item,
          similarity: similarity
        });
      }
    });

    return results.sort((a, b) => b.similarity - a.similarity);
  };

  // Состояние компонента
  const [file, setFile] = useState(null);
  const [results, setResults] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const fileInputRef = useRef(null);

  // Обработка загрузки файла
  const handleFileUpload = (e) => {
    const uploadedFile = e.target.files[0];
    if (!uploadedFile) return;

    setFile(uploadedFile);
    setIsLoading(true);

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const text = e.target.result;
        const plagiarismResults = checkAgainstDatabase(text);
        
        setResults({
          totalSimilarity: plagiarismResults[0]?.similarity || 0,
          details: plagiarismResults,
          originalText: text
        });
      } catch (error) {
        console.error("Ошибка анализа:", error);
        alert("Произошла ошибка при анализе файла");
      } finally {
        setIsLoading(false);
      }
    };
    
    // Чтение как текст (для TXT)
    if (uploadedFile.type === 'text/plain') {
      reader.readAsText(uploadedFile);
    } else {
      alert("Пожалуйста, загрузите текстовый файл (.txt)");
      setIsLoading(false);
    }
  };

  // Подсветка совпадений в тексте
  const highlightMatches = (text, matches) => {
    if (!matches || matches.length === 0) return text;
    
    // Разбиваем текст на предложения для демонстрации
    const sentences = text.split(/[.!?]+/);
    return sentences.map((sentence, i) => {
      const sentenceHash = generateHash(sentence);
      const isMatch = matches.some(work => 
        calculateSimilarity(sentenceHash, generateHash(work.content)) > 50
      );
      
      return isMatch ? (
        <mark key={i}>{sentence}.</mark>
      ) : (
        <span key={i}>{sentence}.</span>
      );
    });
  };

  return (
    <div className="plagiarism-page">
      <h1>Проверка на плагиат</h1>
      <p className="description">
        Загрузите текстовый файл (.txt) для проверки на совпадения с базой студенческих работ
      </p>

      <div 
        className={`upload-area ${file ? 'has-file' : ''}`} 
        onClick={() => fileInputRef.current.click()}
      >
        <input 
          type="file" 
          ref={fileInputRef}
          onChange={handleFileUpload}
          accept=".txt"
          style={{ display: 'none' }}
        />
        
        {file ? (
          <>
            <p>{file.name}</p>
            <span className="file-size">
              {(file.size / 1024).toFixed(1)} KB
            </span>
          </>
        ) : (
          <>
            <p>Перетащите файл или кликните для загрузки</p>
            <span>Поддерживается только .txt</span>
          </>
        )}
      </div>

      {isLoading && (
        <div className="loading">
          <div className="spinner"></div>
          <p>Анализируем документ...</p>
        </div>
      )}

      {results && (
        <div className="results">
          <div className={`summary ${results.totalSimilarity > 50 ? 'warning' : 'safe'}`}>
            <h2>
              {results.totalSimilarity > 50 ? (
                <>⚠️ Возможен плагиат!</>
              ) : (
                <>✓ Низкий уровень заимствований</>
              )}
            </h2>
            <div className="similarity-bar">
              <div 
                className="bar-fill"
                style={{ width: `${results.totalSimilarity}%` }}
              ></div>
              <span>{results.totalSimilarity}%</span>
            </div>
            <p>
              {results.totalSimilarity > 70 ? 
                'Серьезные заимствования. Требуется проверка преподавателем.' :
                results.totalSimilarity > 30 ?
                'Умеренные заимствования. Рекомендуется доработка.' :
                'Работа показала хорошую оригинальность.'}
            </p>
          </div>

          {results.details.length > 0 && (
            <>
              <h3>Найдены совпадения:</h3>
              <ul className="matches-list">
                {results.details.map((work, index) => (
                  <li key={index}>
                    <div className="work-title">{work.title}</div>
                    <div className="work-meta">
                      <span>Автор: {work.author}</span>
                      <span>Дата: {work.date}</span>
                    </div>
                    <div className="similarity-bar">
                      <div 
                        className="bar-fill"
                        style={{ width: `${work.similarity}%` }}
                      ></div>
                      <span>{work.similarity}%</span>
                    </div>
                  </li>
                ))}
              </ul>

              <h3>Анализ текста:</h3>
              <div className="text-analysis">
                <div className="highlighted-text">
                  {highlightMatches(results.originalText, results.details)}
                </div>
                <div className="legend">
                  <span><mark className="sample-mark"></mark> Возможные заимствования</span>
                </div>
              </div>
            </>
          )}
        </div>
      )}
       </div>
  );
}

export default App;