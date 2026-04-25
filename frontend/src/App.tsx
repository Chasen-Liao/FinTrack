import { useState, useEffect, useCallback, useRef } from 'react';
import axios from 'axios';
import { useTranslation } from 'react-i18next';
import StockSelector from './components/StockSelector';
import CandlestickChart from './components/CandlestickChart';
import NewsPanel from './components/NewsPanel';
import NewsCategoryPanel from './components/NewsCategoryPanel';
import RangeAnalysisPanel from './components/RangeAnalysisPanel';
import RangeQueryPopup from './components/RangeQueryPopup';
import RangeNewsPanel from './components/RangeNewsPanel';
import SimilarDaysPanel from './components/SimilarDaysPanel';
import PredictionPanel from './components/PredictionPanel';
import StrategyBacktestPanel from './components/StrategyBacktestPanel';
import './App.css';

interface RangeSelection {
  startDate: string;
  endDate: string;
  priceChange?: number;
  popupX?: number;
  popupY?: number;
}

interface ArticleSelection {
  newsId: string;
  date: string;
}

function App() {
  const { t, i18n } = useTranslation();
  const [activeTickers, setActiveTickers] = useState<string[]>([]);
  const [selectedSymbol, setSelectedSymbol] = useState('');
  const [hoveredDate, setHoveredDate] = useState<string | null>(null);
  const [hoveredOhlc, setHoveredOhlc] = useState<{
    date: string;
    open: number;
    high: number;
    low: number;
    close: number;
    change: number;
  } | null>(null);
  const [selectedRange, setSelectedRange] = useState<RangeSelection | null>(null);
  const [zoomRange, setZoomRange] = useState<RangeSelection | null>(null);
  const [rangeQuestion, setRangeQuestion] = useState<string | null>(null);
  const [selectedDay, setSelectedDay] = useState<string | null>(null);
  const [selectedArticle, setSelectedArticle] = useState<ArticleSelection | null>(null);
  const [rightPanelMode, setRightPanelMode] = useState<'forecast' | 'strategy'>('forecast');

  // Locked article state (click-to-lock)
  const [lockedArticle, setLockedArticle] = useState<ArticleSelection | null>(null);

  // News category filter
  const [activeCategory, setActiveCategory] = useState<string | null>(null);
  const [activeCategoryIds, setActiveCategoryIds] = useState<string[]>([]);
  const [activeCategoryColor, setActiveCategoryColor] = useState<string | null>(null);

  // Chart area ref for popup positioning
  const chartAreaRef = useRef<HTMLDivElement>(null);
  const [chartRect, setChartRect] = useState<DOMRect | undefined>(undefined);

  // Theme state
  const [theme, setTheme] = useState<'dark' | 'light'>(() => {
    const saved = localStorage.getItem('theme');
    return (saved as 'dark' | 'light') || 'dark';
  });

  // Apply theme to document
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
  }, [theme]);

  // Toggle theme
  const toggleTheme = () => {
    setTheme(prev => prev === 'dark' ? 'light' : 'dark');
  };

  // Language switch handler
  const handleLanguageChange = (lang: string) => {
    i18n.changeLanguage(lang);
    localStorage.setItem('language', lang);
  };

  useEffect(() => {
    axios
      .get('/api/stocks')
      .then((res) => {
        const tickers = res.data
          .filter((t: any) => t.last_ohlc_fetch)
          .map((t: any) => t.symbol);
        setActiveTickers(tickers);
        if (tickers.length > 0 && !selectedSymbol) {
          setSelectedSymbol(tickers[0]);
        }
      })
      .catch(console.error);
  }, []);

  // Update chartRect when range is selected (for popup positioning)
  useEffect(() => {
    if (selectedRange && chartAreaRef.current) {
      setChartRect(chartAreaRef.current.getBoundingClientRect());
    }
  }, [selectedRange]);

  const handleHover = useCallback(
    (date: string | null, ohlc?: { date: string; open: number; high: number; low: number; close: number; change: number }) => {
      // Don't update hovered date when locked
      if (!lockedArticle) {
        setHoveredDate(date);
      }
      setHoveredOhlc(ohlc || null);
    },
    [lockedArticle]
  );

  const handleRangeSelect = useCallback((range: RangeSelection | null) => {
    setSelectedRange(range);
    setRangeQuestion(null);
    if (range) {
      setSelectedDay(null);
      setSelectedArticle(null);
      setLockedArticle(null);
    }
  }, []);

  const handleArticleSelect = useCallback((article: ArticleSelection | null) => {
    if (article === null) {
      // Unlock
      setLockedArticle(null);
      setSelectedArticle(null);
      return;
    }
    // Toggle: click same dot → unlock, different dot → lock new
    setLockedArticle((prev) => {
      if (prev && prev.newsId === article.newsId) {
        // Unlock
        setSelectedArticle(null);
        return null;
      }
      // Lock new
      setSelectedArticle(article);
      setSelectedRange(null);
      setRangeQuestion(null);
      setSelectedDay(null);
      setHoveredDate(article.date);
      return article;
    });
  }, []);

  const handleDayClick = useCallback((date: string) => {
    setSelectedDay(date);
    setSelectedRange(null);
    setZoomRange(null);
    setRangeQuestion(null);
    setSelectedArticle(null);
    setLockedArticle(null);
  }, []);

  const handleRangeAsk = useCallback((question: string) => {
    setRangeQuestion(question);
  }, []);

  const handleCategoryChange = useCallback((category: string | null, articleIds: string[], color?: string) => {
    setActiveCategory(category);
    setActiveCategoryIds(articleIds);
    setActiveCategoryColor(color ?? null);
  }, []);

  function handleSelectSymbol(symbol: string) {
    setSelectedSymbol(symbol);
    setHoveredDate(null);
    setHoveredOhlc(null);
    setSelectedRange(null);
    setRangeQuestion(null);
    setSelectedDay(null);
    setSelectedArticle(null);
    setLockedArticle(null);
    setActiveCategory(null);
    setActiveCategoryIds([]);
    setActiveCategoryColor(null);
  }

  function handleAddTicker(symbol: string) {
    if (!activeTickers.includes(symbol)) {
      setActiveTickers((prev) => [...prev, symbol]);
      axios.post('/api/stocks', { symbol }).catch(console.error);
    }
  }

  // Effective date for NewsPanel: locked takes priority
  const effectiveDate = lockedArticle?.date ?? hoveredDate;
  const isLocked = lockedArticle !== null;

  // Right panel priority: rangeQuestion > rangeNews > selectedDay > default NewsPanel
  function renderRightPanel() {
    if (selectedRange && rangeQuestion) {
      return (
        <RangeAnalysisPanel
          symbol={selectedSymbol}
          startDate={selectedRange.startDate}
          endDate={selectedRange.endDate}
          question={rangeQuestion}
          onClear={() => {
            setSelectedRange(null);
            setRangeQuestion(null);
          }}
        />
      );
    }
    if (selectedRange && !rangeQuestion) {
      return (
        <RangeNewsPanel
          symbol={selectedSymbol}
          startDate={selectedRange.startDate}
          endDate={selectedRange.endDate}
          priceChange={selectedRange.priceChange}
          onClose={() => setSelectedRange(null)}
          onAskAI={handleRangeAsk}
        />
      );
    }
    if (selectedDay) {
      return (
        <SimilarDaysPanel
          symbol={selectedSymbol}
          date={selectedDay}
          onClose={() => setSelectedDay(null)}
        />
      );
    }
    return (
      <>
        <NewsPanel
          symbol={selectedSymbol}
          hoveredDate={effectiveDate}
          onFindSimilar={(_newsId: string) => {
            if (effectiveDate) handleDayClick(effectiveDate);
          }}
          highlightedNewsId={selectedArticle?.newsId || null}
          isLocked={isLocked}
          onUnlock={() => {
            setLockedArticle(null);
            setSelectedArticle(null);
          }}
          highlightedCategoryIds={activeCategoryIds.length > 0 ? activeCategoryIds : undefined}
        />
      </>
    );
  }

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-left">
          <h1>{t('app.title')}</h1>
        </div>
        <StockSelector
          activeTickers={activeTickers}
          selectedSymbol={selectedSymbol}
          onSelect={handleSelectSymbol}
          onAdd={handleAddTicker}
        />
        {selectedRange ? (
          <div className="header-ohlc">
            <span className="ohlc-date">{selectedRange.startDate} ~ {selectedRange.endDate}</span>
            <span className="range-badge">{t('header.rangeSelected')}</span>
          </div>
        ) : hoveredOhlc ? (
          <div className="header-ohlc">
            <span className="ohlc-date">{hoveredOhlc.date}</span>
            <span className="ohlc-label">{t('chart.open')}</span>
            <span className="ohlc-val">${hoveredOhlc.open.toFixed(2)}</span>
            <span className="ohlc-label">{t('chart.high')}</span>
            <span className="ohlc-val">${hoveredOhlc.high.toFixed(2)}</span>
            <span className="ohlc-label">{t('chart.low')}</span>
            <span className="ohlc-val">${hoveredOhlc.low.toFixed(2)}</span>
            <span className="ohlc-label">{t('chart.close')}</span>
            <span className="ohlc-val">${hoveredOhlc.close.toFixed(2)}</span>
            <span className={`ohlc-change ${hoveredOhlc.change >= 0 ? 'up' : 'down'}`}>
              {hoveredOhlc.change >= 0 ? '+' : ''}
              {hoveredOhlc.change.toFixed(2)}%
            </span>
          </div>
        ) : null}
        <div className="header-right">
          <span className="header-link" style={{ cursor: 'default' }}>Chasen</span>
          <button
            className="theme-toggle"
            onClick={toggleTheme}
            title={theme === 'dark' ? t('header.switchToLight') : t('header.switchToDark')}
          >
            {theme === 'dark' ? (
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="5"/>
                <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/>
              </svg>
            ) : (
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
              </svg>
            )}
          </button>
          <select
            className="language-switcher"
            value={i18n.language}
            onChange={(e) => handleLanguageChange(e.target.value)}
          >
            <option value="en">{t('language.en')}</option>
            <option value="zh">{t('language.zh')}</option>
          </select>
        </div>
      </header>

      <main className="app-main">
        <div className="chart-area" ref={chartAreaRef}>
          {selectedSymbol ? (
            <>
              <CandlestickChart
                symbol={selectedSymbol}
                lockedNewsId={lockedArticle?.newsId ?? null}
                highlightedArticleIds={activeCategoryIds.length > 0 ? activeCategoryIds : null}
                highlightColor={activeCategoryColor}
                zoomRange={zoomRange}
                onZoomReset={() => setZoomRange(null)}
                onHover={handleHover}
                onRangeSelect={handleRangeSelect}
                onArticleSelect={handleArticleSelect}
                onDayClick={handleDayClick}
              />
              {selectedRange && !rangeQuestion && (
                <RangeQueryPopup
                  range={selectedRange}
                  chartRect={chartRect}
                  onZoom={() => {
                    setZoomRange(selectedRange);
                    setSelectedRange(null);
                    setRangeQuestion(null);
                  }}
                  onAsk={handleRangeAsk}
                  onClose={() => setSelectedRange(null)}
                />
              )}
            </>
          ) : (
            <div className="chart-placeholder">{t('app.selectTicker')}</div>
          )}
        </div>
        {selectedSymbol && (
          <div className="prediction-area">
            <div className="right-mode-tabs">
              <button
                className={`right-mode-tab ${rightPanelMode === 'forecast' ? 'active' : ''}`}
                onClick={() => setRightPanelMode('forecast')}
              >
                {t('header.aiForecast')}
              </button>
              <button
                className={`right-mode-tab ${rightPanelMode === 'strategy' ? 'active' : ''}`}
                onClick={() => setRightPanelMode('strategy')}
              >
                {t('header.strategy')}
              </button>
            </div>
            {rightPanelMode === 'forecast' ? (
              <PredictionPanel symbol={selectedSymbol} />
            ) : (
              <StrategyBacktestPanel />
            )}
          </div>
        )}
        <div className="news-area">
          {selectedSymbol && (
            <NewsCategoryPanel
              symbol={selectedSymbol}
              activeCategory={activeCategory}
              onCategoryChange={handleCategoryChange}
            />
          )}
          {renderRightPanel()}
        </div>
      </main>
    </div>
  );
}

export default App;
