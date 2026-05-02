import { useState, useEffect, useMemo } from 'react';
import axios from 'axios';
import { useTranslation } from 'react-i18next';

interface Driver {
  name: string;
  value: number;
  importance: number;
  z_score: number;
  contribution: number;
}

interface HorizonPrediction {
  direction: 'up' | 'down';
  confidence: number;
  model_type?: string;
  top_drivers: Driver[];
  model_accuracy: number | null;
  baseline_accuracy: number | null;
  roc_auc: number | null;
}

interface SimilarPeriod {
  period_start: string;
  period_end: string;
  similarity: number;
  avg_sentiment: number;
  n_articles: number;
  ret_after_5d: number | null;
  ret_after_10d: number | null;
}

interface Headline {
  date: string;
  title: string;
  sentiment: string;
  summary: string;
}

interface ImpactArticle {
  news_id: string;
  date: string;
  title: string;
  sentiment: string;
  relevance: string | null;
  key_discussion: string;
  ret_t0: number | null;
  ret_t1: number | null;
}

interface NewsSummary {
  total: number;
  positive: number;
  negative: number;
  neutral: number;
  sentiment_ratio: number;
  top_headlines: Headline[];
  top_impact: ImpactArticle[];
}

interface SimilarStats {
  count: number;
  up_ratio_5d: number;
  up_ratio_10d: number;
  avg_ret_5d: number | null;
  avg_ret_10d: number | null;
}

interface DeepAnalysis {
  news_id: string;
  discussion: string;
  growth_reasons: string;
  decrease_reasons: string;
}

interface Forecast {
  symbol: string;
  window_days: number;
  forecast_date: string;
  news_summary: NewsSummary;
  prediction: Record<string, HorizonPrediction>;
  similar_periods: SimilarPeriod[];
  similar_stats: SimilarStats;
  conclusion: string;
}

interface Props {
  symbol: string;
}

function extractKeywords(headlines: Headline[]): string[] {
  const stopwords = new Set([
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
    'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
    'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'between', 'out', 'off', 'over', 'under', 'again', 'further',
    'then', 'once', 'and', 'but', 'or', 'nor', 'not', 'so', 'yet',
    'both', 'either', 'neither', 'each', 'every', 'all', 'any',
    'few', 'more', 'most', 'other', 'some', 'such', 'no', 'only',
    'own', 'same', 'than', 'too', 'very', 'just', 'because', 'about',
    'up', 'its', 'it', 'this', 'that', 'these', 'those', 'he', 'she',
    'they', 'them', 'their', 'what', 'which', 'who', 'whom', 'how',
    'new', 'says', 'said', 'also', 'like', 'now', 'one', 'two',
    'get', 'got', 'make', 'go', 'going', 'set', 'see', 'big', 'still',
  ]);

  const freq = new Map<string, number>();
  for (const h of headlines) {
    const words = h.title.toLowerCase().replace(/[^a-z0-9\s]/g, '').split(/\s+/);
    const seen = new Set<string>();
    for (const w of words) {
      if (w.length < 3 || stopwords.has(w) || seen.has(w)) continue;
      seen.add(w);
      freq.set(w, (freq.get(w) || 0) + 1);
    }
  }

  return Array.from(freq.entries())
    .filter(([, c]) => c >= 2)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 12)
    .map(([w]) => w);
}

/**
 * Parse conclusion text and return styled JSX:
 * - [ModelName] → bold purple badge
 * - bullish/leaning bullish → green bold
 * - bearish/leaning bearish → red bold
 * - +N% / -N% / N% → colored by sign
 * - positive → green, negative → red
 */
function renderStyledText(text: string): React.ReactNode[] {
  // Regex that matches all the patterns we want to style
  const pattern = /(\[[^\]]+\])|(bullish|leaning bullish|Bullish)|(bearish|leaning bearish|Bearish)|(positive)|(negative)|([+-]?\d+\.?\d*%)/gi;

  const parts: React.ReactNode[] = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;
  let key = 0;

  while ((match = pattern.exec(text)) !== null) {
    // Push the plain text before this match
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index));
    }

    const [full, model, bullish, bearish, positive, negative, pct] = match;

    if (model) {
      // Hide model names like [XGBoost] — just skip them
      key++;
    } else if (bullish) {
      parts.push(
        <span key={key++} className="fc-text-bull">{full}</span>
      );
    } else if (bearish) {
      parts.push(
        <span key={key++} className="fc-text-bear">{full}</span>
      );
    } else if (positive) {
      parts.push(
        <span key={key++} className="fc-text-bull">{full}</span>
      );
    } else if (negative) {
      parts.push(
        <span key={key++} className="fc-text-bear">{full}</span>
      );
    } else if (pct) {
      const isNeg = pct.startsWith('-');
      parts.push(
        <span key={key++} className={isNeg ? 'fc-text-pct-down' : 'fc-text-pct-up'}>{full}</span>
      );
    }

    lastIndex = match.index + full.length;
  }

  // Remaining text
  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex));
  }

  return parts;
}

export default function PredictionPanel({ symbol }: Props) {
  const { t } = useTranslation();
  const [forecast7, setForecast7] = useState<Forecast | null>(null);
  const [forecast30, setForecast30] = useState<Forecast | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [expanded, setExpanded] = useState(true);

  // Deep analysis state
  const [deepLoading, setDeepLoading] = useState<string | null>(null);
  const [deepResults, setDeepResults] = useState<Record<string, DeepAnalysis>>({});

  useEffect(() => {
    if (!symbol) return;
    setLoading(true);
    setError('');
    Promise.all([
      axios.get(`/api/predict/${symbol}/forecast?window=7`).then((res) => res.data as Forecast).catch(() => null),
      axios.get(`/api/predict/${symbol}/forecast?window=30`).then((res) => res.data as Forecast).catch(() => null),
    ])
      .then(([f7, f30]) => {
        setForecast7(f7);
        setForecast30(f30);
        if (!f7 && !f30) setError(t('prediction.noModel'));
      })
      .finally(() => setLoading(false));
  }, [symbol, t]);

  const keywords = useMemo(() => {
    const fc = forecast7 || forecast30;
    if (!fc) return [];
    return extractKeywords(fc.news_summary.top_headlines);
  }, [forecast7, forecast30]);

  // Primary forecast for the header summary
  const primaryForecast = forecast7 || forecast30;
  const primary = primaryForecast
    ? (primaryForecast.prediction.t3 || primaryForecast.prediction.t1 || primaryForecast.prediction.t5)
    : null;
  const isUp = primary?.direction === 'up';
  const ns = primaryForecast?.news_summary;

  if (loading) {
    return (
      <div className="pred-panel">
        <div className="pred-header" onClick={() => setExpanded(!expanded)}>
          <span className="pred-title">{t('prediction.title')}</span>
          <span className="pred-loading-dot" />
          <span className="pred-loading-text">{t('prediction.analyzing')}</span>
        </div>
      </div>
    );
  }

  if (error || (!forecast7 && !forecast30)) {
    return (
      <div className="pred-panel">
        <div className="pred-header">
          <span className="pred-title">{t('prediction.title')}</span>
          <span className="pred-no-model">{error || t('prediction.noData')}</span>
        </div>
      </div>
    );
  }

  return (
    <div className={`pred-panel ${expanded ? 'pred-expanded' : ''}`}>
      {/* Header bar */}
      <div className="pred-header" onClick={() => setExpanded(!expanded)}>
        <span className="pred-title">{t('prediction.title')}</span>

        {primary && (
          <>
            <div className={`pred-arrow ${isUp ? 'up' : 'down'}`}>
              {isUp ? '\u2191' : '\u2193'}
            </div>
            <span className={`pred-dir ${isUp ? 'up' : 'down'}`}>
              {isUp ? t('prediction.up') : t('prediction.down')}
            </span>
            <div className="pred-conf-bar">
              <div
                className={`pred-conf-fill ${isUp ? 'up' : 'down'}`}
                style={{ width: `${primary.confidence * 100}%` }}
              />
              <span className="pred-conf-label">{(primary.confidence * 100).toFixed(0)}%</span>
            </div>
          </>
        )}

        {ns && (
          <span className="pred-news-badge">
            {ns.total} {t('prediction.news')} · {ns.positive}+ {ns.negative}-
          </span>
        )}

        <span className="pred-expand-icon">{expanded ? '\u25B2' : '\u25BC'}</span>
      </div>

      {/* Expanded details */}
      {expanded && (
        <div className="pred-details">
          {/* Keyword tags (shared, show once) */}
          {keywords.length > 0 && (
            <div className="fc-keywords-section">
              <div className="fc-section-title">{t('prediction.keyTopics')}</div>
              <div className="fc-keywords">
                {keywords.map((kw) => (
                  <span key={kw} className="fc-keyword-pill">{kw}</span>
                ))}
              </div>
            </div>
          )}

          {/* 7D Forecast Section */}
          {forecast7 && (
            <ForecastSection
              label={t('prediction.sevenDay')}
              forecast={forecast7}
              symbol={symbol}
              deepLoading={deepLoading}
              deepResults={deepResults}
              setDeepLoading={setDeepLoading}
              setDeepResults={setDeepResults}
              t={t}
            />
          )}

          {/* 30D Forecast Section */}
          {forecast30 && (
            <ForecastSection
              label={t('prediction.thirtyDay')}
              forecast={forecast30}
              symbol={symbol}
              deepLoading={deepLoading}
              deepResults={deepResults}
              setDeepLoading={setDeepLoading}
              setDeepResults={setDeepResults}
              t={t}
            />
          )}
        </div>
      )}
    </div>
  );
}

function ForecastSection({
  label,
  forecast,
  symbol,
  deepLoading,
  deepResults,
  setDeepLoading,
  setDeepResults,
  t,
}: {
  label: string;
  forecast: Forecast;
  symbol: string;
  deepLoading: string | null;
  deepResults: Record<string, DeepAnalysis>;
  setDeepLoading: (id: string | null) => void;
  setDeepResults: React.Dispatch<React.SetStateAction<Record<string, DeepAnalysis>>>;
  t: (key: string) => string;
}) {
  const t1 = forecast.prediction.t1;
  const t3 = forecast.prediction.t3;
  const t5 = forecast.prediction.t5;
  const primary = t3 || t1 || t5;
  const isUp = primary?.direction === 'up';
  const ns = forecast.news_summary;
  const stats = forecast.similar_stats;

  const conclusionBullets = forecast.conclusion
    ? forecast.conclusion.split(/(?<=[.!?。])\s*/).filter((s) => s.trim().length > 0)
    : [];

  return (
    <div className="fc-section-block">
      <div className="fc-section-divider">{label} {t('prediction.forecast')}</div>

      {/* AI Prediction Hero */}
      {primary && (
        <div className={`fc-hero ${isUp ? 'fc-hero-up' : 'fc-hero-down'}`}>
          <span className="fc-hero-arrow">{isUp ? '\u2191' : '\u2193'}</span>
          <div className="fc-hero-text">
            <span className="fc-hero-label">{label}:</span>
            <span className="fc-hero-dir">{isUp ? t('prediction.bullish') : t('prediction.bearish')}</span>
          </div>
          <span className="fc-hero-conf">{(primary.confidence * 100).toFixed(0)}%</span>
        </div>
      )}

      {/* Structured analysis bullets */}
      {conclusionBullets.length > 0 && (
        <div className="fc-analysis">
          <div className="fc-section-title">{t('prediction.analysis')}</div>
          <ul className="fc-bullet-list">
            {conclusionBullets.map((bullet, i) => (
              <li key={i} className="fc-bullet-item">{renderStyledText(bullet)}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Prediction cards */}
      <div className="fc-predictions">
        {t1 && <PredictionCard label="T+1" pred={t1} t={t} />}
        {t3 && <PredictionCard label="T+3" pred={t3} t={t} />}
        {t5 && <PredictionCard label="T+5" pred={t5} t={t} />}
      </div>

      {/* Top Impact News */}
      {ns.top_impact && ns.top_impact.length > 0 && (
        <div className="fc-impact-section">
          <div className="fc-section-title">{t('prediction.keyImpactNews')}</div>
          {ns.top_impact.map((article) => {
            const retClass = (article.ret_t0 ?? 0) >= 0 ? 'up' : 'down';
            const deep = deepResults[article.news_id];
            const isAnalyzing = deepLoading === article.news_id;
            return (
              <div key={article.news_id} className={`fc-impact-card fc-impact-${retClass}`}>
                <div className="fc-impact-header">
                  <span className={`fc-impact-ret ${retClass}`}>
                    {article.ret_t0 != null ? `${article.ret_t0 >= 0 ? '+' : ''}${article.ret_t0.toFixed(2)}%` : '-'}
                  </span>
                  <span className={`fc-impact-sentiment ${article.sentiment || 'unknown'}`}>
                    {article.sentiment === 'positive' ? t('prediction.bullish') : article.sentiment === 'negative' ? t('prediction.bearish') : article.sentiment === 'neutral' ? t('prediction.neutral') : 'N/A'}
                  </span>
                  <span className="fc-impact-date">{article.date}</span>
                </div>
                <div className="fc-impact-title">{article.title}</div>
                {article.key_discussion && (
                  <div className="fc-impact-summary">{article.key_discussion}</div>
                )}
                {deep ? (
                  <div className="fc-deep-result">
                    <div className="fc-deep-discussion">{deep.discussion}</div>
                    {deep.growth_reasons && (
                      <div className="fc-deep-reasons fc-deep-bull">
                        <span className="fc-deep-reasons-label">▲ {t('prediction.bullishFactors')}</span>
                        <div className="fc-deep-reasons-text">{deep.growth_reasons}</div>
                      </div>
                    )}
                    {deep.decrease_reasons && (
                      <div className="fc-deep-reasons fc-deep-bear">
                        <span className="fc-deep-reasons-label">▼ {t('prediction.riskFactors')}</span>
                        <div className="fc-deep-reasons-text">{deep.decrease_reasons}</div>
                      </div>
                    )}
                  </div>
                ) : (
                  <button
                    className="fc-deep-btn"
                    disabled={isAnalyzing}
                    onClick={() => {
                      setDeepLoading(article.news_id);
                      axios
                        .post('/api/analysis/deep', { news_id: article.news_id, symbol })
                        .then((res) => {
                          setDeepResults((prev) => ({ ...prev, [article.news_id]: res.data }));
                        })
                        .catch(() => {})
                        .finally(() => setDeepLoading(null));
                    }}
                  >
                    {isAnalyzing ? t('prediction.analyzingNews') : '🔍 ' + t('prediction.deepAnalysis')}
                  </button>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* Similar historical periods */}
      {stats.count > 0 && (
        <div className="fc-similar-section">
          <div className="fc-section-title">{t('prediction.similarHistory')} ({stats.count})</div>
          <div className="fc-similar-stats">
            <div className="fc-stat">
              <span className="fc-stat-label">5D {t('prediction.upRate')}</span>
              <span className={`fc-stat-value ${stats.up_ratio_5d > 0.5 ? 'up' : 'down'}`}>
                {(stats.up_ratio_5d * 100).toFixed(0)}%
              </span>
            </div>
            <div className="fc-stat">
              <span className="fc-stat-label">{t('prediction.avgRet')} 5D</span>
              <span className={`fc-stat-value ${(stats.avg_ret_5d ?? 0) >= 0 ? 'up' : 'down'}`}>
                {stats.avg_ret_5d != null ? `${stats.avg_ret_5d >= 0 ? '+' : ''}${stats.avg_ret_5d.toFixed(1)}%` : '-'}
              </span>
            </div>
            <div className="fc-stat">
              <span className="fc-stat-label">10D {t('prediction.upRate')}</span>
              <span className={`fc-stat-value ${stats.up_ratio_10d > 0.5 ? 'up' : 'down'}`}>
                {(stats.up_ratio_10d * 100).toFixed(0)}%
              </span>
            </div>
            <div className="fc-stat">
              <span className="fc-stat-label">{t('prediction.avgRet')} 10D</span>
              <span className={`fc-stat-value ${(stats.avg_ret_10d ?? 0) >= 0 ? 'up' : 'down'}`}>
                {stats.avg_ret_10d != null ? `${stats.avg_ret_10d >= 0 ? '+' : ''}${stats.avg_ret_10d.toFixed(1)}%` : '-'}
              </span>
            </div>
          </div>

          <div className="fc-periods-list">
            {forecast.similar_periods.slice(0, 5).map((p, i) => (
              <div key={i} className="fc-period-card">
                <div className="fc-period-header">
                  <span className="fc-period-dates">{p.period_start} ~ {p.period_end}</span>
                  <span className="fc-period-sim">{(p.similarity * 100).toFixed(0)}% {t('prediction.match')}</span>
                </div>
                <div className="fc-period-detail">
                  <span>{p.n_articles} {t('prediction.articles')}</span>
                  <span>{t('prediction.sentiment')}: {p.avg_sentiment >= 0 ? '+' : ''}{p.avg_sentiment.toFixed(2)}</span>
                  {p.ret_after_5d != null && (
                    <span className={p.ret_after_5d >= 0 ? 'up' : 'down'}>
                      5D: {p.ret_after_5d >= 0 ? '+' : ''}{p.ret_after_5d.toFixed(1)}%
                    </span>
                  )}
                  {p.ret_after_10d != null && (
                    <span className={p.ret_after_10d >= 0 ? 'up' : 'down'}>
                      10D: {p.ret_after_10d >= 0 ? '+' : ''}{p.ret_after_10d.toFixed(1)}%
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function PredictionCard({ label, pred, t }: { label: string; pred: HorizonPrediction; t: (key: string) => string }) {
  const isUp = pred.direction === 'up';
  const hasAccuracy = pred.model_accuracy != null && pred.baseline_accuracy != null;
  const lift = hasAccuracy ? (pred.model_accuracy! - pred.baseline_accuracy!) : 0;

  return (
    <div className={`fc-pred-card ${isUp ? 'up' : 'down'}`}>
      <div className="fc-pred-header">
        <span className="fc-pred-label">{label}</span>
        {/* model_type hidden — show generic "AI" label */}
        <span className={`fc-pred-dir ${isUp ? 'up' : 'down'}`}>
          {isUp ? '\u2191' : '\u2193'} {isUp ? t('prediction.up') : t('prediction.down')}
        </span>
        <span className="fc-pred-conf">{(pred.confidence * 100).toFixed(0)}%</span>
      </div>
      {hasAccuracy && (
        <div className="fc-pred-meta">
          {t('prediction.accuracy')} {(pred.model_accuracy! * 100).toFixed(1)}% / {t('prediction.baseline')} {(pred.baseline_accuracy! * 100).toFixed(1)}% / {t('prediction.lift')} {lift >= 0 ? '+' : ''}{(lift * 100).toFixed(1)}pp
          {pred.roc_auc != null && (
            <span> / AUC {pred.roc_auc.toFixed(3)}</span>
          )}
        </div>
      )}
      {pred.top_drivers.length > 0 && (
        <div className="fc-drivers" ref={(el) => {
          if (!el || pred.top_drivers.length === 0) return;
          import('d3').then((d3) => {
            el.innerHTML = '';
            const data = pred.top_drivers.slice(0, 4).map(d => ({
              name: d.name,
              importance: d.contribution,
            }));
            const maxVal = Math.max(...data.map(d => d.importance), 0.01);
            const containerWidth = el.clientWidth || 280;
            const barHeight = 20;
            const svgHeight = data.length * barHeight + 8;
            const svg = d3.select(el).append('svg')
              .attr('width', containerWidth)
              .attr('height', svgHeight);
            const x = d3.scaleLinear().domain([0, maxVal]).range([0, containerWidth - 80]);
            const colors = ['#3b82f6', '#60a5fa', '#93c5fd', '#bfdbfe'];
            data.forEach((d, i) => {
              const g = svg.append('g').attr('transform', `translate(0,${i * barHeight + 4})`);
              g.append('rect')
                .attr('x', 0).attr('y', 0)
                .attr('width', x(d.importance)).attr('height', barHeight - 4)
                .attr('fill', colors[i] || colors[colors.length - 1])
                .attr('rx', 2);
              g.append('text')
                .attr('x', Math.min(x(d.importance) + 4, containerWidth - 90))
                .attr('y', barHeight - 6).attr('fill', '#1e3a5f')
                .attr('font-size', '10px')
                .text(d.name.length > 18 ? d.name.slice(0, 16) + '…' : d.name);
            });
          });
        }} />
      )}
    </div>
  );
}
