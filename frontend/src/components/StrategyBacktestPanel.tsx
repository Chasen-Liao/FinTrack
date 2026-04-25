import { useEffect, useMemo, useState } from 'react';
import axios from 'axios';
import { useTranslation } from 'react-i18next';

interface Trade {
  entry_date: string;
  exit_date: string;
  return: number;
}

interface StrategyResult {
  symbol: string;
  horizon: string;
  threshold: number;
  fee_rate: number;
  start_date: string;
  end_date: string;
  annual_return: number;
  max_drawdown: number;
  cumulative_return: number;
  buy_hold_return: number;
  win_rate: number;
  trade_count: number;
  average_trade_return: number;
  best_trade_return: number;
  worst_trade_return: number;
  meets_annual_return_target: boolean;
  meets_drawdown_target: boolean;
  meets_course_target: boolean;
  trades?: Trade[];
}

interface StrategyBestResponse {
  best?: StrategyResult;
  candidate_count: number;
  passing_count: number;
  top_candidates: StrategyResult[];
}

function formatPercent(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '-';
  return `${(value * 100).toFixed(digits)}%`;
}

function thresholdLabel(value: number): string {
  return value.toFixed(2);
}

function candidateKey(item: Pick<StrategyResult, 'symbol' | 'horizon' | 'threshold'>): string {
  return `${item.symbol}-${item.horizon}-${item.threshold}`;
}

function metricTone(value: number): 'up' | 'down' {
  return value >= 0 ? 'up' : 'down';
}

export default function StrategyBacktestPanel() {
  const { t } = useTranslation();
  const [summary, setSummary] = useState<StrategyBestResponse | null>(null);
  const [selected, setSelected] = useState<StrategyResult | null>(null);
  const [selectedKey, setSelectedKey] = useState('');
  const [loading, setLoading] = useState(true);
  const [detailLoading, setDetailLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    setLoading(true);
    setError('');
    axios
      .get('/api/predict/strategy-best')
      .then((res) => {
        const data = res.data as StrategyBestResponse;
        setSummary(data);
        if (data.best) {
          setSelected(data.best);
          setSelectedKey(candidateKey(data.best));
        }
      })
      .catch((err) => {
        setError(err.response?.data?.detail || t('strategy.noResult'));
      })
      .finally(() => setLoading(false));
  }, [t]);

  const candidates = useMemo(() => {
    if (!summary) return [];
    const seen = new Set<string>();
    const rows: StrategyResult[] = [];
    if (summary.best) {
      rows.push(summary.best);
      seen.add(candidateKey(summary.best));
    }
    for (const item of summary.top_candidates || []) {
      const key = candidateKey(item);
      if (seen.has(key)) continue;
      rows.push(item);
      seen.add(key);
    }
    return rows.slice(0, 16);
  }, [summary]);

  function loadCandidate(item: StrategyResult) {
    const key = candidateKey(item);
    setSelectedKey(key);
    setDetailLoading(true);
    setError('');
    axios
      .get(`/api/predict/${item.symbol}/strategy-backtest`, {
        params: { horizon: item.horizon, threshold: item.threshold },
      })
      .then((res) => setSelected(res.data as StrategyResult))
      .catch((err) => {
        setError(err.response?.data?.detail || t('strategy.unableToLoad'));
        setSelected(item);
      })
      .finally(() => setDetailLoading(false));
  }

  if (loading) {
    return (
      <div className="strategy-panel">
        <div className="strategy-header">
          <h2>{t('strategy.title')}</h2>
          <span className="strategy-subtle">{t('strategy.loading')}</span>
        </div>
      </div>
    );
  }

  if (error && !summary) {
    return (
      <div className="strategy-panel">
        <div className="strategy-header">
          <h2>{t('strategy.title')}</h2>
        </div>
        <div className="strategy-empty">
          <div>{error}</div>
          <code>python -m backend.ml.strategy_backtest --scan</code>
        </div>
      </div>
    );
  }

  return (
    <div className="strategy-panel">
      <div className="strategy-header">
        <div>
          <h2>{t('strategy.title')}</h2>
          <span className="strategy-subtle">
            {t('strategy.passingScanned', { passing: summary?.passing_count ?? 0, scanned: summary?.candidate_count ?? 0 })}
          </span>
        </div>
        {selected?.meets_course_target && <span className="strategy-pass-badge">{t('strategy.targetMet')}</span>}
      </div>

      {selected && (
        <div className="strategy-detail">
          <div className="strategy-title-row">
            <div>
              <span className="strategy-symbol">{selected.symbol}</span>
              <span className="strategy-meta">
                {selected.horizon.toUpperCase()} · {t('strategy.threshold')} {thresholdLabel(selected.threshold)}
              </span>
            </div>
            {detailLoading && <span className="strategy-subtle">{t('strategy.updating')}</span>}
          </div>

          <div className="strategy-window">
            {selected.start_date} ~ {selected.end_date}
          </div>

          <div className="strategy-metrics">
            <MetricCard
              label={t('strategy.annualReturn')}
              value={formatPercent(selected.annual_return)}
              tone={selected.annual_return >= 0.2 ? 'up' : 'down'}
            />
            <MetricCard
              label={t('strategy.maxDrawdown')}
              value={formatPercent(selected.max_drawdown)}
              tone={selected.max_drawdown < 0.2 ? 'up' : 'down'}
            />
            <MetricCard
              label={t('strategy.cumulative')}
              value={formatPercent(selected.cumulative_return)}
              tone={metricTone(selected.cumulative_return)}
            />
            <MetricCard
              label={t('strategy.buyHold')}
              value={formatPercent(selected.buy_hold_return)}
              tone={metricTone(selected.buy_hold_return)}
            />
            <MetricCard
              label={t('strategy.winRate')}
              value={formatPercent(selected.win_rate)}
              tone={selected.win_rate >= 0.5 ? 'up' : 'down'}
            />
            <MetricCard label={t('strategy.trades')} value={String(selected.trade_count)} />
          </div>

          <div className="strategy-target-checks">
            <CheckRow label={t('strategy.annualReturnTarget')} passed={selected.meets_annual_return_target} t={t} />
            <CheckRow label={t('strategy.maxDrawdownTarget')} passed={selected.meets_drawdown_target} t={t} />
            <CheckRow label={t('strategy.courseTarget')} passed={selected.meets_course_target} strong t={t} />
          </div>

          <div className="strategy-trade-summary">
            <span>{t('strategy.avg')} {formatPercent(selected.average_trade_return)}</span>
            <span>{t('strategy.best')} {formatPercent(selected.best_trade_return)}</span>
            <span>{t('strategy.worst')} {formatPercent(selected.worst_trade_return)}</span>
            <span>{t('strategy.fee')} {formatPercent(selected.fee_rate, 1)}</span>
          </div>
        </div>
      )}

      {error && summary && <div className="strategy-inline-error">{error}</div>}

      <div className="strategy-list">
        <div className="strategy-list-title">{t('strategy.candidates')}</div>
        {candidates.map((item, index) => {
          const key = candidateKey(item);
          const active = key === selectedKey;
          return (
            <button
              key={`${key}-${index}`}
              className={`strategy-row ${active ? 'active' : ''}`}
              onClick={() => loadCandidate(item)}
            >
              <div className="strategy-row-main">
                <span className="strategy-row-rank">#{index + 1}</span>
                <span className="strategy-row-symbol">{item.symbol}</span>
                <span className="strategy-row-config">
                  {item.horizon.toUpperCase()} / {thresholdLabel(item.threshold)}
                </span>
              </div>
              <div className="strategy-row-stats">
                <span className={item.annual_return >= 0 ? 'up' : 'down'}>
                  {t('strategy.ar')} {formatPercent(item.annual_return, 1)}
                </span>
                <span className={item.max_drawdown < 0.2 ? 'up' : 'down'}>
                  {t('strategy.dd')} {formatPercent(item.max_drawdown, 1)}
                </span>
                <span className={item.meets_course_target ? 'up' : 'down'}>
                  {item.meets_course_target ? t('strategy.pass') : t('strategy.fail')}
                </span>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function MetricCard({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone?: 'up' | 'down';
}) {
  return (
    <div className="strategy-metric-card">
      <span className="strategy-metric-label">{label}</span>
      <span className={`strategy-metric-value ${tone || ''}`}>{value}</span>
    </div>
  );
}

function CheckRow({
  label,
  passed,
  strong = false,
  t,
}: {
  label: string;
  passed: boolean;
  strong?: boolean;
  t: (key: string) => string;
}) {
  return (
    <div className={`strategy-check-row ${strong ? 'strong' : ''}`}>
      <span>{label}</span>
      <span className={passed ? 'up' : 'down'}>{passed ? t('strategy.pass') : t('strategy.fail')}</span>
    </div>
  );
}
