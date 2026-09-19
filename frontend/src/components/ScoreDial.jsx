import { scoreCategoryToStatus } from '../lib/format';

const COLOR_BY_STATUS = { good: 'var(--good)', warn: 'var(--warn)', critical: 'var(--critical)' };

export default function ScoreDial({ score, category }) {
  const status = scoreCategoryToStatus(category);
  const radius = 40;
  const circumference = 2 * Math.PI * radius;
  const fraction = Math.max(0, Math.min(1, score / 100));
  const dash = `${circumference * fraction} ${circumference}`;

  return (
    <div className="score-block">
      <div className="score-ring">
        <svg width="96" height="96" viewBox="0 0 96 96">
          <circle className="score-ring-track" cx="48" cy="48" r={radius} />
          <circle
            className="score-ring-value"
            cx="48" cy="48" r={radius}
            stroke={COLOR_BY_STATUS[status]}
            strokeDasharray={dash}
          />
        </svg>
        <div className="score-ring-number">{score}</div>
      </div>
      <div>
        <div className="score-category" style={{ color: COLOR_BY_STATUS[status] }}>{category}</div>
        <div className="score-caption">ML Readiness Score, out of 100</div>
      </div>
    </div>
  );
}
