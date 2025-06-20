import React, { useState } from 'react';

const ResultDisplay = ({ results }) => {
  if (!results) return null;

  const { approved = [], rejected = [], stats } = results.results;

  return (
    <div style={{ marginTop: '2rem' }}>
      <h2>📝 Resume Screening Report</h2>
      <p><strong>Total Candidates:</strong> {stats.total_candidates}</p>
      <p><strong>Approved:</strong> {stats.approved}</p>
      <p><strong>Rejected:</strong> {stats.rejected}</p>
      <p><strong>Approval Rate:</strong> {stats.approval_rate}%</p>

      <hr />

      {/* ---------- Approved Candidates ---------- */}
      <h3 style={{ color: 'green' }}>✅ Approved Candidates</h3>
      {approved.length > 0 ? (
        approved.map((candidate, index) => (
          <CandidateReport key={index} candidate={candidate} isRejected={false} />
        ))
      ) : (
        <p>No approved candidates found.</p>
      )}

      <hr style={{ margin: '2rem 0' }} />

      {/* ---------- Rejected Candidates ---------- */}
      <h3 style={{ color: 'red' }}>❌ Rejected Candidates</h3>
      {rejected.length > 0 ? (
        rejected.map((candidate, index) => (
          <CandidateReport key={index} candidate={candidate} isRejected={true} />
        ))
      ) : (
        <p>No rejected candidates found.</p>
      )}

      <hr />

      <button
        onClick={() => {
          const blob = new Blob([JSON.stringify(results, null, 2)], { type: 'application/json' });
          const link = document.createElement('a');
          link.href = URL.createObjectURL(blob);
          link.download = 'ranked_resumes.json';
          link.click();
        }}
        style={{ marginTop: '1rem' }}
      >
        📦 Download Full JSON Report
      </button>
    </div>
  );
};

const CandidateReport = ({ candidate, isRejected }) => {
  const [expanded, setExpanded] = useState(false);

  const data = isRejected ? candidate.original_data : candidate;
  const reasons = isRejected ? candidate.reasons : [];
  const { name, score, matched_skills, gaps, resume_data, score_breakdown } = data;

  const toggleExpand = () => setExpanded(prev => !prev);

  const generateReportText = () => {
    return `
${isRejected ? '❌ Rejected' : '✅ Approved'} Candidate: ${name}

${isRejected && reasons.length ? `🚫 Rejection Reasons:\n${reasons.map(r => `- ${r}`).join('\n')}` : ''}

📊 Score: ${score}
Skills: ${score_breakdown?.skills ?? '-'}
Experience: ${score_breakdown?.experience ?? '-'}
Education: ${score_breakdown?.education ?? '-'}
Projects: ${score_breakdown?.projects ?? '-'}

✅ Matched Skills:
${matched_skills?.join(', ') || '—'}

⚠️ Gaps:
${Object.entries(gaps || {})
  .map(([k, v]) => v.length ? `- ${k.replace(/_/g, ' ')}:\n  • ${v.join('\n  • ')}` : '')
  .join('\n')}

💡 Recommendations:
${gaps?.recommendations?.map(r => `- ${r}`).join('\n') || '—'}

📎 Contact Info:
Email: ${resume_data?.contact_info?.email ?? '-'}
Phone: ${resume_data?.contact_info?.phone ?? '-'}
LinkedIn: ${resume_data?.contact_info?.linkedin ?? '-'}
GitHub: ${resume_data?.contact_info?.github ?? '-'}
Location: ${resume_data?.contact_info?.location ?? '-'}
`.trim();
  };

  const downloadReport = () => {
    const blob = new Blob([generateReportText()], { type: 'text/plain' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `${name.replace(/\s/g, '_')}_report.txt`;
    link.click();
  };

  return (
    <div
      style={{
        backgroundColor: isRejected ? '#fff6f6' : '#f6fff9',
        margin: '1rem 0',
        padding: '1rem',
        border: `2px solid ${isRejected ? '#f44336' : '#4caf50'}`,
        borderRadius: '10px',
      }}
    >
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h4 style={{ margin: 0 }}>{isRejected ? '❌' : '✅'} {name}</h4>
        <div>
          <button onClick={toggleExpand} style={{ marginRight: '0.5rem' }}>
            {expanded ? 'Hide Details' : 'View Details'}
          </button>
          <button onClick={downloadReport}>📥 Download Report</button>
        </div>
      </div>

      {/* Expandable Content */}
      {expanded && (
        <div style={{ marginTop: '1rem' }}>
          <p><strong>Score:</strong> {score}</p>

          {isRejected && reasons.length > 0 && (
            <>
              <h5>🚫 Rejection Reasons</h5>
              <ul>{reasons.map((reason, idx) => <li key={idx}>{reason}</li>)}</ul>
            </>
          )}

          <h5>📊 Score Breakdown</h5>
          <ul>
            <li><strong>Skills:</strong> {score_breakdown?.skills ?? '-'}</li>
            <li><strong>Experience:</strong> {score_breakdown?.experience ?? '-'}</li>
            <li><strong>Education:</strong> {score_breakdown?.education ?? '-'}</li>
            <li><strong>Projects:</strong> {score_breakdown?.projects ?? '-'}</li>
          </ul>

          <h5>✅ Matched Skills</h5>
          <p>{matched_skills?.join(', ') || '—'}</p>

          <h5>⚠️ Gaps Identified</h5>
          {Object.entries(gaps || {}).map(([key, value]) =>
            value.length > 0 ? (
              <div key={key}>
                <strong>{key.replace(/_/g, ' ').toUpperCase()}:</strong>
                <ul>{value.map((item, idx) => <li key={idx}>{item}</li>)}</ul>
              </div>
            ) : null
          )}

          {gaps?.recommendations && (
            <>
              <h5>💡 Recommendations</h5>
              <ul>{gaps.recommendations.map((r, i) => <li key={i}>{r}</li>)}</ul>
            </>
          )}

          <h5>📎 Contact Info</h5>
          <ul>
            <li><strong>Email:</strong> {resume_data?.contact_info?.email ?? '-'}</li>
            <li><strong>Phone:</strong> {resume_data?.contact_info?.phone ?? '-'}</li>
            <li><strong>LinkedIn:</strong> {resume_data?.contact_info?.linkedin ?? '-'}</li>
            <li><strong>GitHub:</strong> {resume_data?.contact_info?.github ?? '-'}</li>
            <li><strong>Location:</strong> {resume_data?.contact_info?.location ?? '-'}</li>
          </ul>
        </div>
      )}
    </div>
  );
};

export default ResultDisplay;
