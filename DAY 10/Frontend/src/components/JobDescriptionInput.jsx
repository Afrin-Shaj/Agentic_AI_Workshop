import React from 'react';

const JobDescriptionInput = ({ jd, setJd }) => {
  return (
    <div>
      <label>Job Description:</label>
      <textarea
        rows="6"
        value={jd}
        onChange={(e) => setJd(e.target.value)}
        placeholder="Paste your job description here..."
        style={{ width: '100%' }}
      />
    </div>
  );
};

export default JobDescriptionInput;
