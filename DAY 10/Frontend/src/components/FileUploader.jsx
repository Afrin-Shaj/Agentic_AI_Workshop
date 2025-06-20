import React from 'react';

const FileUploader = ({ setFiles }) => {
  const handleChange = (e) => {
    setFiles(Array.from(e.target.files));
  };

  return (
    <div>
      <label>Upload Resumes (PDF):</label>
      <input type="file" multiple accept=".pdf" onChange={handleChange} />
    </div>
  );
};

export default FileUploader;
