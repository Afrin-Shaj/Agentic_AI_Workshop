import axios from 'axios';

export const rankResumes = async (jdText, files) => {
  const formData = new FormData();
  formData.append('jd', jdText);
  files.forEach(file => formData.append('resumes', file));

  const response = await axios.post('http://localhost:5000/api/rank-resumes', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });

  return response.data;
};
