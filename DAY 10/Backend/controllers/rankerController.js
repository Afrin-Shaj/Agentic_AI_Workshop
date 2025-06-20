const axios = require('axios');
const FormData = require('form-data');
const Resume = require('../models/rankerModel');

const rankResumes = async (req, res) => {
  try {
    const { jd } = req.body;
    const resumes = req.files && req.files['resumes'] ? req.files['resumes'] : [];

    if (!jd || jd.trim() === '' || resumes.length === 0) {
      return res.status(400).json({ error: 'Job description and at least one resume are required' });
    }

    const form = new FormData();
    form.append('jd', jd);
    resumes.forEach((file) => {
      form.append('resumes', file.buffer, file.originalname);
    });

    const response = await axios.post(process.env.FASTAPI_URL, form, {
      headers: {
        ...form.getHeaders(),
      },
    });

    // Prepare candidates array combining approved and rejected
    const candidates = [
      ...response.data.results.approved.map(c => ({ ...c, status: 'approved' })),
      ...response.data.results.rejected.map(c => ({ 
        ...c.original_data, 
        status: 'rejected',
        rejection_reasons: c.reasons 
      }))
    ];

    const resumeData = new Resume({
      job_description: jd,
      candidates: candidates,
      filter_stats: response.data.results.stats,
    });

    await resumeData.save();

    res.status(200).json(response.data);
  } catch (error) {
    console.error('Error in rankResumes:', error.message);
    if (error.response) {
      console.error('FastAPI Response Data:', error.response.data);
    }
    res.status(500).json({ error: 'Internal server error' });
  }
};

module.exports = { rankResumes };