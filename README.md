# Resume-Screening-App
Resume Screening App With Python and Machine Learning 

# Resume Category Prediction

A machine learning application that automatically categorizes resumes into job types. Upload PDF, DOCX or TXT files and get instant predictions.

![Python](https://img.shields.io/badge/python-v3.7+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)

## Features

- Supports PDF, DOCX, and TXT file formats
- Predicts from 12 different job categories
- Built with Streamlit for easy web interface
- Uses Support Vector Machine for classification
- Real-time text extraction and processing

## Installation

Clone the repository:

```bash
git clone https://github.com/Nitya-01/Resume-Screening--streamlit.git
cd Resume-Screening--streamlit
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Generate the machine learning models:

```bash
python create_demo_model.py
```

Run the application:

```bash
streamlit run app.py
```

## Job Categories

The model can classify resumes into these categories:

- **ARTS** - Creative and design roles
- **BUSINESS** - Management and consulting
- **EDUCATION** - Teaching and academic positions
- **ENGINEERING** - All engineering disciplines
- **FINANCE** - Financial and accounting roles
- **FITNESS** - Health and fitness careers
- **HEALTHCARE** - Medical and healthcare positions
- **HR** - Human resources roles
- **INFORMATION-TECHNOLOGY** - Software and IT positions
- **LEGAL** - Legal and law-related roles
- **OPERATIONS** - Operations and logistics
- **SALES** - Sales and customer service

## How It Works

1. **Text Extraction**: Extracts text from uploaded resume files
2. **Text Cleaning**: Removes URLs, special characters and extra whitespace
3. **Vectorization**: Converts text to numerical features using TF-IDF
4. **Classification**: Uses trained SVM model to predict job category
5. **Results**: Displays the predicted category in the web interface

## Technical Details

- **Algorithm**: Support Vector Machine with linear kernel
- **Feature Extraction**: TF-IDF vectorization (max 1000 features)
- **Text Processing**: Regex-based cleaning
- **Training Data**: Synthetic resume samples across 12 categories
- **Libraries**: scikit-learn, Streamlit, PyPDF2, python-docx

## Usage Examples

For a fitness resume containing keywords like "personal trainer", "nutrition", "exercise" -> the model will predict **FITNESS**.

For a technical resume with "python", "programming", "software development" -> it will predict **INFORMATION-TECHNOLOGY**.

## Batch Processing

You can process multiple resumes programmatically:

```python
from app import pred, handle_file_upload
import os

results = {}
for filename in os.listdir("resume_folder"):
    if filename.endswith(('.pdf', '.docx', '.txt')):
        with open(f"resume_folder/{filename}", 'rb') as file:
            text = handle_file_upload(file)
            category = pred(text)
            results[filename] = category
```

### Current Limitations

- Model is trained on synthetic data (being upgraded with real resume data)
- Limited to 12 predefined categories (expanding soon)
- English language only
- Basic feature set (more features in development)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature-name`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add feature'`)
5. Push to the branch (`git push origin feature-name`)
6. Create a Pull Request

## Troubleshooting

**Module import errors**: Install all dependencies with `pip install -r requirements.txt`

**Missing model files**: Run `python create_demo_model.py` to generate them

**File upload issues**: Check that your file is under 200MB and not corrupted

**Poor predictions**: The model uses limited training data. Consider retraining with domain-specific samples.

## Requirements

- Python 3.7+
- streamlit
- scikit-learn
- python-docx
- PyPDF2
- numpy
- pandas

## Development Roadmap
## Current Status & Upcoming Improvements

**Note**: This project is actively being upgraded with improvements for V2:

- **Enhanced Dataset**: Currently integrating a comprehensive Kaggle dataset with real-world resume samples to replace synthetic training data
- **Expanded Categories**: Adding more job categories based on industry standards. Also widenng the categories in place of the current oversimplified ones.
- **Improved Accuracy**: Retraining models with larger and more diverse datasets
- **Additional Features**: Working on confidence scores, skill extraction and experience level detection
- A/B testing with real users/HR professionals

### Planned Features
- [ ] Batch file upload support
- [ ] REST API endpoint
- [ ] Integration with ATS systems
- [ ] Multi-language support
- [ ] Advanced analytics dashboard
- [ ] Export functionality (CSV, JSON)
- [ ] UI/UX integration and deployment for better multiple device usage.
