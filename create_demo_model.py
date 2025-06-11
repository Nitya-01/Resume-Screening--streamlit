import pickle
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

# Sample resume data for demonstration
sample_resumes = [
    # Fitness/Health resumes
    "personal trainer fitness gym workout exercise nutrition health wellness strength training cardio yoga",
    "fitness coach health wellness nutrition exercise science sports medicine rehabilitation physical therapy",
    "gym instructor fitness training workout programs health coaching nutrition advice exercise physiology",
    
    # IT/Tech resumes  
    "software developer python java javascript programming web development database SQL APIs",
    "network security engineer cybersecurity firewall intrusion detection VPN encryption CISSP",
    "data scientist machine learning python R statistics analytics big data visualization",
    
    # Business/Marketing resumes
    "marketing manager digital marketing SEO social media campaigns brand management advertising",
    "business analyst project management strategy consulting data analysis requirements gathering",
    "sales representative client relations CRM lead generation business development account management",
    
    # Healthcare resumes
    "registered nurse patient care medical records hospital clinical experience healthcare",
    "physician doctor medical degree residency patient diagnosis treatment healthcare",
    "pharmacist medication management pharmaceutical care drug interactions healthcare",
    
    # Engineering resumes
    "mechanical engineer CAD design manufacturing quality control project management engineering drawings",
    "civil engineer construction project management structural design AutoCAD surveying infrastructure",
    "electrical engineer circuit design power systems electronics troubleshooting automation control systems",

    # Finance/Accounting resumes
    "financial analyst investment banking financial modeling Excel VBA risk management portfolio analysis",
    "accountant bookkeeping tax preparation QuickBooks financial statements audit compliance CPA",
    "investment advisor wealth management client relations financial planning retirement planning securities",

    # Education resumes
    "teacher classroom management curriculum development lesson planning student assessment education technology",
    "professor research publications academic writing grant funding university teaching higher education",
    "school administrator educational leadership policy development staff management budget planning",

    # Legal resumes
    "attorney litigation contract law legal research court proceedings client representation bar exam",
    "paralegal legal documents case preparation research discovery legal software court filings",
    "corporate lawyer mergers acquisitions compliance regulatory law contract negotiation legal counsel",

    # Creative/Design resumes
    "graphic designer Adobe Creative Suite branding visual design marketing materials web design UX",
    "web developer HTML CSS JavaScript React Node.js responsive design frontend backend development",
    "content writer copywriting SEO content marketing blog writing social media creative writing",

    # Sales/Customer Service resumes
    "sales manager lead generation client acquisition CRM pipeline management territory management quotas",
    "customer service representative call center customer satisfaction problem solving communication skills",
    "retail manager inventory management staff supervision customer experience sales targets merchandising",

    # Human Resources resumes
    "HR manager recruitment employee relations performance management benefits administration policy development",
    "recruiter talent acquisition interviewing candidate screening applicant tracking systems sourcing",
    "training coordinator employee development learning management systems workshop facilitation onboarding",

    # Operations/Logistics resumes
    "operations manager supply chain logistics inventory management process improvement lean manufacturing",
    "logistics coordinator shipping receiving warehouse management transportation freight distribution",
    "project manager agile methodology stakeholder management risk assessment timeline management budgeting"
]

# Corresponding labels
labels = [
    'FITNESS', 'FITNESS', 'FITNESS',
    'INFORMATION-TECHNOLOGY', 'INFORMATION-TECHNOLOGY', 'INFORMATION-TECHNOLOGY', 
    'BUSINESS', 'BUSINESS', 'BUSINESS',
    'HEALTHCARE', 'HEALTHCARE', 'HEALTHCARE',
    'ENGINEERING', 'ENGINEERING', 'ENGINEERING',
    'FINANCE', 'FINANCE', 'FINANCE', 
    'EDUCATION', 'EDUCATION', 'EDUCATION',
    'LEGAL', 'LEGAL', 'LEGAL',
    'ARTS', 'ARTS', 'ARTS',
    'SALES', 'SALES', 'SALES',
    'HR', 'HR', 'HR',
    'OPERATIONS', 'OPERATIONS', 'OPERATIONS'
]

# Text cleaning function (same as in your app)
def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

# Clean the sample data
cleaned_resumes = [cleanResume(resume) for resume in sample_resumes]

# Create TF-IDF vectorizer
print("Creating TF-IDF vectorizer...")
tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
X = tfidf.fit_transform(cleaned_resumes)

# Create label encoder
print("Creating label encoder...")
le = LabelEncoder()
y = le.fit_transform(labels)

# Train SVM model
print("Training SVM model...")
svm_model = SVC(kernel='linear', probability=True, random_state=42)
svm_model.fit(X, y)

# Save the model and components
print("Saving model files...")
with open('clf.pkl', 'wb') as f:
    pickle.dump(svm_model, f)

with open('tfidf.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
    
with open('encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print("✅ Model files created successfully!")
print("Created files:")
print("- clf.pkl (SVM classifier)")
print("- tfidf.pkl (TF-IDF vectorizer)")  
print("- encoder.pkl (Label encoder)")
print("\nCategories the model can predict:")
for category in le.classes_:
    print(f"- {category}")
    
print("\nYou can now run your Streamlit app with: streamlit run app.py")