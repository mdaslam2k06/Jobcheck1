import pandas as pd 
import numpy as np 
import re 
import os 
import warnings 
from collections import Counter 
# NLP Libraries 
import nltk 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer 
# Feature Extraction 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder 
# Visualization 
import matplotlib.pyplot as plt 
import seaborn as sns 
from wordcloud import WordCloud 
# Suppress warnings 
warnings.filterwarnings('ignore') 
# Download required NLTK data 
print("Downloading NLTK resources...") 
nltk.download('punkt', quiet=True) 
nltk.download('stopwords', quiet=True) 
nltk.download('wordnet', quiet=True) 
nltk.download('punkt_tab', quiet=True) 
 
# Set plotting style 
plt.style.use('seaborn-v0_8-whitegrid') 
sns.set_palette("husl") 
 
 
class FakeJobDataProcessor: 
    """ 
    Complete data preprocessing pipeline for fake job detection. 
    """ 
     
    def __init__(self, data_path=None): 
        """ 
        Initialize the data processor. 
         
        Args: 
            data_path: Path to the dataset CSV file 
        """ 
        self.data_path = data_path 
        self.df = None 
        self.df_processed = None 
        self.tfidf_vectorizer = None 
        self.X_train = None 
        self.X_test = None 
        self.y_train = None 
        self.y_test = None 
        self.feature_matrix = None 
         
        # Initialize NLP tools 
        self.lemmatizer = WordNetLemmatizer() 
        self.stop_words = set(stopwords.words('english')) 
         
        # Add custom stop words relevant to job postings 
        self.custom_stop_words = { 
            'job', 'work', 'company', 'position', 'apply', 'experience', 
            'candidate', 'role', 'opportunity', 'team', 'looking' 
        } 
        self.stop_words.update(self.custom_stop_words) 
     
    def load_data(self): 
        """ 
        Load the dataset from CSV file or create sample data for demonstration. 
        """ 
        print("\n" + "="*60, flush=True) 
        print("STEP 1: LOADING DATA", flush=True) 
        print("="*60, flush=True) 
         
        if self.data_path and os.path.exists(self.data_path): 
            self.df = pd.read_csv(self.data_path) 
            print(f"✓ Loaded dataset from {self.data_path}", flush=True) 
        else: 
            # Create sample dataset for demonstration 
            print("✓ Creating sample dataset for demonstration...", flush=True) 
            self.df = self._create_sample_data() 
         
        print(f"✓ Dataset shape: {self.df.shape}", flush=True) 
        print(f"✓ Columns: {list(self.df.columns)}", flush=True) 
         
        return self.df 
     
    def _create_sample_data(self): 
        """ 
        Create a sample dataset mimicking the Kaggle fake job postings dataset. 
        """ 
        np.random.seed(42) 
         
        # Real job posting examples 
        real_jobs = [ 
            { 
                'title': 'Senior Software Engineer', 
                'location': 'San Francisco, CA', 
                'department': 'Engineering', 
                'company_profile': 'We are a leading tech company with 500+ employees focused on building innovative solutions.', 
                'description': 'We are looking for an experienced software engineer to join our team. You will be responsible for designing and implementing scalable systems. Requirements include 5+ years of experience with Python, Java, or C++.', 
                'requirements': 'Bachelor\'s degree in Computer Science. 5+ years of software development experience. Strong problem-solving skills.', 
                'benefits': 'Competitive salary, health insurance, 401k matching, flexible work hours, remote work options.', 
                'employment_type': 'Full-time', 
                'required_experience': 'Mid-Senior level', 
                'required_education': "Bachelor's Degree", 
                'industry': 'Information Technology', 
                'function': 'Engineering', 
                'fraudulent': 0 
            }, 
            { 
                'title': 'Marketing Manager', 
                'location': 'New York, NY', 
                'department': 'Marketing', 
                'company_profile': 'Fortune 500 retail company with a strong presence across North America.', 
                'description': 'Lead our marketing team in developing and executing marketing strategies. Manage a team of 5 marketing specialists.', 
                'requirements': 'MBA preferred. 7+ years of marketing experience. Experience with digital marketing and analytics.', 
                'benefits': 'Base salary plus bonus, comprehensive benefits package, professional development opportunities.', 
                'employment_type': 'Full-time', 
                'required_experience': 'Director', 
                'required_education': "Master's Degree", 
                'industry': 'Retail', 
                'function': 'Marketing', 
                'fraudulent': 0 
            }, 
            { 
                'title': 'Data Analyst', 
                'location': 'Chicago, IL', 
                'department': 'Analytics', 
                'company_profile': 'Fast-growing fintech startup disrupting the payments industry.', 
                'description': 'Analyze large datasets to provide actionable insights. Create dashboards and reports for stakeholders.', 
                'requirements': 'Strong SQL skills. Experience with Python or R. Knowledge of visualization tools like Tableau.', 
                'benefits': 'Stock options, health benefits, unlimited PTO, free lunch.', 
                'employment_type': 'Full-time', 
                'required_experience': 'Entry level', 
                'required_education': "Bachelor's Degree", 
                'industry': 'Financial Services', 
                'function': 'Analyst', 
                'fraudulent': 0 
            }, 
            { 
                'title': 'Product Designer', 
                'location': 'Seattle, WA', 
                'department': 'Design', 
                'company_profile': 'Award-winning design agency working with top brands worldwide.', 
                'description': 'Create user-centered designs for web and mobile applications. Conduct user research and usability testing.', 
                'requirements': 'Portfolio demonstrating UX/UI skills. Proficiency in Figma, Sketch. 3+ years experience.', 
                'benefits': 'Creative environment, competitive salary, annual design conference attendance.', 
                'employment_type': 'Full-time', 
                'required_experience': 'Associate', 
                'required_education': "Bachelor's Degree", 
                'industry': 'Design', 
                'function': 'Design', 
                'fraudulent': 0 
            }, 
            { 
                'title': 'Sales Representative', 
                'location': 'Austin, TX', 
                'department': 'Sales', 
                'company_profile': 'Enterprise software company with 1000+ customers globally.', 
                'description': 'Drive new business development and manage customer relationships. Meet quarterly sales targets.', 
                'requirements': 'Proven sales track record. Excellent communication skills. CRM experience preferred.', 
                'benefits': 'Base plus commission, car allowance, president\'s club trips.', 
                'employment_type': 'Full-time', 
                'required_experience': 'Mid-Senior level', 
                'required_education': "Bachelor's Degree", 
                'industry': 'Computer Software', 
                'function': 'Sales', 
                'fraudulent': 0 
            } 
        ] 
         
        # Fake job posting examples 
        fake_jobs = [ 
            { 
                'title': 'Work From Home - Earn $5000 Weekly!!!', 
                'location': 'Anywhere', 
                'department': '', 
                'company_profile': '', 
                'description': 'AMAZING OPPORTUNITY!!! Make money from home! No experience needed! Start TODAY! Just send us your bank details to get started. This is NOT a scam!', 
                'requirements': 'None! Anyone can do this!', 
                'benefits': 'UNLIMITED INCOME POTENTIAL!!!', 
                'employment_type': 'Contract', 
                'required_experience': 'Not Applicable', 
                'required_education': 'Unspecified', 
                'industry': '', 
                'function': '', 
                'fraudulent': 1 
            }, 
            { 
                'title': 'Personal Assistant Needed URGENTLY', 
                'location': 'Remote', 
                'department': '', 
                'company_profile': 'International business executive', 
                'description': 'Need assistant to handle financial transactions. Will receive checks and wire money. Easy work! Pay $500/week. Send resume with SSN for background check.', 
                'requirements': 'Must have bank account', 
                'benefits': 'Cash payments', 
                'employment_type': 'Part-time', 
                'required_experience': 'Not Applicable', 
                'required_education': 'Unspecified', 
                'industry': '', 
                'function': '', 
                'fraudulent': 1 
            }, 
            { 
                'title': 'Data Entry - $30/hour - Start Immediately!', 
                'location': 'Work from anywhere', 
                'department': '', 
                'company_profile': '', 
                'description': 'Simple data entry work. We provide training. Just need your personal information to register you in our system. Contact us on WhatsApp for more details.', 
                'requirements': 'Computer and internet', 
                'benefits': 'Flexible hours. Weekly pay via Western Union.', 
                'employment_type': 'Part-time', 
                'required_experience': 'Not Applicable', 
                'required_education': 'High School', 
                'industry': '', 
                'function': '', 
                'fraudulent': 1 
            }, 
            { 
                'title': 'Mystery Shopper - FREE Products!', 
                'location': 'Nationwide', 
                'department': '', 
                'company_profile': 'Leading market research company', 
                'description': 'Get paid to shop! Receive $200 for evaluating stores. Send registration fee of $50 to get started. Money back guarantee!!!', 
                'requirements': 'Must pay registration fee upfront', 
                'benefits': 'Free products, cash payments', 
                'employment_type': 'Freelance', 
                'required_experience': 'Not Applicable', 
                'required_education': 'Unspecified', 
                'industry': '', 
                'function': '', 
                'fraudulent': 1 
            }, 
            { 
                'title': 'Administrative Assistant - Foreign Company', 
                'location': 'Remote position', 
                'department': '', 
                'company_profile': 'Overseas company expanding to US', 
                'description': 'Help process payments for our international clients. Receive payments in your account and forward to designated accounts. Commission based.', 
                'requirements': 'US bank account required. Must be 18+.', 
                'benefits': '10% commission on all transactions', 
                'employment_type': 'Contract', 
                'required_experience': 'Not Applicable', 
                'required_education': 'Unspecified', 
                'industry': '', 
                'function': '', 
                'fraudulent': 1 
            } 
        ] 
         
        # Generate more samples by duplicating with variations 
        all_jobs = [] 
         
        # Add more real jobs (multiply by 15 to get ~75 real jobs) 
        for i in range(15): 
            for job in real_jobs: 
                job_copy = job.copy() 
                # Add slight variations 
                if i % 3 == 0: 
                    job_copy['location'] = np.random.choice(['Boston, MA', 'Denver, CO', 'Miami, FL', 'Portland, OR']) 
                all_jobs.append(job_copy) 
         
        # Add more fake jobs (multiply by 3 to get ~15 fake jobs for realistic imbalance) 
        for i in range(3): 
            for job in fake_jobs: 
                job_copy = job.copy() 
                all_jobs.append(job_copy) 
         
        df = pd.DataFrame(all_jobs) 
         
        # Add job_id column 
        df['job_id'] = range(1, len(df) + 1) 
         
        # Reorder columns 
        cols = ['job_id', 'title', 'location', 'department', 'company_profile',  
                'description', 'requirements', 'benefits', 'employment_type', 
                'required_experience', 'required_education', 'industry', 'function', 'fraudulent'] 
        df = df[cols] 
         
        # Shuffle the dataframe 
        df = df.sample(frac=1, random_state=42).reset_index(drop=True) 
         
        print(f"  → Created {len(df)} sample job postings") 
        print(f"  → Real jobs: {len(df[df['fraudulent']==0])}") 
        print(f"  → Fake jobs: {len(df[df['fraudulent']==1])}") 
         
        return df 
     
    def explore_data(self): 
        """ 
        Perform comprehensive Exploratory Data Analysis. 
        """ 
        print("\n" + "="*60, flush=True) 
        print("STEP 2: EXPLORATORY DATA ANALYSIS", flush=True) 
        print("="*60, flush=True) 
         
        # Basic info 
        print("\n--- Dataset Information ---") 
        print(f"Total samples: {len(self.df)}") 
        print(f"Total features: {len(self.df.columns)}") 
         
        # Target distribution 
        print("\n--- Target Distribution ---") 
        target_counts = self.df['fraudulent'].value_counts() 
        print(f"Real jobs (0): {target_counts.get(0, 0)} ({target_counts.get(0, 
0)/len(self.df)*100:.1f}%)") 
        print(f"Fake jobs (1): {target_counts.get(1, 0)} ({target_counts.get(1, 
0)/len(self.df)*100:.1f}%)") 
         
        # Missing values 
        print("\n--- Missing Values ---") 
        missing = self.df.isnull().sum() 
        missing_pct = (missing / len(self.df) * 100).round(2) 
        missing_df = pd.DataFrame({'Missing': missing, 'Percentage': missing_pct}) 
        missing_df = missing_df[missing_df['Missing'] > 0].sort_values('Missing', ascending=False) 
        if len(missing_df) > 0: 
            print(missing_df) 
        else: 
            print("No missing values found!") 
         
        # Text length analysis 
        print("\n--- Text Length Analysis ---") 
        text_cols = ['title', 'description', 'requirements', 'company_profile'] 
        for col in text_cols: 
            if col in self.df.columns: 
                self.df[f'{col}_length'] = self.df[col].fillna('').str.len() 
                mean_len = self.df[f'{col}_length'].mean() 
                print(f"{col}: avg length = {mean_len:.0f} characters") 
         
        # Employment type distribution 
        if 'employment_type' in self.df.columns: 
            print("\n--- Employment Type Distribution ---") 
            print(self.df['employment_type'].value_counts()) 
         
        self._create_visualizations() 
         
        return self.df 
     
    def _create_visualizations(self): 
        """ 
        Create and save EDA visualizations. 
        """ 
        print("\n--- Creating Visualizations ---") 
         
        # Create figure with subplots 
        fig = plt.figure(figsize=(16, 12)) 
         
        # 1. Target distribution 
        ax1 = fig.add_subplot(2, 3, 1) 
        colors = ['#2ecc71', '#e74c3c'] 
        target_counts = self.df['fraudulent'].value_counts() 
        bars = ax1.bar(['Real', 'Fake'], [target_counts.get(0, 0), target_counts.get(1, 0)], 
color=colors) 
        ax1.set_title('Job Post Distribution', fontsize=12, fontweight='bold') 
        ax1.set_ylabel('Count') 
        for bar in bars: 
            height = bar.get_height() 
            ax1.text(bar.get_x() + bar.get_width()/2., height, 
                    f'{int(height)}', ha='center', va='bottom') 
         
        # 2. Description length distribution 
        ax2 = fig.add_subplot(2, 3, 2) 
        if 'description_length' in self.df.columns: 
            real_lens = self.df[self.df['fraudulent']==0]['description_length'] 
            fake_lens = self.df[self.df['fraudulent']==1]['description_length'] 
            ax2.hist([real_lens, fake_lens], bins=20, label=['Real', 'Fake'], color=colors, alpha=0.7) 
            ax2.set_title('Description Length Distribution', fontsize=12, fontweight='bold') 
            ax2.set_xlabel('Character Count') 
            ax2.set_ylabel('Frequency') 
            ax2.legend() 
         
        # 3. Employment type by fraud status 
        ax3 = fig.add_subplot(2, 3, 3) 
        if 'employment_type' in self.df.columns: 
            emp_fraud = pd.crosstab(self.df['employment_type'], self.df['fraudulent']) 
            emp_fraud.plot(kind='bar', ax=ax3, color=colors) 
            ax3.set_title('Employment Type by Status', fontsize=12, fontweight='bold') 
            ax3.set_xlabel('Employment Type') 
            ax3.set_ylabel('Count') 
            ax3.legend(['Real', 'Fake']) 
            ax3.tick_params(axis='x', rotation=45) 
         
        # 4. Word cloud for real jobs 
        ax4 = fig.add_subplot(2, 3, 4) 
        real_text = ' '.join(self.df[self.df['fraudulent']==0]['description'].fillna('').tolist()) 
        if real_text.strip(): 
            wordcloud_real = WordCloud(width=400, height=300, background_color='white', 
                                       colormap='Greens', max_words=50).generate(real_text) 
            ax4.imshow(wordcloud_real, interpolation='bilinear') 
            ax4.set_title('Word Cloud: Real Jobs', fontsize=12, fontweight='bold') 
        ax4.axis('off') 
         
        # 5. Word cloud for fake jobs 
        ax5 = fig.add_subplot(2, 3, 5) 
        fake_text = ' '.join(self.df[self.df['fraudulent']==1]['description'].fillna('').tolist()) 
        if fake_text.strip(): 
            wordcloud_fake = WordCloud(width=400, height=300, background_color='white', 
                                       colormap='Reds', max_words=50).generate(fake_text) 
            ax5.imshow(wordcloud_fake, interpolation='bilinear') 
            ax5.set_title('Word Cloud: Fake Jobs', fontsize=12, fontweight='bold') 
        ax5.axis('off') 
         
        # 6. Missing data heatmap 
        ax6 = fig.add_subplot(2, 3, 6) 
        text_cols = ['title', 'company_profile', 'description', 'requirements', 'benefits'] 
        text_cols = [c for c in text_cols if c in self.df.columns] 
        missing_matrix = self.df[text_cols].isnull().astype(int) 
        sns.heatmap(missing_matrix.T, cmap='YlOrRd', cbar=True, ax=ax6, 
                   yticklabels=text_cols) 
        ax6.set_title('Missing Data Pattern', fontsize=12, fontweight='bold') 
        ax6.set_xlabel('Sample Index') 
         
        plt.tight_layout() 
        plt.savefig('eda_visualizations.png', dpi=150, bbox_inches='tight') 
        plt.close() 
        print("✓ Saved visualizations to 'eda_visualizations.png'") 
     
    def clean_text(self, text): 
        """ 
        Clean and preprocess text data. 
         
        Args: 
            text: Input text string 
             
        Returns: 
            Cleaned text string 
        """ 
        if pd.isna(text) or text == '': 
            return '' 
         
        # Convert to lowercase 
        text = str(text).lower() 
         
        # Remove HTML tags 
        text = re.sub(r'<[^>]+>', '', text) 
         
        # Remove URLs 
        text = re.sub(r'http\S+|www\S+', '', text) 
         
        # Remove email addresses 
        text = re.sub(r'\S+@\S+', '', text) 
         
        # Remove special characters and digits 
        text = re.sub(r'[^a-zA-Z\s]', '', text) 
         
        # Remove extra whitespace 
        text = re.sub(r'\s+', ' ', text).strip() 
         
        # Tokenize 
        tokens = word_tokenize(text) 
         
        # Remove stopwords and lemmatize 
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens  
                 if token not in self.stop_words and len(token) > 2] 
         
        return ' '.join(tokens) 
     
    def preprocess_data(self): 
        """ 
        Apply text preprocessing to all text columns. 
        """ 
        print("\n" + "="*60, flush=True) 
        print("STEP 3: TEXT PREPROCESSING", flush=True) 
        print("="*60, flush=True) 
         
        self.df_processed = self.df.copy() 
         
        # Combine text columns for main feature 
        text_columns = ['title', 'company_profile', 'description', 'requirements', 'benefits'] 
        text_columns = [col for col in text_columns if col in self.df_processed.columns] 
         
        print(f"Combining text from columns: {text_columns}") 
         
        # Combine all text fields 
        self.df_processed['combined_text'] = '' 
        for col in text_columns: 
            self.df_processed['combined_text'] += ' ' + self.df_processed[col].fillna('') 
         
        # Clean the combined text 
        print("Cleaning text data...") 
        self.df_processed['cleaned_text'] = self.df_processed['combined_text'].apply(self.clean_text) 
         
        # Show sample 
        print("\n--- Sample Cleaned Text ---") 
        sample_idx = self.df_processed[self.df_processed['fraudulent']==0].index[0] 
        print(f"Original (truncated): {self.df_processed.loc[sample_idx, 'combined_text'][:200]}...") 
        print(f"Cleaned (truncated): {self.df_processed.loc[sample_idx, 'cleaned_text'][:200]}...") 
         
        # Text statistics after cleaning 
        self.df_processed['word_count'] = self.df_processed['cleaned_text'].str.split().str.len() 
         
        print("\n--- Word Count Statistics ---") 
        print(f"Average words (Real): {self.df_processed[self.df_processed['fraudulent']==0]['word_count'].mean():.0f}") 
        print(f"Average words (Fake): {self.df_processed[self.df_processed['fraudulent']==1]['word_count'].mean():.0f}") 
         
        # Remove empty texts 
        empty_count = (self.df_processed['cleaned_text'] == '').sum() 
        if empty_count > 0: 
            print(f"\n⚠ Found {empty_count} empty texts after cleaning") 
            self.df_processed = self.df_processed[self.df_processed['cleaned_text'] != ''] 
         
        print(f"\n✓ Preprocessing complete. Final dataset size: {len(self.df_processed)}") 
         
        return self.df_processed 
     
    def extract_features(self, max_features=5000): 
        """ 
        Extract TF-IDF features from cleaned text. 
         
        Args: 
            max_features: Maximum number of TF-IDF features 
        """ 
        print("\n" + "="*60, flush=True) 
        print("STEP 4: FEATURE EXTRACTION (TF-IDF)", flush=True) 
        print("="*60, flush=True) 
         
        # Initialize TF-IDF Vectorizer 
        self.tfidf_vectorizer = TfidfVectorizer( 
            max_features=max_features, 
            ngram_range=(1, 2),  # Unigrams and bigrams 
            min_df=2,           # Minimum document frequency 
            max_df=0.95,        # Maximum document frequency 
            sublinear_tf=True   # Apply sublinear tf scaling 
        ) 
         
        # Fit and transform 
        print(f"Extracting TF-IDF features (max_features={max_features})...") 
        self.feature_matrix = self.tfidf_vectorizer.fit_transform(self.df_processed['cleaned_text']) 
         
        print(f"✓ Feature matrix shape: {self.feature_matrix.shape}") 
        print(f"✓ Number of features: {len(self.tfidf_vectorizer.get_feature_names_out())}") 
         
        # Show top features 
        print("\n--- Top 20 TF-IDF Features ---") 
        feature_names = self.tfidf_vectorizer.get_feature_names_out() 
        avg_tfidf = np.array(self.feature_matrix.mean(axis=0)).flatten() 
        top_indices = avg_tfidf.argsort()[-20:][::-1] 
        for i, idx in enumerate(top_indices, 1): 
            print(f"{i}. {feature_names[idx]}: {avg_tfidf[idx]:.4f}") 
         
        return self.feature_matrix 
     
    def prepare_train_test_split(self, test_size=0.2, random_state=42): 
        """ 
        Split data into training and testing sets. 
         
        Args: 
            test_size: Proportion of data for testing 
            random_state: Random seed for reproducibility 
        """ 
        print("\n" + "="*60, flush=True) 
        print("STEP 5: TRAIN/TEST SPLIT", flush=True) 
        print("="*60, flush=True) 
         
        X = self.feature_matrix 
        y = self.df_processed['fraudulent'].values 
         
        # Encode labels (already 0/1, but ensuring consistency) 
        self.label_encoder = LabelEncoder() 
        y_encoded = self.label_encoder.fit_transform(y) 
         
        # Split 
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split( 
            X, y_encoded,  
            test_size=test_size,  
            random_state=random_state, 
            stratify=y_encoded  # Maintain class distribution 
        ) 
         
        print(f"✓ Training set size: {self.X_train.shape[0]} samples") 
        print(f"✓ Testing set size: {self.X_test.shape[0]} samples") 
        print(f"✓ Feature dimensions: {self.X_train.shape[1]}") 
         
        # Class distribution in splits 
        print("\n--- Class Distribution ---") 
        print(f"Training - Real: {sum(self.y_train==0)}, Fake: {sum(self.y_train==1)}") 
        print(f"Testing  - Real: {sum(self.y_test==0)}, Fake: {sum(self.y_test==1)}") 
         
        return self.X_train, self.X_test, self.y_train, self.y_test 
     
    def save_processed_data(self, output_dir='processed_data'): 
        """ 
        Save processed data and artifacts for Milestone 2. 
        """ 
        print("\n" + "="*60, flush=True) 
        print("STEP 6: SAVING PROCESSED DATA", flush=True) 
        print("="*60, flush=True) 
         
        # Create output directory 
        os.makedirs(output_dir, exist_ok=True) 
         
        # Save processed dataframe 
        self.df_processed.to_csv(f'{output_dir}/processed_jobs.csv', index=False) 
        print(f"✓ Saved processed data to '{output_dir}/processed_jobs.csv'") 
         
        # Save train/test splits using numpy 
        np.save(f'{output_dir}/X_train.npy', self.X_train.toarray()) 
        np.save(f'{output_dir}/X_test.npy', self.X_test.toarray()) 
        np.save(f'{output_dir}/y_train.npy', self.y_train) 
        np.save(f'{output_dir}/y_test.npy', self.y_test) 
        print(f"✓ Saved train/test splits to '{output_dir}/'") 
         
        # Save vectorizer using pickle 
        import pickle 
        with open(f'{output_dir}/tfidf_vectorizer.pkl', 'wb') as f: 
            pickle.dump(self.tfidf_vectorizer, f) 
        print(f"✓ Saved TF-IDF vectorizer to '{output_dir}/tfidf_vectorizer.pkl'") 
         
        print("\n✓ All data saved successfully!") 
         
        return output_dir 
     
 
def main(): 
    """ 
    Main function to run the complete Milestone 1 pipeline. 
    """ 
    print("\n" + "="*60, flush=True) 
    print("FAKE JOB POST DETECTION SYSTEM", flush=True) 
    print("Milestone 1: Data Preprocessing & Exploration", flush=True) 
    print("="*60, flush=True) 
     
    # Initialize processor 
    # You can pass a path to your dataset here: 
    # processor = FakeJobDataProcessor('path/to/fake_job_postings.csv') 
    processor = FakeJobDataProcessor() 
     
    # Run pipeline 
    processor.load_data() 
    processor.explore_data() 
    processor.preprocess_data() 
    processor.extract_features(max_features=3000) 
    processor.prepare_train_test_split(test_size=0.2) 
    processor.save_processed_data() 
     
    print("\n" + "="*60, flush=True) 
    print("MILESTONE 1 COMPLETED SUCCESSFULLY!", flush=True) 
    print("="*60, flush=True) 
     
    return processor 
 
 
if __name__ == "__main__": 
    processor = main()