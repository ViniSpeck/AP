import os
import csv
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from email import message_from_string
from dateutil import parser
from dateutil.parser import ParserError

nltk.download('stopwords')
nltk.download('punkt')

class ETL:
    def __init__(self, spam_folder_path, ham_folder_path):
        self.spam_folder_path = spam_folder_path
        self.ham_folder_path = ham_folder_path
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.suspicious_words = ['money','free','ad','advertisement','investment','invest','urgent','deal','loan','click','subscribe','unsubscribe','bank','account','password','credit','card','congratulations','limited time','act now','exclusive','guaranteed','cash','viagra','discount','secret','weight loss']
    
    def extract(self):
        data = []
        for folder_path, is_spam in [(self.spam_folder_path, 1), (self.ham_folder_path, 0)]:
            for filename in os.listdir(folder_path):
                with open(os.path.join(folder_path, filename), 'r', encoding='utf-8', errors='ignore') as f:
                    email_text = f.read()
                    data.append(self.transform(email_text, is_spam))
        return data

    def transform(self, email_text, is_spam):
        email_lower = email_text.lower()

        tokens = word_tokenize(email_lower)
        
        tokens = [token for token in tokens if token not in self.stop_words]
        
        tokens_stem = [self.stemmer.stem(token) for token in tokens]
        
        num_uppercase = sum(1 for char in email_text if char.isupper())
        percent_uppercase = num_uppercase / len(email_text)
        
        msg = message_from_string(email_text)
        date_str = msg['Date']
        if date_str:
            try:
                date = parser.parse(date_str)
                time_only = date.strftime('%H:%M')
            except ParserError:
                time_only = None
        else:
            time_only = None


        num_words = len(tokens_stem)
        num_chars = len(email_text)
        num_hyperlinks = email_text.count('http')
        num_suspicious_words = sum(1 for token in tokens if token.lower() in self.suspicious_words)
        
        
        return {
            'is_spam': is_spam,
            'percent_uppercase': percent_uppercase,
            'num_words': num_words,
            'num_chars': num_chars,
            'num_hyperlinks': num_hyperlinks,
            'num_suspicious_words': num_suspicious_words,
            'sent_time': time_only,
        }
    
    def load(self, data, filename):
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)

spam_folder_path = 'raw_data/spam_emails'
ham_folder_path = 'raw_data/ham_emails'
etl_processor = ETL(spam_folder_path, ham_folder_path)
data = etl_processor.extract()
etl_processor.load(data, 'processed_emails.csv')
