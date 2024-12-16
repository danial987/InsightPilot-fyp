import speech_recognition as sr
import io
import pandas as pd
import openai
import uuid
import streamlit as st

openai.api_key = 'your-openai-api-key-here'

class Chatbot:
    def __init__(self):
        """Initialize the chatbot, setting up session state and chat history."""
        self.initialize_session()
        self.recognizer = sr.Recognizer() 


    def initialize_session(self):
        """Initialize session state variables."""
        if 'session_id' not in st.session_state:
            st.session_state['session_id'] = str(uuid.uuid4())

        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []


    def load_dataset(self):
        """Load the dataset from session state, return the DataFrame and its name."""
        dataset_data = st.session_state.get('df_to_chat', None)
        dataset_name = st.session_state.get('dataset_name_to_chat', None)

        if dataset_data is None:
            st.error("No dataset found in session state.")
            return None, None

        df = self.convert_dataset_to_dataframe(dataset_data)
        return df, dataset_name


    def convert_dataset_to_dataframe(self, dataset_data):
        """Convert the dataset bytes into a pandas DataFrame."""
        try:
            if isinstance(dataset_data, bytes):
                try:
                    return pd.read_csv(io.BytesIO(dataset_data), encoding='utf-8', delimiter=',', on_bad_lines='skip')
                except UnicodeDecodeError:
                    try:
                        return pd.read_csv(io.BytesIO(dataset_data), encoding='ISO-8859-1', delimiter=',', on_bad_lines='skip')
                    except Exception as e:
                        st.error(f"Failed to read CSV file (encoding issue): {str(e)}")
                        return None
                except pd.errors.ParserError as e:
                    st.warning(f"ParserError occurred: {e}. Trying different delimiters.")
                    try:
                        return pd.read_csv(io.BytesIO(dataset_data), encoding='utf-8', delimiter=';', on_bad_lines='skip')  
                    except pd.errors.ParserError as e:
                        try:
                            return pd.read_csv(io.BytesIO(dataset_data), encoding='utf-8', delimiter='\t', on_bad_lines='skip') 
                        except Exception as e:
                            st.error(f"Failed to read CSV with semicolon or tab delimiter: {str(e)}")
                            return None
                except Exception as e:
                    st.error(f"Failed to load CSV file: {str(e)}")
                    return None
            else:
                return dataset_data
        except Exception as e:
            st.error(f"Failed to load dataset: {str(e)}")
            return None


    def call_openai(self, prompt, df):
        """Generate detailed responses using OpenAI API with dataset context."""
        full_prompt = self.build_openai_prompt(prompt, df)
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": "You are an AI that helps answer questions related to datasets, providing detailed, elaborative explanations."},
                          {"role": "user", "content": full_prompt}],
                max_tokens=200,
                temperature=0.6
            )
            return response['choices'][0]['message']['content'].strip()
        except Exception as e:
            return f"Error calling OpenAI API: {str(e)}"


    def build_openai_prompt(self, user_prompt, df):
        """Build the prompt for OpenAI by including detailed context from the dataset."""
        dataset_summary = self.generate_dataset_summary(df)
        full_prompt = f"{user_prompt}\n\nDataset Context:\n{dataset_summary}"
        return full_prompt


    def generate_dataset_summary(self, df):
        """Generate a detailed summary of the dataset, including column details and basic statistics."""
        try:
            column_info = self.get_column_information(df)
            missing_values_info = self.get_missing_values_info(df)
            numeric_summary = self.get_numeric_summary(df)

            dataset_summary = (
                f"Columns in the dataset:\n{column_info}\n\n"
                f"Missing values:\n{missing_values_info}\n\n"
                f"Summary statistics for numeric columns:\n{numeric_summary}\n\n"
                f"First 5 rows of the dataset:\n{df.head(5).to_string(index=False)}"
            )
            return dataset_summary
        except Exception as e:
            return f"Error generating dataset summary: {str(e)}"


    def get_column_information(self, df):
        """Get detailed information about dataset columns, including their types."""
        column_info = []
        for col in df.columns:
            dtype = df[col].dtype
            column_info.append(f"- {col}: {dtype}")
        return "\n".join(column_info)


    def get_missing_values_info(self, df):
        """Check for missing values in the dataset."""
        missing_info = df.isnull().sum()
        if missing_info.sum() == 0:
            return "No missing values in the dataset."
        else:
            missing_report = "\n".join([f"- {col}: {missing_info[col]} missing values" for col in df.columns if missing_info[col] > 0])
            return missing_report


    def get_numeric_summary(self, df):
        """Generate summary statistics for numeric columns in the dataset."""
        numeric_columns = df.select_dtypes(include=['float64', 'int64'])
        if numeric_columns.empty:
            return "No numeric columns found in the dataset."
        else:
            return numeric_columns.describe().to_string()


    def record_audio(self):
        """Record audio and return the transcribed text."""
        with sr.Microphone() as source:
            st.info("Say something...")
            audio = self.recognizer.listen(source)
            try:
                user_input = self.recognizer.recognize_google(audio)
                return user_input
            except sr.UnknownValueError:
                st.error("Sorry, I could not understand the audio.")
                return None
            except sr.RequestError:
                st.error("There was an error with the speech recognition service.")
                return None


    def run(self):
        """Run the chatbot, handling user input and generating responses."""
        with st.spinner("Loading Please Wait ..."):

            df, dataset_name = self.load_dataset()
    
            if df is not None and dataset_name is not None:
                st.header(f"Chat with your Dataset: {dataset_name} 🧠", divider='violet')
    
                user_input = st.chat_input(f"Ask a question about {dataset_name}!")
    
                if user_input:
                    self.append_chat_history('user', user_input)
                    response = self.call_openai(user_input, df)
                    self.append_chat_history('assistant', response)
    
                for chat in st.session_state['chat_history']:
                    st.chat_message(chat['role']).write(chat['message'])
    
                self.voice_input_icon()
    
                self.export_chat_history()
            else:
                st.error("No dataset selected. Please go back and select a dataset.")


    def append_chat_history(self, role, message):
        """Append the user or assistant message to the chat history."""
        st.session_state['chat_history'].append({'role': role, 'message': message})
        st.chat_message(role).write(message)


    def export_chat_history(self):
        """Allow the user to export the chat history as a text file."""
        if st.session_state.get('chat_history'):
            chat_history_text = self.format_chat_history()
            st.download_button(
                label="Download Chat History",
                data=chat_history_text,
                file_name="chat_history.txt",
                mime="text/plain"
            )


    def format_chat_history(self):
        """Format chat history as a readable text."""
        formatted_history = []
        for chat in st.session_state['chat_history']:
            role = "User" if chat['role'] == 'user' else "Assistant"
            formatted_history.append(f"{role}: {chat['message']}\n")
        return "".join(formatted_history)


    def voice_input_icon(self):
        """Display a button styled as a microphone icon. Clicking the button triggers the record_audio method."""
        st.markdown(
            """
            <style>
            .voice-icon-container {
                position: fixed;
                bottom: 30px;
                right: 30px;
                z-index: 1000;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown('<div class="voice-icon-container">', unsafe_allow_html=True)
        if st.button("🎙️", key="voice_button"):
            user_input = self.record_audio()
            if user_input:
                self.append_chat_history('user', user_input)
                response = self.call_openai(user_input, self.load_dataset()[0])
                self.append_chat_history('assistant', response)
            else:
                st.error("Sorry, I didn't catch that. Please try again.")
        st.markdown('</div>', unsafe_allow_html=True)

chatbot = Chatbot()
chatbot.run()
