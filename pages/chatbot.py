from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import queue
import io
import soundfile as sf
import speech_recognition as sr
import pandas as pd
import openai
import uuid
import streamlit as st

openai_api_key = st.text_input("Enter your OpenAI API key:", type="password")
if openai_api_key:
    openai.api_key = openai_api_key
else:
    st.warning("Please provide an API key to proceed.")

class AudioProcessor(AudioProcessorBase):
    """Processes audio frames captured by WebRTC streamer."""
    def __init__(self):
        self.audio_queue = queue.Queue()

    def recv(self, frame):
        self.audio_queue.put(frame.to_ndarray().flatten())
        return frame


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
                return pd.read_csv(io.BytesIO(dataset_data), encoding='utf-8', delimiter=',', on_bad_lines='skip')
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
                messages=[
                    {"role": "system", "content": "You are an AI that helps answer questions about datasets."},
                    {"role": "user", "content": full_prompt},
                ],
                max_tokens=200,
                temperature=0.6
            )
            return response['choices'][0]['message']['content'].strip()
        except Exception as e:
            return f"Error calling OpenAI API: {str(e)}"

    def build_openai_prompt(self, user_prompt, df):
        """Build the prompt for OpenAI by including dataset context."""
        dataset_summary = self.generate_dataset_summary(df)
        full_prompt = f"{user_prompt}\n\nDataset Context:\n{dataset_summary}"
        return full_prompt

    def generate_dataset_summary(self, df):
        """Generate a detailed summary of the dataset."""
        try:
            column_info = self.get_column_information(df)
            numeric_summary = self.get_numeric_summary(df)
            dataset_summary = (
                f"Columns:\n{column_info}\n\n"
                f"Summary statistics for numeric columns:\n{numeric_summary}\n\n"
                f"First 5 rows:\n{df.head(5).to_string(index=False)}"
            )
            return dataset_summary
        except Exception as e:
            return f"Error generating dataset summary: {str(e)}"

    def get_column_information(self, df):
        """Get dataset column details."""
        return "\n".join([f"- {col}: {df[col].dtype}" for col in df.columns])

    def get_numeric_summary(self, df):
        """Generate summary statistics for numeric columns."""
        numeric_columns = df.select_dtypes(include=['float64', 'int64'])
        return numeric_columns.describe().to_string() if not numeric_columns.empty else "No numeric columns found."

    def record_audio():
        """Record audio using streamlit-webrtc and return the transcribed text."""
        st.info("Click the START button below to begin recording.")
    
        # Initialize WebRTC streamer
        audio_processor = webrtc_streamer(
            key="speech_recorder",
            mode=WebRtcMode.SENDRECV,
            audio_processor_factory=AudioProcessor,
            media_stream_constraints={"audio": True, "video": False},
        )
    
        if not audio_processor:
            st.warning("WebRTC not initialized. Please wait.")
            return None
    
        # Check if audio frames are being captured
        st.info("Recording audio... Speak now!")
        audio_frames = []
        if audio_processor.audio_queue:
            try:
                for _ in range(100):  # Collect a finite number of audio chunks
                    frame = audio_processor.audio_queue.get(timeout=1)
                    audio_frames.extend(frame)
            except queue.Empty:
                st.warning("No audio detected. Please ensure your microphone is active.")
    
        # Process recorded audio into WAV format
        if audio_frames:
            st.success("Recording completed. Transcribing...")
            wav_buffer = io.BytesIO()
            with sf.SoundFile(
                wav_buffer, mode="w", samplerate=16000, channels=1, format="WAV"
            ) as wav_file:
                wav_file.write(np.array(audio_frames))
            wav_buffer.seek(0)
    
            # Use SpeechRecognition to transcribe
            recognizer = sr.Recognizer()
            try:
                with sr.AudioFile(wav_buffer) as source:
                    audio_data = recognizer.record(source)
                    return recognizer.recognize_google(audio_data)
            except sr.UnknownValueError:
                st.error("Sorry, I could not understand the audio.")
                return None
            except sr.RequestError as e:
                st.error(f"Speech recognition service error: {e}")
                return None
        else:
            st.warning("No audio captured. Please try again.")
            return None



    def run(self):
        """Run the chatbot."""
        with st.spinner("Loading..."):
            df, dataset_name = self.load_dataset()

            if df is not None and dataset_name is not None:
                st.header(f"Chat with your Dataset: {dataset_name}")
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
        """Append a message to chat history."""
        st.session_state['chat_history'].append({'role': role, 'message': message})
        st.chat_message(role).write(message)

    def export_chat_history(self):
        """Allow chat history export."""
        if st.session_state.get('chat_history'):
            chat_history_text = self.format_chat_history()
            st.download_button(
                label="Download Chat History",
                data=chat_history_text,
                file_name="chat_history.txt",
                mime="text/plain",
            )

    def format_chat_history(self):
        """Format chat history as text."""
        return "\n".join([f"{chat['role'].capitalize()}: {chat['message']}" for chat in st.session_state['chat_history']])

    def voice_input_icon(self):
        """Show voice input button."""
        if st.button("🎙️ Start Voice Input"):
            user_input = self.record_audio()
            if user_input:
                self.append_chat_history('user', user_input)
                response = self.call_openai(user_input, self.load_dataset()[0])
                self.append_chat_history('assistant', response)

chatbot = Chatbot()
chatbot.run()
