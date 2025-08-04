import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import VitsModel, AutoTokenizer
import torch
import soundfile as sf
import io

st.set_page_config(page_title="Text-to-Speech from URL", layout="wide")

st.title("Text-to-Speech from Web Page")

st.info("Enter a URL of a web page, and the app will read its content aloud.")

# Step 1: Load the pre-trained TTS model
@st.cache_resource
def load_model():
    model_name = "facebook/mms-tts-vie"
    model = VitsModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

with st.spinner("Loading Text-to-Speech model... This may take a moment."):
    model, tokenizer = load_model()

# Step 2: Get URL from user
url = st.text_input("Enter the URL of the web page you want to read:", "https://vnexpress.net/hai-nhan-vien-bai-bien-da-nang-tim-lai-nhan-kim-cuong-cho-du-khach-4779350.html")

# Step 3: Create a button to trigger the process
if st.button("Read Aloud"):
    if url:
        try:
            # Step 4: Fetch and parse the web page content
            with st.spinner("Fetching and parsing web page content..."):
                response = requests.get(url, timeout=10)
                response.raise_for_status()  # Raise an exception for bad status codes
                soup = BeautifulSoup(response.content, "html.parser")

                # Extract text from all paragraph tags (<p>) on the page.
                paragraphs = soup.find_all("p")
                if paragraphs:
                    text = "\n".join(p.get_text(strip=True) for p in paragraphs)
                else:
                    st.error("Could not find any paragraph (<p>) content on the page. Falling back to all text.")
                    text = soup.get_text(separator="\n", strip=True)

            if text:
                st.subheader("Extracted Text:")
                st.text_area("This is the text that will be converted to speech.", text, height=200)

                # Step 5: Generate audio from the extracted text
                with st.spinner("Generating audio... Please wait."):
                    inputs = tokenizer(text, return_tensors="pt")
                    with torch.no_grad():
                        output = model(**inputs).waveform

                    # Step 6: Display the audio player
                    st.subheader("Listen to the Audio:")
                    audio_bytes = io.BytesIO()
                    sampling_rate = model.config.sampling_rate
                    sf.write(audio_bytes, output.numpy().T, sampling_rate, format='WAV')
                    audio_bytes.seek(0)
                    st.audio(audio_bytes, format="audio/wav")

                    # Provide a download link
                    st.download_button(
                        label="Download Audio as WAV",
                        data=audio_bytes,
                        file_name="output.wav",
                        mime="audio/wav"
                    )

        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching the URL: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
    else:
        st.warning("Please enter a URL.")

st.sidebar.header("About")
st.sidebar.info(
    "This app uses the Hugging Face Transformers library to convert text from a web page into speech. "
    "It fetches the content of the provided URL, extracts the text, and uses a pre-trained Text-to-Speech model to generate the audio."
)