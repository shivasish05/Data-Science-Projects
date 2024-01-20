# Language Translator - Streamlit App

This Streamlit application uses OpenAI's GPT-3.5 Turbo model to perform English to Hinglish translation. It can be used to convert English text into Hinglish, which is a combination of English and Hindi (in the Devanagari script). It maintains the style of having the Hindi part in Devanagari script and only translates specific words into Hindi while leaving the rest in English.


<img width="960" alt="Screenshot 2023-10-30 172138" src="https://github.com/shivasish05/TextTranslation-Model/assets/98316576/30561754-917f-41a0-8a6f-bf0adee0f737">


## Getting Started

To run this Streamlit app, you'll need to set up your environment and configure the necessary API keys. Here are the steps to get started:

1. Clone the repository to your local machine.

2. Ensure you have Python and pip installed on your system.

3. Install the required dependencies by running the following command:

   ```
   pip install streamlit openai streamlit_chat python-dotenv
   ```

4. Set up your OpenAI API key by creating a `.env` file and adding your API key in the following format:

   ```
   OPENAI_API_KEY=your_api_key_here
   ```

5. Save your `.env` file in the same directory as your application.

6. Run the Streamlit app by executing the following command in your terminal:

   ```
   streamlit run your_app_file.py
   ```

   Replace `your_app_file.py` with the name of your Python script.

## Usage

1. Access the Streamlit app through your web browser.

2. In the provided text area, enter the English text that you want to translate to Hinglish.

3. Click the "Translate" button to initiate the translation process.

4. The translated Hinglish text will be displayed on the screen along with the original English input.

5. You can continue entering new text and translating as needed.

## Example

**English Input:**

```
I had about a 30-minute demo just using this new headset.
```

**Desired Hinglish Output:**

```
मझे थी एक ३०-मिनट की डेमो, using इस नए headset.
```

## Dependencies

- [Streamlit](https://www.streamlit.io/): The app's user interface is created using Streamlit.
- [OpenAI API](https://beta.openai.com/): OpenAI's GPT-3.5 Turbo model is used for the translation.
- [Streamlit Chat](https://pypi.org/project/streamlit-chat/): A Streamlit component for displaying chat messages.

