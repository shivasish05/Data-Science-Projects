
# PowerPoint Presentation Generator

https://github.com/shivasish05/Text_To_PPT_Generator/assets/98316576/2e7e0a38-1a18-439d-98a0-0a373ead830c

Welcome to the PowerPoint Presentation Generator, a Python application that utilizes OpenAI's GPT-3 to create visually appealing PowerPoint presentations. This tool automates the process of generating presentation slides on a given topic, making it easy to create engaging presentations in a matter of minutes.

## Table of Contents

- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Customization](#customization)
- [License](#license)

## Getting Started

Follow the instructions below to set up and use the PowerPoint Presentation Generator on your local machine or deploy it to a web application using Streamlit.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.7+
- An OpenAI API key (You can obtain one from [OpenAI](https://beta.openai.com/signup/))

## Installation

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/shivasish05/powerpoint-presentation-generator.git
   ```

2. Change into the project directory:

   ```bash
   cd powerpoint-presentation-generator
   ```

3. Install the required Python packages using pip:

   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project directory and add your OpenAI API key as follows:

   ```env
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

To use the PowerPoint Presentation Generator, follow these steps:

1. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

2. Access the application in your web browser at `http://localhost:8501`.

3. Enter the topic for your presentation and click "Generate Presentation."

4. The application will utilize GPT-3 to create slide titles and content.

5. Once the presentation is generated, you can download it using the provided download link.

## Customization

You can customize the presentation's appearance and behavior by modifying the code in `app.py`. Here are some areas you might want to customize:

- Slide transitions and backgrounds
- Fonts and font sizes
- Color schemes
- Additional design elements (e.g., images, shapes)

Feel free to explore and adjust the code to create presentations that match your specific requirements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```
Replace `shivasish05/powerpoint-presentation-generator` with the appropriate GitHub repository name you intend to use. This README template provides an overview of your project, installation instructions, and basic usage guidance. You can expand it as needed to provide more detailed information about your project and its features.
