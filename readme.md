# CodeLlama-based XML Node Splitter with Chroma DB

This project is a Streamlit application created by a beginner experimenting with various tools. It uses the CodeLlama model to intelligently split XML nodes into specific categories and integrates Chroma DB for storing and retrieving previous splits.

Please note that the methods and tools used in this project are not optimized for efficiency. The primary goal is to experiment and learn.

## Features

- Split XML nodes into introduction, task, profile, offer, and contact categories
- Use CodeLlama for intelligent content splitting
- Store and retrieve previous splits using Chroma DB
- Explain changes made to the XML structure
- User-friendly interface built with Streamlit

## Requirements

- Python 3.7+
- Streamlit
- lxml
- requests
- chromadb
- An instance of CodeLlama running locally (see Setup section)

## Setup

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/xml-splitter-app.git
   cd xml-splitter-app
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Ensure you have a CodeLlama instance running locally on `http://localhost:11434`. You can use [Ollama](https://ollama.ai/) to run CodeLlama:
   ```
   ollama run codellama
   ```

## Usage

1. Run the Streamlit app:
   ```
   xml_splitter_app_codellama_chroma.py
   ```

2. Open your web browser and go to the URL provided by Streamlit (usually `http://localhost:8501`).

3. Enter your XML content and the node path you want to split.

4. Click "Split Node" to process the XML.

5. View the results, including the original XML, the split result, and an explanation of the changes.

6. You can also view previous splits by clicking the "View Previous Splits" button.

## How it works

1. The app takes XML content and a node path as input.
2. It uses CodeLlama to intelligently split the content of the specified node into introduction, task, profile, offer, and contact categories.
3. The split result is displayed and stored in Chroma DB for future reference.
4. CodeLlama is also used to generate an explanation of the changes made to the XML structure.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.
