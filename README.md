# Grand Avenue Restaurant Chatbot

A Streamlit-based chatbot for Grand Avenue Restaurant that helps customers with menu inquiries, table bookings, and general restaurant information.

## Features

- 🤖 Interactive chatbot powered by Google's Gemini AI
- 📋 Complete restaurant menu with categories
- 🪑 Table booking system
- 📍 Location and contact information
- 🕰️ Operating hours
- 🍽️ Special platters and offers

## Deployment on Hugging Face Spaces

1. Fork this repository
2. Create a new Space on Hugging Face
3. Connect your forked repository
4. Add the following secrets in your Space settings:
   - `GOOGLE_API_KEY`: Your Google API key for Gemini AI

## Local Development

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file with your Google API key:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```
5. Run the application:
   ```bash
   streamlit run app.py
   ```

## Project Structure

- `app.py`: Main application file
- `requirements.txt`: Project dependencies
- `data.pdf`: Restaurant data for the chatbot
- `image.jpeg`, `image2.jpg`, `demo.jpg`: Restaurant images

## Environment Variables

The following environment variables are required:
- `GOOGLE_API_KEY`: Your Google API key for Gemini AI

## License

© 2024 Grand Avenue Restaurant. All rights reserved. 