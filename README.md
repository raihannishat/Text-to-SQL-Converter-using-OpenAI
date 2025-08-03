# Text-to-SQL Converter using OpenAI

A clean, minimal implementation of a dynamic PostgreSQL Text-to-SQL converter using OpenAI's GPT-4o-mini model.

## 🎯 Features

- **🔍 Dynamic PostgreSQL**: Connect to any PostgreSQL database
- **🤖 GPT-4o-mini**: AI-powered natural language to SQL conversion
- **📊 Schema Analysis**: Automatic table and column detection
- **⚡ Token Optimized**: Efficient for large databases
- **🎨 Clean UI**: Simple, focused interface with schema info page

## 📁 Project Structure

```
langchain-example/
├── app_clean.py          # Main application (342 lines)
├── config.env            # Environment configuration
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🚀 Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment**:
   - Copy your OpenAI API key to `config.env`
   - Format: `OPENAI_API_KEY=sk-your-actual-api-key-here`

3. **Run the App**:
   ```bash
   streamlit run app_clean.py
   ```

4. **Connect to PostgreSQL**:
   - Enter your database credentials in the sidebar
   - Click "Connect" to establish connection

## 🎯 Usage

### Main Page
- Enter natural language queries
- Generate SQL using GPT-4o-mini
- Execute queries and view results

### Schema Info Page
- View database metrics
- Explore table structures
- See sample data

## 🔧 Requirements

- Python 3.8+
- PostgreSQL database
- OpenAI API key
- Valid database credentials

## 📦 Dependencies

- `streamlit` - Web interface
- `pandas` - Data handling
- `sqlalchemy` - Database ORM
- `psycopg2-binary` - PostgreSQL adapter
- `langchain-openai` - OpenAI integration
- `python-dotenv` - Environment management

## 🎯 Key Benefits

- **Minimal Code**: Only essential functionality
- **Fast Loading**: No unnecessary imports
- **Clean UI**: Simple, focused interface
- **Token Efficient**: Optimized for large databases
- **Modern Code**: Latest LangChain syntax
- **Error Free**: No deprecation warnings

## 🔍 Perfect Implementation

✅ **Text-to-SQL Converter using OpenAI**  
✅ **GPT-4o-mini** model  
✅ **Fully dynamic** PostgreSQL approach  
✅ **Schema Info Page** for database exploration  
✅ **Clean, minimal code**  
✅ **Token-optimized**  
✅ **Modern LangChain syntax** 