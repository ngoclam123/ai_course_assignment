# Assignment 11: Satellite Image Cloud Detection with CSV Logging

This project is a satellite image classification system using **Azure OpenAI Vision API** and **Streamlit** for the user interface. The system classifies satellite images as either "Clear" or "Cloudy" and automatically logs all results to a CSV file for analysis and tracking.

## ✨ Features

- **🛰️ Satellite Image Analysis**: Upload and classify satellite images for cloud detection
- **🤖 AI-Powered Classification**: Uses Azure OpenAI's vision capabilities for accurate predictions  
- **📊 Confidence Scoring**: Get confidence percentages for each classification
- **📝 Automatic CSV Logging**: All results automatically logged with timestamps
- **📈 Classification History**: View complete history of all classifications
- **💾 Export Functionality**: Download complete log file in CSV format
- **🎨 Modern UI**: Professional Streamlit interface with responsive design

## 🛠️ Technical Stack

- **AI Vision**: Azure OpenAI GPT-4 Vision (or compatible vision model)
- **Web Framework**: Streamlit for interactive web interface
- **Data Logging**: CSV file with pandas integration
- **Image Processing**: PIL (Python Imaging Library)
- **Structured Output**: Pydantic models for reliable response parsing
- **Framework**: LangChain for Azure OpenAI integration

## 🚀 Key Features

### 1. **CSV Logging System**
- Automatic logging of all classification results
- Includes timestamp, filename, prediction, and confidence
- CSV format for easy analysis in Excel or other tools
- Persistent storage across sessions

### 2. **Enhanced User Interface**
- Professional two-column layout for better organization
- Color-coded results (Clear = green ☀️, Cloudy = blue ☁️)
- File information display (filename and size)
- Download button for complete log export

### 3. **Classification History**
- View all previous classifications in a sortable data table
- Easy filtering and analysis of historical results
- Export functionality for further data analysis

### 4. **Robust Error Handling**
- Graceful handling of API failures
- File upload validation
- CSV read/write error management
- User-friendly error messages

## 📁 Project Structure

```
assignment_11/
├── assignment_11.py              # Main application with CSV logging
├── classification_log.csv        # Auto-generated log file (created on first run)
└── README.md                     # This documentation
```

## 🛠️ Setup Instructions

### 1. Prerequisites
- Python 3.10+ with virtual environment
- Azure OpenAI API access with vision-capable model
- Streamlit for web interface

### 2. Install Dependencies
```bash
# Activate virtual environment (if using one)
source ../venv/bin/activate

# Install required packages
pip install -r envPkg.txt
```

### 3. Environment Configuration
Create a `.env` file in the parent directory (`../`) with your Azure OpenAI credentials:
```env
AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint
AZURE_OPENAI_API_KEY=your_azure_openai_api_key
AZURE_OPENAI_LLM_MODEL=your_vision_model_deployment_name
```

### 4. Run the Application
```bash
streamlit run assignment_11.py
```

## 📊 CSV Log Format

The application automatically creates and maintains a CSV log file (`classification_log.csv`) with the following structure:

| Column | Description | Example |
|--------|-------------|---------|
| Timestamp | Date and time of classification | 2025-08-10 14:30:25 |
| Filename | Original filename of uploaded image | satellite_image_001.jpg |
| Prediction | Classification result (Clear/Cloudy) | Cloudy |
| Confidence | Confidence percentage | 88.5% |

### Sample CSV Output:
```csv
Timestamp,Filename,Prediction,Confidence
2025-08-10 14:30:25,satellite_001.jpg,Cloudy,88.5%
2025-08-10 14:32:10,clear_sky.png,Clear,92.3%
2025-08-10 14:35:45,weather_sat.jpg,Cloudy,76.8%
2025-08-10 14:40:12,clear_sky_002.jpg,Clear,89.2%
2025-08-10 14:42:55,storm_clouds.png,Cloudy,94.7%
```

## 🎯 Usage Guide

### **Upload and Classify**
1. Start the application: `streamlit run assignment_11.py`
2. Upload a satellite image (JPG, JPEG, or PNG format)
3. Click "🚀 Classify Image" to get results
4. View the prediction and confidence score
5. Results are automatically logged to CSV immediately

### **View Classification History**
- Scroll down to see the "📊 Classification History" section
- View all previous classifications in a sortable table
- Use the download button to export the complete log

### **Supported Image Formats**
- **JPEG/JPG**: Most common satellite image format
- **PNG**: High-quality images with transparency support
- **File Size**: Optimized for typical satellite image sizes

## 🔧 Technical Implementation

### **Structured Output with Pydantic**
```python
class WeatherResponse(BaseModel):
    accuracy: float = Field(description="The accuracy of the result")
    result: str = Field(description="The result of the classification")
```

### **CSV Logging Functions**
- `initialize_log_file()`: Creates CSV with headers if not exists
- `log_classification()`: Appends new classification results with timestamp
- `display_log_history()`: Shows historical data with download option

### **Azure OpenAI Integration**
```python
llm = AzureChatOpenAI(
    azure_endpoint=azure_endpoint,
    azure_deployment=azure_deployment,
    api_key=azure_api_key,
    api_version="2024-02-15-preview"
)
```

## 📈 Data Analysis Capabilities

The CSV log enables various analyses:

- **Accuracy Tracking**: Monitor confidence scores over time
- **Classification Distribution**: Count of Clear vs Cloudy predictions
- **Temporal Analysis**: Classification patterns by time/date
- **Performance Monitoring**: Track model confidence trends
- **Data Export**: Easy integration with Excel, Python pandas, or other analysis tools

## 🚀 Quick Start Guide

1. **Setup environment**: Configure Azure OpenAI credentials in `.env` file
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Start application**: `streamlit run assignment_11.py`
4. **Upload image**: Drag and drop or browse for satellite image
5. **Get results**: Click classify and view prediction with confidence
6. **Review history**: Check the log section for all past classifications
7. **Export data**: Download CSV for external analysis

## 🛰️ Application Screenshots

### Main Interface
- Clean two-column layout with image upload and results
- Professional styling with emojis and clear labeling
- File information display and progress indicators

### Classification Results
- Color-coded predictions (Clear/Cloudy)
- Confidence scoring with percentage display
- Automatic logging confirmation

### History Section
- Sortable data table with all past classifications
- Export button for CSV download
- Real-time updates after new classifications

## 🔍 Troubleshooting

### Common Issues:
1. **Missing environment variables**: Check `.env` file in parent directory
2. **API key issues**: Ensure Azure OpenAI key has access to vision models
3. **Import errors**: Run `pip install -r requirements.txt`
4. **CSV file errors**: Check file permissions in application directory

### Error Handling:
- The application includes comprehensive error handling
- User-friendly error messages for common issues
- Graceful fallback when services are unavailable

## 📝 Development Notes

- **File Structure**: Modular design with separate functions for logging
- **Error Handling**: Comprehensive exception handling throughout
- **Data Persistence**: CSV file ensures data survives app restarts
- **UI/UX**: Professional design with intuitive workflow
- **Performance**: Efficient image processing and API calls

---

## 🎉 Success!

You now have a complete satellite image classification system with:

✅ **AI-powered cloud detection** with Azure OpenAI Vision  
✅ **Automatic CSV logging** of all results with timestamps  
✅ **Professional web interface** built with Streamlit  
✅ **Classification history** viewing and export functionality  
✅ **Confidence scoring** for each prediction  
✅ **Production-ready error handling** and robust operation

### 🛰️ Ready to Classify!

Upload satellite images and start building your classification dataset!

**Happy classifying! 🌤️**

---

*Built with Azure OpenAI, Streamlit, LangChain, and Python | Enhanced with comprehensive CSV logging for data analysis*
