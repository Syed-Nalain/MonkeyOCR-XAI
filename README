# 🔍 MonkeyOCR XAI Analysis Tool

Explainable AI (XAI) Analysis Tool for MonkeyOCR - Making OCR decisions transparent and interpretable through comprehensive LIME and SHAP explanations.

📋 Table of Contents

Overview

Features

Installation

Quick Start

Usage Guide

MonkeyOCR Integration

Code Structure

Output Files

Advanced Configuration

Performance Optimization

Contributing

License

🎯 Overview

This tool provides Explainable AI (XAI) analysis for MonkeyOCR outputs, helping you understand:

Which image regions contribute most to OCR predictions

What features the OCR model prioritizes

Why certain recognition decisions are made

How confident the model is in different document areas

Why Use This Tool?

🔍 Transparency: Understand OCR model decisions
🐛 Debugging: Identify why OCR fails on specific documents
📊 Quality Assurance: Validate OCR performance across different document types
🎯 Model Improvement: Data-driven insights for OCR enhancement
✅ Trust: Build confidence in OCR systems through interpretability

✨ Features
🔬 LIME Analysis

Custom OCR-Aware Segmentation: Intelligent image segmentation respecting text boundaries

Text-Focused Prediction: OCR-specific importance scoring

Visual Explanations: Color-coded importance maps and overlays

High Fidelity: 94%+ faithfulness scores

🎯 SHAP Analysis

Feature Engineering: 8 OCR-optimized features

Contribution Analysis: Detailed feature importance breakdown

Statistical Insights: Comprehensive feature interaction analysis

Robust Implementation: Error-resistant with fallback mechanisms

📊 Comprehensive Reporting

12+ Visualizations: Multiple perspectives on model behavior

JSON Export: Machine-readable analysis results

Markdown Reports: Human-readable comprehensive documentation

Quality Metrics: Faithfulness, consistency, and reliability scores

🚀 Installation
Prerequisites
# Python 3.8 or higher
python --version

# Required system packages (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y python3-dev python3-pip

Install Dependencies
# Clone this repository (or download the script)
git clone <your-repo-url>
cd monkeyocr-xai-analysis

# Install required packages
pip install -r requirements.txt


Or install manually:

pip install numpy>=1.21.0 matplotlib>=3.5.0 seaborn>=0.11.0 
pip install scikit-learn>=1.0.0 scikit-image>=0.19.0 
pip install lime>=0.2.0.1 shap>=0.41.0 
pip install Pillow>=8.0.0 pandas>=1.3.0

Requirements File (requirements.txt)
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.0.0
scikit-image>=0.19.0
lime>=0.2.0.1
shap>=0.41.0
Pillow>=8.0.0
pandas>=1.3.0

🚀 Quick Start
1. Run MonkeyOCR First
# Process your image with MonkeyOCR
python monkeyocr_script.py --image /path/to/your/image.png --output ./ocr_output/

2. Run XAI Analysis
# Basic analysis
python fixed_xai_analyzer.py /path/to/your/image.png ./ocr_output/

# With task specification  
python fixed_xai_analyzer.py /path/to/image.png ./ocr_output/ --task text

# Quick analysis mode
python fixed_xai_analyzer.py /path/to/image.png ./ocr_output/ --quick

3. View Results
# Results will be saved in:
./ocr_output/xai_analysis/
├── lime_fixed_comprehensive.png      
├── shap_comprehensive_fixed.png      
├── combined_lime_shap_analysis.png   
└── comprehensive_ocr_xai_report.md   

📖 Usage Guide
Command Line Interface
python fixed_xai_analyzer.py <image_path> <monkeyocr_output_dir> [options]

Arguments
Argument	Type	Description
image_path	Required	Path to the original PNG image file
monkeyocr_output_dir	Required	Directory containing MonkeyOCR output files
-t, --task	Optional	Task type: text, formula, table (default: text)
--quick	Optional	Quick analysis mode (fewer LIME samples)
Examples
# Text document analysis
python fixed_xai_analyzer.py ./documents/report.png ./output/report_ocr/ -t text

# Mathematical formula analysis
python fixed_xai_analyzer.py ./math/equation.png ./output/equation_ocr/ -t formula

# Table/spreadsheet analysis  
python fixed_xai_analyzer.py ./tables/data.png ./output/table_ocr/ -t table

# Quick analysis (faster, less detailed)
python fixed_xai_analyzer.py ./docs/scan.png ./output/scan_ocr/ --quick

🔗 MonkeyOCR Integration
Step-by-Step Integration
1. Prepare Your Image
# Ensure image is in supported format
ls -la your_image.png

2. Run MonkeyOCR
cd /path/to/MonkeyOCR/
python main.py --image /path/to/your_image.png --output_dir ./results/img_analysis/

3. Expected MonkeyOCR Output Structure
monkeyocr_output/
├── result.md                           
├── [filename]_content_list.json        
├── detected_text.txt                   
├── analysis_report.json                
└── confidence_metrics.json             

4. Run XAI Analysis
cd /path/to/xai-analysis/
python fixed_xai_analyzer.py /path/to/your_image.png /path/to/MonkeyOCR/results/img_analysis/

🏗️ Code Structure
Main Class: FixedMonkeyOCRXAIAnalyzer

This class runs the entire XAI analysis pipeline. It loads the image and OCR results, then runs both LIME and SHAP analysis before generating visualizations and reports.

Core Components

Data Loading & Preprocessing
Handles reading images and OCR output files, converting them into formats ready for analysis.

Custom LIME Implementation
Uses OCR-aware segmentation to divide images into meaningful text-based regions and score their importance.

SHAP Feature Analysis
Extracts OCR-related features (like brightness, text density, and edge density) and computes their contributions.

Visualization Engine
Produces comprehensive plots that explain how different parts of the image influenced OCR recognition.

Error Handling & Type Safety
Converts complex data types into safe formats for JSON and ensures the pipeline runs smoothly without crashing.

Processing Pipeline
graph TD
    A[Image + OCR Output] --> B[Load & Preprocess]
    B --> C[LIME Analysis]
    B --> D[SHAP Analysis]  
    C --> E[Custom Segmentation]
    C --> F[OCR Prediction Function]
    D --> G[Feature Extraction]
    D --> H[Shapley Value Calculation]
    E --> I[Importance Scoring]
    F --> I
    G --> J[Feature Contribution Analysis]
    H --> J
    I --> K[LIME Visualizations]
    J --> L[SHAP Visualizations]
    K --> M[Combined Analysis]
    L --> M
    M --> N[Comprehensive Report]

📁 Output Files
Directory Structure
xai_analysis/
├── 📊 Visualizations
│   ├── lime_fixed_comprehensive.png          
│   ├── shap_comprehensive_fixed.png          
│   └── combined_lime_shap_analysis.png       
├── 📋 Data Files  
│   ├── lime_fixed_analysis_data.json         
│   ├── shap_analysis_data.json               
│   └── combined_analysis_data.json           
└── 📄 Reports
    └── comprehensive_ocr_xai_report.md        

⚙️ Advanced Configuration
Custom Parameters

You can adjust parameters for LIME and SHAP inside the code, such as number of samples or feature weights, to fine-tune accuracy and speed.

Performance Tuning

Use --quick for faster analysis with fewer samples

Increase the number of LIME samples for high-quality detailed results

Add custom features if your OCR data needs specialized analysis

🚀 Performance Optimization
For Large Images (>2MB)

Images are automatically resized to a maximum of 600px

Use quick mode for faster analysis

Preprocess large images manually if necessary

For Complex Documents

Use specific task types (text, formula, table) for better accuracy

Increase LIME samples for more reliable explanations

Monitor memory usage on very detailed documents

🤝 Contributing

We welcome contributions! Please fork, branch, and submit pull requests. Follow PEP8 guidelines, add docstrings, and include tests for new features.

📝 License

This project is licensed under the MIT License - see the LICENSE
 file for details.

🙏 Acknowledgments

MonkeyOCR Team for the OCR foundation

LIME Authors for interpretability methods

SHAP Authors for feature attribution techniques

scikit-image for image processing tools

📧 Support

🐛 Bug Reports: Open an issue with error details

💡 Feature Requests: Suggest new ideas

📖 Documentation: Refer to README and reports

💬 Community: Share insights and results
