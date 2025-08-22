# 🔍 Explainable AI for MonkeyOCR: 
## Making OCR Decisions Transparent  

MonkeyOCR is a powerful OCR engine, but it has always acted as a **black box** — we knew *what* text it recognized but not *why*.  
This project introduces a **comprehensive Explainable AI (XAI) framework** for MonkeyOCR using **LIME** and **SHAP**, making its decisions transparent and interpretable.  

---

## 🚀 What's Happening  

- **Problem:** OCR works well but provides no explanation of decisions.  
- **Solution:** Implemented **OCR-aware LIME** and **OCR-specific SHAP** explanations.  
- **Impact:** Users can now see *why* text is recognized (or misrecognized).  

---

## 🛠 How It Works  

### 🟢 OCR-Aware LIME Implementation  
Instead of generic LIME, I created a **custom OCR-specific approach**:  

1. **Smart Image Segmentation**  
   - 🔹 **SLIC Superpixels** → coherent text regions  
   - 🔹 **Sobel Edge Detection** → character boundaries  
   - 🔹 **Morphological Processing** → connect broken characters  
   - ✅ Produces **26 meaningful segments** (not random grids)  

2. **OCR-Focused Prediction Function**  
   Evaluates regions using OCR-specific metrics:  
   - Text density  
   - Edge density  
   - Contrast levels  
   - Local binary patterns  
   - Brightness distribution  
   - Horizontal/vertical projections  

3. **LIME Process**  
   - Segment image → perturb regions → re-run OCR → measure confidence drop  
   - Importance = `(Original - Modified) / Original`  
   - **Positive scores** → crucial text regions  
   - **Negative scores** → background/non-text noise  

---

### 🔵 SHAP with OCR-Specific Features  
I engineered **8 OCR features** to measure contribution:  

- **Text Features**: Text density (25%), edge density (20%), dark regions (5%)  
- **Image Quality**: Contrast (15%), brightness mean (10%), brightness std (15%)  
- **Structure**: Uniformity (5%), bright regions (5%)  

**Computation Steps:**  
- Build baseline (median OCR doc features)  
- Compare current doc against baseline  
- Use Shapley values to assign fair contributions  

**Results:**  
- ✅ Text Density (+0.15) → strongest positive driver  
- ✅ Brightness Mean (+0.08) → good lighting improves OCR  

---

## 📊 Key Insights & Impact  

- 🔎 **94.14% faithfulness** → explanations accurately match model behavior  
- 📈 Preprocessing (noise/background removal) improves OCR by ~15%  
- 🛠 Helps predict **OCR failures** before they happen  
- ✅ Increases **trust, transparency, debugging power**  

---

## 📂 Project Structure  
├── 📁 Data Files
│ ├── lime_fixed_analysis_data.json
│ ├── shap_analysis_data.json
│ └── combined_analysis_data.json
├── 📁 Reports
│ └── comprehensive_ocr_xai_report.md


---

## ⚙️ Advanced Configuration  (optional)

You can adjust parameters for LIME & SHAP inside the code:  


### 🚀 Large Image Optimization (>2MB)  
- Images auto-resized to **600px max**  
- Use quick mode for speed  
- Preprocess manually if needed  

### 📑 Complex Documents  
- Use specific task types (`text`, `formula`, `table`)  
- Increase LIME samples for reliable results  

---



