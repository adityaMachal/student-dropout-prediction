# 📸 INT315 Cluster Computing: Screenshot Capturing & Insertion Guide

This guide provides exact terminal commands, browser actions, and filename specifications to capture and insert all 10 required screenshots into your **INT315 Project Report** (`report.pdf`).

All screenshot images should be saved into the **`report/images/`** directory.

---

## 🛠️ Step 1: Capture Execution & Result Screenshots

### 1. Terminal Execution Screenshots (Section 9)

| Screenshot ID | Recommended Filename | How to Produce | What to Capture / Frame |
| :--- | :--- | :--- | :--- |
| **Screenshot 1** | `images/01_pyspark_pipeline_exec.png` | Run in terminal:<br>`source ~/apps/miniconda3/etc/profile.d/conda.sh`<br>`conda activate ML`<br>`cd ~/Documents/projects/student-dropout-prediction`<br>`python3 -m src.pipeline_runner` | Top section of terminal showing `SparkSession` startup, `[Task 97] Cleaned records: 4,424`, and the `80/20 Train-Test Split` summary. |
| **Screenshot 2** | `images/02_model_training_log.png` | Look at the next section of the same terminal output from `pipeline_runner`. | Middle section showing `Training Logistic Regression Pipeline...`, `Saved Logistic Regression model`, `Training Random Forest Classifier...`, and `Saved Random Forest model`. |
| **Screenshot 3** | `images/03_evaluation_metrics_log.png` | Look at the bottom section of the terminal output from `pipeline_runner`. | The formatted benchmark table in terminal comparing Logistic Regression vs Random Forest (`ROC-AUC 0.9082`, `F1 0.8584`, `Dropout Recall 0.6775`), plus the `Early Warning Cohort Distribution` counts. |
| **Screenshot 4** | `images/04_pytest_verification.png` | Run in terminal:<br>`source ~/apps/miniconda3/etc/profile.d/conda.sh`<br>`conda activate ML`<br>`cd ~/Documents/projects/student-dropout-prediction`<br>`pytest -v` | Terminal output showing all **14 tests passing** in green (`14 passed in 18.98s`). |

---

### 2. Streamlit Dashboard Output Screenshots (Section 10)

Launch the dashboard in your terminal:
```bash
source ~/apps/miniconda3/etc/profile.d/conda.sh
conda activate ML
cd ~/Documents/projects/student-dropout-prediction
streamlit run dashboard/app.py --server.port 8501
```
Open **`http://localhost:8501`** in your browser and capture the following tabs:

| Screenshot ID | Recommended Filename | Dashboard Tab | What to Capture / Frame |
| :--- | :--- | :--- | :--- |
| **Screenshot 5** | `images/05_dashboard_risk_categories.png` | **Tab 1: 🚨 1. Risk Categories** *(Compulsory 1)* | Capture the header KPIs, the **Risk Tier Donut Chart** (High, Moderate, Low Risk), and the **Predicted Dropout Probability Distribution** histogram with threshold dashed lines. |
| **Screenshot 6** | `images/06_dashboard_actual_vs_predicted.png` | **Tab 2: ⚖️ 2. Actual vs. Predicted Outcomes** *(Compulsory 2)* | Capture the interactive **Confusion Matrix Heatmap** (TP: 191, FP: 36, TN: 536, FN: 80), Diagnostic Summary, and the Model Benchmark Comparison table. |
| **Screenshot 7** | `images/07_dashboard_academic_programs.png` | **Tab 3: 🏫 3. Dropout Rates by Academic Program** *(Compulsory 3)* | Capture the grouped horizontal bar chart displaying **Actual vs. Predicted Dropout Rate (%) across Academic Programs** (Nursing, Management, Informatics Engineering, etc.). |
| **Screenshot 8** | `images/08_dashboard_live_serving.png` | **Tab 4: 🔮 4. Live Student Risk Calculator** | Fill in sample student details (e.g., Informatics Engineering, Evening, 0 approved units in Sem 1), click **Run PySpark Dropout Risk Assessment**, and capture the resulting **High Risk badge**, probability score, and advisory card. |
| **Screenshot 9** | `images/09_dashboard_advisor_roster.png` | **Tab 5: 📋 5. Advisor Early Warning Roster** | Capture the prioritized table of High-Risk students ($P \ge 0.60$) and the **Download High-Risk Advisor Outreach Roster (CSV)** button. |

---

### 3. Plagiarism Assessment Screenshot (Section 13)

| Screenshot ID | Recommended Filename | How to Produce | What to Capture |
| :--- | :--- | :--- | :--- |
| **Screenshot 10** | `images/10_plagiarism_certificate.png` | Export or screenshot from your institution's Turnitin / DrillBit / Urkund portal. | Official certificate/summary page verifying similarity index $\le 7\%$. |

---

## 🔄 Step 2: How to Insert Screenshots into `report.tex`

Once you save the image files into `report/images/`, simply open `report/report.tex` and update the corresponding figure block.

For example, in **Figure 1**:
```latex
% BEFORE (Showing placeholder box):
\begin{figure}[H]
    \centering
    % \includegraphics[width=0.95\textwidth]{images/01_pyspark_pipeline_exec.png}
    \begin{tcolorbox}[colback=gray!10,colframe=gray!60,...]
        ...
    \end{tcolorbox}
    \caption{...}
\end{figure}
```

Change it to:
```latex
% AFTER (Showing actual screenshot):
\begin{figure}[H]
    \centering
    \includegraphics[width=0.95\textwidth]{images/01_pyspark_pipeline_exec.png}
    \caption{PySpark Data Ingestion and Dataset Partitioning Execution}
    \label{fig:exec_ingestion}
\end{figure}
```
*(Simply uncomment the `\includegraphics` line and remove or comment out the `\begin{tcolorbox} ... \end{tcolorbox}` block!)*

---

## ⚡ Step 3: Recompile the PDF

After inserting your screenshots, recompile the report with one command:
```bash
cd ~/Documents/projects/student-dropout-prediction/report
./build.sh
```
Or open the project in **Overleaf** by uploading `report.tex` and the `images/` folder!
