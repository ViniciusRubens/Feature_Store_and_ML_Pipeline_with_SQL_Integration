# Feature Store & ML Pipeline with SQL Integration

### About the project
This project implements a robust, end-to-end Machine Learning pipeline focused on the engineering aspects of a **Feature Store**. Unlike traditional academic scripts, this solution simulates a production environment where feature consistency, data lineage, and automated artifact management are prioritized.

The system generates synthetic data, persists it into a Relational Database (SQLite) acting as an **Offline Feature Store**, and orchestrates the training of a Random Forest classifier.

### Problem
In many Machine Learning lifecycles, a disconnect exists between data engineering and data science. Common challenges include:
* **Training-Serving Skew:** Discrepancies between data used for training and data used during inference.
* **Lack of Reproducibility:** Difficulty in tracking exactly which dataset version generated a specific model.
* **Inefficient Data Management:** Reliance on flat files (CSVs) which lack schema enforcement and query capabilities.

### Solution
This project addresses these issues by implementing:
1.  **SQL-Based Feature Store:** Replaces flat files with **SQLite/SQLAlchemy**, providing a structured, queryable, and persistent "Offline Store".
2.  **Automated Artifact Generation:** EDA (Exploratory Data Analysis) charts are generated and saved as static image artifacts, allowing for historical logging without halting execution.
3.  **Metadata Tracking:** Every pipeline run generates a JSON manifest linking the trained model, the dataset source, and performance metrics.
4.  **Modular Architecture:** Separation of concerns between Data Generation, Visualization, Training, and Artifact Management.

### Tech stack
* **Language:** Python 3.12+
* **Data Manipulation:** Pandas, NumPy
* **Feature Store / Database:** SQLAlchemy, SQLite
* **Machine Learning:** Scikit-Learn (Random Forest Classifier)
* **Visualization:** Matplotlib, Seaborn
* **Serialization:** Joblib, JSON
* **Packaging:** Setuptools (via `pyproject.toml`)

---

### Data source and Pipeline

The pipeline follows a strict flow to ensure data integrity:

1.  **Data Generation (Ingestion):**
    * Synthetic data is created with 3 groups of features.

2.  **Feature Storage:**
    * The raw dataframe is persisted into the **SQLite Feature Store** (`feature_store.db`).

3.  **Pipeline Orchestration:**
    * **Extraction:** Data is read back from SQL using `read_sql` queries.
    * **EDA:** Distribution and Correlation plots are generated and saved to `pipeline_runs/`.
    * **Training:** A Random Forest Classifier is trained on the extracted features.
    * **Evaluation:** Accuracy and Classification Reports are calculated.

4.  **Artifact Deployment:**
    * The model is serialized to `.joblib`.
    * Predictions and Run Metadata are archived.

---

## Getting Started

Follow these instructions to set up the project locally.

### Prerequisites

* **Operating System:** Linux (Ubuntu/Debian recommended), macOS, or Windows (WSL).
* **Python:** Version 3.12 or higher.
* **Package Manager:** `pip` and `venv`.

#### Installation

1.  **Clone the repository** (or download https://github.com/ViniciusRubens/Feature_Store_and_ML_Pipeline_with_SQL_Integration the files to a local folder).
    ```bash
    git clone git@github.com:ViniciusRubens/Feature_Store_and_ML_Pipeline_with_SQL_Integration.git
    cd your-repository-name
    ```

2.  **Create a Virtual Environment**
    Isolate the project dependencies by creating a virtual environment named `my_env`.
    ```bash
    python3 -m venv my_env
    ```

3.  **Activate the Environment**
    ```bash
    source my_env/bin/activate
    # Note: On Windows, use: my_env\Scripts\activate
    ```

4.  **Install Dependencies**
    First, install the standard dependencies listed in `requirements.txt`. Then, install the project itself in **editable mode**.
    
    * `pip install -e .`: This command links the virtual environment directly to the source code defined in `pyproject.toml`. It ensures that any changes made to the source code are immediately reflected in the environment without requiring re-installation.

    ```bash
    pip install -r requirements.txt
    pip install -e .
    ```

---

## Usage

To execute the Machine Learning pipeline, follow the steps below. This process will trigger data generation, SQL persistence, model training, and artifact saving.

1.  **Activate the Environment (if not already active)**
    ```bash
    source my_env/bin/activate
    ```

2.  **Run the Orchestrator**
    Execute the main Python script located in the `src` directory.
    ```bash
    python src/run.py
    ```

3.  **Deactivate**
    Once the pipeline execution is complete, you can exit the virtual environment to return to the system's global scope.
    ```bash
    deactivate
    ```
    
---

## License

Distributed under the MIT License. See `LICENSE` file for more information.