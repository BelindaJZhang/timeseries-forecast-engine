# **Time Series Forecast Engine**

*A modular, multi-horizon forecasting pipeline for commodity, energy, and financial markets.*

---

## **Overview**

This repository provides a **scalable and reusable forecasting engine** designed to handle:

* multiple time series (e.g., Henry Hub, TTF, JKM, PVB)
* multiple forecasting horizons (30, 60, 90 days)

The project evolved from my earlier LNG price forecasting & trading simulation work, but is redesigned as a **clean, modular, production-style framework** focused on:

* reproducibility
* automation
* modular engineering
* consistent evaluation across markets
* clean artifact management

This engine demonstrates both **data science depth** and  **software engineering discipline** .

---

## **Key Features**

### **1. Modular Forecast Engine**

* A unified pipeline for any time series
* Clear separation between training, validation, and retraining
* Swappable components (dataset, model, evaluation, saving)

### **2. Multi-Horizon Forecasting**

* Supports 30-day, 60-day, and 90-day prediction horizons
* Horizon-aware windowing (Conv1D + LSTM hybrid)

### **3. Conv1D–LSTM Hybrid Neural Architecture**

* Conv1D for local feature extraction
* LSTM layers for temporal dependency
* Dense layers for non-linear mapping

### **4. Robust Validation Metrics**

Tracked and saved for each market & horizon:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* Mean Absolute Percentage Error (MAPE)
* Coefficient of Determination (R²)

### **5. Reproducible Outputs**

Each run automatically stores:

* trained models
* scalers
* validation metrics
* predictions
* future horizon forecast

### **6. Future Horizon Prediction**

After best model selection, the engine:

* retrains on full history
* predicts future value at forecast horizon (t + h)

---

## **Repository Structure**

---

## **Quickstart**

### **1. Clone the repository**

<pre class="overflow-visible!" data-start="3088" data-end="3203"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>git </span><span>clone</span><span> https://github.com/BelindaJZhang/timeseries-forecast-engine.git
</span><span>cd</span><span> timeseries-forecast-engine
</span></span></code></div></div></pre>

### **2. Create the environment**

<pre class="overflow-visible!" data-start="3239" data-end="3310"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>conda </span><span>env</span><span> create -f environment.yml
conda activate forecast
</span></span></code></div></div></pre>

### **3. Run a demo forecast**

<pre class="overflow-visible!" data-start="3343" data-end="3386"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>python examples/run_henryhub.py
</span></span></code></div></div></pre>

### **4. Run all markets & horizons**

<pre class="overflow-visible!" data-start="3426" data-end="3472"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>python examples/run_all_markets.py
</span></span></code></div></div></pre>

---

## **How the Engine Works**

### **Step 1 — Train/Validation Split**

A fixed cutoff date prevents leakage.

### **Step 2 — Scaling**

MinMaxScaler is fitted on  *training data only* .

### **Step 3 — Model Selection**

A Conv1D–LSTM neural network trains on train/validation.

Includes:

* EarlyStopping
* ModelCheckpoint
* Automatic selection of best model

### **Step 4 — Validation Evaluation**

Outputs include:

* predicted vs actual validation values
* validation dates reconstructed
* MAE, RMSE, MAPE, R² stored in JSON

### **Step 5 — Full-History Retraining**

Retrain the best architecture on the entire dataset.

### **Step 6 — Future Horizon Forecast**

Use final window (e.g., last 90 days) to predict t + horizon.

### **Step 7 — Save All Artifacts**

Models, scalers, results, and metrics saved to reproducible folders.

---

## **Architecture Diagram**

<pre class="overflow-visible!" data-start="4399" data-end="4453"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre!"><span><span>!</span><span>[Architecture Diagram]</span><span>(images/architecture.png)
</span></span></code></div></div></pre>

---

## **Example Forecast Plot**

*(Add a plot under `/docs/hh_30day_forecast.png`)*

<pre class="overflow-visible!" data-start="4542" data-end="4597"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre!"><span><span>!</span><span>[Example Forecast]</span><span>(docs/hh_30day_forecast.png)
</span></span></code></div></div></pre>

---

## **Example Metrics (Henry Hub)**

| Horizon | MAE   | RMSE  | MAPE   | R²   |
| ------- | ----- | ----- | ------ | ----- |
| 30 days | 0.869 | 1.297 | 25.48% | 0.620 |
| 60 days | 0.950 | 1.378 | 30.28% | 0.523 |
| 90 days | 1.003 | 1.393 | 33.47% | 0.452 |

---

## **Use Cases**

This forecasting engine can be applied to:

* LNG and natural gas price forecasting
* Energy trading & hedging
* Commodity markets
* Financial time series
* Demand forecasting
* Automated ML pipelines

---

## **Related Work**

This engine evolved from my earlier project:

**LNG Price Forecasting & Profitability Simulation**

[https://github.com/BelindaJZhang/lng-price-forecasting-and-profitability-simulation](https://github.com/BelindaJZhang/lng-price-forecasting-and-profitability-simulation)

The previous project focuses on:

* business context
* simulation
* event-driven market analysis

This repository extends that work into a  **general-purpose, production-style forecasting framework** .

---

## **Roadmap**

* [ ] Multi-step forecasting (direct, recursive, seq2seq)
* [ ] Support for external features (weather, storage, freight, FX)
* [ ] Transformer-based forecasting
* [ ] Hyperparameter search
* [ ] W&B or MLflow tracking
* [ ] Docker and API integration
* [ ] Deployment examples (FastAPI / Streamlit)

---

## **License**

MIT License.
