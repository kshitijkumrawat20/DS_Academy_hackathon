# Methodology Note: India Predicts 2026 Election Outcome Prediction

## Executive Summary
This document outlines the methodology used to forecast the assembly election outcomes for five states: Kerala, Tamil Nadu, West Bengal, Assam, and Puducherry. Driven by historical electoral data and candidate metadata, our approach utilizes a binary classification engine powered by XGBoost. Our prediction targets the binary outcome—Win (1) or Loss (0) for candidates—in the 2026 assembly elections.

## Data Sources & Feature Engineering
We rely entirely on publicly accessible government and transparency platforms to source our features:
1. **Election Commission of India (ECI):** Extracted historical constituency-level results, vote margins, and past turnout statistics.
2. **MyNeta / Association for Democratic Reforms (ADR):** Scraped detailed candidate profiles to leverage crucial "X-factor" features. This includes wealth/assets, declared criminal records, educational attainment, and incumbent status. 

These datasets were merged on constituency IDs and candidate names. Significant feature engineering was performed:
* **Incumbency Advantage Factor:** Created by cross-referencing PRS Legislative Research histories.
* **Wealth-to-Margin Ratio:** Calculated by correlating ADR asset declarations with historical margin victories.
* **Criminal Record Ticker:** A binary flag and categorical risk score based on the severity of disclosures.

## Model Selection & Rationale
We selected **XGBoost (Extreme Gradient Boosting)** because of its proven resilience against non-linear relationships and missing candidate data (which is frequent in historical archives). Alternative models like Logistic Regression and Random Forest were tested; however, XGBoost consistently outperformed them on validation sets across states.

## Strategy & State-Level Tuning
To comply with the rulebook and maximize our score:
1. **West Bengal Prioritization:** Recognizing West Bengal's position as the largest state (294 seats) and its use as Tiebreaker #3, the XGBoost hyperparameters (learning rate, depth) were independently fine-tuned on West Bengal historical strata.
2. **Breadth of Accuracy (Tiebreaker #2):** Instead of overfitting on a single state, we employed Stratified K-Fold cross-validation across all 5 states independently, ensuring our precision-recall curve maintains uniformly above the 75% accuracy threshold in each region.

## Limitations & Risks
Our model operates under strict historical and structural assumptions:
* **Volatility of Alliances:** We implicitly assume voter loyalty transfers uniformly during coalition shifts, which may struggle if major unanticipated pre-poll alliances emerge.
* **Asset Valuation Inflation:** The model relies on self-declared assets, which often lag behind true market valuations, adding minor noise to the wealth factor.
* **Novelty Candidates:** The system may underestimate newly introduced independent candidates with no established ECI/ADR history.

Our predictions represent data-driven projections of behavioral voting patterns rather than political endorsements.
