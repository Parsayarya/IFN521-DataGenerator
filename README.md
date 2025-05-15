# IFN521-DataGenerator
This repository contains code for generating and analyzing synthetic survey data of emergency department practitioners' attitudes toward AI triage systems, with a focus on transparency, trust, and disclosure preferences.
* Overview
The code simulates a survey of emergency department practitioners (physicians, nurses, administrators, etc.) regarding their attitudes toward AI systems used in triage. It focuses specifically on:

Preferences for transparency in AI systems
Trust in AI systems that disclose their limitations
Preferences for various types of information disclosure about AI limitations

Features
Comprehensive Data Generation

Realistic Role Distribution: Generates practitioners across six different ED roles (Emergency Medicine Physicians, ED Nurses, Triage Nurses, Physician Assistants, Nurse Practitioners, and Administrators) with proportions reflecting actual ED staffing patterns
Experience Stratification: Creates a realistic distribution of experience levels from less than 1 year to more than 20 years with appropriate weighting toward mid-career professionals
AI Exposure Modeling: Simulates varying levels of exposure to different AI tools including triage systems, diagnostic tools, decision support systems, AI-embedded EHRs, and risk scoring systems
Technology Comfort Simulation: Models practitioners' comfort with technology on a 5-point scale with appropriate correlations to AI usage
AI Understanding Variation: Generates varying levels of AI understanding that realistically correlate with comfort, training, and use frequency
Likert Scale Responses: Creates 30 detailed Likert-scale (1-5) responses across three categories (transparency attitudes, trust dimensions, and disclosure preferences)
Role-Based Effects: Incorporates realistic role-based variation where certain roles (e.g., triage nurses) care more about transparency than others
Experience Effects: Models how practitioners' attitudes toward AI change with experience level
Noise Inclusion: Adds realistic noise and outliers to response patterns (4% chance of outlier responses) to better reflect real-world data

Advanced Statistical Analysis

Correlation Analysis: Examines relationships between key composite variables including transparency preferences, disclosure preferences, and trust measures
Simple Linear Regression: Tests the primary hypothesis that preference for limitation disclosure predicts trust in transparent AI systems
Group Comparison: Segments practitioners into high and low disclosure preference groups for comparative analysis
Welch's T-test: Performs robust statistical testing between high and low preference groups
ANOVA Implementation: Conducts analysis of variance to assess group differences
Multiple Regression: Controls for experience, AI understanding, and technology comfort to isolate the effects of disclosure preference on trust
Effect Size Calculation: Quantifies the magnitude of differences between groups using Cohen's d

Comprehensive Visualization Suite

Regression Visualization: Creates scatter plots with fitted regression lines to illustrate relationships between key variables
Group Comparison Plots: Generates bar charts with confidence intervals comparing high and low preference groups
Correlation Heatmaps: Produces color-coded correlation matrices of all key variables
Response Distribution Analysis: Creates histograms showing the distribution of responses to key survey items
Role-Based Comparison: Generates boxplots showing how attitudes vary by practitioner role
Experience Trend Analysis: Produces line plots showing how trust and transparency preferences change with experience level
AI Understanding Comparison: Creates side-by-side boxplots comparing practitioners with high vs. low AI understanding

Detailed Report Generation

Executive Summary: Provides concise overview of key findings
Statistical Highlights: Reports critical statistical measures (r-squared, p-values, effect sizes)
Practitioner Segmentation: Identifies which practitioner types show highest/lowest trust in transparent systems
Practical Implications: Translates statistical findings into practical design recommendations for AI triage systems
Methodology Summary: Includes detailed breakdown of sample composition and analysis approach

Methods
Data Generation Methodology

Demographic Generation

Creates a representative sample of ED practitioners with realistic role distributions
Generates experience levels using weighted random sampling to match real-world ED staffing patterns
Maps categorical experience levels to numeric years for statistical analysis


AI Experience Simulation

Simulates prior exposure to 0-4 different AI tools with 12% having no exposure
Models frequency of AI use with correlations to number of tools used
Generates technology comfort levels that correlate with use frequency but include realistic variation
Creates AI understanding levels that correlate with both comfort and use frequency
Models formal AI training levels that correlate with understanding but include appropriate noise


Survey Response Generation

Transparency Attitudes: Models 10 questions about preferences for AI transparency with realistic correlations to demographics
Trust Dimensions: Generates responses to 10 questions about trust factors with careful modeling of how experience and AI understanding affect different dimensions of trust
Disclosure Preferences: Creates responses to 10 questions about preferred levels of information disclosure with strong correlations to transparency preferences
Special Handling of Reverse-Coded Items: Properly handles reversed items with appropriate negative correlations
Composite Score Creation: Calculates aggregate scores across question categories, excluding reverse-coded items


Correlation Introduction

Enhances the correlation between limitation disclosure preference and trust in transparent systems based on healthcare literature
Adds role-specific effects where triage nurses show higher trust in transparent systems (+0.2) while administrators show lower trust (-0.1)
Introduces small random noise (4% outlier chance) to simulate real-world response patterns



Analysis Methodology

Correlation Analysis

Calculates Pearson correlation coefficients between all key composite variables
Visualizes relationships using heatmaps with numerical annotations


Regression Analysis

Performs simple linear regression with limitation disclosure preference as predictor and trust as outcome
Tests significance of the relationship with p-value calculation
Evaluates model fit using R-squared and adjusted R-squared metrics
Controls for confounding variables in multiple regression (experience, AI understanding, tech comfort)


Group Comparison

Divides practitioners into high and low disclosure preference groups using a cutpoint of 3.8 on the 5-point scale
Calculates descriptive statistics (mean, standard deviation) for each group
Performs Welch's t-test to accommodate potential unequal variances
Calculates Cohen's d to quantify effect size independent of sample size


Visualization Methodology

Creates seven distinct visualization types to illustrate different aspects of the findings
Uses appropriate color schemes, error bars, and statistical annotations
Implements proper subplot arrangement for multi-panel figures
Saves all visualizations as high-resolution (300 dpi) PNG files




Research Foundation
This simulation is based on research by:

Sibbald et al. (2022) on clinician experiences with Electronic Diagnostic Support
Townsend et al. (2023) on Diagnostic AI Systems for Robot-Assisted Triage
Nord-Bronzyk et al. (2025) on ethical issues of AI limitation disclosure
Soltan et al. (2022) on AI systems that communicate confidence levels
Feretzakis et al. (2024) on detailed explanations from AI systems

License
MIT
