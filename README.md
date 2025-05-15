# IFN521-DataGenerator

## Overview

The code simulates a survey of emergency department practitioners (physicians, nurses, administrators, etc.) regarding their attitudes toward AI systems used in triage. It focuses specifically on:

- Preferences for transparency in AI systems
- Trust in AI systems that disclose their limitations
- Preferences for various types of information disclosure about AI limitations

## Features

###  Data Generation
- **Role Distribution**: Generates practitioners across six different ED roles (Emergency Medicine Physicians, ED Nurses, Triage Nurses, Physician Assistants, Nurse Practitioners, and Administrators) with proportions reflecting actual ED staffing patterns
- **Experience Stratification**: Creates a realistic distribution of experience levels from less than 1 year to more than 20 years with appropriate weighting toward mid-career professionals
- **AI Exposure Modelling**: Simulates  levels of exposure to different AI tools, including triage systems, diagnostic tools, decision support systems, AI-embedded EHRS, and risk scoring systems
- **Technology Comfort Simulation**: Models practitioners' comfort with technology on a 5-point scale with correlations to AI usage
- **AI Understanding Variation**: Generates levels of AI understanding that realistically correlate with comfort, training, and use frequency
- **Likert Scale Responses**: Creates 30 Likert-scale (1-5) responses across three categories (transparency attitudes, trust dimensions, and disclosure preferences)
- **Role-Based Effects**: Incorporates role-based variation where certain roles (e.g., triage nurses) care more about transparency than others
- **Experience Effects**: Models how practitioners' attitudes toward AI change with experience level
- **Noise Inclusion**: Adds noise and outliers to response patterns (4% chance of outlier responses) to better reflect real-world data

### Statistical Analysis
- **Correlation Analysis**: Examines relationships between variables, including transparency preferences, disclosure preferences, and trust measures
- **Linear Regression**: Tests the hypothesis that preference for limitation disclosure predicts trust in transparent AI systems
- **Group Comparison**: Segments practitioners into high and low disclosure preference groups for comparative analysis
- **Welch's T-test**: Performs statistical testing between high and low preference groups
- **ANOVA**: Conducts analysis of variance to assess group differences
- **Multiple Regression**: Controls for experience, AI understanding, and technology comfort to isolate the effects of disclosure preference on trust
- **Effect Size Calculation**: Quantifies the differences between groups using Cohen's d

###  Visualisation 
- **Regression Visualisation**: Creates scatter plots with fitted regression lines to show relationships between key variables
- **Group Comparison Plots**: Generates bar charts with confidence intervals comparing high and low preference groups
- **Correlation Heatmaps**: Produces correlation matrices of all variables
- **Response Distribution Analysis**: Creates histograms showing the distribution of responses to the survey
- **Role-Based Comparison**: Generates boxplots showing how attitudes vary by practitioner role
- **Experience Trend Analysis**: Produces line plots showing how trust and transparency preferences change with experience level
- **AI Understanding Comparison**: Creates side-by-side boxplots comparing practitioners with high vs. low AI understanding

### Report Generation
- **Executive Summary**: Provides an overview of key findings
- **Statistical Highlights**: Reports statistical measures (r-squared, p-values, effect sizes)
- **Practitioner Segmentation**: Identifies which types show highest/lowest trust in transparent systems

## Research Foundation

This simulation is based on research by:
- Sibbald et al. (2022) on clinician experiences with Electronic Diagnostic Support
- Townsend et al. (2023) on Diagnostic AI Systems for Robot-Assisted Triage
- Nord-Bronzyk et al. (2025) on ethical issues of AI limitation disclosure
- Soltan et al. (2022) on AI systems that communicate confidence levels
- Feretzakis et al. (2024) on detailed explanations from AI systems

## License

MIT
