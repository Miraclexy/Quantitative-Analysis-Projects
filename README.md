# Quantitative-Analysis-Projects
1. back-testing framework for factor models

Given factor data of stocks, stock prices, trading status, max up or down, frequency of position adjustment we want, quantities we want to divide stocks into and whether we want a long-short-position or a long-only-position, the framework can return cumulative return and compare it with universe or benchmark cumulative returns and show the plot.

2. factors

Factor descriptions are got from websites and research papers

3. double moving average timing strategy

Analyze the relationship between moving average with different rolling window period and stock prices; utilize short and long moving average and construct strategy based on the status of these two moving average on the CSI 300 index.

4. financial time series analysis based on hidden markov model and kalman filter

Produce Gaussian Hidden Markov model to predict hidden states of financial market; select and standardize characteristic variables using Box-Cox transformation; test model suitability among large cap stock, small cap stock and financial blue chip stock; 
initiated Kalman Filter model to optimize performance of HMM model by solving the problem of high noises in financial data; performed out-of-sample empirical analysis through Kalman+HMM combination model based on the SSE 50 index
