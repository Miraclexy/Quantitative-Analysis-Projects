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

5. statistical arbitrage in the U.S equity market

Divide stock pool of S&P500 components into different groups based on sectors.
For each sector:

(1): use ETF in this sector as market risk factor

(2): use stocks’ daily returns data to do OLS regression on corresponding ETF’s daily returns; get correlation beta and residual. Assume that the residual we get in the last process follows mean reverting O-U process, so we use test data to get the parameters in O-U process. In discrete time, O-U process becomes AR(1) model, so in practice, we use AR(1) model.

(3): use test data in order to get parameters in O-U process and get s_score which is the trading signal.

(4): define trading strategy. Calculate cumulative returns and benchmark’s cumulative returns.

(5) plot the cumulative returns.

6. frequency arbitrage

Construct two portfolios trading options and delta hedge daily and weekly respectively based on the difference of the volatility signature of underlying asset with different time intervals;

Prove that we can only trade on the underlying asset to replicate the whole strategy by introducing log contract whose payoff is the log of St.

7. fractional brownian motion & hurst exponent

Assume Stock price follows Fractional Brownian motion with a Hurst exponent rather than Geometric Brownian motion, and find out the market feature(momentum/mean reverting/Brownian motion) based on the relationship between H and 0.5; try practical method to estimate Hurst exponent.

8. hedge ratios for spread trading

Show the potential problem about calculation hedge ratio in spread trading using OLS regression and point out a better way to do that which makes the hedge ratio symmetrica.
