import streamlit as st
import numpy as np
from numpy import log, sqrt, exp
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from  matplotlib.colors import LinearSegmentedColormap

class BlackScholes:
    def __init__(self, spot: float, strike: float, interest: float, time_to_maturity: float, volatility: float):
        self.spot = spot
        self.strike = strike
        self.interest = interest
        self.time_to_maturity = time_to_maturity
        self.volatility = volatility

    def calc_d1(self):
        numerator = log(self.spot/self.strike) + (self.interest + (self.volatility**2 / 2)) * self.time_to_maturity
        denominator = self.volatility * sqrt(self.time_to_maturity)
        d1 = numerator / denominator
        return d1

    def calc_d2(self):
        d2 = self.calc_d1() - self.volatility * sqrt(self.time_to_maturity)
        return d2

    def calc_call(self):
        left = (norm.cdf(self.calc_d1()) * self.spot)
        right = (norm.cdf(self.calc_d2()) * self.strike * exp(-1 * self.interest * self.time_to_maturity))
        call_price = left - right
        return call_price

    def calc_put(self):
        left = self.strike * exp(-1 * self.interest * self.time_to_maturity) 
        right = self.spot - self.calc_call()
        put_price = left - right
        return put_price

def plot_heatmap(bs: BlackScholes, spot_range, vol_range, purchase_price, is_call: bool):
    "2D heatmap showing varying spot prices and volatility"
    pnls = np.zeros((len(vol_range), len(spot_range)))

    for i, vol in enumerate(vol_range):
        for j, spot in enumerate(spot_range):
            bs_result = BlackScholes(
                spot=spot,
                strike=bs.strike,
                interest=bs.interest,
                time_to_maturity=bs.time_to_maturity,
                volatility=vol
            )
        
            if is_call:
                pnls[i, j] = bs_result.calc_call() - purchase_price
            else:
                pnls[i, j] = bs_result.calc_put() - purchase_price

    # Plotting PnL Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#121212")
    ax.set_facecolor("#121212")  

    sns.heatmap(
        pnls, 
        xticklabels=np.round(spot_range, 2), 
        yticklabels=np.round(vol_range, 2), 
        annot=True, 
        fmt=".2f", 
        cmap=LinearSegmentedColormap.from_list('rg',["lightcoral", "white", "palegreen"], N=256) , 
        ax=ax
    )

    ax.set_title(('Call' if is_call else 'Put') + ' Option PnL', color="white")
    ax.set_xlabel('Spot Price', color="white")
    ax.set_ylabel('Volatility', color="white")
    
    return fig

def main():
    st.title("📈 Black-Scholes Option Pricer")
    
    # Sidebar for inputs
    st.sidebar.header("Option Parameters")
    
    with st.sidebar:
        spot = st.number_input(
            "Spot Price ($)",
            min_value=0.01,
            value=100.0,
            step=0.01,
        )
        
        strike = st.number_input(
            "Strike Price ($)",
            min_value=0.01,
            value=100.0,
            step=0.01,
        )
        
        interest = st.number_input(
            "Risk-Free Interest Rate (%)",
            min_value=0.0,
            max_value=100.0,
            value=5.0,
            step=0.1,
        ) / 100  # Convert percentage to decimal
        
        time_to_maturity = st.number_input(
            "Time to Maturity (Years)",
            min_value=0.01,
            value=1.0,
            step=0.01,
        )
        
        volatility = st.number_input(
            "Volatility (%)",
            min_value=0.01,
            max_value=500.0,
            value=20.0,
            step=0.1,
        ) / 100  # Convert percentage to decimal
    
    bs = BlackScholes(spot, strike, interest, time_to_maturity, volatility)
    call_price = bs.calc_call()
    put_price = bs.calc_put()
    
    with st.expander("💡 See Black-Scholes Formula..."):
        st.markdown(r"""
        ### Call Option
        $$
        C = S_0 N(d_1) - K e^{-rT} N(d_2)
        $$
                    
        ---

        ### Put Option
        $$
        P = K e^{-rT} N(-d_2) - S_0 N(-d_1)
        $$

        ---
        
        $$
        \text{where } d_1 = \frac{\ln\left(\frac{S_0}{K}\right) + \left(r + \frac{\sigma^2}{2}\right)T}{\sigma \sqrt{T}},
        \qquad
        \text{and } d_2 = d_1 - \sigma \sqrt{T}
        $$

        ---

        * $S_0$ = Current stock price  
        * $K$ = Strike price  
        * $T$ = Time to maturity  
        * $r$ = Risk-free interest rate  
        * $\sigma$ = Volatility  
        * $N(x)$ = Cumulative distribution function of the standard normal distribution
        """)

    st.subheader("📊 Option Prices")
        
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Call option
        st.metric(
            label="Call Option Price",
            value=f"${call_price:.2f}",
        )

    with col2:
        # Put option
        st.metric(
            label="Put Option Price",
            value=f"${put_price:.2f}",
        )

    st.divider()


    # PnL Heatmap
    spot_range = np.linspace(spot*0.5, spot*1.5, 10)
    vol_range = np.linspace(volatility*0.5, volatility*1.5, 10)

    col1, col2 = st.columns([1,1], gap="small")

    with col1:
        st.subheader("📞 Call - PnL Heatmap")
        call_purchase = st.number_input(
            "Call Option Purchase Price ($)",
            min_value=0.01,
            value=call_price,
            step=0.01,
        )
        heatmap_fig_call = plot_heatmap(bs, spot_range, vol_range, call_purchase, True)
        st.pyplot(heatmap_fig_call)

    with col2:
        st.subheader("🏷️ Put - PnL Heatmap")
        put_purchase = st.number_input(
            "Put Option Purchase Price ($)",
            min_value=0.01,
            value=put_price,
            step=0.01,
        )
        heatmap_fig_put = plot_heatmap(bs, spot_range, vol_range, put_purchase, False)
        st.pyplot(heatmap_fig_put)

if __name__ == "__main__":
    main()