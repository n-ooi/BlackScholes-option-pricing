import streamlit as st
from numpy import log, sqrt, exp
from scipy.stats import norm

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
            value=105.0,
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
            value=0.25,
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

if __name__ == "__main__":
    main()