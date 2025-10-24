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
        denominator = volatility * sqrt(time_to_maturity)
        d1 = numerator / denominator
        return d1

    def calc_d2(self):
        d2 = self.calc_d1() - volatility * sqrt(time_to_maturity)
        return d2

    def calc_call(self):
        left = (norm.cdf(self.calc_d1()) * spot)
        right = (norm.cdf(self.calc_d2()) * strike * exp(-1 * self.interest * self.time_to_maturity))
        call_price = left - right
        return call_price

    def calc_put(self):
        pass

if __name__ == "__main__":
    spot = 100              # current price
    strike = 90             # price avaliable to exercise option
    interest = 0.04         # annualised risk-free interest rate
    time_to_maturity = 3    # remaining life of the option in years
    volatility = 0.2        # degree of variation of a trading price series

    black_scholes = BlackScholes(spot, strike, interest, time_to_maturity, volatility)

    print(black_scholes.calc_call())
