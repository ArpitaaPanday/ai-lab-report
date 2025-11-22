import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt
from datetime import datetime

def get_data(ticker="AAPL", years=10):
    end = datetime.today()
    start = end.replace(year=end.year - years)
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    df.columns = [str(c).lower() for c in df.columns]
    close = [c for c in df.columns if "close" in c][0]
    df = df.rename(columns={close: "close"})
    df["ret"] = df["close"].pct_change()
    return df.dropna()

def train_hmm(returns, n_states=2):
    X = returns.reshape(-1, 1)
    model = GaussianHMM(n_components=n_states, n_iter=500, covariance_type="full", random_state=42)
    model.fit(X)
    states = model.predict(X)
    return model, states

def plot_states(df, states, ticker):
    df["state"] = states
    plt.figure(figsize=(14, 7))
    for s in np.unique(states):
        m = df["state"] == s
        plt.plot(df.index[m], df["close"][m], label=f"State {s}")
    plt.legend()
    plt.title(f"{ticker} Price by Hidden States")
    plt.savefig(f"{ticker}_price_states.png", dpi=300)
    plt.close()
    plt.figure(figsize=(14, 5))
    plt.scatter(df.index, df["ret"], c=states, cmap="coolwarm", s=10)
    plt.title("Returns by State")
    plt.savefig(f"{ticker}_returns_states.png", dpi=300)
    plt.close()

def report(model, states):
    print("\nTransition Matrix:\n", model.transmat_)
    for i in range(model.n_components):
        print(f"State {i}: mean={model.means_[i][0]:.5f}, var={model.covars_[i][0][0]:.6f}")
    print("\nLast State:", states[-1])
    print("Next State Probabilities:", model.transmat_[states[-1]])

def main():
    ticker = "AAPL"
    df = get_data(ticker)
    model, states = train_hmm(df["ret"].values)
    report(model, states)
    plot_states(df, states, ticker)

if __name__ == "__main__":
    main()
