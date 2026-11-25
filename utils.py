"""
utils.py
--------
Outils pour le projet de séries temporelles.

Classes :
    - DataDownload : téléchargement des cours, rendements log, découpage train / test
    - ARMA         : sélection, estimation et prévision d'un ARMA(p, q)
    - AR           : cas particulier ARMA(p, 0)
    - MA           : cas particulier ARMA(0, q)
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def _finish_plot(title, ylabel):
    """Mise en forme commune des graphiques."""
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(ylabel)
    plt.legend(loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


class DataDownload:
    """
    Télécharge l'historique d'un actif via yfinance.
        1. download()   -> la série étudiée est le prix de clôture
        2. to_returns() -> (optionnel) la série devient 100 * ln(P_t / P_{t-1})
        3. split()      -> découpage train / test
    """

    PRICE = "Prix de clôture"
    RETURN = "Rendement log (%)"

    def __init__(self, ticker, start, end):
        self.ticker, self.start, self.end = ticker, start, end
        self.df = self.series = self.train = self.test = None

    def download(self):
        """Télécharge les cours bruts ; la série étudiée est le prix de clôture."""
        self.df = yf.Ticker(self.ticker).history(start=self.start, end=self.end)
        if self.df.empty:
            raise ValueError(f"Aucune donnée pour {self.ticker}.")
        self.series = self.df["Close"].rename(self.PRICE)
        print(f"{self.ticker} : {len(self.df)} obs. ({self.start} → {self.end})")
        return self.df

    def to_returns(self):
        """La série étudiée devient le rendement log en % (première date retirée)."""
        self.df["Log_Return_Pct"] = 100 * np.log(self.df["Close"]).diff()
        self.df = self.df.dropna(subset=["Log_Return_Pct"])
        self.series = self.df["Log_Return_Pct"].rename(self.RETURN)
        return self.series

    def split(self, train_end, test_start):
        """Train = [start, train_end], test = [test_start, end]."""
        self.train = self.series.loc[:train_end]
        self.test = self.series.loc[test_start:]
        for nom, s in (("Train", self.train), ("Test ", self.test)):
            print(f"{nom} : {len(s)} obs. ({s.index[0].date()} → {s.index[-1].date()})")
        return self.train, self.test

    def adf_test(self):
        """Test de Dickey-Fuller augmenté sur le train. H0 : racine unitaire."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, pval, lags, _, crit, _ = adfuller(self.train.values, autolag="AIC")
        verdict = "non stationnaire (H0 non rejetée)" if pval > 0.05 else "stationnaire (H0 rejetée)"
        print(f"ADF sur {self.train.name} : {verdict} au seuil de 5 %")
        return pd.Series({"Statistique ADF": stat, "p-valeur": pval, "Lags utilisés": lags,
                          "Valeur critique 5 %": crit["5%"]}).round(4).to_frame("ADF")

    def plot(self, title):
        """Trace la série étudiée (train / test si le découpage est fait)."""
        plt.figure(figsize=(12, 5))
        if self.train is not None:
            plt.plot(self.train, label="Train")
            plt.plot(self.test, color="tab:orange", label="Test")
            plt.axvline(self.test.index[0], color="black", linestyle=":")
        else:
            plt.plot(self.series, label=self.series.name)
        if self.series.name == self.RETURN:
            plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        _finish_plot(title, self.series.name)


class ARMA:
    """
    Modèle ARMA(p, q), ordre k = (p, q) :
        X_t = c + sum_i phi_i X_{t-i} + eps_t + sum_j theta_j eps_{t-j},   eps_t ~ BB(0, sigma^2)
    """

    def __init__(self, train, test):
        self.train, self.test = train, test
        self.k = self.model = self.scores = None

    # Seules méthodes redéfinies par AR et MA
    def _order(self, k):
        return (k[0], 0, k[1])

    def _name(self, k):
        return f"ARMA({k[0]},{k[1]})"

    @property
    def label(self):
        return self._name(self.k)

    def _fit(self, k):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return ARIMA(self.train.values, order=self._order(k)).fit()

    # ------------------------------------------------------------------ #
    # Sélection de modèle
    # ------------------------------------------------------------------ #
    def _correlogram(self, func, plot_func, nlags, **kw):
        fig, ax = plt.subplots(figsize=(10, 4))
        plot_func(self.train.values, lags=nlags, ax=ax, zero=False, **kw)
        ax.set_title(f"{func.__name__.upper()} de la série (train) - IC à 95 %")
        ax.grid(True, linestyle="--", alpha=0.6)
        plt.show()
        val, ic = func(self.train.values, nlags=nlags, alpha=0.05, **kw)
        lags = [k for k in range(1, nlags + 1) if abs(val[k]) > ic[k, 1] - val[k]]
        print(f"Lags significatifs (hors IC 95 %) : {lags or 'aucun'}")
        return lags

    def plot_acf(self, nlags=20):
        """ACF empirique du train ; renvoie les lags significatifs."""
        return self._correlogram(acf, plot_acf, nlags)

    def plot_pacf(self, nlags=20):
        """PACF empirique du train ; renvoie les lags significatifs."""
        return self._correlogram(pacf, plot_pacf, nlags, method="ywm")

    def select(self, orders, criterion="bic"):
        """Estime chaque ordre, renvoie AIC / BIC et retient le minimum de `criterion`."""
        fits = {k: self._fit(k) for k in orders}
        self.scores = pd.DataFrame({"AIC": [f.aic for f in fits.values()],
                                    "BIC": [f.bic for f in fits.values()]},
                                   index=pd.Index([self._name(k) for k in fits], name="Modèle")).round(2)
        for c in ("aic", "bic"):
            print(f"Meilleur selon le {c.upper()} : {self._name(min(fits, key=lambda k: getattr(fits[k], c)))}")
        self.k = min(fits, key=lambda k: getattr(fits[k], criterion))
        self.model = fits[self.k]
        print(f"-> Modèle retenu ({criterion.upper()}) : {self.label}")
        return self.scores

    def use(self, k):
        """Estime et retient le modèle d'ordre k."""
        self.k, self.model = k, self._fit(k)
        return self

    # ------------------------------------------------------------------ #
    # Modèle retenu
    # ------------------------------------------------------------------ #
    def summary(self):
        """Équation estimée et tableau des paramètres."""
        m = self.model
        mu, sigma2 = m.params[0], m.params[-1]
        c = mu * (1 - m.arparams.sum())

        def terms(coefs, var):
            return "".join(f" {'+' if v >= 0 else '-'} {abs(v):.4f} {var}_(t-{i})"
                           for i, v in enumerate(coefs, 1))

        unit = " %" if self.train.name == DataDownload.RETURN else ""
        print(f"Modèle retenu : {self.label}")
        print(f"  X_t = {c:.4f}{terms(m.arparams, 'X')} + eps_t{terms(m.maparams, 'eps')},"
              f"   eps_t ~ BB(0, {sigma2:.4f})")
        print(f"  Moyenne mu = {mu:.4f}{unit}")
        print(f"  Écart-type du bruit sigma = {np.sqrt(sigma2):.4f}{unit}\n")

        names = [n.replace("const", "mu (const)").replace("ar.L", "phi_").replace("ma.L", "theta_")
                 for n in m.param_names]
        return pd.DataFrame({"Estimation": m.params, "Écart-type": m.bse, "p-valeur": m.pvalues},
                            index=names).round(4)

    def roots(self):
        """Racines des polynômes AR et MA (module > 1 : stationnaire / inversible)."""
        rows = [("AR  Phi(z)", r) for r in self.model.arroots] + \
               [("MA  Theta(z)", r) for r in self.model.maroots]
        df = pd.DataFrame(rows, columns=["Polynôme", "Racine"])
        df["Module"] = df["Racine"].abs()
        return df.round(4)

    # ------------------------------------------------------------------ #
    # Prévision
    # ------------------------------------------------------------------ #
    def forecast(self, alpha=0.05):
        """Prévision sur tout le test avec IC à (1 - alpha)."""
        fc = self.model.get_forecast(steps=len(self.test))
        ic = fc.conf_int(alpha=alpha)
        return pd.DataFrame({"Prévision": fc.predicted_mean, "IC_inf": ic[:, 0], "IC_sup": ic[:, 1]},
                            index=self.test.index)

    def plot_forecast(self, title, n_train_last=60):
        """Fin du train, test réel et prévision avec son IC."""
        pred = self.forecast()
        plt.figure(figsize=(13, 5))
        plt.plot(self.train.iloc[-n_train_last:], color="tab:blue", alpha=0.8, label="Train (réel)")
        plt.plot(self.test, color="tab:gray", label="Test (réel)")
        plt.plot(pred["Prévision"], color="tab:red", linewidth=2.2, label=f"Prévision {self.label}")
        plt.fill_between(pred.index, pred["IC_inf"], pred["IC_sup"], color="tab:red", alpha=0.12,
                         label="IC 95 %")
        plt.axvline(pred.index[0], color="black", linestyle=":")
        _finish_plot(title, self.train.name)

    def one_step(self):
        """Prévision glissante à 1 pas sur le test (paramètres du train, non ré-estimés)."""
        full = np.concatenate([self.train.values, self.test.values])
        pred = self.model.apply(full).get_prediction(start=len(self.train)).predicted_mean
        return pd.Series(pred, index=self.test.index, name="Prévision 1 pas")

    def plot_rolling(self, title):
        """Prévision glissante à 1 pas contre le test réel."""
        plt.figure(figsize=(13, 5))
        plt.plot(self.test, color="tab:gray", label="Test (réel)")
        plt.plot(self.one_step(), color="tab:red", linewidth=1.5, label=f"Prévision 1 pas {self.label}")
        plt.axhline(self.model.params[0], color="black", linestyle="--", linewidth=0.8, label="mu estimé")
        _finish_plot(title, self.train.name)

    # ------------------------------------------------------------------ #
    # Résultats
    # ------------------------------------------------------------------ #
    def evaluate(self):
        """RMSE, MAE, couverture de l'IC et dispersion de la prévision."""
        pred, y = self.forecast(), self.test.values
        err = y - pred["Prévision"].values
        couverture = ((y >= pred["IC_inf"]) & (y <= pred["IC_sup"])).mean()
        return pd.Series({
            "RMSE": np.sqrt(np.mean(err ** 2)),
            "MAE": np.mean(np.abs(err)),
            "Couverture IC 95 % (%)": 100 * couverture,
            "Prévision min": pred["Prévision"].min(),
            "Prévision max": pred["Prévision"].max(),
            "Écart-type de la prévision": pred["Prévision"].std(),
            "Moyenne empirique du train": self.train.mean(),
        }).round(4).to_frame(self.label)

    def compare_naive(self):
        """Prévision du modèle vs naïf (dernière valeur du train répétée)."""
        y = self.test.values
        preds = {self.label: self.forecast()["Prévision"].values,
                 "Naïf (dernière valeur)": np.full(len(y), self.train.iloc[-1])}
        return pd.DataFrame({nom: {"RMSE": np.sqrt(np.mean((y - f) ** 2)), "MAE": np.mean(np.abs(y - f)),
                                   "Prévision finale": f[-1], "Valeur réelle finale": y[-1]}
                             for nom, f in preds.items()}).round(2)

    def rolling_forecast(self):
        """RMSE et % de bon sens de variation : prévision 1 pas vs constante mu
        (et vs naïf X_{t-1} pour une série de prix)."""
        y = self.test.values
        veille = np.concatenate([[self.train.iloc[-1]], y[:-1]])
        is_returns = self.train.name == DataDownload.RETURN
        ref = 0.0 if is_returns else veille

        def scores(pred):
            return {"RMSE": np.sqrt(np.mean((y - pred) ** 2)),
                    "Bon sens de variation (%)": 100 * np.mean(np.sign(pred - ref) == np.sign(y - ref))}

        cols = {f"{self.label} glissant 1 pas": scores(self.one_step().values),
                "Constante mu": scores(np.full(len(y), self.model.params[0]))}
        if not is_returns:
            cols["Naïf (veille)"] = {**scores(veille), "Bon sens de variation (%)": np.nan}
        return pd.DataFrame(cols).round(4)

    def plot_price_from_returns(self, data, title, n_train_last=365):
        """Modèle sur rendements log (%) : prix prévu P_T * exp(cumsum(prévisions) / 100)
        comparé au prix réel. `data` est le DataDownload d'origine."""
        prix = data.df["Close"]
        prix_train = prix.loc[:self.train.index[-1]]
        prix_prevu = prix_train.iloc[-1] * np.exp(self.forecast()["Prévision"].cumsum() / 100)
        prix_test = prix.loc[self.test.index]

        plt.figure(figsize=(13, 5))
        plt.plot(prix_train.iloc[-n_train_last:], color="tab:blue", alpha=0.8, label="Prix train (réel)")
        plt.plot(prix_test, color="tab:gray", label="Prix test (réel)")
        plt.plot(prix_prevu, color="tab:red", linewidth=2.2, label=f"Prix reconstruit depuis {self.label}")
        plt.axvline(prix_prevu.index[0], color="black", linestyle=":")
        _finish_plot(title, DataDownload.PRICE)
        print(f"Prix final prévu : {prix_prevu.iloc[-1]:,.0f} | Prix final réel : {prix_test.iloc[-1]:,.0f}")


class AR(ARMA):
    """Modèle AR(p) : X_t = c + phi_1 X_{t-1} + ... + phi_p X_{t-p} + eps_t."""

    def _order(self, p):
        return (p, 0, 0)

    def _name(self, p):
        return f"AR({p})"


class MA(ARMA):
    """Modèle MA(q) : X_t = mu + eps_t + theta_1 eps_{t-1} + ... + theta_q eps_{t-q}."""

    def _order(self, q):
        return (0, 0, q)

    def _name(self, q):
        return f"MA({q})"
